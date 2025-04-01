from flask import Flask, render_template, request, Response, send_file, flash
from werkzeug.utils import secure_filename
import os
# includes for zmq comms
import zmq, logging, sys
# includes for spawn new API
import subprocess, shutil

## init logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config.from_prefixed_env()

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

# Server init variables
REQUEST_TIMEOUT = 2500
REQUEST_RETRIES = 3
SERVER_ENDPOINT = "tcp://localhost:5555"

def spawnAPIServer():
    p = subprocess.Popen([shutil.which("python"), os.path.join(projectRoot, "webapp/torchServer.py")])
    return p

with app.app_context():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("Connecting to AudioLDM2 process through zmq...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(SERVER_ENDPOINT)
    
    variableElements = {"inferencePath": "fake.mp3", "displayInferenceAudio": False}
    
def sendToServer(message, retries=REQUEST_RETRIES):
    global socket
    socket.send_string(message)
    
    retries_left = retries
    
    while True:
        if (socket.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
            reply = socket.recv_string()
            if reply == "ack":
                logging.info("Server replied OK (%s)", reply)
                retries_left = REQUEST_RETRIES
                return True
            else:
                logging.error("Failed: %s", reply)
                return True

        retries_left -= 1
        logging.warning("No response from server")
        # Socket is confused. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        
        logging.info("Reconnecting to server…")
        # Create new connection
        socket = context.socket(zmq.REQ)
        socket.connect(SERVER_ENDPOINT)
        logging.info("Resending (%s)", request)
        
        if retries_left == 0:
            logging.error("Server seems to be offline, abandoning request")
            return False
            
        socket.send_string(message)
        
## Send a message to server and expect a parseable response
## assume server will respond with ack;REST_OF_MESSAGE
## Only returns up until the next ";"
def getFromServer(message):
    global socket
    socket.send_string(message)
    
    retries_left = REQUEST_RETRIES
    while True:
        if (socket.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
            reply = socket.recv_string()
            replyArr = reply.split(";")
            if replyArr[0] == "ack":
                logging.info("Server replied OK (%s)", reply)
                retries_left = REQUEST_RETRIES
                return replyArr[1]
            else:
                logging.error("Did not receive ack: %s", reply)
                return False

        retries_left -= 1
        logging.warning("No response from server")
        # Socket is confused. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        
        logging.info("Reconnecting to server…")
        # Create new connection
        socket = context.socket(zmq.REQ)
        socket.connect(SERVER_ENDPOINT)
        logging.info("Resending (%s)", request)
        
        if retries_left == 0:
            logging.error("Server seems to be offline, abandoning request")
            return False
            
        socket.send_string(message)


@app.route("/", methods=["GET", "POST"])
def index():
    # handle all the forms
    if request.method == "POST":
        if "archiveUploadForm" in request.form:
            print("archiveUploadForm")
            archiveUpload()
        elif "setParameterForm" in request.form:
            print("setParameterForm")
            setParameter()
        elif "startFineTuningForm" in request.form:
            print("startFineTuningForm")
            startFineTuning()
        elif "inferSingleForm" in request.form:
            print("inferSingleForm")
            inferSingle()
        elif "downloadCheckpointLatestForm" in request.form:
            print("downloadCheckpointLatestForm")
            downloadCheckpointLatest()
        elif "restartAPIForm" in request.form:
            print("restartAPIForm")
            restartAPIServer()
        elif "debugForm" in request.form:
            print("debugForm")
            debugFunc()
    return render_template("index.html", variableElements=variableElements)

## handle the upload of an archive file through a submitted form
## saves the file, then calls an unzip helper function
### REQUIRES: werkzeug.utils.secure_filename, flask.request
def archiveUpload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            fileName = secure_filename(file.filename)
            savePath = './webapp/cache/' + fileName
            file.save(savePath)
            
            if(sendToServer("handleDataUpload;" + savePath)):
                flash("successful upload")
                return True
            else:
                flash("failed upload")
                return False

    flash("No file uploaded")
    return False

def setParameter():
    textInput = request.form['save_checkpoint_every_n_steps']
    
    if(sendToServer("set_parameter;step,save_checkpoint_every_n_steps;" + textInput)):
        return "Successfully set parameter"
    return "Failed to set parameter"

def startFineTuning():
    sendToServer("finetune")
    return "Fine tuning started. Please reference torchServer.log for progress."

def inferSingle():
    prompt = request.form['prompt']
    waveformpath = getFromServer("inferSingle;PROMPT:" + prompt)
    
    if not waveformpath:
        return False
    
    variableElements["inferencePath"] = os.path.join("static", os.path.basename(waveformpath)) # set relative path to static/filename
    shutil.copy(waveformpath, os.path.join(projectRoot, "webapp", variableElements["inferencePath"])) # copy inferred file to static
    variableElements["displayInferenceAudio"] = True # tell browser to display the audio
    
    return True

def downloadCheckpointLatest():
    checkpointPath = os.path.join(projectRoot,getFromServer("prepareCheckpointDownload"))
    return send_file(checkpointPath)

def restartAPIServer():
    if(sendToServer("ping", 5)):
        return "API still running"
    else:
    ## boot the API?
        apiServerProcess = spawnAPIServer()
        return "API not responding. New API instance booted"
    
def debugFunc():
    sendToServer("debug")
    return True

# if __name__ == "__main__":
#     app.run(debug=True)