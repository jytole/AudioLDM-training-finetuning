"""Script to define and facilite AudioLDM2 Interface Flask App

Current implementation serves the flask-based webapp through
gunicorn with an nginx proxy.
Serves rendered templates to user with some context included.
Connects to client for interactive content via flask_socketio.
Webapp connects to AudioLDM2 script via zmq.

**BAD PRACTICE**: cors_allowed_origins="*", but web security
is hard.


Expected environment variables:
    To connect "flash" messages and otherwise authorize client comms:
    FLASK_SECRET_KEY

    To enable WANDB logging and avoid failure:
    WANDB_API_KEY
"""

from flask import Flask, render_template, request, Response, send_file, flash, make_response
from werkzeug.utils import secure_filename
import os

# includes for zmq comms
import zmq, logging

# flask socket stuff
from flask_socketio import SocketIO
import threading
from flask_cors import CORS

import time

# includes for spawn new API
import subprocess, shutil

## init logging
logger = logging.getLogger(__name__)

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config.from_prefixed_env()
CORS(app)

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Torch server init variables
REQUEST_TIMEOUT = 2500
REQUEST_RETRIES = 3
SERVER_ENDPOINT = "tcp://localhost:5555"


def spawnAPIServer():
    """Method to spawn the torchServer, to be used if it crashes.

    Returns:
        Popen: Subprocess running `python webapp/torchServer.py`
    """
    p = subprocess.Popen(
        [shutil.which("python"), os.path.join(projectRoot, "webapp/torchServer.py")]
    )
    return p

# global app init stuff
with app.app_context():
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Connecting to AudioLDM2 process through zmq...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(SERVER_ENDPOINT)

    current_state = {"inferencePath": "fake.mp3", 
                     "displayInferenceAudio": False,
                     "params": {
                         "step,save_checkpoint_every_n_steps": 5000,
                         "reload_from_ckpt": "./data/checkpoints/audioldm-m-full.ckpt",
                         "model,params,evaluation_params,unconditional_guidance_scale": 3.5,
                         "model,params,evaluation_params,ddim_sampling_steps": 200,
                         "model,params,evaluation_params,n_candidates_per_samples": 3,
                     },
                    }

## Socket handling code
# TODO figure out why socket closes on all submitted forms (?)
    
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("connect")  # log all new clients
def handle_connect():
    """log a message when client connects"""
    logger.info("Client connected to SocketIO")
    
@socketio.on("debug")
def debugSocket():
    """log socketio debug message"""
    logger.info("Socket Debug Triggered")

# https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file
def follow(logFile):
    """Helper function to monitor a constantly growing file.
    
    Yields lines of the file as they come in.

    Args:
        logFile (stream): File to be monitored

    Yields:
        lines: Lines, as they arrive. A "generator"
    """
    logFile.seek(0,2)
    while True:
        line = logFile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line
    
# TODO consider what messages to look for in a log, maybe progress updates?
def torchServer_monitor():
    """Monitor the latest torchServer log file, emit updates to clients

    Returns:
        success: boolean value
    """
    logDir = os.path.join(projectRoot, "webapp/logs")
    files = os.listdir(logDir)
    if len(files) <= 0:
        return False
    paths = []
    for basename in files:
        if "torchServer-" in basename:
            paths.append(os.path.join(logDir, basename))
    logFilePath = max(paths, key=os.path.getctime)
    
    logger.info("monitoring: " + logFilePath)
    
    logFile = open(logFilePath)
    logLines = follow(logFile)
    # socketio.emit("monitor", "Monitoring torch log file!")
    for line in logLines:
        if "CUDA is not available" in line:
            logger.info("monitor found CUDA fail")
            socketio.emit("monitor", "Operation Failed! CUDA is not available.")
            return False
        elif "Traceback (most recent call last):" in line:
            logger.info("monitor found traceback fail")
            socketio.emit("monitor", "Traceback found. Likely crash.")
            return False
            
def watch_torchServer():
    """Start a thread with the torchServer monitor;
    allows user to continue interacting with the webapp 

    Returns:
        success: True when begun
    """
    torchServer_watch_thread = threading.Thread(target=torchServer_monitor, daemon=True)
    torchServer_watch_thread.start()
    return True

def wait_for_inference(attempt_limit=100):
    """Poll the torchServer for inference completion 
    
    Timeout is ATTEMPT_LIMIT * 2 seconds
    
    Args:
        attempt_limit (int): Number of times to poll the server for completion, defaults to 100

    Returns:
        success: boolean value
    """
    # server polling loop
    waveformpath = "nack"
    serverComplete = False
    attempts = 0
    while not serverComplete:
        time.sleep(2)
        logger.info("Waiting for inference to complete...")
        waveformpath = getFromServer("checkInferenceComplete", retries=1)
        if waveformpath != "nack":
            logger.info("Inference complete")
            serverComplete = True
        if attempts > attempt_limit:
            return False
        attempts += 1

    # update current state when waveformpath is received
    current_state["inferencePath"] = os.path.join(
        "static", os.path.basename(waveformpath)
    )  # set relative path to static/filename
    shutil.copy(
        waveformpath,
        os.path.join(projectRoot, "webapp", current_state["inferencePath"]),
    )  # copy inferred file to static
    current_state["displayInferenceAudio"] = (
        True  # tell browser to display the audio
    )
    logger.info("inference path updated: " + current_state["inferencePath"])
    
    emitCurrentState()
    socketio.emit("monitor", "Inference complete!")
    flash("Inference complete!")
    return True

def emitCurrentState():
    """emits the current_state dictionary"""
    socketio.emit("current_state_update", current_state)

def sendToServer(message, retries=REQUEST_RETRIES):
    """Sends message to torchServer, expect "ack" in return

    Args:
        message (str): Message to be sent to the torchServer via zmq
        retries (int, optional): Number of times to try resending message. Defaults to REQUEST_RETRIES (3).

    Returns:
        success: boolean flag
    """
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
        # Socket is not answering. Close and remove it.
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
def getFromServer(message, retries=REQUEST_RETRIES):
    """Sends message to torchServer, expect "ack;<data>" in return

    Args:
        message (str): Message to be sent to the torchServer via zmq
        retries (int, optional): Number of times to try resending message. Defaults to REQUEST_RETRIES (3).

    Returns:
        success: boolean flag
    """
    global socket
    socket.send_string(message)

    retries_left = retries
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
        # Socket is not answering. Close and remove it.
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
    """http request handler for index (/)
    
    Renders template with current_state; answers requests in the form of forms

    Returns:
        response: http response containing the rendered template
    """
    # handle all the forms
    if request.method == "POST":
        if "archiveUploadForm" in request.form:
            logger.debug("archiveUploadForm")
            archiveUpload()
        elif "setParameterForm" in request.form:
            logger.debug("setParameterForm")
            setParameter()
        elif "startFineTuningForm" in request.form:
            logger.debug("startFineTuningForm")
            startFineTuning()
        elif "inferSingleForm" in request.form:
            logger.debug("inferSingleForm")
            inferSingle()
        elif "downloadCheckpointLatestForm" in request.form:
            logger.debug("downloadCheckpointLatestForm")
            downloadCheckpointLatest()
        elif "debugForm" in request.form:
            logger.debug("debugForm")
            debugFunc()
            
    rendered_template = render_template("index.html", current_state=current_state)
    response = make_response(rendered_template)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    return response


## handle the upload of an archive file through a submitted form
## saves the file, then calls an unzip helper function
### REQUIRES: werkzeug.utils.secure_filename, flask.request
def archiveUpload():
    """Function to handle the upload of a data archive

    Returns:
        success: boolean flag
    """
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            fileName = secure_filename(file.filename)
            savePath = "./webapp/cache/" + fileName
            file.save(savePath)

            if sendToServer("handleDataUpload;" + savePath):
                flash("successful upload")
                return True
            else:
                flash("failed upload")
                return False

    flash("No file uploaded")
    return False

def setParameter():
    """Sets the parameter requested in the form
    
    Expects a form submitted containing "parameter" and "value" fields

    Returns:
        success: boolean flag
    """
    paramPathInput = request.form["parameter"]
    valInput = request.form["value"]
    
    message = "set_parameter;" + paramPathInput + ";" + valInput

    if sendToServer(message):
        current_state["params"][paramPathInput] = valInput
        emitCurrentState()
        flash("Successfully set parameter")
        return True
    else:
        flash("Failed to set parameter")
    return False


def startFineTuning():
    """Start the finetuning process on torchServer.
    
    Flashes message(s) with progress updates

    Returns:
        success: boolean flag
    """
    if sendToServer("finetune"):
        flash("Fine tuning started.")
        watch_torchServer()
        return True
    else:
        flash("AudioLDM2 not available. Is it running?")
        return False

def inferSingle():
    """Perform a single inference on torchServer
    
    Shares the audiofile in current_state

    Returns:
        success: boolean flag
    """
    prompt = request.form["prompt"]
    # waveformpath = getFromServer("inferSingle;PROMPT:" + prompt) # TODO handle long render times (sendToServer?)
    
    sendToServer("inferSingle;PROMPT:" + prompt)
    
    wait_for_inference_thread = threading.Thread(target=wait_for_inference, daemon=True)
    wait_for_inference_thread.start()

    flash("Inference begun")
    return True

## TODO add possibility of checkpoint handling being google cloud based, rather than huge file transfer
def downloadCheckpointLatest():
    """Download the latest checkpoint

    Returns:
        send_file: a file transfer for the client, handled by flask
    """
    path = getFromServer("prepareCheckpointDownload")
    if not path:
        flash("No checkpoint available")
        return False
    checkpointPath = os.path.join(
        projectRoot, path
    )
    return send_file(checkpointPath)

def debugFunc():
    """debug function to show how requests are acting

    Returns:
        success: boolean flag
    """
    sendToServer("debug")
    socketio.emit("debug")
    return True