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
                     "inferencePrompt": "fake prompt",
                     "params": {
                        "seed": 0,
                        "step,validation_every_n_epochs": 5,
                        "step,save_checkpoint_every_n_steps": 5000,
                        "reload_from_ckpt": "./data/checkpoints/audioldm-m-full.ckpt",
                        "model,params,evaluation_params,unconditional_guidance_scale": 3.5,
                        "model,params,evaluation_params,ddim_sampling_steps": 200,
                        "model,params,evaluation_params,n_candidates_per_samples": 3,
                        "preprocessing,audio,duration": 10.24,
                     },
                     "datasets": [
                         
                     ],
                     "checkpoints": [
                        "./data/checkpoints/audioldm-m-full.ckpt",
                     ],
                     "inferenceCheckpoints": [
                        "./log/latent_diffusion/2025_03_27_api_default_finetune/default_finetune/checkpoints/checkpoint-fad-133.00-global_step=4999.ckpt",
                     ],
                     "tab": "finetune",
                     "monitor": {
                        "torchServerStatus": "idle",
                        "epoch": -1,
                        "ddimStep": -1,
                        "globalStep": -1,
                     },
                    }

## Socket handling code
# TODO figure out why socket closes on all submitted forms (?)
    
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def connect(sid):
    logger.info('connect ' + str(sid))

@socketio.on('tab_change')
def tab_change(data):
    current_state["tab"] = data
    emitCurrentState()
    logger.info('tab_change ' + str(data))
    # socketio.emit('my response', {'response': 'my response'})
    
@socketio.on('disconnect')
def disconnect(sid):
    logger.info('disconnect ' + str(sid))

@socketio.on("scanSystem")
def scanSystem():
    logger.debug("Socket scanSystem")
    scanFileSystem()

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
def torchServer_monitor(timeout=100):
    """Monitor the latest torchServer log file, emit updates to clients
    
    Args:
        timeout (int): Seconds torchServer must not print to the log before monitor deems it to be idle

    Returns:
        success: boolean value
    """
    global current_state, socketio, projectRoot, logger
    
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
    
    current_state["monitor"]["epoch"] = -1
    current_state["monitor"]["ddimStep"] = -1
    current_state["monitor"]["globalStep"] = -1
    # t = threading.Timer(timeout, logLines.close())
    # t.start()
    for line in logLines:
        # t.cancel()
        if "CUDA is not available" in line:
            logger.info("monitor found CUDA fail")
            socketio.emit("monitor", "Operation Failed! CUDA is not available.")
            logLines.close()
            current_state["monitor"]["torchServerStatus"] = "crashed"
            socketio.emit("current_state_update", current_state)
            return
        elif "Epoch" in line:
            postEmit = False
            numStart = line.find("Epoch ") + 6
            numEnd = line.find(":", numStart)
            epochNew = int(line[numStart:numEnd])
            if epochNew != current_state["monitor"]["epoch"]:
                current_state["monitor"]["epoch"] = epochNew
                postEmit = True
            if "global_step=" in line:
                numStart = line.find("global_step=") + 12
                numEnd = line.find(".", numStart)
                globalStepNew = int(line[numStart:numEnd])
                if globalStepNew != current_state["monitor"]["globalStep"]:
                    current_state["monitor"]["globalStep"] = globalStepNew
                    postEmit = True
            if postEmit:
                socketio.emit("current_state_update", current_state)
        elif "DDIM Sampler" in line:
            if "/" in line:
                numEnd = line.find("/")
                numStart = line.rfind(" ", 0, numEnd) + 1
                ddimStepNew = int(line[numStart:numEnd])
                if ddimStepNew != current_state["monitor"]["ddimStep"]:
                    current_state["monitor"]["ddimStep"] = ddimStepNew
                    socketio.emit("current_state_update", current_state)
        elif "Traceback (most recent call last):" in line:
            logger.info("monitor found traceback fail")
            socketio.emit("monitor", "Traceback found. Likely crash.")
            logLines.close()
            current_state["monitor"]["torchServerStatus"] = "crashed"
            socketio.emit("current_state_update", current_state)
            return
        elif "Received request: " in line:
            logger.info("torchServer idle, accepting requests again")
            socketio.emit("monitor", "Operation complete!")
            logLines.close()
        # t = threading.Timer(timeout, logLines.close())
        # t.start()
    
    current_state["monitor"]["torchServerStatus"] = "idle"
    socketio.emit("current_state_update", current_state)
    
            
def watch_torchServer(timeout=100):
    """Start a thread with the torchServer monitor;
    allows user to continue interacting with the webapp 
    
    Args:
        timeout (int): Seconds torchServer must not print to the log before monitor deems it to be idle

    Returns:
        success: True when begun
    """
    torchServer_watch_thread = threading.Thread(target=torchServer_monitor, args=[timeout], daemon=True)
    torchServer_watch_thread.start()
    return True

def wait_for_inference(attempt_limit=5000):
    """Poll the torchServer for inference completion 
    
    Timeout is ATTEMPT_LIMIT * 5 seconds
    
    Args:
        attempt_limit (int): Number of times to poll the server for completion, defaults to 100

    Returns:
        success: boolean value
    """
    # server polling loop
    waveformpath = False
    serverComplete = False
    attempts = 0
    while not serverComplete:
        time.sleep(5)
        logger.info("Waiting for inference to complete...")
        waveformpath = getFromServer("checkInferenceComplete", retries=1)
        if waveformpath and waveformpath != "nack":
            logger.info("Inference complete (n=" + str(attempts) + ")")
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
    current_state["displayInferenceAudio"] = True  # tell browser to display the audio
    logger.info("inference path updated: " + current_state["inferencePath"])
    
    current_state["monitor"]["torchServerStatus"] = "idle"
    
    emitCurrentState()
    socketio.emit("monitor", "Inference complete!")
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
                logger.debug("Server replied OK (%s)", reply)
                retries_left = REQUEST_RETRIES
                return True
            else:
                logger.debug("Failed: %s", reply)
                return True

        retries_left -= 1
        logger.debug("No response from server")
        # Socket is not answering. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()

        logger.debug("Reconnecting to server…")
        # Create new connection
        socket = context.socket(zmq.REQ)
        socket.connect(SERVER_ENDPOINT)
        logger.debug("Resending (%s)", message)

        if retries_left == 0:
            logger.debug("Server seems to be offline, abandoning request")
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
                logger.debug("Server replied OK (%s)", reply)
                retries_left = REQUEST_RETRIES
                return replyArr[1]
            else:
                logger.debug("Did not receive ack: %s", reply)
                return False

        retries_left -= 1
        logger.debug("No response from server")
        # Socket is not answering. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()

        logger.info("Reconnecting to server…")
        # Create new connection
        socket = context.socket(zmq.REQ)
        socket.connect(SERVER_ENDPOINT)
        logger.debug("Resending (%s)", message)

        if retries_left == 0:
            logger.debug("Server seems to be offline, abandoning request")
            return False

        socket.send_string(message)


@app.route("/", methods=["GET", "POST"])
def index():
    """http request handler for index (/)
    
    Renders template with current_state; answers requests in the form of forms

    Returns:
        response: http response containing the rendered template
    """     
    rendered_template = render_template("index.html", current_state=current_state)
    response = make_response(rendered_template)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    return response

@app.route("/finetuning", methods=["GET", "POST"])
def finetuningPage():
    """http request handler for finetuning (/finetuning)
    
    Renders template with current_state

    Returns:
        response: http response containing the rendered template
    """
    if request.method == "POST":
        if "archiveUploadForm" in request.form:
            logger.debug("/finetuning: archiveUploadForm")
            archiveUpload() # process dataset
        elif "startFineTuningBulkForm" in request.form:
            logger.debug("/finetuning: startFineTuningBulkForm")
            bulkParamForm() # Handle parameters from form
            startFineTuning() # Sets checkpoint and begins fine-tuning 
            ## Handle checkpoint, parameters, and begin finetuning
        elif "startEvalForm" in request.form:
            logger.debug("/finetuning: startEvalForm")
            startEval()
        elif "downloadCheckpointLatestForm" in request.form:
            logger.debug("/finetuning: downloadCheckpointLatestForm")
            downloadCheckpointLatest()
    rendered_template = render_template("finetuning.html", current_state=current_state)
    response = make_response(rendered_template)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    return response

@app.route("/inference", methods=["GET", "POST"])
def inferencePage():
    """http request handler for finetuning (/finetuning)
    
    Renders template with current_state

    Returns:
        response: http response containing the rendered template
    """
    if request.method == "POST":
        if "startFineTuningForm" in request.form:
            logger.debug("/inference: startFineTuningForm")
        elif "beginInferenceBulkForm" in request.form:
            logger.debug("/inference: beginInferenceBulkForm")
            bulkParamForm()
            inferSingle()
    rendered_template = render_template("inference.html", current_state=current_state)
    response = make_response(rendered_template)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    return response

@app.route("/dev", methods=["GET", "POST"])
def devPage():
    """http request handler for the dev interface (/dev)
    
    Renders template with current_state

    Returns:
        response: http response containing the rendered template
    """
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
        elif "scanFileSystemForm" in request.form:
            logger.debug("scanFileSystemForm")
            scanFileSystem()
        elif "processImportedDatasetForm" in request.form:
            logger.debug("processImportedDatasetForm")
            processImportedDataset()
        elif "startEvalForm" in request.form:
            logger.debug("startEvalForm")
            startEval()
        elif "debugForm" in request.form:
            logger.debug("debugForm")
            debugFunc()
    rendered_template = render_template("dev.html", current_state=current_state)
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
                flash("Upload begun. Please wait.")
                current_state["monitor"]["torchServerStatus"] = "processing dataset"
                emitCurrentState()
                watch_torchServer(10) # watch the server, call it idle after 10 sec inactivity
                return True
            else:
                flash("Failed upload")
                return False

    flash("No file uploaded")
    return False

def processImportedDataset():
    """Function to handle the processing of an imported dataset

    Returns:
        success: boolean flag
    """
    savePath = "./webapp/static/datasets/" + request.form["importedDatasetZip"]
    if sendToServer("handleDataUpload;" + savePath):
        flash("Processing begun. Please wait.")
        current_state["monitor"]["torchServerStatus"] = "processing dataset"
        emitCurrentState()
        watch_torchServer(10) # watch the server, call it idle after 10 sec inactivity
        return True
    else:
        flash("Failed to reach server.")
        return False

def scanFileSystem():
    """Function to scan the filesystem for imports (datasets, checkpoints) and
    change the current_state to reflect the results.
    Emits current_state after scan.
    
    Checks static/checkpoints and static/datasets for files.
    
    Returns:
        success: boolean flag
    """
    checkpoints_path = os.path.join(projectRoot, "webapp/static/checkpoints")
    datasets_path = os.path.join(projectRoot, "webapp/static/datasets")
    
    current_state["inferenceCheckpoints"] = []
    
    current_state["checkpoints"] = [os.path.join(checkpoints_path, f) for f in os.listdir(checkpoints_path) if (os.path.isfile(os.path.join(checkpoints_path, f)) and os.path.splitext(f)[1] == ".ckpt")]
    current_state["checkpoints"].append("./data/checkpoints/audioldm-m-full.ckpt")
    current_state["inferenceCheckpoints"] = [os.path.join(checkpoints_path, f) for f in os.listdir(checkpoints_path) if (os.path.isfile(os.path.join(checkpoints_path, f)) and os.path.splitext(f)[1] == ".ckpt")]
    current_state["datasets"] = [f for f in os.listdir(datasets_path) if (os.path.isfile(os.path.join(datasets_path, f)) and os.path.splitext(f)[1] == ".zip")]
    checkpointDir = getFromServer("getResumeCheckpointDir")
    if checkpointDir:
        for f in os.listdir(checkpointDir):
            if os.path.isfile(os.path.join(checkpointDir, f)) and os.path.splitext(f)[1] == ".ckpt":
                current_state["checkpoints"].append(os.path.join(checkpointDir, f))
                current_state["inferenceCheckpoints"].append(os.path.join(checkpointDir, f))
    emitCurrentState()
    return True

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

def bulkParamForm():
    """Handle all parameters in one form.
    
    Returns:
        success: boolean flag
    """
    # Dictionary of label: arrayKey
    parameters = {
        "seed": "seed",
        "validation_every_n_epochs": "step,validation_every_n_epochs",
        "save_checkpoint_every_n_steps": "step,save_checkpoint_every_n_steps",
        "unconditional_guidance_scale": "model,params,evaluation_params,unconditional_guidance_scale",
        "ddim_sampling_steps": "model,params,evaluation_params,ddim_sampling_steps",
        "n_candidates_per_samples": "model,params,evaluation_params,n_candidates_per_samples",        
        "audio_duration": "preprocessing,audio,duration",
    }
    
    for key, val in parameters:
        valInput = request.form[key]
        message = "set_parameter;" + val + ";" + valInput
        if sendToServer(message):
            current_state["params"][val] = valInput

def startFineTuning():
    """Start the finetuning process on torchServer.
    
    Flashes message(s) with progress updates

    Returns:
        success: boolean flag
    """
    valInput = request.form["checkpointSelect"]
    message = "set_parameter;reload_from_ckpt;" + valInput
    
    if sendToServer(message):
        current_state["params"]["reload_from_ckpt"] = valInput
        emitCurrentState()
    else:
        flash("Failed to set checkpoint parameter")
        return False
    
    if sendToServer("finetune"):
        flash("Fine tuning started.")
        current_state["monitor"]["torchServerStatus"] = "finetuning"
        emitCurrentState()
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
    valInput = request.form["checkpointSelect"]
    message = "set_parameter;reload_from_ckpt;" + valInput
    
    if sendToServer(message):
        current_state["params"]["reload_from_ckpt"] = valInput
        emitCurrentState()
    else:
        flash("Failed to set checkpoint parameter")
        return False
    
    prompt = request.form["prompt"]
    
    if sendToServer("inferSingle;PROMPT:" + prompt):
        current_state["displayInferenceAudio"] = False
        current_state["inferencePath"] = "fake.mp3"
        current_state["inferencePrompt"] = prompt
        current_state["monitor"]["torchServerStatus"] = "performing inference"
        emitCurrentState()
        watch_torchServer(30) # watch the server, call it idle after 30 sec inactivity
        
        wait_for_inference_thread = threading.Thread(target=wait_for_inference, daemon=True)
        wait_for_inference_thread.start()

        flash("Inference begun")
        return True
    else:
        flash("Inference failed to begin")
        return False

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

def startEval():
    if sendToServer("eval"):
        flash("Evaluation started.")
        current_state["monitor"]["torchServerStatus"] = "evaluating"
        emitCurrentState()
        watch_torchServer(10)
        return True
    else:
        flash("AudioLDM2 not available. Is it running?")
        return False

def debugFunc():
    """debug function to show how requests are acting

    Returns:
        success: boolean flag
    """
    current_state["monitor"]["torchServerStatus"] = "finetuning"
    emitCurrentState()
    watch_torchServer()
    return True