"""
File to define and facilite AudioLDM2 Interface Flask App

Interfaces with torchServer.py
"""
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, Response, send_file, flash, make_response
from werkzeug.utils import secure_filename
import os, signal

# includes for zmq comms
import zmq, logging, sys

# flask socket stuff
from flask_socketio import SocketIO
import redis
import threading
from flask_cors import CORS
from flask_redis import FlaskRedis

import time

# includes for spawn new API
import subprocess, shutil

## init logging
logger = logging.getLogger(__name__)

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config.from_prefixed_env()
CORS(app)
redis_client = FlaskRedis(app)

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Torch server init variables
REQUEST_TIMEOUT = 2500
REQUEST_RETRIES = 3
SERVER_ENDPOINT = "tcp://localhost:5555"


def spawnAPIServer():
    p = subprocess.Popen(
        [shutil.which("python"), os.path.join(projectRoot, "webapp/torchServer.py")]
    )
    return p


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

try:
    # Test Redis connection
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
    redis_client.ping()  # Check if Redis is reachable
    logger.info("Redis is running and reachable")
    
except redis.exceptions.ConnectionError as e:
    logger.error("Redis is not running or unreachable: %s", e)
    logger.error("Shutting down master")
    os.kill(os.getppid(), signal.SIGTERM)
    
# socketio = SocketIO(app, message_queue="redis://localhost:6379")
socketio = SocketIO(app, message_queue="redis://localhost:6379", cors_allowed_origins="*")

# TODO fix: redis triggers an error loop that creates a very large log file very fast

## DEBUG could un-tab the following section after start_redis_listener() is moved out

@socketio.on("connect")  # log all new clients
def handle_connect():
    """log a message when client connects"""
    logger.info("Client connected to SocketIO")
    
@socketio.on("debug")
def debugSocket():
    logger.info("Socket Debug Triggered")
    print("socket debug triggered")
    
# Background thread to listen for Redis messages
# Emits message as "child_message"
def redis_listener():
    pubsub = redis_client.pubsub()
    pubsub.subscribe("flask-socketio")  # Subscribe to the Redis channel

    for message in pubsub.listen():
        if message["type"] == "current_state_update":
            data = message["data"].decode("utf-8")
            socketio.emit("current_state_update", data)  # Emit the message to connected clients
        elif message["type"] == "debug":
            socketio.emit("debugReceived")
            logger.info("listener received debug message")

# https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file
def follow(logFile):
    logFile.seek(0,2)
    while True:
        line = logFile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line
    
def torchServer_monitor():
    # pubsub = redis_client.pubsub()
    # pubsub.subscribe("flask-socketio")  # Subscribe to the Redis channel
    logDir = os.path.join(projectRoot, "webapp/logs")
    files = os.listdir(logDir)
    if len(files) <= 0:
        return False
    paths = []
    for basename in files:
        if "torchServer-" in basename:
            paths.append(os.path.join(logDir, basename))
    logFilePath = max(paths, key=os.path.getctime)
    
    logFile = open(logFilePath)
    logLines = follow(logFile)
    socketio.emit("monitor", "Monitoring torch log file!")
    for line in logLines:
        if "CUDA is not available" in line:
            # flash("Operation Failed! CUDA is not available.")
            socketio.emit("monitor", "Operation Failed! CUDA is not available.")
            return False
        elif "Traceback (most recent call last):" in line:
            # flash("Traceback found. Likely crash.")
            socketio.emit("monitor", "Traceback found. Likely crash.")
            return False
            
def watch_torchServer():
    redis_listener_thread = threading.Thread(target=torchServer_monitor, daemon=True)
    return True

## DEBUG SECTION
# Start redis_listener
@socketio.on("start_task")
def start_redis_listener():
    redis_listener_thread = threading.Thread(target=redis_listener, daemon=True).start()
    socketio.emit("task_update", {"data": "Task started!"})
    logger.info("redis_listener started")
    
    return redis_listener_thread

## In prod, the redis_listener would be initialized to read the 
## logfile and emit a message when specific messages are received

## TODO use this example
def child_process_example():
    redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
    message = {"event": "current_state_update",
            "data": current_state}
    redis_client.publish("flask-socketio", message)

def emitCurrentState():
    socketio.emit("current_state_update", current_state)

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
            
    rendered_template = render_template("index.html", current_state=current_state)
    response = make_response(rendered_template)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    return response


## handle the upload of an archive file through a submitted form
## saves the file, then calls an unzip helper function
### REQUIRES: werkzeug.utils.secure_filename, flask.request
def archiveUpload():
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
    paramPathInput = request.form["parameter"]
    valInput = request.form["value"]
    
    message = "set_parameter;" + paramPathInput + ";" + valInput

    if sendToServer(message):
        current_state["params"][paramPathInput] = valInput
        emitCurrentState()
        flash("Successfully set parameter")
        return "Successfully set parameter"
    else:
        flash("Failed to set parameter")
    return "Failed to set parameter"


def startFineTuning():
    if sendToServer("finetune"):
        flash("Fine tuning started. Please reference torchServer.log for progress")
        watch_torchServer()
    else:
        flash("AudioLDM2 not available. Is it running?")

def monitorFineTune():
    # monitor the latest file in webapp/logs beginning with "torchServer"
    return True

def inferSingle():
    prompt = request.form["prompt"]
    waveformpath = getFromServer("inferSingle;PROMPT:" + prompt)

    if not waveformpath:
        flash("Failed to perform inference")
        return False

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

    flash("Inference complete")
    return True


def downloadCheckpointLatest():
    path = getFromServer("prepareCheckpointDownload")
    if not path:
        flash("No checkpoint available")
        return False
    checkpointPath = os.path.join(
        projectRoot, path
    )
    return send_file(checkpointPath)


def restartAPIServer():
    if sendToServer("ping", 5):
        return "API still running"
    else:
        ## boot the API?
        apiServerProcess = spawnAPIServer()
        return "API not responding. New API instance booted"


def debugFunc():
    sendToServer("debug")
    # child_process_example()
    # socketio.emit("debug")
    return True


# if __name__ == "__main__":
#     socketio.run(app, debug=True)
