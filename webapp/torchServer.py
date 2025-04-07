"""Script to host and log an internal server for AudioLDM2 functionality.

Assumes that the webapp init scripts launch this application once.
Makes it possible to send and receive messages from a single
AudioLDM2 instance, minimizing memory overloading and crashes.

Interfaces with audioldm2_api.py"""

import time
import zmq
import zipfile
import os
import logging, sys
from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject
from datetime import datetime

## logger init
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
webappFolder = os.path.dirname(os.path.realpath(__file__))

## log messages to file with format
handler = logging.FileHandler(
    os.path.join(webappFolder, "logs", "torchServer-" + current_time + ".log")
)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)
logger.addHandler(handler)


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    This enables logging all std print statements (notably from audioldm2_api)
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


sys.stdout = StreamToLogger(root, logging.INFO)
sys.stderr = StreamToLogger(root, logging.ERROR)

apiInstance = AudioLDM2APIObject()

## Server init
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

killFlag = False  # NOTE change flag to True to make sphinx documentation

## Message loop
## accepts requests in the format <functionName>;args as follows:
# handleDataUpload;zipPath
#   handles the extraction of data folder zipPath
# set_parameter;path,to,targetParam;val
#   sets parameter targetParam to val
# finetune
#   begins finetuning
# inferSingle;PROMPT:prompt text to evaluate which can contain semicolons
#   performs inference with prompt
# prepareCheckpointDownload
#   prepares the latest checkpoint for download; ack;path/to/zip
# ping
#   returns an ack
# debug
#   prints a debug message
# kill
#   kills the message loop

## TODO fix every request being triggered twice

waveformpath = "nack"

while not killFlag:
    message = socket.recv_string()  #  Wait for next request from client
    messageArr = message.split(";")
    reply = "nack"
    post_loop_finetune = False
    post_loop_infer = False

    # Assumes message format: <functionName>;<args>
    logger.info(f"Received request: {message}")
    if messageArr[0] == "handleDataUpload":
        apiInstance.handleDataUpload(messageArr[1])
        reply = "ack"
    elif messageArr[0] == "set_parameter":
        paramPath = messageArr[1].split(",")  # path where "," = "/"
        if apiInstance.set_parameter(paramPath, messageArr[2]):  # TODO implement non-int param setting...
            reply = "ack"
        else:
            reply = messageArr[1]
    elif messageArr[0] == "finetune":
        reply = "ack"
        post_loop_finetune = True
    elif messageArr[0] == "inferSingle":
        # format inferSingle;PROMPT:<prompt> to support ";" in prompts
        # waveformpath = apiInstance.inferSingle(message.split(";PROMPT:")[1])
        # reply = "ack;" + apiInstance.inferSingle(message.split(";PROMPT:")[1])
        reply = "ack"
        post_loop_infer = True
    elif messageArr[0] == "checkInferenceComplete":
        reply = "ack;" + waveformpath
    elif messageArr[0] == "prepareCheckpointDownload":
        path = apiInstance.prepareCheckpointDownload()
        if not path:
            logger.debug("No checkpoint available")
        else:
            reply = "ack;" + path
    elif messageArr[0] == "ping":
        reply = "ack"
    elif messageArr[0] == "debug":
        logger.debug("torchServer debug function triggered")
        apiInstance.debugFunc()
        reply = "ack"
    elif messageArr[0] == "kill":
        reply = "ack"
        killFlag = True

    #  Send reply back to client
    socket.send_string(reply)

    if post_loop_finetune:
        logger.debug("torchServer beginning finetune")
        # This will freeze the server for the duration of the finetune, but the flask server should still work
        apiInstance.finetune()
        post_loop_finetune = False
        logger.debug("torchServer finetune complete")
        
    if post_loop_infer:
        waveformpath = apiInstance.inferSingle(message.split(";PROMPT:")[1])
        post_loop_infer = False

logger.info("torchServer shut down")