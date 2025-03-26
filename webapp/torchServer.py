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
handler = logging.FileHandler(os.path.join(webappFolder, "torchServer-" + current_time + ".log"))
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logger.addHandler(handler)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

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
## TODO accept CLI to set port, so that flaskApp can spawn the server if it dies

## Message loop
while True:
    #  Wait for next request from client
    message = socket.recv_string()
    messageArr = message.split(";")
    reply = ""
    post_loop_finetune = False
    
    ## Assumes function names are separated from arguments by only semicolon and space
    logger.info(f"Received request: {message}")
    if(messageArr[0] == "handleDataUpload"):
        apiInstance.handleDataUpload(messageArr[1])
        reply = "ack"
    elif(messageArr[0] == "set_parameter"):
        paramPath = messageArr[1].split(",")
        if(apiInstance.set_parameter(paramPath, int(messageArr[2]))):
            reply = "ack"
        else:
            reply = messageArr[1]
    elif(messageArr[0] == "finetune"):
        reply = "ack"
        post_loop_finetune = True
    elif(messageArr[0] == "inferSingle"):
        reply = "ack;" + apiInstance.inferSingle(messageArr[1])
    elif(messageArr[0] == "prepareCheckpointDownload"):
        reply = "ack;" + apiInstance.prepareCheckpointDownload()
    elif(messageArr[0] == "ping"):
        reply = "ack"

    #  Do some 'work'
    # time.sleep(1)

    #  Send reply back to client
    socket.send_string(reply)
    
    if(post_loop_finetune):
        ## This will freeze the server for the duration of the finetune, but the flask server should still work
        apiInstance.finetune()