from flask import Flask, render_template, request, Response, send_file
from werkzeug.utils import secure_filename
import zipfile
import os
from multiprocessing import Process
from threading import Lock

from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

apiInstance = AudioLDM2APIObject()

# Lock to only allow one API call at a time
api_lock = Lock()

@app.route("/")
def index():
    return render_template("index.html")

## Helper function for archiveUpload()
## Extracts the zip file at "savePath"
## Returns the contents of the first file
### REQUIRES: os, zipfile
def testUnzip(savePath):
    with zipfile.ZipFile(savePath,"r") as zip_ref:
        extractPath = './webapp/cache/extract/'
        zip_ref.extractall(extractPath)
        
        readFile = open("./webapp/cache/extract/" + os.listdir("./webapp/cache/extract/")[0], "r")
        return readFile.readline()

## Set up a function to handle the upload of an archive file
## Accepts POST requests to the upload URL including any files, 
## saves the file, then calls an unzip helper function
### REQUIRES: werkzeug.utils.secure_filename, flask.request
@app.route("/archiveUpload", methods=['POST'])
def archiveUpload():
    if not api_lock.acquire(blocking=False):
        return "API is currently in use. Please try again later"
    try:
        if 'file' in request.files:
            file = request.files['file']
            fileName = secure_filename(file.filename)
            savePath = './webapp/cache/' + fileName
            file.save(savePath)
            
            return apiInstance.handleDataUpload(savePath)

        return 'No file uploaded'
    finally:
        api_lock.release()

@app.route("/setParameter", methods=['POST'])
def setParameter():
    if not api_lock.acquire(blocking=False):
        return "API is currently in use. Please try again later"
    
    try:
        text = request.form['save_checkpoint_every_n_steps']
        
        apiInstance.set_parameter(["step", "save_checkpoint_every_n_steps"], int(text))
        
        return "Successfully set parameter"
    finally:
        api_lock.release()

@app.route("/startFineTuning", methods=['POST'])
def startFineTuning():
    if not api_lock.acquire(blocking=False):
        return "API is currently in use. Please try again later"
    
    try:
        apiInstance.finetune()
        return "Fine tuning started. Please reference host console for progress."
    finally:
        api_lock.release()

@app.route("/inferSingle", methods=['POST'])
def inferSingle():
    if not api_lock.acquire(blocking=False):
        return "API is currently in use. Please try again later"
    try:
        prompt = request.form['prompt']
        
        apiInstance.inferSingle(prompt)
        
        return "Inference Complete."
    finally:
        api_lock.release()

@app.route('/downloadCheckpoint/latest')
def downloadCheckpointLatest():
    if not api_lock.acquire(blocking=False):
        return "API is currently in use. Please try again later"
    try:
        checkpointPath = os.path.join(projectRoot,apiInstance.prepareCheckpointDownload())
        return send_file(checkpointPath)
    finally:
        api_lock.release()

# if __name__ == "__main__":
#     app.run(debug=True)