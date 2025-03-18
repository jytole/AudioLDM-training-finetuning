from flask import Flask, render_template, request, Response, send_file
from werkzeug.utils import secure_filename
import zipfile
import os

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from shelljob import proc

from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject

app = Flask(__name__)

apiInstance = AudioLDM2APIObject()

@app.route("/")
def index():
    return render_template("index.html")

## Helper function for archiveUpload()
## Extracts the zip file at "savePath"
## Returns the contents of the first file
### REQUIRES: os, zipfile
def testUnzip(savePath):
    with zipfile.ZipFile(savePath,"r") as zip_ref:
        extractPath = './cache/extract/'
        zip_ref.extractall(extractPath)
        
        readFile = open("./cache/extract/" + os.listdir("./cache/extract/")[0], "r")
        return readFile.readline()

## Set up a function to handle the upload of an archive file
## Accepts POST requests to the upload URL including any files, 
## saves the file, then calls an unzip helper function
### REQUIRES: werkzeug.utils.secure_filename, flask.request
@app.route("/archiveUpload", methods=['POST'])
def archiveUpload():
    if 'file' in request.files:
        file = request.files['file']
        fileName = secure_filename(file.filename)
        savePath = './cache/' + fileName
        file.save(savePath)
        
        return testUnzip(savePath)

    return 'No file uploaded'

@app.route("/setParameter", methods=['POST'])
def setParameter():
    text = request.form['save_checkpoint_every_n_steps']
    
    apiInstance.set_parameter(["step", "save_checkpoint_every_n_steps"], int(text))
    
    return "Successfully set parameter"

@app.route("/startFineTuning", methods=['POST'])
def startFineTuning():
    apiInstance.finetune()
    return "Fine tuning started. Please reference host console for progress."

@app.route("/inferSingle", methods=['POST'])
def inferSingle():
    prompt = request.form['prompt']
    
    apiInstance.inferSingle(prompt)
    
    return "Inference Complete."

@app.route('/downloadCheckpoint/latest')
def downloadCheckpointLatest():
    checkpointPath = os.path.join(projectRoot,apiInstance.prepareCheckpointDownload())
    return send_file(checkpointPath)

if __name__ == "__main__":
    app.run(debug=True)