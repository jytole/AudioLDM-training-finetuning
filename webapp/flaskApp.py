from flask import Flask, render_template, request, Response, send_file
from werkzeug.utils import secure_filename
import zipfile
import os

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject

app = Flask(__name__)

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

# Implement lock to only allow one instance of api
def unlock_api():
    global apiInstance
    if os.path.exists("./webapp/audioldm_api.lock"):
        os.remove("./webapp/audioldm_api.lock")
        apiInstance = None

from flask import appcontext_tearing_down
appcontext_tearing_down.connect(unlock_api, app)

# Only allow one instance of the API to be running at a time
def start_api():
    apiInstance = None
    if not os.path.exists("./webapp/audioldm_api.lock"):
        apiInstance = AudioLDM2APIObject()
        with open("./webapp/audioldm_api.lock", "w") as lockFile:
            lockFile.write("1")
    return apiInstance

apiInstance = start_api()

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
    if 'file' in request.files:
        file = request.files['file']
        fileName = secure_filename(file.filename)
        savePath = './webapp/cache/' + fileName
        file.save(savePath)
        
        return apiInstance.handleDataUpload(savePath)

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