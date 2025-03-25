from flask import Flask, render_template, request, Response, send_file, after_this_request
from werkzeug.utils import secure_filename
import zipfile
import os
from multiprocessing import Process, Manager, set_start_method
import asyncio

from audioldm_train.utilities.audioldm2_api import AudioLDM2APIObject

projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# nginx prod deployment (tell the app it's behind a 1-layer proxy)
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

## TODO: Figure out if this works / how to make this work
# global manager
# global shared_dict

manager = None
shared_dict = None
apiInstance = None

def post_worker_init(worker):
    apiInstance = AudioLDM2APIObject()
    pass

with app.app_context():
    set_start_method("spawn")
    # manager = Manager()
    # shared_dict = manager.dict()
    # shared_dict["apiInstance"] = AudioLDM2APIObject()
    # shared_dict["api_lock"] = manager.Lock()
    # apiInstance = AudioLDM2APIObject()

# apiInstance = AudioLDM2APIObject()

# Lock to only allow one API call at a time
# api_lock = Lock()

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
    lock_acquired = shared_dict["api_lock"].acquire(blocking=False)
    if not lock_acquired:
        return "API is currently in use. Please try again later"
    try:
        if 'file' in request.files:
            file = request.files['file']
            fileName = secure_filename(file.filename)
            savePath = './webapp/cache/' + fileName
            file.save(savePath)
            
            return shared_dict["apiInstance"].handleDataUpload(savePath)

        return 'No file uploaded'
    finally:
        # api_lock.release()
        if lock_acquired:
            shared_dict["api_lock"].release()

@app.route("/setParameter", methods=['POST'])
def setParameter():
    lock_acquired = shared_dict["api_lock"].acquire(blocking=False)
    if not lock_acquired:
        return "API is currently in use. Please try again later"
    try:
        text = request.form['save_checkpoint_every_n_steps']
        
        shared_dict["apiInstance"].set_parameter(["step", "save_checkpoint_every_n_steps"], int(text))
        
        return "Successfully set parameter"
    finally:
        if lock_acquired:
            shared_dict["api_lock"].release()

# attempt 7334 (or something): https://stackoverflow.com/questions/18082683/need-to-execute-a-function-after-returning-the-response-in-flask/66675113#66675113
# @app.route("/startFineTuning", methods=['POST'])
# def fineTuningWrapper():
#     lock_acquired = shared_dict["api_lock"].acquire(blocking=False)
#     if not lock_acquired:
#         return "API is currently in use. Please try again later"
#     # asyncio.to_thread(startFineTuning(lock_acquired)) # start thread with finetuning operation, leave current thread to handle response
#     # ## Note: release lock in startFineTuning() function, not this wrapper
#     # return "Fine tuning started. Please reference host console for progress."
#     # try:
#     @after_this_request
#     def add_close_action(response):
#         @response.call_on_close
#         def process_after_request():
#             shared_dict["apiInstance"].finetune()
#             shared_dict["api_lock"].release()
#         return response
#     return "Fine tuning started. Please reference host console for progress."
#     # finally:
#     #     if lock_acquired:
#     #         shared_dict["api_lock"].release()

# def startFineTuning(lock_acquired):
#     try:
#         shared_dict["apiInstance"].finetune()
#     finally:
#         if lock_acquired:
#             shared_dict["api_lock"].release()
            
@app.route("/startFineTuning", methods=['POST'])
def startFineTuning():
    return '''<div>start</div>
    <script>
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/fineTuneAction', true);
        xhr.onreadystatechange = function(e) {
            var div = document.createElement('div');
            div.innerHTML = '' + this.readyState + ':' + this.responseText;
            document.body.appendChild(div);
        };
        xhr.send();
    </script>
    '''

@app.route("/fineTuneAction")
def fineTuneAction():
    def generate():
        app.logger.info("Starting fine tuning")
        # shared_dict["apiInstance"].finetune()
        apiInstance.finetune()
        app.logger.info("Finished fine tuning")
        yield ''
    return Response(generate(), mimetype='text/plain')


@app.route("/inferSingle", methods=['POST'])
def inferSingle():
    lock_acquired = shared_dict["api_lock"].acquire(blocking=False)
    if not lock_acquired:
        return "API is currently in use. Please try again later"
    try:
        prompt = request.form['prompt']
        
        shared_dict["apiInstance"].inferSingle(prompt)
        
        return "Inference Complete."
    finally:
        if lock_acquired:
            shared_dict["api_lock"].release()

@app.route('/downloadCheckpoint/latest')
def downloadCheckpointLatest():
    lock_acquired = shared_dict["api_lock"].acquire(blocking=False)
    if not lock_acquired:
        return "API is currently in use. Please try again later"
    try:
        checkpointPath = os.path.join(projectRoot,shared_dict["apiInstance"].prepareCheckpointDownload())
        return send_file(checkpointPath)
    finally:
        if lock_acquired:
            shared_dict["api_lock"].release()

# if __name__ == "__main__":
#     app.run(debug=True)