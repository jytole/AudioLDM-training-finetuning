from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import zipfile
import os

app = Flask(__name__)

@app.route("/")
def hello_world():
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