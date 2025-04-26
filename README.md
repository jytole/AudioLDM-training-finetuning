# Readme

This project contains an API which handles an instance of AudioLDM2. This project also has a webserver to interface with users and an additional server to handle calling functions in the API. This project contains all of the AudioLDM2 dependencies from the original repository at https://github.com/haoheliu/AudioLDM-training-finetuning with some minor modifications.

## Installation

### Quick Start

The entire installation process has been completed and thoroughly tested in a custom container image at the link https://hub.docker.com/repository/docker/jytole/runpod_audioldm/. The suggested way to install and utilize this application is to pull this repository and host it on RunPod. The installation section details the manual installation process if you would like to deploy elsewhere. The container image can be pulled and run from docker hub using the below command. This image has been tested on RunPod with the configuration listed in the Deploying on RunPod subsection.

```bash
   (base) $ docker pull jytole/runpod_audioldm:latest
   (base) $ docker run -it --gpus all -p 8000:8000 jytole/runpod_audioldm:latest bash
```

#### Deploying on RunPod

[RunPod](https://www.runpod.io/) was used to deploy the docker container for testing and demonstration. The following settings were used to deploy a "[GPU Pod](https://www.runpod.io/console/deploy)" on runpod:

- GPU Type: NVIDIA A40
- GPU Count: 1
- VRAM: 48 GB
- RAM: 50 GB
- vCPU Count: 9
- Container Image: docker.io/jytole/runpod_audioldm:latest
- Container Disk: 32 GB
- Volume Disk: 40 GB
- Expose HTTP Ports: 80, 8000
- Expose TCP Ports: 22, 8080
- Environment Variables:

   - FLASK_SECRET_KEY: <anything, as long as it is not empty>
   - WANDB_API_KEY: <[wandb api key](https://wandb.ai/authorize)>

### Full Installation

#### Base Container Image

The starting point for the AudioLDM2 pipeline and interface requires system level dependencies to be installed which allow pytorch to run on the GPU. Currently the system has been installed and tested beginning with the below container image, but host systems with similar dependencies may also work, with some exceptions due to version differences.

- RunPod Docker Image: torch 1.13.0, CUDA 11.7.1, ubuntu 22.04
   - [runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04](https://hub.docker.com/layers/runpod/pytorch/1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04/images/sha256-c4075bfad940d8042966fdac95d4049f017f2611f3ff29b70fa0b129c2a0018b)

On a local machine, this container can be downloaded and booted with the following command:

```bash
   (base) $ docker pull runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04
   (base) $ docker run -it --gpus all -p 8000:8000 runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04 bash
```

In order to commit any changes to this container image, give it an additional tag (latest), and push them to docker hub, the following commands can be used:

```bash
   (base) $ docker ps
   (base) $ docker commit <container_id> <dockerhub_username>/<image_name>:<tag>
   (base) $ docker tag <dockerhub_username>/<image_name>:<tag> <dockerhub_username>/<image_name>:latest
   (base) $ docker push -a <dockerhub_username>/<image_name>
```

#### Prerequisites

The following dependencies were configured in the container during the testing of the AudioLDM2 pipeline and interface, prior to preparing the python running environment.

- Miniconda

   - [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

- Python 3.10

   - [Python 3.10](https://www.python.org/downloads/release/python-31017/)

- Gunicorn

   - [Gunicorn](https://gunicorn.org/#quick-start)

```bash
   (base) $ apt-get update; apt-get install gunicorn
```
- Recommended (not required) HTTP proxy: NGINX

   - [NGINX](https://nginx.org/en/docs/install.html)

#### Python Environment

To use the AudioLDM2 Pipeline, first install the prerequisites above. Once these are configured, it is possible to clone the repository and run the interface with the below commands. The repository can be found at the link https://github.com/jytole/AudioLDM-training-finetuning.

First, create and activate a conda environment. Install poetry in the conda environment. Poetry is a dependency management tool for Python that allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

```bash
   (base) $ conda create -n audioldm_train python=3.10
   (base) $ conda activate audioldm_train
   (audioldm_train) $ pip install poetry
```
Then, clone the repository into the desired directory and install the dependencies using poetry.

```bash
   (audioldm_train) $ git clone https://github.com/jytole/AudioLDM-training-finetuning.git
   (audioldm_train) $ cd AudioLDM-training-finetuning
   (audioldm_train) $ poetry install
```

#### Starting the Server

The webapp can be run using the below commands. The script *post_start.sh* in the repository can be moved into the root of the container to run these automatically in the background after the container starts. These commands are configured to suppress all output because the two processes are, by default, configured to log into the ./webapp/logs/ folder.

```bash
   (audioldm_train) $ python webapp/torchServer.py >/dev/null 2>/dev/null &
   (audioldm_train) $ gunicorn -c  webapp/gunicorn-conf.py webapp.flaskApp:app >/dev/null 2>/dev/null &
```

If *post_start.sh* is moved into the root of the container, every time the container starts, these commands will be run automatically to start the server and webapp.

## Using the Interface

This section describes how to use the AudioLDM2 fine-tuning pipeline and interface for fine-tuning and audio generation. This section reflects the interface in version [1.9.2.1](https://hub.docker.com/repository/docker/jytole/runpod_audioldm/tags/1.9.2.1/sha256-3d6570b303fb9f8d0cf23a367407e4c168696e5f8ee01474966c81020e92673a) of the Docker container as of April 18, 2025.

The system is designed to be run in a container, as described in the Installation section, with an exposed web port to access the hosted flask application. When hosted on RunPod, it is possible to access the http interface from the [RunPod console](https://www.runpod.io/console/pods) with the "connect" button on the pod.

### Process Overview

The AudioLDM2 pipeline and interface is intended to be used to either fine-tune the AudioLDM2 model, or to generate audio from a previously fine-tuned model. The process is thus divided into two main sections: fine-tuning and audio generation. In order to fine-tune, a checkpoint of AudioLDM2 and a dataset in .zip format are needed as input. A checkpoint of AudioLDM2 is included in the suggested docker container. In order to generate sounds, a checkpoint of AudioLDM2 is needed as input. A flowchart of the process is shown below.

<img src="docs/source/images/usageDiagram.png" alt="Usage Diagram" width="50%" margin-left="auto" margin-right="auto" />

### Fine-Tuning

In order to initiate the fine-tuning process, the user must upload a dataset in .zip format and select a checkpoint of AudioLDM2. The dataset must be in the format specified in the Dataset Formatting subsection below. The following steps should be followed to fine-tune the model:

1. **Prepare Dataset**: The dataset must be in .zip format and follow the guidelines in the Dataset Formatting subsection below.
2. **Upload Dataset**: The dataset can be uploaded using the "Choose File" button in the interface. "Submit" once the file is uploaded.
3. **Select Checkpoint**: The checkpoint can be selected from the dropdown menu in the "Start Fine-Tuning" field.
4. **Adjust Parameters**: The parameters for the fine-tuning process can be adjusted in the "Options" tab and are discussed in the "Parameters" section below.
5. **Start Fine-Tuning**: Click the "Start" button in the "Start Fine-Tuning" field to begin the fine-tuning process. 
6. **Monitor Progress**: The progress of the fine-tuning process can be monitored at the top of the page in the "AudioLDM2 Monitor" field.

#### Import Files from Cloud Storage

In order to speed up the process of uploading large files like the dataset or the checkpoint, the interface can scan for datasets and checkpoints that have been manually transferred to the container. RunPod supports transferring files from cloud storage. If a checkpoint is placed into the /home/AudioLDM-training-finetuning/webapp/static/checkpoints/ directory, it can be detected and added to the checkpoint dropdown menu. Similarly, if a dataset is placed into the /home/AudioLDM-training-finetuning/webapp/static/datasets/ directory, it can be detected and added to the dataset dropdown menu. The interface will then allow the user to select these files from the dropdown menus instead of uploading them.

In order to initiate this transfer and allow the interface to detect the files, you can refer to the [RunPod documentation](https://docs.runpod.io/pods/configuration/export-data) on transferring files between the container and cloud storage. Ensure that you are transferring checkpoints with the extension .ckpt into the ./webapp/static/checkpoints/ directory, and datasets with the extension .zip into the ./webapp/static/datasets/ directory.

In order to tell the interface to detect the files, there is a scan button on the Options page. This will scan the directories and add any files that are found to the dropdown menus.

#### Dataset Formatting

The dataset must be in .zip format and contain all the audio files. The audio files must be in .wav format. The audiofiles may be in a subdirectory. The dataset must also contain a .csv file with columns for "audio" and "caption". The "audio" column must contain the relative path to an audio file from the root of the dataset. The caption should be a description of the contents of the audio file for use in training. The .csv file must be named metadata.csv and be located in the root of the dataset. An example dataset might be structured as follows:

<img src="docs/source/images/datasetStructure.png" alt="Structure of Dataset" width="25%" margin-left="auto" margin-right="auto" />

The metadata.csv file for this dataset would look like this:

<img src="docs/source/images/datasetCSV.png" alt="Dataset CSV File Structure" width="50%" margin-left="auto" margin-right="auto" />

#### Accessing checkpoints

The fine-tuning page of the interface contains a button to download the latest checkpoint, but this 8 GB download may take a significant amount of time, so it is recommended to export the checkpoint to cloud storage according to the [RunPod documentation](https://docs.runpod.io/pods/configuration/export-data).

You may find checkpoints in the /home/AudioLDM-training-finetuning/log/latent-diffusion/ directory. By default within this directory, the checkpoint is located in ./2025_03_27_api_default_finetune/default_finetune/checkpoints/. The latest checkpoint should be saved with the "global step" count labeled in the filename. By default, the checkpoint is saved every 5000 global steps, but this parameter can be adjusted in the options tab.

#### Manually Stop Fine-Tuning

With the default configuration parameters, fine-tuning may take a long time depending on the size of the dataset and the parameters configured. If you would like to stop the fine-tuning process, it is recommended to first export the latest checkpoint according to the instructions above before stopping the process. This checkpoint can then be imported to generate sounds.

Fine-tuning can be stopped by searching for the process running "python torchServer.py" in the container and killing the process with the lowest process ID. This can be done with the following commands:

```bash
   (audioldm_train) $ ps aux | grep torchServer.py
   (audioldm_train) $ kill <process_id>
```

It is recommended to also kill the process running "gunicorn" int he container, as this process should be running alongside the webapp. Follow the same process as above to find the process ID and kill it. The commands should look like this:

```bash
   (audioldm_train) $ ps aux | grep gunicorn
   (audioldm_train) $ kill <process_id>
```

Once this process is killed, the system may be rebooted to start it again, or the webapp script may be restarted manually without rebooting using the following commands:

```bash
   (audioldm_train) $ export FLASK_SECRET_KEY=$(openssl rand -hex 16)
   (audioldm_train) $ /post_start.sh
```

### Generating sounds

In order to generate sounds using the AudioLDM2 interface, the inference tab can be used. The dropdown box inside the Generate Sound section of the interface should be used to select the desired checkpoint to generate sounds from. The text box should be used to input a prompt, and “Submit” can be pressed to generate a sound.

For example, “Therabot barking” could be used as a prompt for a checkpoint trained on sounds labeled with “Therabot” in order to generate a sound similar to those that were used in training.

### Parameters

The following parameters are available in the Options tab to change the way that the model behaves. A short description of each parameter is included.

- Seed
    - Default: 0
    - Changes the seed for random number generation. 
    - The same seed with the same prompt and the same checkpoint will generate the same sound.
- Save Checkpoint Every N Steps
    - Default: 5000
    - Number of global steps after which to save a checkpoint
- Validation Every N Epochs
    - Default: 5
    - Number of epochs after which to perform the validation loop
    - The validation loop performs inference and adjusts the learning rate according to results
- Evaluation: Unconditional Guidance Scale
    - Default: 3.5
    - A lower value indicates more creativity
    - A higher value indicates less creativity and more obedience to the prompt
    - It is not recommended to go above 15
- Evaluation: DDIM Sampling Steps
    - Default: 200
    - Denoising steps
    - More steps means a clearer sound
    - After around 50 steps, the increase in clarity is less substantial up to 200 steps (according to the original AudioLDM paper)
- Evaluation: N Candidates Per Sample
    - Default: 3
    - Number of sounds to generate during inference before taking the top candidate

### Interface images

<img src="docs/source/images/finetuneInterface.png" alt="Fine-Tuning Page of Interface" width="70%" margin-left="auto" margin-right="auto" />

<img src="docs/source/images/inferenceInterface.png" alt="Inference Page of Interface" width="70%" margin-left="auto" margin-right="auto" />

<img src="docs/source/images/optionsInterface.png" alt="Options Page of the Interface" width="70%" margin-left="auto" margin-right="auto" />
