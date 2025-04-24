Installation
==========================

Welcome to the AudioLDM2 Pipeline documentation! This project contains an API which handles an instance of AudioLDM2. This project also has a webserver to interface with users and an additional server to handle calling functions in the API.

Quick Start
-----------
The entire installation process has been completed and thoroughly tested in a custom container image at the link https://hub.docker.com/repository/docker/jytole/runpod_audioldm/. The suggested way to install and utilize this application is to pull this repository and host it on RunPod. The installation section details the manual installation process if you would like to deploy elsewhere. The container image can be pulled and run from docker hub using the below command. This image has been tested on RunPod with the configuration listed in the Deploying on RunPod subsection.

.. code-block:: console

   (base) $ docker pull jytole/runpod_audioldm:latest
   (base) $ docker run -it --gpus all -p 8000:8000 jytole/runpod_audioldm:latest bash

Deploying on RunPod
^^^^^^^^^^^^^^^^^^^^^^^^
`RunPod <https://www.runpod.io/>`_ was used to deploy the docker container for testing and demonstration. The following settings were used to deploy a "`GPU Pod <https://www.runpod.io/console/deploy>`_" on runpod:

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
   - WANDB_API_KEY: <`wandb api key <https://wandb.ai/authorize>`_>

Full Installation
-----------------

Base Container Image
^^^^^^^^^^^^^^^^^^^^
The starting point for the AudioLDM2 pipeline and interface requires system level dependencies to be installed which allow pytorch to run on the GPU. Currently the system has been installed and tested beginning with the below container image, but host systems with similar dependencies may also work, with some exceptions due to version differences.

- RunPod Docker Image: torch 1.13.0, CUDA 11.7.1, ubuntu 22.04
   - `runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04 <https://hub.docker.com/layers/runpod/pytorch/1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04/images/sha256-c4075bfad940d8042966fdac95d4049f017f2611f3ff29b70fa0b129c2a0018b>`_

On a local machine, this container can be downloaded and booted with the following command:

.. code-block:: console

   (base) $ docker pull runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04
   (base) $ docker run -it --gpus all -p 8000:8000 runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04 bash

In order to commit any changes to this container image, give it an additional tag (latest), and push them to docker hub, the following commands can be used:

.. code-block:: console

   (base) $ docker ps
   (base) $ docker commit <container_id> <dockerhub_username>/<image_name>:<tag>
   (base) $ docker tag <dockerhub_username>/<image_name>:<tag> <dockerhub_username>/<image_name>:latest
   (base) $ docker push -a <dockerhub_username>/<image_name>

Prerequisites
^^^^^^^^^^^^^
The following dependencies were configured in the container during the testing of the AudioLDM2 pipeline and interface, prior to preparing the python running environment.

- Miniconda

   - `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_

- Python 3.10

   - `Python 3.10 <https://www.python.org/downloads/release/python-31017/>`_

- Gunicorn

   - `Gunicorn <https://gunicorn.org/#quick-start>`_

.. code-block:: console

   (base) $ apt-get update; apt-get install gunicorn

- Recommended (not required) HTTP proxy: NGINX

   - `NGINX <https://nginx.org/en/docs/install.html>`_

Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use the AudioLDM2 Pipeline, first install the prerequisites above. Once these are configured, it is possible to clone the repository and run the interface with the below commands. The repository can be found at the link https://github.com/jytole/AudioLDM-training-finetuning.

First, create and activate a conda environment. Install poetry in the conda environment. Poetry is a dependency management tool for Python that allows you to declare the libraries your project depends on and it will manage (install/update) them for you.

.. code-block:: console

   (base) $ conda create -n audioldm_train python=3.10
   (base) $ conda activate audioldm_train
   (audioldm_train) $ pip install poetry

Then, clone the repository into the desired directory and install the dependencies using poetry.

.. code-block:: console

   (audioldm_train) $ git clone https://github.com/jytole/AudioLDM-training-finetuning.git
   (audioldm_train) $ cd AudioLDM-training-finetuning
   (audioldm_train) $ poetry install

Starting the Server
^^^^^^^^^^^^^^^^^^^^

The webapp can be run using the below commands. The script *post_start.sh* in the repository can be moved into the root of the container to run these automatically in the background after the container starts. These commands are configured to suppress all output because the two processes are, by default, configured to log into the ./webapp/logs/ folder.

.. code-block:: console

   (audioldm_train) $ python webapp/torchServer.py >/dev/null 2>/dev/null &
   (audioldm_train) $ gunicorn -c  webapp/gunicorn-conf.py webapp.flaskApp:app >/dev/null 2>/dev/null &

If *post_start.sh* is moved into the root of the container, every time the container starts, these commands will be run automatically to start the server and webapp.