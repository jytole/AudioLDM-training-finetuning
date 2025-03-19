#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/AudioLDM-training-finetuning
conda activate audioldm_train
poetry install
gunicorn -c  gunicorn-conf.py webapp.flaskApp:app &
echo "gunicorn started with logfiles /home/AudioLDM-training-finetuning/flask*Logs.log"