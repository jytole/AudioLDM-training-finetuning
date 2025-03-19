#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/AudioLDM-training-finetuning
conda activate audioldm_train
poetry install
gunicorn --timeout 0 --access-logfile 'flaskAccessLogs.log' --error-logfile 'flaskErrorLogs.log' --capture-output -w 1 webapp.flaskApp:app &
echo "gunicorn started with logfiles /home/AudioLDM-training-finetuning/flask*Logs.log"