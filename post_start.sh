#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/AudioLDM-training-finetuning
conda activate audioldm_train
poetry install
python webapp/torchServer.py &
echo "torch server started with logfiles /home/AudioLDM-training-finetuning/webapp/torchServer*.log"
gunicorn -c  webapp/gunicorn-conf.py webapp.flaskApp:app &
echo "gunicorn started with logfiles /home/AudioLDM-training-finetuning/webapp/flask*Logs*.log"