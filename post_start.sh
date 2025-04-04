#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/AudioLDM-training-finetuning
conda activate audioldm_train
poetry install
python webapp/torchServer.py >/dev/null 2>/dev/null &
echo "torch server started with logfiles /home/AudioLDM-training-finetuning/webapp/logs/torchServer*.log"
gunicorn -c  webapp/gunicorn-conf.py webapp.flaskApp:app >/dev/null 2>/dev/null &
echo "gunicorn started with logfiles /home/AudioLDM-training-finetuning/webapp/logs/flask*Logs*.log"