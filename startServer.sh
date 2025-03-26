python webapp/torchServer.py &
echo "torch server started with logfiles"
gunicorn -c  webapp/gunicorn-conf.py webapp.flaskApp:app &
echo "gunicorn started with logfiles /home/AudioLDM-training-finetuning/webapp/flask*Logs*.log"