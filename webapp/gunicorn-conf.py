import os
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
webappFolder = os.path.dirname(os.path.realpath(__file__))

# Worker Settings
workers = 1
# worker_class = 'eventlet'  # Use gevent async workers
# worker_connections = 5  # Maximum concurrent connections per worker
# timeout = 0 # disable timeouts, to allow API (training, finetuning, inference) to run indefinitely
timeout = 90  # enable timeout to avoid cloudflare shutting down the endpoint

# Server Settings
bind = "0.0.0.0:8000"

# Logging
accesslog = os.path.join(
    webappFolder, "logs", "flaskAccessLogs-" + current_time + ".log"
)
errorlog = os.path.join(webappFolder, "logs", "flaskErrorLogs-" + current_time + ".log")
capture_output = True
