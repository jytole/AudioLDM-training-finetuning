# Worker Settings
workers = 1
# worker_class = 'eventlet'  # Use gevent async workers
# worker_connections = 5  # Maximum concurrent connections per worker
#timeout = 0 # disable timeouts, to allow API (training, finetuning, inference) to run indefinitely
timeout = 90 # enable timeout to avoid cloudflare shutting down the endpoint

# Server Settings
bind = "0.0.0.0:8000"

# Logging
accesslog = "flaskAccessLogs.log"
errorlog = "flaskErrorLogs.log"
capture_output = True