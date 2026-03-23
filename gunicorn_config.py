import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Logging
accesslog = "-"
errorlog  = "-"
loglevel  = "info"

# Process naming
proc_name = "placement_ai"

# Preload app for faster worker spawn
preload_app = True
