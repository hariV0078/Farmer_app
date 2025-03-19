#!/bin/bash

# Start the application with Gunicorn
# Uses Uvicorn worker for ASGI support
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
