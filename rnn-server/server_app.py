#!/usr/bin/env python3
"""
Standalone entry point for RNN Trading Server
This script runs the FastAPI server on a local port (default: 8000)
"""
import uvicorn
import sys
import os

# Ensure the script directory is in the path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = os.path.dirname(sys.executable)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, application_path)

def main():
    """Run the FastAPI server"""
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '127.0.0.1')

    print(f"Starting RNN Trading Server on {host}:{port}")
    print(f"Application path: {application_path}")
    print(f"Health check: http://{host}:{port}/health-check")
    print(f"Press Ctrl+C to stop the server\n")

    # Import the FastAPI app
    from main import app

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
