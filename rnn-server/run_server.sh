#!/bin/bash
# Quick-start script to run the RNN Server executable

# Check if executable exists
if [ -f "dist/server_app.dist/rnn-server" ]; then
    echo "Starting RNN Trading Server..."
    echo "Server will be available at http://127.0.0.1:8000"
    echo "Press Ctrl+C to stop"
    echo ""
    cd dist/server_app.dist && ./rnn-server
elif [ -f "server_app.py" ]; then
    echo "Executable not found. Running development server instead..."
    echo "To build the executable, run: ./build_executable.sh"
    echo ""
    echo "Starting development server..."
    uv run fastapi dev main.py
else
    echo "Error: Neither executable nor development files found"
    echo "Please ensure you're in the rnn-server directory"
    exit 1
fi
