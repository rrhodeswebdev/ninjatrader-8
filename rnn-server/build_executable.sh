#!/bin/bash
# Build script for creating RNN Server executable with Nuitka

set -e  # Exit on error

echo "Building RNN Server Executable with Nuitka..."
echo "=============================================="
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist server_app.dist server_app.build server_app.onefile-build

# Run Nuitka compilation (standalone mode, faster than onefile)
echo "Starting Nuitka compilation..."
echo "This will create a standalone directory with the executable."
echo ""

uv run python -m nuitka \
    --standalone \
    --output-filename=rnn-server \
    --output-dir=dist \
    --include-data-dir=models=models \
    --nofollow-import-to=matplotlib \
    --nofollow-import-to=PIL \
    --nofollow-import-to=IPython \
    --nofollow-import-to=jupyter \
    --nofollow-import-to=pytest \
    --nofollow-import-to=setuptools \
    --assume-yes-for-downloads \
    --show-progress \
    --show-memory \
    --python-flag=no_site \
    server_app.py

echo ""
echo "=============================================="
echo "Build complete!"
echo ""
echo "Executable location: dist/server_app.dist/rnn-server"
echo ""
echo "The standalone folder contains:"
echo "  - rnn-server (executable)"
echo "  - All required dependencies"
echo "  - models/ directory with trained models"
echo ""
echo "To run the server:"
echo "  cd dist/server_app.dist"
echo "  ./rnn-server"
echo ""
echo "Or create a symlink:"
echo "  ln -s dist/server_app.dist/rnn-server rnn-server"
echo "  ./rnn-server"
echo ""
echo "The server will start on http://127.0.0.1:8000"
echo "Health check: http://127.0.0.1:8000/health-check"
echo ""
