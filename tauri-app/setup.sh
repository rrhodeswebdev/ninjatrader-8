#!/bin/bash

echo "========================================"
echo "RNN Trading Server Desktop App Setup"
echo "========================================"
echo ""

echo "[1/4] Checking prerequisites..."
echo ""

if ! command -v node &> /dev/null; then
    echo "❌ ERROR: Node.js is not installed!"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi
echo "✓ Node.js found"

if ! command -v cargo &> /dev/null; then
    echo "❌ ERROR: Rust is not installed!"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi
echo "✓ Rust found"

if ! command -v uv &> /dev/null; then
    echo "⚠️  WARNING: uv is not installed!"
    echo "The app needs uv to run the Python server."
    echo "Install it with: pip install uv"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ uv found"
fi

echo ""
echo "[2/4] Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to install Node.js dependencies"
    exit 1
fi

echo ""
echo "[3/4] Generating default icons..."
npx @tauri-apps/cli icon
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Failed to generate icons"
    echo "You may need to create icons manually"
    echo "See generate-icons.md for instructions"
fi

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo ""
echo "Development mode (macOS):"
echo "  npm run dev"
echo ""
echo "Build for Windows (requires cross-compilation):"
echo "  See BUILD_FOR_WINDOWS.md"
echo ""
echo "========================================"

