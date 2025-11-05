# NinjaTrader 8 Algorithmic Trading System

A comprehensive algorithmic trading system for NinjaTrader 8 with AI-powered
trading signals via a Python RNN server.

## Project Structure

```
ninjatrader-8/
├── strategies/          # NinjaTrader 8 C# strategies
├── indicators/          # Custom technical indicators (empty)
├── rnn-server/         # Python FastAPI server for AI predictions
└── tauri-app/          # Desktop app for managing the RNN server
```

## Components

### 1. NinjaTrader 8 Strategies (C#)

Custom trading strategies that integrate with the RNN server for AI-driven
trading decisions.

**Location:** `strategies/`

**Setup:**

1. Copy `.cs` files to NinjaTrader's custom strategy folder
2. Compile in NinjaTrader's NinjaScript Editor (F5)

### 2. RNN Server (Python)

FastAPI-based machine learning inference server providing trading signals.

**Location:** `rnn-server/`

**Quick Start:**

```bash
cd rnn-server
uv run fastapi dev main.py
```

**Documentation:** See `rnn-server/RNN_SERVER_DOCUMENTATION.md`

### 3. Desktop App (Tauri) ⭐ NEW

Windows desktop application for easy RNN server management.

**Location:** `tauri-app/`

**Features:**

-   🚀 One-click server start/stop
-   📊 Real-time status monitoring
-   🔍 Built-in connection testing
-   💻 Native Windows application

**Quick Start:**

```bash
cd tauri-app
setup.bat          # First-time setup
npm run dev        # Development mode
npm run build      # Build installer
```

**Documentation:** See `tauri-app/README.md` and `tauri-app/QUICK_START.md`

## Getting Started

### For Traders (End Users)

1. **Install the Desktop App:**

    - Download and install the RNN Trading Server desktop app
    - Launch it and click "Start Server"
    - Server runs at `http://127.0.0.1:8000`

2. **Set up NinjaTrader:**

    - Copy strategies from `strategies/` to NinjaTrader
    - Compile in NinjaScript Editor
    - Apply strategy to your chart

3. **Start Trading:**
    - Ensure RNN server is running
    - NinjaTrader strategy will connect automatically
    - Monitor trades and AI signals

### For Developers

1. **RNN Server Development:**

    ```bash
    cd rnn-server
    uv run fastapi dev main.py
    ```

2. **Desktop App Development:**

    ```bash
    cd tauri-app
    npm install
    npm run dev
    ```

3. **NinjaTrader Strategy Development:**
    - Edit `.cs` files in `strategies/`
    - Compile in NinjaTrader 8
    - Test on simulator

## Requirements

### For End Users

-   Windows 10 or later
-   NinjaTrader 8
-   Python 3.10 or 3.11
-   `uv` package manager: `pip install uv`

### For Developers

-   All of the above, plus:
-   Rust (for Tauri development)
-   Node.js (for Tauri development)

## Architecture

```
┌─────────────────┐
│  NinjaTrader 8  │
│   (Strategies)  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Tauri Desktop  │─────►│   RNN Server     │
│      App        │      │   (FastAPI)      │
│  (Start/Stop)   │      │   Port 8000      │
└─────────────────┘      └──────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  ML Model (RNN) │
                         │  Trading Signals│
                         └─────────────────┘
```

## Documentation

-   **Main Project:** `CLAUDE.md` - Development guidelines
-   **RNN Server:** `rnn-server/RNN_SERVER_DOCUMENTATION.md`
-   **Desktop App:** `tauri-app/README.md` and `tauri-app/QUICK_START.md`

## License

[Your License Here]
