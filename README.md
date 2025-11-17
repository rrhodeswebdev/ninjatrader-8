# NinjaTrader 8 Algorithmic Trading System

A comprehensive algorithmic trading system for NinjaTrader 8 with AI-powered
trading signals via a Python RNN server.

## Project Structure

```
ninjatrader-8/
├── strategies/          # NinjaTrader 8 C# strategies
├── indicators/          # Custom technical indicators (empty)
└── rnn-server/          # Python FastAPI server for AI predictions
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

## Getting Started

### For Traders (End Users)

1. **Start the RNN Server:**

    - Open a terminal
    - Run:

    ```bash
    cd rnn-server
    uv run fastapi dev main.py
    ```

2. **Set up NinjaTrader:**

    - Copy strategies from `strategies/` to NinjaTrader
    - Compile in NinjaScript Editor
    - Apply strategy to your chart

3. **Start Trading:**
    - Ensure the RNN server is running
    - NinjaTrader strategy will connect automatically
    - Monitor trades and AI signals

### For Developers

1. **RNN Server Development:**

    ```bash
    cd rnn-server
    uv run fastapi dev main.py
    ```

2. **NinjaTrader Strategy Development:**
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

-   All of the above

## Architecture

```
┌─────────────────┐
│  NinjaTrader 8  │
│   (Strategies)  │
└────────┬────────┘
         │ HTTP
         ▼
┌──────────────────┐
│   RNN Server     │
│   (FastAPI)      │
└────────┬─────────┘
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

## License

[Your License Here]
