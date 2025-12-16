# NinjaTrader 8 Algorithmic Trading System

A comprehensive algorithmic trading system for NinjaTrader 8 with AI-powered
trading signals via a Python RNN server.

## Project Structure

```
ninjatrader-8/
├── strategies/          # NinjaTrader 8 C# strategies
├── indicators/          # Custom technical indicators (empty)
└── rnn-server/          # Python FastAPI server for AI predictions
    ├── core/            # Pure function modules (validation, transformations, etc.)
    ├── services/        # Request handlers
    ├── features/        # Feature engineering (price action, order flow, multi-timeframe)
    ├── models/          # Trained model files (.pth)
    ├── tests/           # Test suite
    └── utils/           # Functional programming utilities
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

## Getting Started

### For Traders (End Users)

1. **Start the RNN Server:**

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

-   Windows 10 or later (for NinjaTrader)
-   NinjaTrader 8
-   Python 3.10+
-   `uv` package manager: `pip install uv`

## Architecture

```
┌─────────────────┐
│  NinjaTrader 8  │
│   (Strategies)  │
└────────┬────────┘
         │ HTTP POST /analysis
         ▼
┌──────────────────┐
│   RNN Server     │
│   (FastAPI)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│  (LSTM+Attn)    │
│  Trading Signals│
└─────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **FastAPI Server** | HTTP API handling requests from NinjaTrader |
| **TradingModel** | Wrapper managing data, training, and inference |
| **ImprovedTradingRNN** | LSTM + Attention neural network for predictions |
| **Market Regime** | Classifies market conditions (trending/choppy/volatile) |
| **MTF Filter** | Prevents counter-trend trades using 5-min alignment |
| **Signal Stability** | Prevents over-trading by tracking signal changes |
| **Risk Management** | Position sizing, stop loss, and take profit calculation |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analysis` | POST | Main endpoint for historical data & real-time predictions |
| `/health-check` | GET | Server status & model state |
| `/training-status` | GET | Training progress |
| `/save-model` | POST | Persist model to disk |
| `/load-model` | POST | Load model from disk |

### Prediction Pipeline

1. **Data Validation** - Append bar to buffer, validate fields
2. **Multi-Timeframe Filter** - Compare 1-min vs 5-min trends
3. **Market Regime Detection** - Analyze volatility and trend strength
4. **Feature Engineering** - Extract ~87 features (price, time, volatility, multi-TF, volume)
5. **Neural Network** - LSTM + Attention model inference
6. **Confidence Filter** - Apply regime-adjusted confidence threshold
7. **Signal Stability** - Prevent whipsaw trading
8. **Risk Management** - Calculate position size, stops, and targets

### Neural Network Architecture

- **Input:** 15 bars × 87 features
- **LSTM Layers:** 2 layers, 128 hidden units, dropout 0.3
- **Attention:** Self-attention with 4 heads + positional encoding
- **FC Layers:** 128 → 64 → 32 → 3 with BatchNorm and dropout
- **Output:** Softmax → [Long, Short, Hold] + Confidence Score

## Documentation

-   **Development Guidelines:** `CLAUDE.md`

## License

[Your License Here]
