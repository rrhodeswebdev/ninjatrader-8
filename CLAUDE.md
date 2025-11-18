# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NinjaTrader 8 algorithmic trading system with a Python-based RNN server for AI-driven trading decisions. The architecture consists of:

1. **NinjaTrader 8 Components** (C#): Custom strategies and indicators that run within the NinjaTrader platform
2. **RNN Server** (Python/FastAPI): Machine learning inference server that provides trading signals
3. **Backtesting Framework** (Python): Integrated backtesting with two complementary approaches:
   - RNN event-driven backtester for rapid model iteration
   - backintime framework for production-grade validation

## Architecture

### NinjaTrader 8 Integration
- **Strategies**: C# classes in `strategies/` that inherit from `NinjaTrader.NinjaScript.Strategies.Strategy`
- **Indicators**: Custom technical indicators in `indicators/` (currently empty)
- Strategies communicate with the RNN server via HTTP to receive AI-generated trading signals

### RNN Server
- FastAPI-based Python server in `rnn-server/`
- Provides REST endpoints for health checks and trading predictions
- Uses `uv` for Python package management
- Includes integrated backtesting capabilities

### Backtesting Framework
- **Location**: `/backtester` (backintime framework) + `/rnn-server` (RNN integration)
- **Two complementary approaches**:
  1. **RNN Event-Driven Backtester** (`rnn-server/backtester.py`):
     - Fast iteration during model development
     - Direct integration with ML model predictions
     - Comprehensive ML metrics (Sharpe, Sortino, MFE/MAE)
  2. **backintime Framework** (`/backtester`):
     - Production-grade futures backtesting
     - Realistic order execution and fills
     - Margin management and session handling
     - Sample strategies: mean reversion, trend following, MACD
- **Integration**: `rnn-server/backintime_rnn_adapter.py` bridges RNN models with backintime
- **Data utilities**: `rnn-server/data_loaders.py` for format conversions

## Development Commands

### Python RNN Server
```bash
cd rnn-server
uv run fastapi dev main.py  # Run development server with hot reload
```

### Backtesting

**Quick RNN backtest** (for model development):
```bash
cd rnn-server
python3 run_backtest.py
```

**Production validation** (backintime framework):
```bash
cd rnn-server
# First, install backintime framework
python3 -m pip install ../backtester/src
python3 -m pip install -r ../backtester/requirements.txt

# Run comparison of both approaches
python3 examples/compare_backtesting.py
```

**Traditional strategies** (from /backtester):
```bash
cd backtester
python3 strategies/mean_reversion/strategy.py
python3 strategies/trend_following_style_2/strategy.py
```

### NinjaTrader 8 Strategies
NinjaTrader strategies must be compiled within the NinjaTrader 8 platform:
1. Copy `.cs` files from `strategies/` to NinjaTrader's custom strategy folder (typically `Documents/NinjaTrader 8/bin/Custom/Strategies/`)
2. Open NinjaTrader 8
3. Open NinjaScript Editor (Tools > Edit NinjaScript > Strategy)
4. Compile via F5 or Tools > Compile

## Key Conventions

### NinjaTrader Strategy Structure
- Strategies must be in the `NinjaTrader.NinjaScript.Strategies` namespace
- Core lifecycle methods:
  - `OnStateChange()`: Initialize strategy parameters and settings when `State == State.SetDefaults`
  - `OnBarUpdate()`: Main trading logic executed on each bar update
- Important strategy properties set in `SetDefaults`:
  - `BarsRequiredToTrade`: Minimum bars needed before trading (default: 20)
  - `Calculate`: When strategy calculations occur (typically `Calculate.OnBarClose`)
  - Entry/exit handling, slippage, and order management settings

### Python Server
- Uses Python 3.13+
- FastAPI with standard extras for async HTTP server
- Package management via `uv` (modern Python package installer)

### Backtesting
- **RNN Backtester**: Event-driven, integrates with `TradingModel.predict()`
  - Daily P&L limits and risk management
  - Trade-by-trade analysis with MFE/MAE
  - Ideal for rapid model iteration
- **backintime Framework**: Professional futures backtesting
  - Install from `/backtester/src` directory
  - Realistic broker simulation (market/limit/TP/SL orders)
  - Futures margin management (initial, maintenance, overnight)
  - Session-based trading with timezone support
  - Multiple timeframe support
- **Integration**: Use `backintime_rnn_adapter.py` to run RNN models in backintime
- **Data conversion**: Use `data_loaders.py` to convert between formats
- See `rnn-server/BACKTESTING_INTEGRATION.md` for complete guide

### When to Use Which Backtester
- **Use RNN backtester** when:
  - Developing and tuning ML models
  - Quick iterations needed
  - Testing different confidence thresholds
  - Analyzing ML-specific metrics
- **Use backintime** when:
  - Final validation before live trading
  - Comparing RNN vs traditional strategies
  - Testing with realistic order execution
  - Validating margin requirements
  - Production-grade results needed