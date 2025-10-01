# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NinjaTrader 8 algorithmic trading system with a Python-based RNN server for AI-driven trading decisions. The architecture consists of:

1. **NinjaTrader 8 Components** (C#): Custom strategies and indicators that run within the NinjaTrader platform
2. **RNN Server** (Python/FastAPI): Machine learning inference server that provides trading signals

## Architecture

### NinjaTrader 8 Integration
- **Strategies**: C# classes in `strategies/` that inherit from `NinjaTrader.NinjaScript.Strategies.Strategy`
- **Indicators**: Custom technical indicators in `indicators/` (currently empty)
- Strategies communicate with the RNN server via HTTP to receive AI-generated trading signals

### RNN Server
- FastAPI-based Python server in `rnn-server/`
- Provides REST endpoints for health checks and trading predictions
- Uses `uv` for Python package management

## Development Commands

### Python RNN Server
```bash
cd rnn-server
uv run fastapi dev main.py  # Run development server with hot reload
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