# Backtesting Usage Guide

This guide covers the three backtesting approaches available in this project.

## Overview

| Tool | Purpose | Speed | Use Case |
|------|---------|-------|----------|
| **RNN Backtester** | Fast ML model iteration | Fast | Model development, tuning |
| **backintime Framework** | Production-grade validation | Slower | Final validation, realistic fills |
| **compare_backtesting.py** | Side-by-side comparison | Medium | Comparing approaches |

---

## 1. RNN Event-Driven Backtester

**File:** `backtester.py`

### Quick Start

```python
from backtester import Backtester
from model import TradingModel
from data_loaders import DataLoader

# Load data
loader = DataLoader()
df = loader.load_ninjatrader_csv('path/to/data.csv')

# Train model
model = TradingModel(sequence_length=15)
model.train(df, epochs=30)

# Run backtest
backtester = Backtester(
    initial_capital=25000.0,
    commission_per_contract=0.70,  # Tradovate MNQ round-trip
    slippage_ticks=1,
    contract='MNQ'
)

results = backtester.run(df, model)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 25000.0 | Starting capital |
| `commission_per_contract` | 0.70 | Round-trip commission (Tradovate: $0.35/side) |
| `slippage_ticks` | 1 | Slippage per side in ticks |
| `contract` | 'MNQ' | Contract type (MNQ, NQ, ES, etc.) |
| `daily_goal` | 500.0 | Daily profit target (stops trading if hit) |
| `daily_max_loss` | 250.0 | Daily loss limit (stops trading if hit) |

### Contract Specifications

Defined in `risk_management.py`:

| Contract | Point Value | Tick Size | Tick Value |
|----------|-------------|-----------|------------|
| MNQ | $2.0 | 0.25 | $0.50 |
| NQ | $20.0 | 0.25 | $5.00 |
| ES | $50.0 | 0.25 | $12.50 |
| MES | $5.0 | 0.25 | $1.25 |

### Warm-up Requirements

The backtester requires **115 bars** of warm-up data for Hurst calculation:
- `sequence_length` (15) + Hurst buffer (100) = 115 bars
- With 1-minute data: ~2 hours of pre-trading data

### Trading Hours Gate

Trading is gated to RTH (Regular Trading Hours): **9:30 AM - 4:00 PM**
- Pre-market data is used for warm-up only
- Matches live trading behavior in `AITrader.cs`

---

## 2. backintime Framework

**File:** `backintime_rnn_adapter.py`

### Quick Start

```python
from backintime_rnn_adapter import run_rnn_backtest
from model import TradingModel

# Train or load model
model = TradingModel(sequence_length=15)
model.load('models/trading_model.pth')

# Run production-grade backtest
results = run_rnn_backtest(
    model=model,
    data_file='path/to/data.csv',
    contract='MNQ',
    initial_capital=25000.0
)
```

### Features

- Realistic order execution (market, limit, stop, take-profit)
- Futures margin management (initial, maintenance, overnight)
- Session-based trading with timezone support
- Proper handling of partial fills and slippage

### Installation

```bash
# Install backintime framework
cd backtester
pip install -e src/
pip install -r requirements.txt
```

### Session Configuration

```python
# RTH session (default)
session_start=timedelta(hours=9, minutes=30),  # 9:30 AM
session_end=timedelta(hours=16, minutes=0),     # 4:00 PM
```

---

## 3. Comparison Script

**File:** `examples/compare_backtesting.py`

### Quick Start

```bash
cd rnn-server
uv run python examples/compare_backtesting.py
```

### Configuration

Edit the script to customize:

```python
# Data file (full historical data)
data_file = Path('../backtester/data/MNQ12-25_Sept15-Dec12.csv')

# Date-based split
train_cutoff = pd.Timestamp('2025-12-05 16:00:00')  # Last training day
test_start = pd.Timestamp('2025-12-08 07:30:00')    # Include pre-market

# Contract
CONTRACT = 'MNQ'
MODEL_SEQUENCE_LENGTH = 15
```

### Data Split Strategy

The script splits data by **date** (not ratio) to match live trading:

1. **Training**: All data up to cutoff date
2. **Validation**: Last trading day before test period
3. **Test**: Test period with pre-market data for warm-up

Example for Dec 8-12 testing:
- Train: Sept 15 - Dec 4
- Validate: Dec 5
- Test: Dec 8-12 (7:30 AM - 4:00 PM)

### Output

Results are saved to:
- `results/rnn_backtester_trades.csv` - Trade log
- Console output with comparison metrics

---

## Data Format

### NinjaTrader Format (Semicolon-delimited)

```
YYYYMMDD HHMMSS;open;high;low;close;volume
20251208 093000;25793.50;25795.00;25791.25;25793.75;1234
```

Load with:
```python
df = loader.load_ninjatrader_csv('path/to/data.csv')
```

### Standard CSV Format

```csv
time,open,high,low,close,volume
2025-12-08 09:30:00,25793.50,25795.00,25791.25,25793.75,1234
```

Load with:
```python
df = loader.load_csv('path/to/data.csv')
```

### Auto-detection

The comparison script auto-detects format:
```python
with open(data_file, 'r') as f:
    first_line = f.readline()
if ';' in first_line:
    df = loader.load_ninjatrader_csv(str(data_file))
else:
    df = loader.load_csv(str(data_file))
```

---

## Common Workflows

### 1. Model Development (Fast Iteration)

```python
# Use RNN backtester for quick tests
from backtester import Backtester
from model import TradingModel

model = TradingModel(sequence_length=15)
model.train(train_df, epochs=30)

backtester = Backtester(contract='MNQ')
results = backtester.run(test_df, model)
```

### 2. Production Validation

```python
# Use backintime for final validation
from backintime_rnn_adapter import run_rnn_backtest

results = run_rnn_backtest(
    model=trained_model,
    data_file='data.csv',
    contract='MNQ'
)
```

### 3. Comparing to Live Executions

```python
# Use trade_log_matcher.py to parse NinjaTrader executions
from trade_log_matcher import parse_ninja_executions, print_comparison_report

actual_trades = parse_ninja_executions('executions.csv')
print_comparison_report(actual_trades, backtest_results)
```

---

## Configuration Reference

### config.py Settings

```python
# Model
MODEL_SEQUENCE_LENGTH = 15

# Backtesting
BACKTEST_INITIAL_CAPITAL = 25000.0
BACKTEST_COMMISSION_PER_CONTRACT = 0.70  # Tradovate MNQ
BACKTEST_SLIPPAGE_TICKS = 1

# Risk Management
DAILY_PROFIT_TARGET = 500.0
DAILY_MAX_LOSS = 250.0
```

### Environment Setup

```bash
# Install dependencies
cd rnn-server
uv sync

# Run with uv
uv run python examples/compare_backtesting.py

# Or activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
python examples/compare_backtesting.py
```

---

## Troubleshooting

### "Not enough bars" Error

Ensure your data has at least 115 bars before the first trading time:
- For 9:30 AM trading start, data should begin at ~7:30 AM
- Warm-up: 115 bars = ~2 hours of 1-minute data

### backintime Not Installed

```bash
cd backtester
pip install -e src/
pip install -r requirements.txt
```

### NinjaTrader Data Format Issues

Check for:
- Semicolon delimiter (`;`)
- No header row
- Format: `YYYYMMDD HHMMSS;open;high;low;close;volume`

### Different Results Between Backtesters

Normal due to:
- Order execution models differ
- Slippage handling varies
- Session boundary handling

Use backintime for production validation as it's more realistic.

---

## Example: Custom Date Range Backtest

```python
import pandas as pd
from pathlib import Path
from backtester import Backtester
from model import TradingModel
from data_loaders import DataLoader

# Load data
loader = DataLoader()
df = loader.load_ninjatrader_csv('../backtester/data/MNQ12-25_Sept15-Dec12.csv')

# Define date range
train_end = pd.Timestamp('2025-11-29 16:00:00')
test_start = pd.Timestamp('2025-12-02 07:30:00')
test_end = pd.Timestamp('2025-12-06 16:00:00')

# Split data
train_df = df[df['time'] <= train_end].copy()
test_df = df[(df['time'] >= test_start) & (df['time'] <= test_end)].copy()

# Filter test to include warm-up + RTH
test_df = loader.filter_trading_hours(test_df, start_time="07:30", end_time="16:00")

# Train model
model = TradingModel(sequence_length=15)
model.train(train_df, epochs=30)

# Run backtest
backtester = Backtester(
    initial_capital=25000.0,
    commission_per_contract=0.70,
    contract='MNQ'
)
results = backtester.run(test_df, model)
```

---

## File Reference

| File | Description |
|------|-------------|
| `backtester.py` | RNN event-driven backtester |
| `backintime_rnn_adapter.py` | backintime framework adapter |
| `examples/compare_backtesting.py` | Comparison script |
| `data_loaders.py` | Data loading utilities |
| `trade_log_matcher.py` | NinjaTrader execution parser |
| `config.py` | Configuration settings |
| `risk_management.py` | Contract specs and risk rules |
| `model.py` | TradingModel (RNN) implementation |
