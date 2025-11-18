# Backtesting Integration Guide

This document explains how to use the integrated backtesting capabilities in this project.

## Overview

This project now supports **two complementary backtesting approaches**:

### 1. **RNN Event-Driven Backtester** (`backtester.py`)
- **Purpose**: Rapid iteration during ML model development
- **Strengths**:
  - Direct integration with `TradingModel.predict()`
  - Comprehensive ML-specific metrics (Sharpe, Sortino, MFE/MAE)
  - Daily P&L limits and trading constraints
  - Fast iteration for model tuning
- **Use when**: Training and evaluating RNN models

### 2. **backintime Framework** (`/backtester`)
- **Purpose**: Production-grade strategy backtesting
- **Strengths**:
  - Realistic order execution (market, limit, TP, SL orders)
  - Futures margin management (initial, maintenance, overnight)
  - Session-based trading (RTH, overnight)
  - Multiple timeframe support
  - Professional broker simulation
- **Use when**: Final validation, comparing strategies, production testing

## Installation

### Install backintime Framework

The backintime framework must be installed from the local `/backtester/src` directory:

```bash
cd /home/user/backtester-pro/rnn-server

# Install backintime
python3 -m pip install ../backtester/src

# Install backtester requirements
python3 -m pip install -r ../backtester/requirements.txt
```

Or using `uv`:

```bash
cd /home/user/backtester-pro/rnn-server
uv pip install ../backtester/src
uv pip install -r ../backtester/requirements.txt
```

### Verify Installation

```python
# Test backintime import
python3 -c "import backintime; print('backintime installed successfully')"
```

## Usage Examples

### Option 1: RNN Event-Driven Backtesting (Existing)

**Best for**: Model development and quick iterations

```python
from model import TradingModel
from backtester import Backtester
import pandas as pd

# Load data
df = pd.read_csv('historical_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Split data
train_df = df.iloc[:2000]
test_df = df.iloc[2000:]

# Train model
model = TradingModel(sequence_length=40)
model.train(train_df, epochs=50)

# Backtest
backtester = Backtester(
    initial_capital=25000.0,
    commission_per_contract=2.50,
    slippage_ticks=1,
    daily_goal=500.0,
    daily_max_loss=250.0
)

results = backtester.run(test_df, model, verbose=True)

# Analyze results
print(f"Win Rate: {results['win_rate']*100:.1f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total P&L: ${results['total_pnl']:.2f}")
```

### Option 2: backintime Framework (Production-Grade)

**Best for**: Final validation with realistic execution

```python
from model import TradingModel
from backintime_rnn_adapter import run_rnn_backtest
from data_loaders import DataLoader
from datetime import datetime

# Load and prepare data
loader = DataLoader()
df = loader.load_csv('historical_data.csv')

# Split data
train_df, val_df, test_df = loader.split_train_test(df)

# Train model
model = TradingModel(sequence_length=40)
model.train(train_df, epochs=50)

# Convert test data to backintime format
test_data_file = loader.save_for_backintime(
    test_df,
    'backintime_test_data.csv'
)

# Run production-grade backtest
results = run_rnn_backtest(
    model=model,
    data_file=test_data_file,
    since=test_df['time'].min(),
    until=test_df['time'].max(),
    initial_capital=25000.0,
    atr_multiplier=2.0,
    results_dir='./results'
)

# Results automatically exported to ./results/
```

### Option 3: Combined Approach (Recommended)

**Best for**: Comprehensive validation

```python
from model import TradingModel
from backtester import Backtester as RNNBacktester
from backintime_rnn_adapter import run_rnn_backtest
from data_loaders import DataLoader

# 1. Load and prepare data
loader = DataLoader()
df = loader.load_csv('historical_data.csv')
train_df, val_df, test_df = loader.split_train_test(df)

# 2. Train model
model = TradingModel(sequence_length=40)
model.train(train_df, epochs=50)

# 3. Quick validation with RNN backtester
print("\n=== QUICK VALIDATION (RNN Backtester) ===")
rnn_backtester = RNNBacktester(initial_capital=25000.0)
quick_results = rnn_backtester.run(test_df, model, verbose=True)

# 4. If quick validation looks good, run production backtest
if quick_results['sharpe_ratio'] > 1.0 and quick_results['win_rate'] > 0.50:
    print("\n=== PRODUCTION VALIDATION (backintime) ===")

    # Convert to backintime format
    test_file = loader.save_for_backintime(test_df, 'bt_test_data.csv')

    # Run production backtest
    prod_results = run_rnn_backtest(
        model=model,
        data_file=test_file,
        since=test_df['time'].min(),
        until=test_df['time'].max(),
        initial_capital=25000.0,
        results_dir='./results'
    )
else:
    print("\n⚠️  Quick validation failed. Improve model before production testing.")
```

## Data Format Conversion

Use the `DataLoader` class to convert between formats:

```python
from data_loaders import DataLoader

loader = DataLoader()

# Load any format
df = loader.load_csv('data.csv')

# Save for RNN backtester
loader.save_for_rnn_backtester(df, 'rnn_data.csv')

# Save for backintime
loader.save_for_backintime(df, 'backintime_data.csv')

# Or use convenience function
from data_loaders import convert_between_formats

convert_between_formats(
    'input.csv',
    'output.csv',
    output_format='backintime'  # or 'rnn'
)
```

## Data Utilities

### Filter to Regular Trading Hours

```python
from data_loaders import DataLoader

loader = DataLoader()
df = loader.load_csv('data.csv')

# Keep only 9:30 AM - 4:00 PM ET
df_rth = loader.filter_trading_hours(df, "09:30", "16:00")
```

### Resample Timeframes

```python
# Convert 1-min to 5-min bars
df_5min = loader.resample_timeframe(df, '5min')

# Convert to hourly
df_1h = loader.resample_timeframe(df, '1H')
```

## Comparing Traditional vs RNN Strategies

You can now compare your RNN strategies against traditional indicator-based strategies:

```python
# Run RNN strategy (as shown above)
rnn_results = run_rnn_backtest(...)

# Run traditional strategy from /backtester/strategies
# Example: mean reversion
from backtester.strategies.mean_reversion.strategy import run_with_params

traditional_results = run_with_params(
    since=test_df['time'].min(),
    until=test_df['time'].max()
)

# Compare results
print(f"RNN Sharpe: {rnn_results.get_stats()['sharpe']}")
print(f"Mean Reversion Sharpe: {traditional_results.get_stats()['sharpe']}")
```

## Configuration Reference

### RNN Backtester Parameters

```python
Backtester(
    initial_capital=25000.0,          # Starting capital
    commission_per_contract=2.50,     # Round-trip commission
    slippage_ticks=1,                 # Slippage per side
    tick_value=12.50,                 # ES: $12.50/tick
    daily_goal=500.0,                 # Stop trading after profit
    daily_max_loss=250.0,             # Stop trading after loss
    max_trades_per_day=10             # Trade limit per day
)
```

### backintime Parameters

```python
run_rnn_backtest(
    model=model,                      # Trained TradingModel
    data_file='data.csv',             # backintime format CSV
    since=datetime(...),              # Start date
    until=datetime(...),              # End date
    initial_capital=10000.0,          # Starting capital
    atr_multiplier=2.0,               # ATR multiplier for stops
    session_start=timedelta(hours=9, minutes=30),
    session_end=timedelta(hours=16, minutes=0),
    session_timezone='America/New_York',
    results_dir='./results'           # Export directory
)
```

## File Organization

After integration, your directory structure:

```
rnn-server/
├── backtester.py                    # RNN event-driven backtester
├── backtesting_framework.py.bak     # Archived (superseded by backintime)
├── backintime_rnn_adapter.py        # NEW: RNN adapter for backintime
├── data_loaders.py                  # NEW: Unified data utilities
├── run_backtest.py                  # Quick RNN backtest example
├── examples/
│   └── compare_backtesting.py       # NEW: Compare both approaches
└── results/                         # Backtest results

backtester/                          # backintime framework
├── src/                             # backintime source (install this)
├── strategies/                      # Sample traditional strategies
│   ├── mean_reversion/
│   ├── trend_following_style_1/
│   └── trend_following_style_2/
└── data/                            # Sample data
```

## Best Practices

1. **Development Workflow**:
   - Use RNN backtester during model development
   - Use backintime for final validation
   - Always test on out-of-sample data

2. **Data Management**:
   - Keep original data in standard format
   - Use `DataLoader` for conversions
   - Filter to trading hours for realistic results

3. **Performance Comparison**:
   - Compare both backtesting engines on same data
   - Verify similar results (accounting for execution differences)
   - Use backintime results as the "ground truth"

4. **Production Deployment**:
   - Validate with backintime before live trading
   - Check margin requirements
   - Test session handling (RTH vs overnight)

## Troubleshooting

### Import Error: backintime not found

```bash
# Make sure backintime is installed
cd /home/user/backtester-pro/rnn-server
python3 -m pip install ../backtester/src
```

### Data Format Errors

```python
# Use DataLoader to standardize formats
from data_loaders import DataLoader
loader = DataLoader()
df = loader.load_csv('your_data.csv')  # Auto-detects and standardizes
```

### Different Results Between Backtestors

This is expected due to:
- **Order execution**: backintime simulates realistic fills
- **Margin management**: backintime enforces margin requirements
- **Session handling**: backintime respects trading hours

backintime results are more realistic and should be considered authoritative.

## Next Steps

1. Install backintime framework
2. Try the example scripts
3. Compare RNN vs traditional strategies
4. Validate your model with production-grade backtesting

For more information:
- See `/backtester/README.md` for backintime documentation
- See example strategies in `/backtester/strategies/`
- Check `examples/compare_backtesting.py` for comparison workflow
