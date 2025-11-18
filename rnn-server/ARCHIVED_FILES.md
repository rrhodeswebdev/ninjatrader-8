# Archived Files

This document lists files that have been archived during the backtesting integration.

## backtesting_framework.py.bak

**Original file**: `backtesting_framework.py`
**Archived on**: 2025-11-18
**Reason**: Superseded by backintime framework integration

**What it did**:
- Simple walk-forward validation wrapper
- Basic accuracy-based performance metrics
- Limited to ML model evaluation

**Replaced by**:
- `backintime_rnn_adapter.py` - Full RNN integration with backintime
- `data_loaders.py` - Unified data loading utilities
- `/backtester` - Professional backintime framework with:
  - Realistic order execution
  - Futures margin management
  - Session-based trading
  - Multiple strategy examples

**Why archived**:
The backintime framework provides superior capabilities:
- Professional broker simulation
- Realistic fills (market, limit, TP, SL orders)
- Margin management (initial, maintenance, overnight)
- Trading session handling
- Better performance metrics
- Production-grade validation

**If you need the old functionality**:
The file is preserved as `backtesting_framework.py.bak` for reference.
For walk-forward validation, use the backintime framework's built-in capabilities
or the RNN backtester's comprehensive metrics.

## Migration Path

If you were using `backtesting_framework.py`:

**Old code**:
```python
from backtesting_framework import BacktestingFramework

framework = BacktestingFramework(model_class, config)
results = framework.run_backtest(data, train_window=252, test_window=63)
```

**New code (Option 1 - RNN Backtester)**:
```python
from backtester import Backtester
from model import TradingModel

model = TradingModel(sequence_length=40)
model.train(train_data)

backtester = Backtester(initial_capital=25000.0)
results = backtester.run(test_data, model, verbose=True)
```

**New code (Option 2 - backintime)**:
```python
from backintime_rnn_adapter import run_rnn_backtest
from data_loaders import DataLoader

model = TradingModel(sequence_length=40)
model.train(train_data)

loader = DataLoader()
test_file = loader.save_for_backintime(test_data, 'test.csv')

results = run_rnn_backtest(
    model=model,
    data_file=test_file,
    since=test_data['time'].min(),
    until=test_data['time'].max()
)
```

See `BACKTESTING_INTEGRATION.md` for complete integration guide.
