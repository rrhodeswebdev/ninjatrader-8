# Backtesting Examples

This directory contains example scripts demonstrating the integrated backtesting capabilities.

## Available Examples

### compare_backtesting.py

**Purpose**: Compare RNN event-driven backtester vs backintime framework

**What it does**:
1. Generates or loads historical data
2. Splits into train/validation/test sets
3. Trains an RNN model
4. Runs quick validation with RNN backtester
5. Runs production validation with backintime (if installed)
6. Compares results and provides recommendations

**Usage**:
```bash
cd /home/user/backtester-pro/rnn-server

# Quick test with synthetic data
python3 examples/compare_backtesting.py

# Or provide your own data as historical_data.csv
# Format: time,open,high,low,close,volume
python3 examples/compare_backtesting.py
```

**Requirements**:
- Basic: Works without backintime (RNN backtester only)
- Full: Install backintime for production validation

**Install backintime**:
```bash
cd /home/user/backtester-pro/rnn-server
python3 -m pip install ../backtester/src
python3 -m pip install -r ../backtester/requirements.txt
```

## Expected Output

The script will:
1. Load/generate data
2. Train model (may take a few minutes)
3. Run RNN backtester (fast)
4. Run backintime backtest (if available)
5. Display comparison summary
6. Provide recommendations based on performance

## Data Format

If providing your own `historical_data.csv`:

```csv
time,open,high,low,close,volume
2024-01-02 09:30:00,4500.00,4502.50,4499.75,4501.25,1000
2024-01-02 09:31:00,4501.25,4503.00,4500.50,4502.00,1200
...
```

## Interpreting Results

### RNN Backtester Metrics
- **Sharpe Ratio**: >1.0 is good, >1.5 is excellent
- **Win Rate**: >50% is profitable, >55% is strong
- **Profit Factor**: >1.5 is good, >2.0 is excellent
- **Max Drawdown**: <10% is good, <5% is excellent

### backintime Results
- More conservative due to realistic execution
- Accounts for slippage and realistic fills
- Enforces margin requirements
- Use these results for production decisions

## Next Steps

After running the comparison:

1. **Good results** (Sharpe >1.0, Win Rate >50%):
   - Review risk parameters
   - Consider live paper trading
   - Validate on more recent data

2. **Marginal results** (Sharpe 0.5-1.0):
   - Tune confidence threshold
   - Optimize risk management
   - Try different training parameters

3. **Poor results** (Sharpe <0.5):
   - Retrain with more/better data
   - Review feature engineering
   - Consider different model architecture

## Additional Examples

You can create your own examples by following these patterns:

**RNN backtesting only**:
```python
from model import TradingModel
from backtester import Backtester

model = TradingModel(sequence_length=40)
model.train(train_data)

backtester = Backtester(initial_capital=25000.0)
results = backtester.run(test_data, model)
```

**backintime integration**:
```python
from backintime_rnn_adapter import run_rnn_backtest
from data_loaders import DataLoader

loader = DataLoader()
test_file = loader.save_for_backintime(test_data, 'test.csv')

results = run_rnn_backtest(
    model=model,
    data_file=test_file,
    since=test_data['time'].min(),
    until=test_data['time'].max()
)
```

See `../BACKTESTING_INTEGRATION.md` for complete documentation.
