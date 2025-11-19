# Sample Trading Data

This directory contains sample ES futures data for testing and demonstrations.

## Quick Start

Generate sample data by running:

```bash
cd /home/user/backtester-pro/rnn-server/sample_data
python3 generate_sample_data.py
```

This will create:
- `es_sample_data.csv` - Full dataset (~10 days, RNN format)
- `es_sample_data_backintime.csv` - backintime format
- `es_sample_quick.csv` - Quick test file (1 day)

## Files Generated

### es_sample_data.csv
- **Format**: RNN backtester format
- **Columns**: time, open, high, low, close, volume, bid_volume, ask_volume
- **Duration**: ~10 trading days
- **Bars**: ~3,900 (1-minute bars, 9:30 AM - 4:00 PM ET)
- **Use for**: Training models, testing backtesting, development

### es_sample_data_backintime.csv
- **Format**: backintime framework format
- **Columns**: open_time, open, high, low, close, volume, close_time (no header)
- **Duration**: Same as above
- **Use for**: backintime backtesting, production validation

### es_sample_quick.csv
- **Format**: RNN format
- **Duration**: 1 trading day (~390 bars)
- **Use for**: Quick tests, CI/CD, debugging

## Data Characteristics

### Realistic Features
- ✓ Regular trading hours (9:30 AM - 4:00 PM ET)
- ✓ Weekday trading only
- ✓ Higher volatility at open (9:30-10:30) and close (3:00-4:00)
- ✓ Lower volatility during lunch (12:00-2:00)
- ✓ Volatility clustering
- ✓ Volume patterns (higher at open/close)
- ✓ Order flow imbalance (bid/ask volume)
- ✓ ES tick size (0.25 points)

### Price Action
- Starting price: ~$4,800
- Slight upward drift (realistic market tendency)
- Intraday mean reversion
- Trend persistence with noise

## Usage Examples

### Load for RNN Backtesting
```python
import pandas as pd

df = pd.read_csv('sample_data/es_sample_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Use for training/testing
from model import TradingModel
model = TradingModel(sequence_length=40)
model.train(df.iloc[:2000])

from backtester import Backtester
backtester = Backtester(initial_capital=25000.0)
results = backtester.run(df.iloc[2000:], model)
```

### Load for backintime
```python
from backintime_rnn_adapter import run_rnn_backtest

results = run_rnn_backtest(
    model=trained_model,
    data_file='sample_data/es_sample_data_backintime.csv',
    since=datetime(2024, 1, 2, 9, 30),
    until=datetime(2024, 1, 15, 16, 0),
    initial_capital=25000.0
)
```

### Quick Test
```python
# Use quick file for rapid tests
df_quick = pd.read_csv('sample_data/es_sample_quick.csv')
# Single day, perfect for debugging
```

## Regenerating Data

To regenerate with different parameters:

```bash
cd sample_data
python3 generate_sample_data.py
```

Or customize in Python:
```python
from generate_sample_data import generate_realistic_es_data

df = generate_realistic_es_data(
    n_days=20,              # More days
    start_date="2024-03-01",
    start_price=5000.0,
    output_dir="custom_data"
)
```

## Notes

- This is synthetic data for **testing only**
- Not suitable for production trading decisions
- Patterns are simplified but realistic
- Use real historical data for actual model training
- Seed is fixed (42) for reproducibility

## Data Quality

✓ No missing values
✓ Proper OHLC relationships (High >= Open/Close, Low <= Open/Close)
✓ Valid timestamps (no gaps during trading hours)
✓ Consistent tick sizes
✓ Realistic volume distributions

## Integration with Examples

The comparison example automatically uses this data if no `historical_data.csv` is found:

```bash
cd /home/user/backtester-pro/rnn-server

# Generate sample data
cd sample_data && python3 generate_sample_data.py && cd ..

# Run comparison
python3 examples/compare_backtesting.py
```
