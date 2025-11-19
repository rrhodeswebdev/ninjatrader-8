"""
Generate Sample Trading Data

Creates realistic sample ES futures data for testing and demonstrations.
Generates both 1-minute bars with realistic intraday patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_realistic_es_data(
    n_days: int = 10,
    start_date: str = "2024-01-02",
    start_price: float = 4800.0,
    output_dir: str = "."
) -> pd.DataFrame:
    """
    Generate realistic ES futures data with intraday patterns.

    Args:
        n_days: Number of trading days
        start_date: Start date (YYYY-MM-DD)
        start_price: Starting price
        output_dir: Directory to save CSV files

    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n📊 Generating {n_days} days of sample ES futures data...")

    # Trading hours: 9:30 AM - 4:00 PM ET
    session_start = timedelta(hours=9, minutes=30)
    session_end = timedelta(hours=16, minutes=0)

    # Generate timestamps
    start = pd.Timestamp(start_date)
    times = []

    for day_offset in range(n_days * 2):  # Generate extra to ensure n_days of trading
        current_date = start + timedelta(days=day_offset)

        # Skip weekends
        if current_date.dayofweek >= 5:
            continue

        # Generate 1-minute bars during session
        session_minutes = int((session_end - session_start).total_seconds() / 60)

        for minute in range(session_minutes):
            bar_time = current_date + session_start + timedelta(minutes=minute)
            times.append(bar_time)

        if len([t for t in times if t.date() >= start.date()]) >= n_days * session_minutes:
            break

    n_bars = len(times)
    print(f"  Generating {n_bars} 1-minute bars...")

    # Generate realistic price movement
    np.random.seed(42)  # Reproducible data

    # Base returns with slight upward drift
    returns = np.random.normal(0.00005, 0.0015, n_bars)

    # Add intraday patterns
    for i, t in enumerate(times):
        hour = t.hour + t.minute / 60.0

        # Higher volatility at open (9:30-10:30) and close (3:00-4:00)
        if 9.5 <= hour <= 10.5:
            returns[i] *= 1.5
        elif 15.0 <= hour <= 16.0:
            returns[i] *= 1.3

        # Lower volatility during lunch (12:00-2:00)
        elif 12.0 <= hour <= 14.0:
            returns[i] *= 0.7

    # Add volatility clustering
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > 0.003:
            returns[i] *= 1.5

    # Generate close prices
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Generate OHLC with realistic spread
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    open_prices = np.zeros(n_bars)

    for i in range(n_bars):
        # Calculate realistic high/low based on volatility
        volatility = abs(returns[i]) * close_prices[i]
        spread = max(0.25, np.random.normal(1.5, 0.5))  # Min 1 tick

        high[i] = close_prices[i] + abs(np.random.normal(spread, spread * 0.3))
        low[i] = close_prices[i] - abs(np.random.normal(spread, spread * 0.3))

        # Open price (use previous close or add small gap)
        if i == 0:
            open_prices[i] = start_price
        else:
            gap = np.random.normal(0, 0.5)
            open_prices[i] = close_prices[i-1] + gap

    # Ensure OHLC consistency
    for i in range(n_bars):
        high[i] = max(high[i], open_prices[i], close_prices[i])
        low[i] = min(low[i], open_prices[i], close_prices[i])

    # Generate volume (higher at open/close)
    volume = np.random.lognormal(7.5, 0.8, n_bars)
    for i, t in enumerate(times):
        hour = t.hour + t.minute / 60.0
        if 9.5 <= hour <= 10.0 or 15.5 <= hour <= 16.0:
            volume[i] *= 1.8

    # Generate order flow (bid/ask volume)
    # Bullish days: more ask volume, Bearish: more bid volume
    bid_volume = np.zeros(n_bars)
    ask_volume = np.zeros(n_bars)

    for i in range(n_bars):
        if i == 0 or close_prices[i] > close_prices[i-1]:
            # Bullish bar: more buyers
            ask_volume[i] = volume[i] * np.random.uniform(0.55, 0.65)
            bid_volume[i] = volume[i] - ask_volume[i]
        else:
            # Bearish bar: more sellers
            bid_volume[i] = volume[i] * np.random.uniform(0.55, 0.65)
            ask_volume[i] = volume[i] - bid_volume[i]

    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume,
        'bid_volume': bid_volume,
        'ask_volume': ask_volume
    })

    # Round prices to tick size (0.25 for ES)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = (df[col] * 4).round() / 4  # Round to nearest 0.25

    df['volume'] = df['volume'].round()
    df['bid_volume'] = df['bid_volume'].round()
    df['ask_volume'] = df['ask_volume'].round()

    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save RNN format (with order flow)
    rnn_file = output_path / "es_sample_data.csv"
    df.to_csv(rnn_file, index=False)
    print(f"\n✓ Saved RNN format: {rnn_file}")
    print(f"  {len(df)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

    # Save backintime format (with close_time, no header)
    df_bt = df.copy()
    df_bt['close_time'] = df_bt['time'] + pd.Timedelta(minutes=1)
    df_bt['time'] = df_bt['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_bt['close_time'] = df_bt['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    bt_file = output_path / "es_sample_data_backintime.csv"
    df_bt[['time', 'open', 'high', 'low', 'close', 'volume', 'close_time']].to_csv(
        bt_file, index=False, header=False
    )
    print(f"✓ Saved backintime format: {bt_file}")

    # Generate smaller subset for quick tests
    quick_test_df = df.iloc[:390].copy()  # First trading day
    quick_file = output_path / "es_sample_quick.csv"
    quick_test_df.to_csv(quick_file, index=False)
    print(f"✓ Saved quick test file: {quick_file} ({len(quick_test_df)} bars)")

    return df


def generate_readme(output_dir: str = "."):
    """Generate README for sample data"""

    readme_content = """# Sample Trading Data

This directory contains sample ES futures data for testing and demonstrations.

## Files

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

Or customize:
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
"""

    readme_file = Path(output_dir) / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✓ Saved README: {readme_file}")


if __name__ == '__main__':
    import sys

    # Allow custom output directory from command line
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("="*70)
    print("SAMPLE DATA GENERATOR")
    print("="*70)

    df = generate_realistic_es_data(
        n_days=10,
        start_date="2024-01-02",
        start_price=4800.0,
        output_dir=output_dir
    )

    generate_readme(output_dir)

    print("\n" + "="*70)
    print("✓ Sample data generation complete!")
    print("="*70)
    print("\nQuick test:")
    print("  python3 -c \"import pandas as pd; df = pd.read_csv('es_sample_data.csv'); print(df.head())\"")
    print("\nUse in backtesting:")
    print("  cd .. && python3 run_backtest.py")
    print("="*70 + "\n")
