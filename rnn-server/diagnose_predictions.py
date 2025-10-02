"""
Diagnostic script to understand why model only predicts HOLD
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_data_for_trading_signals(csv_file=None, sample_data=None):
    """Analyze data to see what signals would be generated"""

    if csv_file:
        df = pd.read_csv(csv_file)
    elif sample_data is not None:
        df = sample_data
    else:
        # Generate sample data
        print("No data provided, generating sample...")
        np.random.seed(42)
        n_bars = 500
        start_time = datetime.now() - timedelta(minutes=n_bars)
        times = [start_time + timedelta(minutes=i) for i in range(n_bars)]

        current_price = 100.0
        prices = []
        for i in range(n_bars):
            # Random walk
            change = np.random.randn() * 0.3
            current_price = current_price * (1 + change / 100)

            prices.append({
                'time': times[i],
                'open': current_price,
                'high': current_price * (1 + abs(np.random.randn()) * 0.002),
                'low': current_price * (1 - abs(np.random.randn()) * 0.002),
                'close': current_price * (1 + np.random.randn() * 0.001)
            })

        df = pd.DataFrame(prices)

    print("="*70)
    print("DATA ANALYSIS FOR TRADING SIGNALS")
    print("="*70)

    # Calculate bar-to-bar price changes
    df['price_change_pct'] = df['close'].pct_change() * 100

    print(f"\nTotal bars: {len(df)}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Overall change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")

    print(f"\nBar-to-bar price changes:")
    print(f"  Mean: {df['price_change_pct'].mean():.4f}%")
    print(f"  Std: {df['price_change_pct'].std():.4f}%")
    print(f"  Min: {df['price_change_pct'].min():.4f}%")
    print(f"  Max: {df['price_change_pct'].max():.4f}%")

    # Test different thresholds
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    print("\n" + "="*70)
    print("SIGNAL DISTRIBUTION AT DIFFERENT THRESHOLDS")
    print("="*70)

    for threshold in thresholds:
        short = (df['price_change_pct'] < -threshold).sum()
        long = (df['price_change_pct'] > threshold).sum()
        hold = len(df) - short - long

        print(f"\nThreshold: {threshold}%")
        print(f"  SHORT: {short:4d} ({short/len(df)*100:5.1f}%)")
        print(f"  HOLD:  {hold:4d} ({hold/len(df)*100:5.1f}%)")
        print(f"  LONG:  {long:4d} ({long/len(df)*100:5.1f}%)")

        if hold / len(df) > 0.9:
            print(f"  ⚠️  WARNING: {hold/len(df)*100:.0f}% HOLD - threshold too strict!")
        elif short < 10 or long < 10:
            print(f"  ⚠️  WARNING: Insufficient SHORT or LONG signals")
        else:
            print(f"  ✅ Good distribution")

    # Recommend threshold
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    volatility = df['price_change_pct'].std()
    recommended_threshold = max(0.01, volatility * 0.5)

    print(f"\nPrice volatility (std): {volatility:.4f}%")
    print(f"Recommended threshold: {recommended_threshold:.4f}%")
    print(f"\nThis would give:")
    short = (df['price_change_pct'] < -recommended_threshold).sum()
    long = (df['price_change_pct'] > recommended_threshold).sum()
    hold = len(df) - short - long
    print(f"  SHORT: {short:4d} ({short/len(df)*100:5.1f}%)")
    print(f"  HOLD:  {hold:4d} ({hold/len(df)*100:5.1f}%)")
    print(f"  LONG:  {long:4d} ({long/len(df)*100:5.1f}%)")

    return recommended_threshold

if __name__ == "__main__":
    print("\nRunning diagnostic on sample data...")
    threshold = analyze_data_for_trading_signals()

    print("\n" + "="*70)
    print("INSTRUCTIONS")
    print("="*70)
    print(f"""
To fix the HOLD-only predictions:

1. Update the threshold in model.py line 195-200:
   Change from: if price_change_pct > 0.05:
   Change to:   if price_change_pct > {threshold:.4f}:

2. If using real data, run this script with your data:

   import pandas as pd
   df = pd.read_csv('your_data.csv')
   # Make sure df has 'close' column
   analyze_data_for_trading_signals(sample_data=df)

3. The threshold should be ~0.5 to 1x your instrument's volatility

4. After updating, retrain the model with fresh data
""")
