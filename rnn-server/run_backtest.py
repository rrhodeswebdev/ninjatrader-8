"""
Example script to run backtesting on historical data

This demonstrates how to:
1. Load historical data
2. Train the model
3. Run a comprehensive backtest
4. Analyze results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from model import TradingModel
from backtester import Backtester
from risk_management import RiskManager
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    MODEL_SEQUENCE_LENGTH,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_COMMISSION_PER_CONTRACT,
    BACKTEST_SLIPPAGE_TICKS,
    DAILY_GOAL,
    DAILY_MAX_LOSS,
    MAX_TRADES_PER_DAY,
    CONTRACT
)


def load_sample_data():
    """
    Load historical data for backtesting

    Format expected:
    - time: datetime
    - open, high, low, close: OHLC prices
    - volume: trading volume (optional)
    """

    # Check if historical data file exists
    data_file = Path('historical_data.csv')

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("\nTo run backtesting, you need historical data in CSV format with columns:")
        print("  - time (datetime)")
        print("  - open, high, low, close (float)")
        print("  - volume (float, optional)")
        print("\nExample:")
        print("  time,open,high,low,close,volume")
        print("  2025-01-01 09:30:00,4500.00,4502.50,4499.75,4501.25,1000")
        return None

    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])

    print(f"\n Loaded {len(df)} bars from {data_file}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

    return df


def create_synthetic_data(n_bars=5000):
    """
    Create synthetic price data for testing (if no real data available)

    This generates random walk price data with realistic intraday patterns
    """
    print(f"\n  No historical data found. Generating {n_bars} bars of synthetic data for testing...")

    # Start parameters
    start_price = 4500.0
    start_time = pd.Timestamp('2025-01-01 09:30:00')

    # Generate time series (1-minute bars during trading hours)
    times = pd.date_range(start=start_time, periods=n_bars, freq='1min')

    # Random walk with drift
    returns = np.random.normal(0.0001, 0.002, n_bars)  # Slight upward drift
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Add intraday volatility pattern
    for i, t in enumerate(times):
        hour = t.hour + t.minute / 60.0
        # Higher volatility at open and close
        if hour < 10.5 or hour > 15.0:
            close_prices[i] += np.random.normal(0, 2.0)

    # Generate OHLC from close
    high = close_prices + np.abs(np.random.normal(1, 0.5, n_bars))
    low = close_prices - np.abs(np.random.normal(1, 0.5, n_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price

    # Generate volume
    volume = np.random.lognormal(8, 1, n_bars)

    df = pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    })

    print(f" Generated synthetic data:")
    print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

    return df


def plot_results(results: dict, save_path: str = 'backtest_results.png'):
    """
    Plot backtest results: equity curve and trade distribution
    """
    if results.get('total_trades', 0) == 0:
        print("No trades to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')

    # 1. Equity Curve
    equity_curve = results['equity_curve']
    ax1 = axes[0, 0]
    ax1.plot(equity_curve, linewidth=2, color='#2E86AB')
    ax1.fill_between(range(len(equity_curve)), equity_curve.min(), equity_curve,
                      alpha=0.3, color='#2E86AB')
    ax1.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Trade P&L Distribution
    trades_df = results['trades_df']
    ax2 = axes[0, 1]
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']

    ax2.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})')
    ax2.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax2.set_title('Trade P&L Distribution', fontweight='bold')
    ax2.set_xlabel('P&L ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Exit Reasons
    ax3 = axes[1, 0]
    exit_counts = trades_df['exit_reason'].value_counts()
    colors = {'stop': '#EE6C4D', 'target': '#06A77D', 'eod': '#F4A259', 'signal': '#4ECDC4'}
    exit_colors = [colors.get(reason, 'gray') for reason in exit_counts.index]

    ax3.bar(exit_counts.index, exit_counts.values, color=exit_colors, alpha=0.8)
    ax3.set_title('Exit Reasons', fontweight='bold')
    ax3.set_xlabel('Exit Type')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Performance Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
     PERFORMANCE SUMMARY

    Total Trades:        {results['total_trades']:>6d}
    Win Rate:            {results['win_rate']*100:>6.1f}%

    Total P&L:           ${results['total_pnl']:>8,.2f}
    Total Return:        {results['total_return_pct']:>6.1f}%

    Sharpe Ratio:        {results['sharpe_ratio']:>8.2f}
    Sortino Ratio:       {results['sortino_ratio']:>8.2f}
    Profit Factor:       {results['profit_factor']:>8.2f}

    Max Drawdown:        {results['max_drawdown']:>6.1f}%
    Expectancy:          ${results['expectancy']:>8,.2f}

    Avg Win:             ${results['avg_win']:>8,.2f}
    Avg Loss:            ${results['avg_loss']:>8,.2f}

    Largest Win:         ${results['largest_win']:>8,.2f}
    Largest Loss:        ${results['largest_loss']:>8,.2f}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Results chart saved to {save_path}")


def main():
    """
    Main backtesting workflow
    """
    print("\n" + "="*60)
    print("  RNN TRADING SYSTEM - BACKTESTING")
    print("="*60)

    # Step 1: Load data
    df = load_sample_data()

    if df is None:
        df = create_synthetic_data(n_bars=3000)

    if df is None or len(df) < 1000:
        print("\nError: Not enough data for backtesting (need at least 1000 bars)")
        return

    # Step 2: Split data into training and testing periods
    train_size = int(len(df) * 0.7)  # Train on first 70%
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    print(f"\n DATA SPLIT:")
    print(f"  Training:   {len(df_train)} bars ({df_train['time'].min()} to {df_train['time'].max()})")
    print(f"  Testing:    {len(df_test)} bars ({df_test['time'].min()} to {df_test['time'].max()})")

    # Step 3: Train model
    print(f"\n TRAINING MODEL...")
    print(f"  Using sequence length: {MODEL_SEQUENCE_LENGTH}")
    model = TradingModel(sequence_length=MODEL_SEQUENCE_LENGTH)

    # Train on historical data
    model.train(df_train, epochs=50, batch_size=32)

    # Step 4: Run backtest
    print(f"\n RUNNING BACKTEST ON TEST DATA...")

    backtester = Backtester(
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        commission_per_contract=BACKTEST_COMMISSION_PER_CONTRACT,
        slippage_ticks=BACKTEST_SLIPPAGE_TICKS,
        daily_goal=DAILY_GOAL,
        daily_max_loss=DAILY_MAX_LOSS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        contract=CONTRACT
    )

    results = backtester.run(df_test, model, verbose=True)

    # Step 5: Additional analysis
    if results.get('total_trades', 0) > 0:
        print(f"\n ADDITIONAL ANALYSIS:")
        print(f"  Average holding time: {results['avg_bars_held']:.1f} minutes")
        print(f"  Average MFE: {results['avg_mfe']:.2f} points")
        print(f"  Average MAE: {results['avg_mae']:.2f} points")
        print(f"  MFE/MAE ratio: {results['avg_mfe']/max(results['avg_mae'], 0.01):.2f}")

        # Check if matplotlib is available
        try:
            plot_results(results)
        except ImportError:
            print("\n  matplotlib not available. Install with: uv add matplotlib seaborn")

    # Step 6: Save results
    if results.get('total_trades', 0) > 0:
        results_file = Path('backtest_results.csv')
        results['trades_df'].to_csv(results_file, index=False)
        print(f"\n Trade details saved to {results_file}")

    print(f"\n{'='*60}")
    print("  BACKTESTING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
