"""
Compare RNN Backtesting Approaches

This script demonstrates both backtesting methods:
1. RNN Event-Driven Backtester (fast iteration)
2. backintime Framework (production-grade)

It trains an RNN model and tests it using both approaches, comparing results.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import TradingModel
from backtester import Backtester as RNNBacktester
from data_loaders import DataLoader


def generate_sample_data(n_bars: int = 3000) -> pd.DataFrame:
    """Generate synthetic ES futures data for testing"""
    print(f"\n📊 Generating {n_bars} bars of synthetic ES data...")

    start_price = 4500.0
    start_time = pd.Timestamp('2024-01-02 09:30:00')

    # Generate 1-minute bars during trading hours
    times = []
    current_time = start_time

    while len(times) < n_bars:
        # Skip weekends
        if current_time.dayofweek < 5:
            # Trading hours: 9:30 AM - 4:00 PM
            if 9.5 <= current_time.hour + current_time.minute / 60.0 <= 16.0:
                times.append(current_time)
        current_time += pd.Timedelta(minutes=1)

    times = times[:n_bars]

    # Random walk with realistic intraday patterns
    returns = np.random.normal(0.0001, 0.002, n_bars)
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Add volatility clustering
    for i in range(1, len(close_prices)):
        if abs(returns[i - 1]) > 0.003:  # High volatility begets high volatility
            close_prices[i] += np.random.normal(0, 3.0)

    # Generate OHLC
    high = close_prices + np.abs(np.random.normal(1.5, 0.5, n_bars))
    low = close_prices - np.abs(np.random.normal(1.5, 0.5, n_bars))
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

    print(f"✓ Generated {len(df)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")

    return df


def main():
    """Main comparison workflow"""

    print("\n" + "="*70)
    print("  RNN BACKTESTING COMPARISON")
    print("  Testing both RNN and backintime approaches")
    print("="*70)

    # ========================================================================
    # STEP 1: Generate or load data
    # ========================================================================

    print("\n[STEP 1] Loading data...")

    # Try to load real data, fall back to synthetic
    data_file = Path('historical_data.csv')

    if data_file.exists():
        loader = DataLoader()
        df = loader.load_csv(str(data_file))
        print(f"✓ Loaded {len(df)} bars from {data_file}")
    else:
        print(f"⚠️  {data_file} not found, generating synthetic data")
        df = generate_sample_data(n_bars=3000)

    # ========================================================================
    # STEP 2: Split data
    # ========================================================================

    print("\n[STEP 2] Splitting data...")

    loader = DataLoader()
    train_df, val_df, test_df = loader.split_train_test(
        df,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    # ========================================================================
    # STEP 3: Train RNN model
    # ========================================================================

    print("\n[STEP 3] Training RNN model...")

    model = TradingModel(sequence_length=40)
    model.train(train_df, epochs=30, batch_size=32)

    print("✓ Model training complete")

    # ========================================================================
    # STEP 4: Quick validation with RNN backtester
    # ========================================================================

    print("\n" + "="*70)
    print("[STEP 4] QUICK VALIDATION - RNN Event-Driven Backtester")
    print("="*70)

    rnn_backtester = RNNBacktester(
        initial_capital=25000.0,
        commission_per_contract=2.50,
        slippage_ticks=1,
        daily_goal=500.0,
        daily_max_loss=250.0,
        max_trades_per_day=10
    )

    rnn_results = rnn_backtester.run(test_df, model, verbose=True)

    # ========================================================================
    # STEP 5: Production validation with backintime (if available)
    # ========================================================================

    print("\n" + "="*70)
    print("[STEP 5] PRODUCTION VALIDATION - backintime Framework")
    print("="*70)

    try:
        # Import backintime adapter
        from backintime_rnn_adapter import run_rnn_backtest

        # Convert test data to backintime format
        test_data_file = '/tmp/backintime_test_data.csv'
        loader.save_for_backintime(test_df, test_data_file)

        # Run backintime backtest
        backintime_results = run_rnn_backtest(
            model=model,
            data_file=test_data_file,
            since=test_df['time'].min(),
            until=test_df['time'].max(),
            initial_capital=25000.0,
            atr_multiplier=2.0,
            results_dir='./results'
        )

        has_backintime = True

    except ImportError as e:
        print("\n⚠️  backintime not installed")
        print("   Install with: python3 -m pip install ../backtester/src")
        print("   Skipping production validation...")
        has_backintime = False
        backintime_results = None

    # ========================================================================
    # STEP 6: Compare results
    # ========================================================================

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    if rnn_results.get('total_trades', 0) > 0:
        print("\n🔹 RNN Event-Driven Backtester Results:")
        print(f"  Total Trades:     {rnn_results['total_trades']:>6d}")
        print(f"  Win Rate:         {rnn_results['win_rate']*100:>6.1f}%")
        print(f"  Total P&L:        ${rnn_results['total_pnl']:>8,.2f}")
        print(f"  Total Return:     {rnn_results['total_return_pct']:>6.1f}%")
        print(f"  Sharpe Ratio:     {rnn_results['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:    {rnn_results['sortino_ratio']:>8.2f}")
        print(f"  Profit Factor:    {rnn_results['profit_factor']:>8.2f}")
        print(f"  Max Drawdown:     {rnn_results['max_drawdown']:>6.1f}%")
        print(f"  Avg Trade P&L:    ${rnn_results['avg_trade_pnl']:>8,.2f}")
    else:
        print("\n⚠️  RNN backtester: No trades executed")

    if has_backintime and backintime_results:
        print("\n🔹 backintime Framework Results:")
        stats = backintime_results.get_stats()
        print(f"  Total Trades:     {stats.get('total_trades', 'N/A'):>6}")
        print(f"  Win Rate:         {stats.get('win_rate', 0)*100:>6.1f}%")
        print(f"  Total P&L:        ${stats.get('total_pnl', 0):>8,.2f}")
        print(f"  Sharpe Ratio:     {stats.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Max Drawdown:     {stats.get('max_drawdown', 0):>6.1f}%")

        print("\n📊 Execution Differences:")
        print("  backintime provides:")
        print("    ✓ Realistic order fills (market/limit)")
        print("    ✓ Margin management")
        print("    ✓ Session-based trading")
        print("    ✓ Professional broker simulation")
    else:
        print("\n⚠️  backintime results not available")

    # ========================================================================
    # STEP 7: Recommendations
    # ========================================================================

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if rnn_results.get('total_trades', 0) > 0:
        sharpe = rnn_results['sharpe_ratio']
        win_rate = rnn_results['win_rate']
        profit_factor = rnn_results['profit_factor']

        print("\n📋 Based on RNN backtester results:\n")

        if sharpe > 1.5 and win_rate > 0.55 and profit_factor > 1.5:
            print("  ✅ EXCELLENT - Strategy shows strong performance")
            print("     → Ready for production validation with backintime")
            print("     → Consider live paper trading")

        elif sharpe > 1.0 and win_rate > 0.50:
            print("  ✓ GOOD - Strategy shows promise")
            print("     → Validate with backintime before live trading")
            print("     → Consider parameter optimization")

        elif sharpe > 0.5:
            print("  ⚠️  MARGINAL - Strategy needs improvement")
            print("     → Review risk management parameters")
            print("     → Consider more training data")
            print("     → Adjust confidence threshold")

        else:
            print("  ❌ POOR - Strategy not recommended")
            print("     → Retrain model with different features")
            print("     → Review signal quality")
            print("     → Consider different market regimes")

        print(f"\n  Key Metrics:")
        print(f"    Sharpe Ratio:   {sharpe:.2f} (target: >1.0)")
        print(f"    Win Rate:       {win_rate*100:.1f}% (target: >50%)")
        print(f"    Profit Factor:  {profit_factor:.2f} (target: >1.5)")

    print("\n" + "="*70)
    print("  Comparison complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
