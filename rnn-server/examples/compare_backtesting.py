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


def generate_sample_data(n_bars: int = 15000) -> pd.DataFrame:
    """Generate synthetic stock market data for testing (9:30 AM - 4:00 PM ET)"""
    print(f"\n Generating {n_bars} bars of synthetic stock market data...")

    start_price = 4500.0
    # Start on Monday at 9:30 AM (market open)
    start_time = pd.Timestamp('2024-01-02 09:30:00')  # Tuesday, Jan 2, 2024

    # Generate 1-minute bars during stock market hours
    times = []
    current_time = start_time

    while len(times) < n_bars:
        # Stock market: Monday-Friday, 9:30 AM - 4:00 PM
        if current_time.dayofweek < 5:  # Monday-Friday (0-4)
            # Trading hours: 9:30 AM - 4:00 PM
            hour_decimal = current_time.hour + current_time.minute / 60.0
            if 9.5 <= hour_decimal < 16.0:
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

    print(f"OK: Generated {len(df)} bars")
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
        print(f"OK: Loaded {len(df)} bars from {data_file}")
    else:
        print(f"WARNING:  {data_file} not found, generating synthetic stock market data")
        df = generate_sample_data(n_bars=15000)  # Increased for better coverage

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

    print(f"  Train: {len(train_df)} bars")
    print(f"  Validation: {len(val_df)} bars")
    print(f"  Test: {len(test_df)} bars")

    # ========================================================================
    # STEP 3: Train RNN model
    # ========================================================================

    print("\n[STEP 3] Training RNN model...")
    print(f"  Using sequence length: {MODEL_SEQUENCE_LENGTH}")
    print(f"  Contract: {CONTRACT}")

    model = TradingModel(sequence_length=MODEL_SEQUENCE_LENGTH)
    model.train(train_df, epochs=30, batch_size=32)

    print("OK: Model training complete")

    # ========================================================================
    # STEP 4: Quick validation with RNN backtester
    # ========================================================================

    print("\n" + "="*70)
    print("[STEP 4] QUICK VALIDATION - RNN Event-Driven Backtester")
    print("="*70)

    rnn_backtester = RNNBacktester(
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        commission_per_contract=BACKTEST_COMMISSION_PER_CONTRACT,
        slippage_ticks=BACKTEST_SLIPPAGE_TICKS,
        daily_goal=DAILY_GOAL,
        daily_max_loss=DAILY_MAX_LOSS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        contract=CONTRACT
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
        # IMPORTANT: backintime requires data to start at session boundaries (9:30 AM)
        # We need to find the first 9:30 AM timestamp and include buffer before it

        print(f"\n  Preparing backintime data:")

        # Combine validation and test for analysis
        combined_temp = pd.concat([val_df, test_df], ignore_index=True)
        combined_temp['time'] = pd.to_datetime(combined_temp['time'])
        combined_temp = combined_temp.sort_values('time').reset_index(drop=True)

        # Find first session boundary (9:30 AM) in test data
        test_df_temp = test_df.copy().reset_index(drop=True)  # Reset index to use positions
        test_df_temp['time'] = pd.to_datetime(test_df_temp['time'])

        # Find first bar at 9:30 AM in test data
        test_df_temp['hour'] = test_df_temp['time'].dt.hour
        test_df_temp['minute'] = test_df_temp['time'].dt.minute
        session_start_mask = (test_df_temp['hour'] == 9) & (test_df_temp['minute'] == 30)

        if session_start_mask.any():
            # Get POSITION (not index) of first session boundary
            first_session_pos = test_df_temp[session_start_mask].index[0]
            print(f"    Found session boundary at position {first_session_pos}")
            print(f"    Time: {test_df_temp.loc[first_session_pos, 'time']}")

            # Take test data from first session boundary using POSITION
            test_df_reset = test_df.reset_index(drop=True)
            aligned_test_df = test_df_reset.iloc[first_session_pos:].copy()
        else:
            print(f"    No 9:30 AM boundary found, using original test data")
            aligned_test_df = test_df.copy()

        # Find session-aligned buffer from validation data
        # We want buffer to END at 4:00 PM (market close) the day before test starts
        val_df_temp = val_df.copy().reset_index(drop=True)
        val_df_temp['time'] = pd.to_datetime(val_df_temp['time'])
        val_df_temp['hour'] = val_df_temp['time'].dt.hour
        val_df_temp['minute'] = val_df_temp['time'].dt.minute

        # Find all 9:30 AM timestamps in validation data
        val_session_starts = val_df_temp[
            (val_df_temp['hour'] == 9) & (val_df_temp['minute'] == 30)
        ]

        if len(val_session_starts) > 0:
            # Take the last full day from validation (start at 9:30 AM)
            last_session_pos = val_session_starts.index[-1]
            print(f"    Using validation data from position {last_session_pos}")
            print(f"    Validation session start: {val_df_temp.loc[last_session_pos, 'time']}")

            val_buffer = val_df.reset_index(drop=True).iloc[last_session_pos:].copy()
        else:
            # Fallback: use last 400 bars
            print(f"    No session boundary in validation, using last 400 bars")
            val_buffer = val_df.tail(400).copy()

        print(f"    Buffer bars from validation: {len(val_buffer)}")
        print(f"    Test bars (aligned): {len(aligned_test_df)}")

        # Combine buffer + aligned test data
        combined_df = pd.concat([val_buffer, aligned_test_df], ignore_index=True)
        combined_df['time'] = pd.to_datetime(combined_df['time'])
        combined_df = combined_df.sort_values('time').reset_index(drop=True)

        print(f"    Combined total: {len(combined_df)}")
        print(f"    Date range: {combined_df['time'].iloc[0]} to {combined_df['time'].iloc[-1]}")

        # Verify session alignment
        first_time = combined_df['time'].iloc[0]
        print(f"    First bar time: {first_time.strftime('%Y-%m-%d %H:%M:%S')} ({first_time.strftime('%A')})")
        if combined_df['time'].iloc[-1].hour >= 16:
            print(f"    WARNING:  Last bar is after 4:00 PM - may cause issues")

        # Use OS-independent temp directory
        import tempfile
        test_data_file = os.path.join(tempfile.gettempdir(), 'backintime_test_data.csv')
        print(f"    Saving to: {test_data_file}")
        loader.save_for_backintime(combined_df, test_data_file)

        # Verify the file was saved correctly
        with open(test_data_file, 'r') as f:
            csv_lines = f.readlines()
            print(f"    Verified CSV file has {len(csv_lines)} lines")
            if len(csv_lines) > 0:
                print(f"    First line: {csv_lines[0][:80]}...")
                print(f"    Last line: {csv_lines[-1][:80]}...")

        # Use the FIRST timestamp in combined_df as since (for prefetching)
        # and keep until as the last test timestamp
        since_dt = combined_df['time'].iloc[0]
        until_dt = combined_df['time'].iloc[-1]

        print(f"    Backtest period: {since_dt} to {until_dt}")

        # Remove timezone info if present
        if hasattr(since_dt, 'tzinfo') and since_dt.tzinfo is not None:
            since_dt = since_dt.tz_localize(None)
        if hasattr(until_dt, 'tzinfo') and until_dt.tzinfo is not None:
            until_dt = until_dt.tz_localize(None)

        # Convert to Python datetime objects
        since_dt = since_dt.to_pydatetime()
        until_dt = until_dt.to_pydatetime()

        # Run backintime backtest
        backintime_results = run_rnn_backtest(
            model=model,
            data_file=test_data_file,
            since=since_dt,
            until=until_dt,
            initial_capital=25000.0,
            atr_multiplier=2.0,
            results_dir='./results'
        )

        has_backintime = True

    except ImportError as e:
        print("\nWARNING:  backintime not installed")
        print("   Install with: python3 -m pip install ../backtester/src")
        print("   Skipping production validation...")
        has_backintime = False
        backintime_results = None

    # ========================================================================
    # STEP 6: Compare results
    # ========================================================================

    if has_backintime and backintime_results:
        # Use the comparison utility
        from data_loaders import compare_backtest_results

        comparison_df = compare_backtest_results(
            rnn_results,
            backintime_results,
            verbose=True
        )
    else:
        # Manual display if backintime not available
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        if rnn_results.get('total_trades', 0) > 0:
            print("\n* RNN Event-Driven Backtester Results:")
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
            print("\nWARNING:  RNN backtester: No trades executed")

        print("\nWARNING:  backintime results not available")
        print("   Install with: python3 -m pip install ../backtester/src")

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

        print("\n Based on RNN backtester results:\n")

        if sharpe > 1.5 and win_rate > 0.55 and profit_factor > 1.5:
            print("  OK EXCELLENT - Strategy shows strong performance")
            print("     -> Ready for production validation with backintime")
            print("     -> Consider live paper trading")

        elif sharpe > 1.0 and win_rate > 0.50:
            print("  OK: GOOD - Strategy shows promise")
            print("     -> Validate with backintime before live trading")
            print("     -> Consider parameter optimization")

        elif sharpe > 0.5:
            print("  WARNING:  MARGINAL - Strategy needs improvement")
            print("     -> Review risk management parameters")
            print("     -> Consider more training data")
            print("     -> Adjust confidence threshold")

        else:
            print("  X POOR - Strategy not recommended")
            print("     -> Retrain model with different features")
            print("     -> Review signal quality")
            print("     -> Consider different market regimes")

        print(f"\n  Key Metrics:")
        print(f"    Sharpe Ratio:   {sharpe:.2f} (target: >1.0)")
        print(f"    Win Rate:       {win_rate*100:.1f}% (target: >50%)")
        print(f"    Profit Factor:  {profit_factor:.2f} (target: >1.5)")

    print("\n" + "="*70)
    print("  Comparison complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
