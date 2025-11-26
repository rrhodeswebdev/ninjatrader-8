"""
Standalone script to run ONLY backintime backtesting with the RNN model.

This script:
1. Loads historical data
2. Loads a trained RNN model
3. Runs backintime backtest only (no RNN event-driven backtester)
4. Displays detailed results

Usage:
    cd rnn-server
    python examples/run_backintime_only.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TradingModel
from data_loaders import DataLoader
from backintime_rnn_adapter import run_rnn_backtest
from compare_backtesting import generate_sample_data
from config import (
    MODEL_SEQUENCE_LENGTH,
    CONTRACT
)


def main():
    """Run backintime backtest only"""

    print("\n" + "="*70)
    print("  BACKINTIME BACKTESTING - RNN Strategy")
    print("="*70)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================

    print("\n[STEP 1] Loading data...")

    # Try to load real data, fall back to synthetic
    data_file = Path('../backtester/data/historical_data.csv') #NQ_1m_20251013_20251112

    if data_file.exists():
        loader = DataLoader()
        df = loader.load_csv(str(data_file))
        print(f"✓ Loaded {len(df)} bars from {data_file}")
    else:
        print(f"⚠️  {data_file} not found, generating synthetic stock market data")
        df = generate_sample_data(n_bars=15000)  # Increased for better coverage

        # Save synthetic data to temporary file for backintime to read
        # Must use backintime format: no header, 7 columns with close_time
        temp_data_file = Path('../backtester/data/synthetic_data_temp.csv')
        loader = DataLoader()
        loader.save_for_backintime(df, str(temp_data_file))
        data_file = temp_data_file
        print(f"✓ Saved synthetic data in backintime format to {data_file}")

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
    # STEP 4: Load Trained Model
    # ========================================================================

    print("\n[STEP 4] Loading trained model...")

    # Check if model is trained
    if not model.is_trained:
        print("⚠️  Model is not trained!")
        print("\nTo train the model first, run:")
        print("  python examples/compare_backtesting.py")
        print("\nOr train manually:")
        print("  from model import TradingModel")
        print("  model = TradingModel(sequence_length=20)")
        print("  model.train(training_data)")
        return

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # ========================================================================
    # STEP 5: Run backintime Backtest
    # ========================================================================

    print("\n[STEP 5] Running backintime backtest...")
    print("  Using backintime framework for realistic execution simulation")
    print("  - Limit orders with realistic fills")
    print("  - Margin management")
    print("  - Session-based trading")

    # Determine date range from data
    since = df['time'].min()
    until = df['time'].max()

    print(f"\n  Backtest period: {since} to {until}")
    print(f"  Total bars: {len(df)}")

    # Run backintime backtest
    try:
        results = run_rnn_backtest(
            model=model,
            data_file=str(data_file),
            since=since,
            until=until,
            initial_capital=25000,
            results_dir='./results'
        )

        print("\n✓ Backintime backtest completed successfully")

    except Exception as e:
        print(f"\n❌ Backtest failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # STEP 6: Display Results
    # ========================================================================

    print("\n" + "="*70)
    print("BACKINTIME RESULTS SUMMARY")
    print("="*70)

    try:
        stats = results.get_stats()

        # Stats dataclass uses direct attributes
        # Note: trades_count is NOT stored in stats, calculate from wins + losses
        wins = stats.wins_count
        losses = stats.losses_count
        total_trades = wins + losses

        print(f"\n📊 Trade Statistics:")
        print(f"  Total Trades:        {total_trades:>6}")
        print(f"  Winning Trades:      {wins:>6}")
        print(f"  Losing Trades:       {losses:>6}")

        # win_rate is a Decimal representing percentage (not string)
        win_rate = float(stats.win_rate) if not stats.win_rate.is_nan() else 0.0
        print(f"  Win Rate:            {win_rate:>6.1f}%")

        print(f"\n💰 P&L Metrics:")
        # Calculate total gain from average_profit_all and total trades
        avg_all = float(stats.average_profit_all) if not stats.average_profit_all.is_nan() else 0.0
        total_gain = avg_all * total_trades
        print(f"  Total Gain:          ${total_gain:>10,.2f}")

        # Calculate total return percentage from average_profit_all_percents
        avg_pct = float(stats.average_profit_all_percents) if not stats.average_profit_all_percents.is_nan() else 0.0
        print(f"  Total Return:        {avg_pct:>6.1f}%")

        profit_factor = float(stats.profit_factor) if not stats.profit_factor.is_nan() else 0.0
        print(f"  Profit Factor:       {profit_factor:>8.2f}")

        # Expectancy may not be available on all stats objects
        if hasattr(stats, 'expectancy'):
            expectancy = float(stats.expectancy) if not stats.expectancy.is_nan() else 0.0
            print(f"  Expectancy:          ${expectancy:>10,.2f}")
        else:
            print(f"  Expectancy:          N/A")

        print(f"\n💵 Trade P&L:")
        # Use correct attribute names
        avg_profit_all = float(stats.average_profit_all) if not stats.average_profit_all.is_nan() else 0.0
        avg_loss = float(stats.average_loss) if not stats.average_loss.is_nan() else 0.0

        # TradeProfit objects - access .absolute_profit attribute directly
        if stats.best_deal_absolute and hasattr(stats.best_deal_absolute, 'absolute_profit'):
            best_deal = float(stats.best_deal_absolute.absolute_profit)
        else:
            best_deal = 0.0

        if stats.worst_deal_absolute and hasattr(stats.worst_deal_absolute, 'absolute_profit'):
            worst_deal = float(stats.worst_deal_absolute.absolute_profit)
        else:
            worst_deal = 0.0

        print(f"  Average All Trades:  ${avg_profit_all:>10,.2f}")
        print(f"  Average Loss:        ${avg_loss:>10,.2f}")
        print(f"  Best Deal:           ${best_deal:>10,.2f}")
        print(f"  Worst Deal:          ${worst_deal:>10,.2f}")

    except Exception as e:
        print(f"⚠️  Could not retrieve detailed stats: {e}")
        print("   Results object structure may differ from expected format")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    try:
        # Use the stats object we already have from above
        total_return = float(stats.average_profit_all_percents) if not stats.average_profit_all_percents.is_nan() else 0.0
        win_rate = float(stats.win_rate) if not stats.win_rate.is_nan() else 0.0
        profit_factor = float(stats.profit_factor) if not stats.profit_factor.is_nan() else 0.0

        # Use profit factor and win rate for recommendations (sharpe may not be available)
        if profit_factor > 2.0 and win_rate > 55:
            print("\n  ✅ EXCELLENT - Strategy shows strong performance")
            print("     → Ready for paper trading validation")
            print("     → Consider position sizing optimization")

        elif profit_factor > 1.5 and win_rate > 50:
            print("\n  ✅ GOOD - Strategy shows promise")
            print("     → Run additional tests with different market conditions")
            print("     → Consider parameter optimization")

        elif profit_factor > 1.0:
            print("\n  ⚠️  MARGINAL - Strategy needs improvement")
            print("     → Review entry/exit logic")
            print("     → Consider more training data")
            print("     → Optimize risk parameters")

        else:
            print("\n  ❌ WEAK - Strategy not ready for live trading")
            print("     → Significant improvements needed")
            print("     → Review model architecture")
            print("     → Consider different features or timeframes")

        if total_return < 0:
            print("\n  ⚠️  Negative returns detected")
            print("     → Strategy is losing money overall")
            print("     → Do NOT trade live until profitable")

    except Exception as e:
        print(f"\n  Could not generate recommendations: {e}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("\n  📁 Results saved to: ./results/")
    print("     - stats.csv (performance metrics)")
    print("     - trades.csv (individual trade details)")
    print("     - orders.csv (order execution details)")

    print("\n  📊 Compare with RNN backtester:")
    print("     python examples/compare_backtesting.py")

    print("\n  🔧 Adjust parameters in backintime_rnn_adapter.py:")
    print("     - initial_capital (default: $25,000)")
    print("     - atr_multiplier (default: 2.0 for stops)")
    print("     - percentage_amount (default: 15% for position sizing)")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
