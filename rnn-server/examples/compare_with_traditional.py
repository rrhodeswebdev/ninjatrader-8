"""
Compare RNN Backtester with Traditional Strategy Baseline

This compares your RNN approach against a simple traditional strategy
to show the value-add of the ML model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
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


class SimpleEMACrossStrategy:
    """Traditional EMA crossover strategy for comparison"""

    def __init__(self, fast_period=9, slow_period=21):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.sequence_length = slow_period + 1

    def predict(self, df):
        """Generate signal based on EMA crossover"""
        if len(df) < self.slow_period:
            return 'hold', 0.5

        # Calculate EMAs
        fast_ema = df['close'].ewm(span=self.fast_period).mean()
        slow_ema = df['close'].ewm(span=self.slow_period).mean()

        # Get last two values
        fast_current = fast_ema.iloc[-1]
        fast_prev = fast_ema.iloc[-2]
        slow_current = slow_ema.iloc[-1]
        slow_prev = slow_ema.iloc[-2]

        # Crossover logic
        if fast_prev <= slow_prev and fast_current > slow_current:
            return 'long', 0.7  # Bullish crossover
        elif fast_prev >= slow_prev and fast_current < slow_current:
            return 'short', 0.7  # Bearish crossover
        else:
            return 'hold', 0.5


def main():
    print("\n" + "="*70)
    print("  RNN vs TRADITIONAL STRATEGY COMPARISON")
    print("  Testing ML-driven vs Rule-based approaches")
    print("="*70)

    # Load data
    print("\n[STEP 1] Loading data...")
    data_file = Path('historical_data.csv')

    if not data_file.exists():
        print(f"❌ {data_file} not found")
        print("   Please run the backtest with historical_data.csv first")
        return

    loader = DataLoader()
    df = loader.load_csv(str(data_file))
    print(f"✓ Loaded {len(df)} bars")

    # Split data
    print("\n[STEP 2] Splitting data...")
    train_df, val_df, test_df = loader.split_train_test(df, train_ratio=0.6, validation_ratio=0.2)
    print(f"  Training:   {len(train_df)} bars")
    print(f"  Validation: {len(val_df)} bars")
    print(f"  Testing:    {len(test_df)} bars")

    # ========================================================================
    # BASELINE: Traditional EMA Crossover Strategy
    # ========================================================================

    print("\n" + "="*70)
    print("[STEP 3] BASELINE - Traditional EMA Crossover")
    print("="*70)

    traditional_strategy = SimpleEMACrossStrategy(fast_period=9, slow_period=21)

    backtester_traditional = RNNBacktester(
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        commission_per_contract=BACKTEST_COMMISSION_PER_CONTRACT,
        slippage_ticks=BACKTEST_SLIPPAGE_TICKS,
        daily_goal=DAILY_GOAL,
        daily_max_loss=DAILY_MAX_LOSS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        contract=CONTRACT
    )

    traditional_results = backtester_traditional.run(test_df, traditional_strategy, verbose=False)

    # ========================================================================
    # RNN: Machine Learning Strategy
    # ========================================================================

    print("\n" + "="*70)
    print("[STEP 4] RNN - Machine Learning Strategy")
    print("="*70)

    print(f"  Training model with sequence length: {MODEL_SEQUENCE_LENGTH}")
    model = TradingModel(sequence_length=MODEL_SEQUENCE_LENGTH)
    model.train(train_df, epochs=30, batch_size=32)

    backtester_rnn = RNNBacktester(
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        commission_per_contract=BACKTEST_COMMISSION_PER_CONTRACT,
        slippage_ticks=BACKTEST_SLIPPAGE_TICKS,
        daily_goal=DAILY_GOAL,
        daily_max_loss=DAILY_MAX_LOSS,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        contract=CONTRACT
    )

    rnn_results = backtester_rnn.run(test_df, model, verbose=False)

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "="*70)
    print("COMPARISON: RNN vs Traditional")
    print("="*70)

    comparison = pd.DataFrame({
        'Metric': [
            'Total Return ($)',
            'Total Return (%)',
            'Sharpe Ratio',
            'Win Rate (%)',
            'Profit Factor',
            'Max Drawdown (%)',
            'Total Trades',
            'Avg Win ($)',
            'Avg Loss ($)',
            'Final Equity ($)'
        ],
        'Traditional (EMA)': [
            f"${traditional_results['total_return']:.2f}",
            f"{traditional_results['total_return_pct']:.2f}%",
            f"{traditional_results['sharpe_ratio']:.2f}",
            f"{traditional_results['win_rate']*100:.1f}%",
            f"{traditional_results.get('profit_factor', 0):.2f}",
            f"{traditional_results['max_drawdown']*100:.1f}%",
            f"{traditional_results['total_trades']}",
            f"${traditional_results.get('avg_win', 0):.2f}",
            f"${traditional_results.get('avg_loss', 0):.2f}",
            f"${traditional_results['final_equity']:.2f}"
        ],
        'RNN (ML)': [
            f"${rnn_results['total_return']:.2f}",
            f"{rnn_results['total_return_pct']:.2f}%",
            f"{rnn_results['sharpe_ratio']:.2f}",
            f"{rnn_results['win_rate']*100:.1f}%",
            f"{rnn_results.get('profit_factor', 0):.2f}",
            f"{rnn_results['max_drawdown']*100:.1f}%",
            f"{rnn_results['total_trades']}",
            f"${rnn_results.get('avg_win', 0):.2f}",
            f"${rnn_results.get('avg_loss', 0):.2f}",
            f"${rnn_results['final_equity']:.2f}"
        ]
    })

    print("\n" + comparison.to_string(index=False))

    # Calculate improvement
    print("\n" + "="*70)
    print("RNN IMPROVEMENT OVER TRADITIONAL")
    print("="*70)

    return_improvement = ((rnn_results['total_return'] - traditional_results['total_return']) /
                          abs(traditional_results['total_return']) * 100 if traditional_results['total_return'] != 0 else 0)
    sharpe_improvement = rnn_results['sharpe_ratio'] - traditional_results['sharpe_ratio']

    print(f"Return Improvement:     {return_improvement:+.1f}%")
    print(f"Sharpe Improvement:     {sharpe_improvement:+.2f}")
    print(f"Win Rate Difference:    {(rnn_results['win_rate'] - traditional_results['win_rate'])*100:+.1f}%")

    if rnn_results['total_return'] > traditional_results['total_return']:
        print("\n✓ RNN strategy outperformed traditional strategy!")
    else:
        print("\n⚠️  RNN strategy underperformed - consider retraining or tuning")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
