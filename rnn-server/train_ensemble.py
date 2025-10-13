"""
Train and Evaluate Ensemble Model

This script trains multiple models with different random seeds
and evaluates the ensemble's performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ensemble import EnsemblePredictor
from backtester import Backtester
import argparse


def load_data(file_path: str = 'historical_data.csv'):
    """Load historical data"""
    data_file = Path(file_path)

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("\nGenerating synthetic data for testing...")
        return generate_synthetic_data()

    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'])

    print(f"\n✓ Loaded {len(df)} bars from {data_file}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    return df


def generate_synthetic_data(n_bars=3000):
    """Generate synthetic data for testing"""
    start_price = 4500.0
    start_time = pd.Timestamp('2025-01-01 09:30:00')

    times = pd.date_range(start=start_time, periods=n_bars, freq='1min')
    returns = np.random.normal(0.0001, 0.002, n_bars)
    close_prices = start_price * np.exp(np.cumsum(returns))

    high = close_prices + np.abs(np.random.normal(1, 0.5, n_bars))
    low = close_prices - np.abs(np.random.normal(1, 0.5, n_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    volume = np.random.lognormal(8, 1, n_bars)

    df = pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    })

    print(f"✓ Generated {n_bars} bars of synthetic data")
    return df


def main():
    parser = argparse.ArgumentParser(description='Train ensemble trading model')
    parser.add_argument('--n-models', type=int, default=5, help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per model')
    parser.add_argument('--data-file', type=str, default='historical_data.csv', help='Path to data file')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training data split')
    parser.add_argument('--run-backtest', action='store_true', help='Run backtest after training')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ENSEMBLE MODEL TRAINING")
    print("="*60)

    # Load data
    df = load_data(args.data_file)

    if df is None or len(df) < 1000:
        print("\nError: Not enough data for training (need at least 1000 bars)")
        return

    # Split data
    train_size = int(len(df) * args.train_split)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    print(f"\n📊 DATA SPLIT:")
    print(f"  Training:   {len(df_train)} bars")
    print(f"  Testing:    {len(df_test)} bars")

    # Create ensemble
    print(f"\n🔧 Creating ensemble with {args.n_models} models...")
    ensemble = EnsemblePredictor(n_models=args.n_models, sequence_length=40)

    # Train ensemble
    ensemble.train_ensemble(
        df_train,
        epochs=args.epochs,
        batch_size=32,
        verbose=True
    )

    # Save ensemble
    ensemble.save_ensemble()

    # Evaluate on test data
    print("\n📊 Evaluating ensemble on test data...")
    eval_results = ensemble.evaluate_ensemble(df_test)

    # Compare single model vs ensemble
    print("\n" + "="*60)
    print("SINGLE MODEL VS ENSEMBLE COMPARISON")
    print("="*60)

    print(f"\n  Model Disagreement: {eval_results['disagreement_rate']*100:.1f}%")
    print(f"    (Higher = more diverse ensemble)")

    print(f"\n  Average Confidence:")
    print(f"    Single Model:  {eval_results['avg_individual_confidence']*100:.1f}%")
    print(f"    Ensemble:      {eval_results['avg_ensemble_confidence']*100:.1f}%")
    print(f"    Boost:         {eval_results['confidence_boost']:.2f}x")

    # Run backtest if requested
    if args.run_backtest:
        print("\n" + "="*60)
        print("RUNNING BACKTEST WITH ENSEMBLE")
        print("="*60)

        # Create a wrapper that uses ensemble predictions
        class EnsembleModelWrapper:
            def __init__(self, ensemble):
                self.ensemble = ensemble
                self.sequence_length = ensemble.sequence_length

            def predict(self, df):
                return self.ensemble.predict(df, voting_strategy='soft')

        ensemble_wrapper = EnsembleModelWrapper(ensemble)

        backtester = Backtester(
            initial_capital=25000,
            commission_per_contract=2.50,
            slippage_ticks=1,
            daily_goal=500,
            daily_max_loss=250
        )

        results = backtester.run(df_test, ensemble_wrapper, verbose=True)

        # Save results
        if results.get('total_trades', 0) > 0:
            results_file = Path('ensemble_backtest_results.csv')
            results['trades_df'].to_csv(results_file, index=False)
            print(f"\n✓ Ensemble backtest results saved to {results_file}")

    print("\n" + "="*60)
    print("  ENSEMBLE TRAINING COMPLETE")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Review ensemble performance metrics")
    print("  2. Compare with single model backtests")
    print("  3. If satisfied, deploy ensemble for live trading")
    print("\nTo use ensemble in production:")
    print("  from ensemble import EnsemblePredictor")
    print("  ensemble = EnsemblePredictor(n_models=5)")
    print("  ensemble.load_ensemble()")
    print("  signal, confidence = ensemble.predict(df)")


if __name__ == '__main__':
    main()
