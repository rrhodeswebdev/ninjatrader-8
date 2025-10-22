"""
Walk-Forward Optimization System
=================================

Implements proper walk-forward validation to prevent overfitting.

Process:
1. Split data into windows (e.g., train on 3 months, test on 1 month)
2. Train model on training window
3. Test on out-of-sample test window
4. Roll forward and repeat
5. Aggregate results across all windows

This gives realistic performance estimates and prevents curve-fitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json


class WalkForwardOptimizer:
    """
    Walk-forward validation for trading models
    """

    def __init__(
        self,
        train_window_days: int = 90,  # Train on 3 months
        test_window_days: int = 30,   # Test on 1 month
        step_days: int = 30,           # Roll forward 1 month
        min_samples_per_window: int = 1000
    ):
        """
        Args:
            train_window_days: Days of data for training
            test_window_days: Days of data for testing
            step_days: Days to roll forward for next window
            min_samples_per_window: Minimum bars required per window
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_samples_per_window = min_samples_per_window

        self.results = []

    def create_windows(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create train/test window pairs for walk-forward validation

        Args:
            df: Full dataset with 'time' column

        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        windows = []
        start_date = df['time'].min()
        end_date = df['time'].max()

        current_train_start = start_date

        while True:
            # Calculate window boundaries
            train_end = current_train_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Extract windows
            train_df = df[(df['time'] >= current_train_start) & (df['time'] < train_end)].copy()
            test_df = df[(df['time'] >= test_start) & (df['time'] < test_end)].copy()

            # Validate window sizes
            if len(train_df) >= self.min_samples_per_window and len(test_df) >= self.min_samples_per_window // 3:
                windows.append((train_df, test_df))
                print(f"Window {len(windows)}: Train [{current_train_start.date()} to {train_end.date()}] "
                      f"({len(train_df)} bars), Test [{test_start.date()} to {test_end.date()}] "
                      f"({len(test_df)} bars)")

            # Roll forward
            current_train_start += timedelta(days=self.step_days)

        return windows

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        model_class,
        model_kwargs: Dict = None,
        train_kwargs: Dict = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete walk-forward validation

        Args:
            df: Full dataset
            model_class: TradingModel class or similar
            model_kwargs: Kwargs for model initialization
            train_kwargs: Kwargs for model.train()
            verbose: Print progress

        Returns:
            Dictionary with aggregated results
        """
        from trading_metrics import (
            calculate_sharpe_ratio,
            calculate_profit_factor,
            calculate_win_rate,
            calculate_max_drawdown,
            calculate_expectancy
        )

        model_kwargs = model_kwargs or {}
        train_kwargs = train_kwargs or {'epochs': 50, 'batch_size': 32, 'validation_split': 0.2}

        print("\n" + "="*70)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*70)
        print(f"Train window: {self.train_window_days} days")
        print(f"Test window: {self.test_window_days} days")
        print(f"Step size: {self.step_days} days")
        print("="*70 + "\n")

        # Create windows
        windows = self.create_windows(df)

        if len(windows) == 0:
            return {
                'error': 'Not enough data for walk-forward validation',
                'windows': 0
            }

        print(f"\nCreated {len(windows)} walk-forward windows\n")

        # Run validation on each window
        self.results = []

        for i, (train_df, test_df) in enumerate(windows):
            print(f"\n{'='*70}")
            print(f"WINDOW {i+1}/{len(windows)}")
            print(f"{'='*70}")

            # Initialize fresh model
            model = model_class(**model_kwargs)

            # Train on training window
            print(f"\nTraining on {len(train_df)} bars...")
            model.train(train_df, **train_kwargs)

            # Test on out-of-sample window
            print(f"\nTesting on {len(test_df)} bars (OUT-OF-SAMPLE)...")
            predictions = []
            actuals = []
            returns = []
            confidences = []

            for j in range(model.sequence_length, len(test_df) - 5):
                # Get sequence up to current bar
                sequence_data = test_df.iloc[:j+1]

                try:
                    signal, confidence = model.predict(sequence_data)

                    # Map signal to numeric
                    signal_map = {'short': 0, 'hold': 1, 'long': 2}
                    pred = signal_map[signal]

                    # Calculate actual return
                    current_price = test_df.iloc[j]['close']
                    next_price = test_df.iloc[j+1]['close']
                    ret = (next_price - current_price) / current_price

                    # Determine actual label
                    threshold = 0.0005
                    if ret > threshold:
                        actual = 2  # Long
                    elif ret < -threshold:
                        actual = 0  # Short
                    else:
                        actual = 1  # Hold

                    predictions.append(pred)
                    actuals.append(actual)
                    returns.append(ret)
                    confidences.append(confidence)

                except Exception as e:
                    if verbose:
                        print(f"Prediction error: {e}")
                    continue

            # Calculate metrics for this window
            if len(predictions) > 0:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                returns = np.array(returns)
                confidences = np.array(confidences)

                # Filter to trading signals only
                trade_mask = predictions != 1
                trade_returns = returns[trade_mask] if np.sum(trade_mask) > 0 else np.array([])

                accuracy = np.mean(predictions == actuals)
                win_rate = calculate_win_rate(trade_returns) if len(trade_returns) > 0 else 0
                sharpe = calculate_sharpe_ratio(trade_returns, periods_per_year=252*390) if len(trade_returns) > 10 else 0
                profit_factor = calculate_profit_factor(trade_returns) if len(trade_returns) > 0 else 0
                expectancy = calculate_expectancy(trade_returns) if len(trade_returns) > 0 else 0

                cumulative_returns = np.cumsum(trade_returns) if len(trade_returns) > 0 else np.array([0])
                max_dd = calculate_max_drawdown(cumulative_returns)

                window_result = {
                    'window': i + 1,
                    'train_start': train_df['time'].min(),
                    'train_end': train_df['time'].max(),
                    'test_start': test_df['time'].min(),
                    'test_end': test_df['time'].max(),
                    'train_bars': len(train_df),
                    'test_bars': len(test_df),
                    'num_predictions': len(predictions),
                    'num_trades': len(trade_returns),
                    'accuracy': accuracy,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe,
                    'profit_factor': profit_factor,
                    'expectancy': expectancy,
                    'max_drawdown': max_dd,
                    'total_return': np.sum(trade_returns) if len(trade_returns) > 0 else 0,
                    'avg_confidence': np.mean(confidences) if len(confidences) > 0 else 0
                }

                self.results.append(window_result)

                print(f"\n📊 WINDOW {i+1} RESULTS (OUT-OF-SAMPLE):")
                print(f"  Predictions: {len(predictions)} | Trades: {len(trade_returns)}")
                print(f"  Accuracy: {accuracy:.2%}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Sharpe Ratio: {sharpe:.3f}")
                print(f"  Profit Factor: {profit_factor:.3f}")
                print(f"  Max Drawdown: {max_dd*100:.2f}%")
                print(f"  Avg Confidence: {np.mean(confidences):.2%}")

            else:
                print(f"\n⚠️  Window {i+1}: No valid predictions")

        # Aggregate results across all windows
        print(f"\n{'='*70}")
        print("WALK-FORWARD SUMMARY (AGGREGATED)")
        print(f"{'='*70}\n")

        if len(self.results) == 0:
            return {'error': 'No valid windows', 'windows': len(windows)}

        # Calculate average metrics
        avg_accuracy = np.mean([r['accuracy'] for r in self.results])
        avg_win_rate = np.mean([r['win_rate'] for r in self.results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in self.results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in self.results])
        avg_expectancy = np.mean([r['expectancy'] for r in self.results])
        avg_max_dd = np.mean([r['max_drawdown'] for r in self.results])
        total_trades = sum([r['num_trades'] for r in self.results])
        total_return = sum([r['total_return'] for r in self.results])

        # Consistency metrics
        sharpe_std = np.std([r['sharpe_ratio'] for r in self.results])
        win_rate_std = np.std([r['win_rate'] for r in self.results])
        positive_windows = sum([1 for r in self.results if r['sharpe_ratio'] > 0])

        summary = {
            'total_windows': len(windows),
            'successful_windows': len(self.results),
            'positive_sharpe_windows': positive_windows,
            'avg_accuracy': avg_accuracy,
            'avg_win_rate': avg_win_rate,
            'avg_sharpe_ratio': avg_sharpe,
            'sharpe_std': sharpe_std,
            'avg_profit_factor': avg_profit_factor,
            'avg_expectancy': avg_expectancy,
            'avg_max_drawdown': avg_max_dd,
            'win_rate_std': win_rate_std,
            'total_trades': total_trades,
            'total_return': total_return,
            'consistency_score': positive_windows / len(self.results) if len(self.results) > 0 else 0,
            'window_results': self.results
        }

        print(f"📈 AGGREGATED METRICS (Across {len(self.results)} windows):")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        print(f"  Average Win Rate: {avg_win_rate:.2%} (±{win_rate_std:.2%})")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.3f} (±{sharpe_std:.3f})")
        print(f"  Average Profit Factor: {avg_profit_factor:.3f}")
        print(f"  Average Max Drawdown: {avg_max_dd*100:.2f}%")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Return: {total_return:.6f}")
        print(f"\n🎯 CONSISTENCY:")
        print(f"  Positive Sharpe Windows: {positive_windows}/{len(self.results)} ({positive_windows/len(self.results)*100:.1f}%)")
        print(f"  Sharpe Consistency: {sharpe_std:.3f} (lower is better)")
        print(f"\n{'='*70}\n")

        # Red flags
        if avg_sharpe < 1.0:
            print("⚠️  WARNING: Average Sharpe < 1.0 - Model may not be profitable")
        if sharpe_std > 1.0:
            print("⚠️  WARNING: High Sharpe variability - Inconsistent performance")
        if positive_windows / len(self.results) < 0.7:
            print("⚠️  WARNING: < 70% positive windows - Low consistency")
        if avg_max_dd < -0.15:
            print("⚠️  WARNING: Average drawdown > 15% - High risk")

        return summary

    def save_results(self, filepath: str):
        """Save walk-forward results to JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to strings for JSON serialization
        results_serializable = []
        for r in self.results:
            r_copy = r.copy()
            for key in ['train_start', 'train_end', 'test_start', 'test_end']:
                if key in r_copy and hasattr(r_copy[key], 'isoformat'):
                    r_copy[key] = r_copy[key].isoformat()
            results_serializable.append(r_copy)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"✅ Results saved to {filepath}")


# Example usage
if __name__ == '__main__':
    print("Walk-Forward Optimization System")
    print("\nUsage example:")
    print("""
    from walk_forward_optimizer import WalkForwardOptimizer
    from model import TradingModel
    import pandas as pd

    # Load data
    df = pd.read_csv('historical_data.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Initialize optimizer
    optimizer = WalkForwardOptimizer(
        train_window_days=90,  # Train on 3 months
        test_window_days=30,   # Test on 1 month
        step_days=30           # Roll forward 1 month
    )

    # Run walk-forward validation
    results = optimizer.run_walk_forward(
        df,
        model_class=TradingModel,
        model_kwargs={'sequence_length': 40},
        train_kwargs={'epochs': 50, 'batch_size': 32}
    )

    # Save results
    optimizer.save_results('models/walk_forward_results.json')

    print(f"Average Sharpe: {results['avg_sharpe_ratio']:.3f}")
    print(f"Win Rate: {results['avg_win_rate']:.2%}")
    """)
