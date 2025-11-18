"""
Complete Backtesting Framework with Walk-Forward Validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from walk_forward_strategy import WalkForwardTrainingStrategy
from performance_dashboard import PerformanceMetrics


class BacktestingFramework:
    """Complete backtesting with walk-forward validation"""

    def __init__(self, model_class, config: Dict):
        self.model_class = model_class
        self.config = config
        self.wf_strategy = WalkForwardTrainingStrategy(model_class, config)

    def run_backtest(
        self,
        data: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21
    ) -> Dict:
        """
        Run complete backtest with walk-forward validation

        Args:
            data: Historical data
            train_window: Training window size
            test_window: Testing window size
            step_size: Step size for walk-forward

        Returns:
            Complete backtest results
        """
        print("\n" + "="*70)
        print("BACKTESTING WITH WALK-FORWARD VALIDATION")
        print("="*70)

        # Run walk-forward optimization
        wf_results = self.wf_strategy.walk_forward_train(
            data=data,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size
        )

        # Calculate performance metrics
        metrics = self._calculate_backtest_metrics(wf_results)

        # Generate report
        self._print_report(metrics)

        return {
            'walk_forward_results': wf_results,
            'performance_metrics': metrics
        }

    def _calculate_backtest_metrics(self, wf_results: Dict) -> Dict:
        """Calculate performance metrics from walk-forward results"""
        accuracies = [r['metrics']['accuracy'] for r in wf_results['period_results']]

        return {
            'mean_accuracy': wf_results['mean_accuracy'],
            'std_accuracy': wf_results['std_accuracy'],
            'min_accuracy': wf_results['min_accuracy'],
            'max_accuracy': wf_results['max_accuracy'],
            'num_periods': wf_results['num_periods'],
            'consistency': 1 - (wf_results['std_accuracy'] / (wf_results['mean_accuracy'] + 1e-10))
        }

    def _print_report(self, metrics: Dict):
        """Print backtest report"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"Std Accuracy: {metrics['std_accuracy']:.4f}")
        print(f"Min Accuracy: {metrics['min_accuracy']:.4f}")
        print(f"Max Accuracy: {metrics['max_accuracy']:.4f}")
        print(f"Consistency Score: {metrics['consistency']:.4f}")
        print(f"Number of Periods: {metrics['num_periods']}")
        print("="*70 + "\n")
