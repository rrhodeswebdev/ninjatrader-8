"""
Hyperparameter Optimization using Optuna

Automatically finds optimal hyperparameters for the trading model:
- Hidden size
- Dropout rates
- Learning rate
- Sequence length
- Number of LSTM layers
- Batch size
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional
import torch

# Import will be done dynamically to avoid circular imports


class HyperparameterOptimizer:
    """
    Optimize trading model hyperparameters using Bayesian optimization (Optuna)
    """

    def __init__(self, df: pd.DataFrame, df_secondary: Optional[pd.DataFrame] = None,
                 validation_split: float = 0.2, n_trials: int = 50,
                 study_name: str = "trading_model_optimization"):
        """
        Initialize optimizer

        Args:
            df: Training data (primary timeframe)
            df_secondary: Optional secondary timeframe data
            validation_split: Validation set size
            n_trials: Number of optimization trials
            study_name: Name for the optimization study
        """
        self.df = df
        self.df_secondary = df_secondary
        self.validation_split = validation_split
        self.n_trials = n_trials
        self.study_name = study_name

        # Results
        self.study = None
        self.best_params = None
        self.best_value = None

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation Sharpe ratio (metric to maximize)
        """
        # Import here to avoid circular imports
        from model import TradingModel

        # Suggest hyperparameters
        params = {
            'sequence_length': trial.suggest_int('sequence_length', 10, 25),
            'hidden_size': trial.suggest_int('hidden_size', 64, 256, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'fc_hidden_size': trial.suggest_int('fc_hidden_size', 32, 128, step=32),
        }

        print(f"\n{'='*70}")
        print(f"Trial {trial.number} - Testing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print('='*70)

        try:
            # Create model with suggested parameters
            model = TradingModel(
                sequence_length=params['sequence_length'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )

            # Train with early stopping (reduced epochs for faster optimization)
            model.train(
                self.df,
                epochs=30,  # Reduced for optimization speed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                validation_split=self.validation_split,
                verbose=False  # Reduce output
            )

            # Get validation Sharpe ratio
            val_sharpe = model.last_val_sharpe if hasattr(model, 'last_val_sharpe') else 0.0

            # Also get validation accuracy as secondary metric
            val_acc = model.last_val_accuracy if hasattr(model, 'last_val_accuracy') else 0.0

            print(f"\nTrial {trial.number} Results:")
            print(f"  Validation Sharpe: {val_sharpe:.3f}")
            print(f"  Validation Accuracy: {val_acc:.3f}")

            # Report intermediate value for pruning
            trial.report(val_sharpe, step=30)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return val_sharpe

        except Exception as e:
            print(f"\n❌ Trial {trial.number} failed: {e}")
            return -999.0  # Return very poor score for failed trials

    def optimize(self, timeout: Optional[int] = None) -> Dict:
        """
        Run hyperparameter optimization

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Dictionary with best parameters and results
        """
        print("\n" + "="*70)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        print(f"Configuration:")
        print(f"  Number of trials: {self.n_trials}")
        print(f"  Validation split: {self.validation_split}")
        print(f"  Timeout: {timeout if timeout else 'None'}")
        print("="*70)

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',  # Maximize Sharpe ratio
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\nBest Sharpe Ratio: {self.best_value:.3f}")
        print(f"\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

        # Save results
        self.save_results()

        # Print top 5 trials
        self.print_top_trials(n=5)

        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }

    def print_top_trials(self, n: int = 5):
        """Print top N trials"""
        print(f"\n{'='*70}")
        print(f"TOP {n} TRIALS")
        print('='*70)

        # Sort trials by value
        sorted_trials = sorted(self.study.trials, key=lambda t: t.value if t.value else -999, reverse=True)

        for i, trial in enumerate(sorted_trials[:n]):
            if trial.value is None:
                continue

            print(f"\n#{i+1} - Trial {trial.number}")
            print(f"  Sharpe: {trial.value:.3f}")
            print(f"  Parameters:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

    def save_results(self, output_dir: str = 'optimization_results'):
        """Save optimization results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save best parameters
        with open(output_path / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        # Save study statistics
        stats = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'n_complete_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
        }

        with open(output_path / 'optimization_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # Save all trial results
        trial_data = []
        for trial in self.study.trials:
            trial_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })

        df_trials = pd.DataFrame(trial_data)
        df_trials.to_csv(output_path / 'all_trials.csv', index=False)

        print(f"\n✓ Saved results to {output_path}/")

    def plot_optimization_history(self, output_dir: str = 'optimization_results'):
        """Plot optimization history"""
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances

            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_image(str(output_path / 'optimization_history.png'))

            # Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_image(str(output_path / 'param_importances.png'))

            print(f"✓ Saved plots to {output_path}/")

        except ImportError:
            print("⚠️  Cannot create plots: plotly or kaleido not installed")
            print("   Install with: pip install plotly kaleido")

    def load_best_params(self, params_file: str = 'optimization_results/best_params.json') -> Dict:
        """Load best parameters from file"""
        with open(params_file, 'r') as f:
            return json.load(f)


def quick_hyperparameter_search(df: pd.DataFrame, n_trials: int = 20) -> Dict:
    """
    Convenience function for quick hyperparameter search

    Args:
        df: Training data
        n_trials: Number of trials (20 for quick search, 50+ for thorough)

    Returns:
        Best parameters
    """
    optimizer = HyperparameterOptimizer(df, n_trials=n_trials)
    results = optimizer.optimize()
    return results['best_params']


def comprehensive_hyperparameter_search(df: pd.DataFrame,
                                       df_secondary: Optional[pd.DataFrame] = None,
                                       n_trials: int = 100,
                                       timeout_hours: int = 6) -> Dict:
    """
    Comprehensive hyperparameter search (long-running)

    Args:
        df: Primary timeframe data
        df_secondary: Secondary timeframe data
        n_trials: Number of trials
        timeout_hours: Maximum hours to run

    Returns:
        Best parameters and full results
    """
    print("Starting comprehensive hyperparameter search...")
    print(f"This may take up to {timeout_hours} hours")

    optimizer = HyperparameterOptimizer(
        df,
        df_secondary,
        n_trials=n_trials,
        study_name="comprehensive_optimization"
    )

    results = optimizer.optimize(timeout=timeout_hours * 3600)

    # Plot results
    optimizer.plot_optimization_history()

    return results


if __name__ == '__main__':
    print("Hyperparameter Optimization Module")
    print("="*70)
    print("\nUsage:")
    print("  from hyperparameter_optimization import HyperparameterOptimizer")
    print("  optimizer = HyperparameterOptimizer(df, n_trials=50)")
    print("  results = optimizer.optimize()")
    print("\nOr use convenience functions:")
    print("  quick_hyperparameter_search(df, n_trials=20)")
    print("  comprehensive_hyperparameter_search(df, n_trials=100)")
