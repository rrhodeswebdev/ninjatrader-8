"""
Feature Importance Analysis
============================

Identifies which features contribute most to model predictions.
Use this to prune weak features and reduce overfitting.

Methods:
1. Permutation Importance: Shuffle feature values and measure performance drop
2. Ablation Testing: Remove features one-by-one and test performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for trading models
    """

    def __init__(self, feature_names: List[str]):
        """
        Args:
            feature_names: List of feature names matching model input
        """
        self.feature_names = feature_names
        self.importance_scores = None

    def permutation_importance(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 10,
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Calculate permutation importance

        Args:
            model: Trained model with predict() method
            X_val: Validation features (n_samples, sequence_length, n_features)
            y_val: Validation labels
            n_repeats: Number of times to shuffle each feature
            metric: 'accuracy', 'sharpe', or 'profit_factor'

        Returns:
            Dictionary mapping feature names to importance scores
        """
        print("\n" + "="*70)
        print("PERMUTATION IMPORTANCE ANALYSIS")
        print("="*70)
        print(f"Validation samples: {len(X_val)}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Metric: {metric}")
        print(f"Repeats per feature: {n_repeats}\n")

        # Calculate baseline performance
        print("Calculating baseline performance...")
        baseline_score = self._calculate_score(model, X_val, y_val, metric)
        print(f"✓ Baseline {metric}: {baseline_score:.4f}\n")

        # Calculate importance for each feature
        importance_dict = {}

        print("Testing feature importance:")
        for i, feature_name in enumerate(tqdm(self.feature_names)):
            scores = []

            for _ in range(n_repeats):
                # Make a copy and shuffle this feature
                X_permuted = X_val.copy()
                # Shuffle across all samples and sequence positions for this feature
                X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i].flatten()).reshape(X_permuted[:, :, i].shape)

                # Calculate performance with shuffled feature
                score = self._calculate_score(model, X_permuted, y_val, metric)
                scores.append(score)

            # Importance = baseline - average permuted score
            # (positive = feature is helpful, negative = feature hurts)
            avg_permuted_score = np.mean(scores)
            importance = baseline_score - avg_permuted_score

            importance_dict[feature_name] = {
                'importance': importance,
                'std': np.std(scores),
                'avg_score_when_shuffled': avg_permuted_score
            }

        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1]['importance'], reverse=True)

        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE RESULTS")
        print(f"{'='*70}\n")

        print(f"{'Feature':<40} {'Importance':<15} {'Std':<10} {'Status'}")
        print("-"*70)

        for feat_name, stats in sorted_features:
            imp = stats['importance']
            std = stats['std']

            # Status indicator
            if imp > 0.01:
                status = "✅ Important"
            elif imp > 0:
                status = "→  Weak"
            elif imp < -0.01:
                status = "❌ Harmful"
            else:
                status = "≈  Neutral"

            print(f"{feat_name:<40} {imp:>+.5f}      {std:<10.5f} {status}")

        # Store results
        self.importance_scores = importance_dict

        return importance_dict

    def _calculate_score(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metric: str
    ) -> float:
        """Calculate model score using specified metric"""
        # Get predictions
        predictions = []

        for i in range(len(X)):
            # Create a temporary dataframe (model expects this format)
            # This is a simplified version - adapt to your model's predict() interface
            try:
                # For batch prediction (if model supports it)
                pred = model.model.predict(X[i:i+1])  # Single sample
                pred_class = np.argmax(pred)
                predictions.append(pred_class)
            except:
                # Fallback: assume model returns signal directly
                predictions.append(1)  # Hold as default

        predictions = np.array(predictions)

        if metric == 'accuracy':
            score = np.mean(predictions == y)

        elif metric == 'sharpe':
            # Calculate returns based on predictions
            returns = []
            for pred, actual_label in zip(predictions, y):
                if pred == 2:  # Long
                    ret = 0.001 if actual_label == 2 else -0.001
                elif pred == 0:  # Short
                    ret = 0.001 if actual_label == 0 else -0.001
                else:  # Hold
                    ret = 0.0
                returns.append(ret)

            returns = np.array(returns)
            returns = returns[returns != 0]  # Filter holds

            if len(returns) > 1 and np.std(returns) > 0:
                score = np.mean(returns) / np.std(returns) * np.sqrt(252*390)
            else:
                score = 0.0

        elif metric == 'profit_factor':
            # Similar to Sharpe calculation
            returns = []
            for pred, actual_label in zip(predictions, y):
                if pred == 2:
                    ret = 0.001 if actual_label == 2 else -0.001
                elif pred == 0:
                    ret = 0.001 if actual_label == 0 else -0.001
                else:
                    ret = 0.0
                returns.append(ret)

            returns = np.array(returns)
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))

            score = gross_profit / gross_loss if gross_loss > 0 else 999.99

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return score

    def plot_importance(self, save_path: str = None, top_n: int = 25):
        """
        Plot feature importance

        Args:
            save_path: Optional path to save plot
            top_n: Number of top features to show
        """
        if self.importance_scores is None:
            print("❌ No importance scores calculated. Run permutation_importance() first.")
            return

        # Sort by importance
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: abs(x[1]['importance']),
            reverse=True
        )[:top_n]

        feature_names = [f[0] for f in sorted_features]
        importances = [f[1]['importance'] for f in sorted_features]
        stds = [f[1]['std'] for f in sorted_features]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(feature_names) * 0.3)))

        y_pos = np.arange(len(feature_names))
        colors = ['green' if imp > 0 else 'red' for imp in importances]

        ax.barh(y_pos, importances, xerr=stds, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {len(feature_names)} Feature Importance (Permutation Method)', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def suggest_features_to_remove(self, threshold: float = 0.0) -> List[str]:
        """
        Suggest features to remove based on importance

        Args:
            threshold: Remove features with importance < threshold

        Returns:
            List of feature names to remove
        """
        if self.importance_scores is None:
            return []

        features_to_remove = [
            feat_name for feat_name, stats in self.importance_scores.items()
            if stats['importance'] < threshold
        ]

        print(f"\n📋 FEATURES TO REMOVE (importance < {threshold}):")
        print(f"Found {len(features_to_remove)} features:\n")

        for feat in features_to_remove:
            imp = self.importance_scores[feat]['importance']
            print(f"  - {feat:<40} (importance: {imp:+.5f})")

        return features_to_remove

    def save_results(self, filepath: str):
        """Save importance scores to file"""
        if self.importance_scores is None:
            print("❌ No results to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame for easy saving
        results_df = pd.DataFrame([
            {
                'feature': feat_name,
                'importance': stats['importance'],
                'std': stats['std'],
                'avg_score_when_shuffled': stats['avg_score_when_shuffled']
            }
            for feat_name, stats in self.importance_scores.items()
        ])

        results_df = results_df.sort_values('importance', ascending=False)
        results_df.to_csv(filepath, index=False)

        print(f"✅ Results saved to {filepath}")


# Example usage
if __name__ == '__main__':
    print("Feature Importance Analyzer")
    print("="*50)
    print("\nUsage:")
    print("""
    from feature_importance_analyzer import FeatureImportanceAnalyzer
    from model import TradingModel
    import pandas as pd

    # Load validation data
    df_val = pd.read_csv('validation_data.csv')

    # Train model
    model = TradingModel(sequence_length=40)
    model.train(df_train)

    # Prepare validation data
    X_val, y_val = model.prepare_data(df_val, fit_scaler=False)

    # Define feature names (must match model input order)
    feature_names = [
        'open', 'high', 'low', 'close',
        'hurst_H', 'atr', 'velocity', 'acceleration',
        # ... all 62 features ...
    ]

    # Analyze importance
    analyzer = FeatureImportanceAnalyzer(feature_names)
    importance = analyzer.permutation_importance(
        model, X_val, y_val,
        n_repeats=10,
        metric='accuracy'
    )

    # Plot results
    analyzer.plot_importance('models/feature_importance.png', top_n=25)

    # Get features to remove
    to_remove = analyzer.suggest_features_to_remove(threshold=0.0)

    # Save results
    analyzer.save_results('models/feature_importance.csv')

    print(f"\\nRemove {len(to_remove)} weak features to simplify model")
    """)
