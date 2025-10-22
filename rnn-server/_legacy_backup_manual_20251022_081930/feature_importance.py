#!/usr/bin/env python3
"""
Feature Importance Analysis via Ablation Testing
Identifies which feature groups contribute most to model performance
"""

import numpy as np
import pandas as pd
from model import TradingModel
from trading_metrics import evaluate_trading_performance
import torch


# Define feature groups and their indices (59 features total after reduction)
# REMOVED: daily_pnl (3 features), 6 order_flow features, 10 deviation features, tf2_delta
# Total: 4+2+1+2+15+13+1+5+5+4+9+1 = 62 features (recounted from actual stacking)
FEATURE_GROUPS = {
    'ohlc': list(range(0, 4)),            # Open, High, Low, Close (4)
    'hurst': list(range(4, 6)),           # Hurst H and C (2)
    'atr': [6],                           # Average True Range (1)
    'price_momentum': list(range(7, 9)),  # Velocity, acceleration (2)
    'price_patterns': list(range(9, 24)), # Range, wicks, gaps, swings, stats, etc. (15)
    'price_deviations': list(range(24, 37)),  # Windows 20 & 50 (10) + vol_accel, high_dev, low_dev (3) = 13
    'order_flow': [37],                   # Volume ratio only (1, removed 6 bid/ask features)
    'time_of_day': list(range(38, 43)),   # Hour, session period, etc. (5)
    'microstructure': list(range(43, 48)), # Volume surge, spreads, VWAP (5)
    'volatility_regime': list(range(48, 52)), # Vol regime, Parkinson vol, etc. (4)
    'multi_timeframe': list(range(52, 61)),   # Secondary timeframe features (9, removed tf2_delta)
    'price_change_mag': [61],             # Recent volatility indicator (1)
}


def measure_feature_importance(model, val_data, val_labels, val_price_changes):
    """
    Measure importance of each feature group via ablation

    Args:
        model: Trained TradingModel
        val_data: Validation sequences (numpy array)
        val_labels: True labels
        val_price_changes: Actual price changes

    Returns:
        Dictionary of feature group importances
    """
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS (Ablation Testing)")
    print("="*70)

    # Get baseline performance (all features)
    baseline_metrics = evaluate_with_features(
        model, val_data, val_labels, val_price_changes, ablate_indices=None
    )
    baseline_sharpe = baseline_metrics['sharpe_ratio']
    baseline_accuracy = baseline_metrics['accuracy']

    print(f"\nBaseline Performance (all features):")
    print(f"  Sharpe Ratio: {baseline_sharpe:.3f}")
    print(f"  Accuracy: {baseline_accuracy:.2%}")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2%}")

    # Test each feature group
    importance_results = {}

    print(f"\n{'='*70}")
    print("Testing Feature Groups (removing each group)")
    print(f"{'='*70}")

    for group_name, feature_indices in FEATURE_GROUPS.items():
        print(f"\nRemoving feature group: {group_name} ({len(feature_indices)} features)")

        # Evaluate with this group ablated (set to zero)
        ablated_metrics = evaluate_with_features(
            model, val_data, val_labels, val_price_changes,
            ablate_indices=feature_indices
        )

        # Calculate performance drop
        sharpe_drop = baseline_sharpe - ablated_metrics['sharpe_ratio']
        accuracy_drop = baseline_accuracy - ablated_metrics['accuracy']

        importance_results[group_name] = {
            'num_features': len(feature_indices),
            'sharpe_drop': sharpe_drop,
            'accuracy_drop': accuracy_drop,
            'ablated_sharpe': ablated_metrics['sharpe_ratio'],
            'ablated_accuracy': ablated_metrics['accuracy'],
        }

        print(f"  Sharpe drop: {sharpe_drop:+.3f} ({ablated_metrics['sharpe_ratio']:.3f})")
        print(f"  Accuracy drop: {accuracy_drop:+.2%} ({ablated_metrics['accuracy']:.2%})")

        # Interpret importance
        if sharpe_drop > 0.3:
            print(f"  ⭐⭐⭐ CRITICAL: Removing causes major performance loss!")
        elif sharpe_drop > 0.15:
            print(f"  ⭐⭐ HIGH IMPORTANCE: Significant contribution to performance")
        elif sharpe_drop > 0.05:
            print(f"  ⭐ MODERATE IMPORTANCE: Useful but not critical")
        elif sharpe_drop > -0.05:
            print(f"  ⚪ LOW IMPORTANCE: Minimal impact")
        else:
            print(f"  ❌ NEGATIVE IMPACT: Removing improves performance!")

    # Rank by importance
    print(f"\n{'='*70}")
    print("FEATURE GROUP RANKING (by Sharpe drop)")
    print(f"{'='*70}")

    ranked = sorted(importance_results.items(),
                   key=lambda x: x[1]['sharpe_drop'],
                   reverse=True)

    for rank, (group_name, metrics) in enumerate(ranked, 1):
        print(f"{rank}. {group_name:20s} - Sharpe drop: {metrics['sharpe_drop']:+.3f} "
              f"(Accuracy drop: {metrics['accuracy_drop']:+.2%})")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    print("\n✅ KEEP (High Importance - Sharpe drop > 0.15):")
    for group_name, metrics in ranked:
        if metrics['sharpe_drop'] > 0.15:
            print(f"  - {group_name}: {metrics['sharpe_drop']:+.3f}")

    print("\n⚠️  CONSIDER REMOVING (Low/Negative Importance - Sharpe drop < 0.05):")
    for group_name, metrics in ranked:
        if metrics['sharpe_drop'] < 0.05:
            print(f"  - {group_name}: {metrics['sharpe_drop']:+.3f}")
            if metrics['sharpe_drop'] < 0:
                print(f"    → Removing this group IMPROVES performance!")

    return importance_results


def evaluate_with_features(model, val_data, val_labels, val_price_changes, ablate_indices=None):
    """
    Evaluate model performance with specified features ablated (set to zero)

    Args:
        model: Trained model
        val_data: Validation data (N, sequence_length, num_features)
        val_labels: True labels
        val_price_changes: Price changes
        ablate_indices: List of feature indices to zero out (None = use all features)

    Returns:
        Dictionary of performance metrics
    """
    # Copy data to avoid modifying original
    data_copy = val_data.copy()

    # Zero out specified features
    if ablate_indices is not None:
        data_copy[:, :, ablate_indices] = 0

    # Get predictions
    model.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(data_copy).to(model.device)
        outputs = model.model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Calculate accuracy
    accuracy = np.mean(predictions == val_labels)

    # Calculate trading metrics
    daily_pnl_config = {'dailyGoal': 500.0, 'dailyMaxLoss': 250.0}
    trading_metrics = evaluate_trading_performance(
        predictions, val_labels, val_price_changes, daily_pnl_config
    )

    return {
        'accuracy': accuracy,
        'sharpe_ratio': trading_metrics['sharpe_ratio'],
        'win_rate': trading_metrics['win_rate'],
        'profit_factor': trading_metrics['profit_factor'],
    }


def suggest_reduced_feature_set(importance_results, sharpe_threshold=0.05):
    """
    Suggest a reduced feature set based on importance analysis

    Args:
        importance_results: Results from measure_feature_importance
        sharpe_threshold: Minimum Sharpe drop to keep feature group

    Returns:
        List of feature indices to keep
    """
    keep_features = []
    remove_groups = []

    for group_name, metrics in importance_results.items():
        if metrics['sharpe_drop'] >= sharpe_threshold:
            # Keep this group
            keep_features.extend(FEATURE_GROUPS[group_name])
        else:
            remove_groups.append(group_name)

    print(f"\nReduced Feature Set Suggestion (threshold: {sharpe_threshold}):")
    print(f"  Original features: 87")
    print(f"  Reduced features: {len(keep_features)}")
    print(f"  Reduction: {87 - len(keep_features)} features ({(87-len(keep_features))/87*100:.1f}%)")
    print(f"\nGroups removed: {', '.join(remove_groups)}")

    return sorted(keep_features)


if __name__ == "__main__":
    print("Feature Importance Analysis Script")
    print("\nThis script requires:")
    print("1. A trained model")
    print("2. Validation data prepared with prepare_data()")
    print("\nExample usage:")
    print("  model = TradingModel()")
    print("  model.load_model()")
    print("  X_val, y_val = model.prepare_data(validation_df, fit_scaler=False)")
    print("  results = measure_feature_importance(model, X_val, y_val, price_changes)")
