#!/usr/bin/env python3
"""
Walk-Forward Validation for RNN Trading Model
Tests model robustness across different time periods
"""

import pandas as pd
import numpy as np
from model import TradingModel
from trading_metrics import evaluate_trading_performance
import json
from pathlib import Path


def walk_forward_validation(df, n_splits=5, min_train_size=1000, test_size=400):
    """
    Walk-forward validation with expanding window

    Args:
        df: Full dataset
        n_splits: Number of validation folds
        min_train_size: Minimum bars needed for training
        test_size: Number of bars in each test period

    Returns:
        Dictionary with aggregated results across all folds
    """
    print("="*70)
    print("WALK-FORWARD VALIDATION")
    print("="*70)

    results = []
    total_bars = len(df)

    # Calculate fold boundaries
    print(f"\nTotal bars: {total_bars}")
    print(f"Minimum training size: {min_train_size}")
    print(f"Test size per fold: {test_size}")
    print(f"Number of folds: {n_splits}\n")

    for fold in range(n_splits):
        # Expanding window: train on all data up to test period
        train_end = min_train_size + (fold * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, total_bars)

        # Need enough data for test
        if test_end - test_start < 100:
            print(f"Fold {fold+1}: Insufficient test data ({test_end - test_start} bars), skipping")
            break

        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{n_splits}")
        print(f"{'='*70}")
        print(f"Training:   bars 0 to {train_end} ({train_end} bars)")
        print(f"Testing:    bars {test_start} to {test_end} ({test_end - test_start} bars)")

        # Split data
        train_data = df.iloc[:train_end].copy()
        test_data = df.iloc[test_start:test_end].copy()

        # Train fresh model on this fold
        model = TradingModel(sequence_length=40)

        print(f"\nTraining model on fold {fold+1}...")
        model.train(train_data, epochs=50, batch_size=32, validation_split=0.2)

        # Evaluate on test period
        print(f"\nEvaluating on test period...")
        fold_metrics = evaluate_fold(model, test_data, fold+1)

        results.append({
            'fold': fold + 1,
            'train_size': train_end,
            'test_size': test_end - test_start,
            'metrics': fold_metrics
        })

        # Save fold results
        save_fold_results(results, fold+1)

    # Aggregate results across folds
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS ACROSS ALL FOLDS")
    print(f"{'='*70}")

    aggregate_results(results)

    return results


def evaluate_fold(model, test_data, fold_num):
    """
    Evaluate model on a single test fold

    Returns:
        Dictionary of performance metrics
    """
    predictions = []
    actual_signals = []
    confidences = []
    price_changes = []

    # Generate predictions for test period
    for i in range(len(test_data) - model.sequence_length - 5):
        # Get sequence up to current bar
        sequence_data = test_data.iloc[:model.sequence_length + i]

        # Get prediction
        try:
            signal, confidence = model.predict(sequence_data)

            # Map signal to class
            signal_to_class = {'short': 0, 'hold': 1, 'long': 2}
            pred_class = signal_to_class[signal]

            predictions.append(pred_class)
            confidences.append(confidence)

            # Calculate actual price change for next bar
            current_idx = model.sequence_length + i
            if current_idx + 1 < len(test_data):
                current_price = test_data.iloc[current_idx]['close']
                next_price = test_data.iloc[current_idx + 1]['close']
                price_change = (next_price - current_price) / current_price
                price_changes.append(price_change)

                # Determine actual label (simplified)
                threshold = 0.0005  # 0.05%
                if price_change > threshold:
                    actual_signals.append(2)  # Long
                elif price_change < -threshold:
                    actual_signals.append(0)  # Short
                else:
                    actual_signals.append(1)  # Hold

        except Exception as e:
            print(f"Error on prediction {i}: {e}")
            continue

    # Calculate metrics
    predictions = np.array(predictions)
    actual_signals = np.array(actual_signals)
    confidences = np.array(confidences)
    price_changes = np.array(price_changes)

    # Overall accuracy
    accuracy = np.mean(predictions == actual_signals)

    # High-confidence accuracy
    high_conf_mask = confidences >= 0.65
    if np.sum(high_conf_mask) > 0:
        high_conf_accuracy = np.mean(predictions[high_conf_mask] == actual_signals[high_conf_mask])
        high_conf_pct = np.mean(high_conf_mask) * 100
    else:
        high_conf_accuracy = 0.0
        high_conf_pct = 0.0

    # Trading metrics
    daily_pnl_config = {'dailyGoal': 500.0, 'dailyMaxLoss': 250.0}
    trading_metrics = evaluate_trading_performance(
        predictions, actual_signals, price_changes, daily_pnl_config
    )

    # Print fold results
    print(f"\nFold {fold_num} Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  High-Conf Accuracy: {high_conf_accuracy:.2%} ({high_conf_pct:.1f}% of predictions)")
    print(f"  Sharpe Ratio: {trading_metrics['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {trading_metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {trading_metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")

    return {
        'accuracy': accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'high_conf_pct': high_conf_pct,
        'sharpe_ratio': trading_metrics['sharpe_ratio'],
        'win_rate': trading_metrics['win_rate'],
        'profit_factor': trading_metrics['profit_factor'],
        'max_drawdown': trading_metrics['max_drawdown'],
        'num_trades': trading_metrics['num_trades'],
        'total_return': trading_metrics['total_return']
    }


def aggregate_results(results):
    """Aggregate and display results across all folds"""

    # Extract metrics
    accuracies = [r['metrics']['accuracy'] for r in results]
    high_conf_accs = [r['metrics']['high_conf_accuracy'] for r in results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    win_rates = [r['metrics']['win_rate'] for r in results]
    profit_factors = [r['metrics']['profit_factor'] for r in results]
    max_drawdowns = [r['metrics']['max_drawdown'] for r in results]

    # Calculate statistics
    print(f"\nMetric Averages Across {len(results)} Folds:")
    print(f"  Accuracy:           {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print(f"  High-Conf Accuracy: {np.mean(high_conf_accs):.2%} ± {np.std(high_conf_accs):.2%}")
    print(f"  Sharpe Ratio:       {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
    print(f"  Win Rate:           {np.mean(win_rates):.2%} ± {np.std(win_rates):.2%}")
    print(f"  Profit Factor:      {np.mean(profit_factors):.3f} ± {np.std(profit_factors):.3f}")
    print(f"  Max Drawdown:       {np.mean(max_drawdowns):.2%} ± {np.std(max_drawdowns):.2%}")

    # Consistency check
    print(f"\nConsistency Analysis:")
    print(f"  Sharpe > 1.0 in {sum([1 for s in sharpes if s > 1.0])}/{len(sharpes)} folds")
    print(f"  Accuracy > 45% in {sum([1 for a in accuracies if a > 0.45])}/{len(accuracies)} folds")
    print(f"  Win Rate > 50% in {sum([1 for w in win_rates if w > 0.50])}/{len(win_rates)} folds")

    # Overall assessment
    avg_sharpe = np.mean(sharpes)
    avg_accuracy = np.mean(accuracies)

    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    if avg_sharpe > 1.5 and avg_accuracy > 0.48:
        print("✅ EXCELLENT: Model is robust and ready for production")
    elif avg_sharpe > 1.0 and avg_accuracy > 0.45:
        print("✅ GOOD: Model shows promise, consider additional tuning")
    elif avg_sharpe > 0.5:
        print("⚠️  MARGINAL: Model needs improvement before live trading")
    else:
        print("❌ POOR: Model not ready for live trading")


def save_fold_results(results, fold_num):
    """Save intermediate results to JSON"""
    output_dir = Path('validation_results')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'walk_forward_results_fold{fold_num}.json'

    # Convert numpy types to Python types for JSON serialization
    serializable_results = []
    for r in results:
        serializable_results.append({
            'fold': int(r['fold']),
            'train_size': int(r['train_size']),
            'test_size': int(r['test_size']),
            'metrics': {k: float(v) for k, v in r['metrics'].items()}
        })

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Saved results to {output_file}")


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Walk-Forward Validation Script")
    print("This script requires historical market data")
    print("\nTo use:")
    print("1. Load your historical data into a DataFrame")
    print("2. Ensure it has columns: time, open, high, low, close, volume")
    print("3. Call: results = walk_forward_validation(df, n_splits=5)")
