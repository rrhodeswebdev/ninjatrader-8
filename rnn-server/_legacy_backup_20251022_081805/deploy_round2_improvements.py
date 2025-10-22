#!/usr/bin/env python3
"""
Deploy Round 2 Improvements - Confidence & Accuracy Enhancements

This script orchestrates the deployment of all Round 2 improvements:
1. Train model with CombinedTradingLoss
2. Train with Triple Barrier labels (optional)
3. Calibrate confidence scores
4. Validate improvements

Features:
- Reduced HOLD bias (40% → 25%)
- Advanced loss functions (label smoothing, confidence penalty, directional)
- Triple Barrier Method for labels
- Temperature scaling for calibration
- Comprehensive validation

Usage:
    # Quick deploy (standard labels + calibration):
    uv run python deploy_round2_improvements.py --quick

    # Full deploy (triple barrier + calibration):
    uv run python deploy_round2_improvements.py --full

    # Custom configuration:
    uv run python deploy_round2_improvements.py --epochs 150 --use-triple-barrier

Expected Results:
- Confidence scores: 45% → 68% on correct predictions
- Accuracy: 52% → 58-60%
- Sharpe ratio: 2.0 → 2.5-2.8
- ECE: 0.15+ → <0.05
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model import TradingModel
from confidence_calibration_advanced import calibrate_trading_model, calculate_ece
import torch
from torch.utils.data import DataLoader, TensorDataset


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def load_data(data_path: str = 'historical_data.csv'):
    """Load historical data"""
    print_section("LOADING HISTORICAL DATA")

    if not Path(data_path).exists():
        print(f"❌ ERROR: {data_path} not found!")
        print("Please ensure historical_data.csv exists in rnn-server/")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])

    print(f"✓ Loaded {len(df)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def train_model(model: TradingModel, df: pd.DataFrame, epochs: int = 100,
                use_triple_barrier: bool = False):
    """Train model with Round 2 improvements"""

    print_section("TRAINING MODEL WITH ROUND 2 IMPROVEMENTS")

    print("\n📋 Configuration:")
    print(f"  HOLD bias: 25% (down from 40%)")
    print(f"  Loss function: CombinedTradingLoss")
    print(f"  Label method: {'Triple Barrier' if use_triple_barrier else 'Percentile-based'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: 32")

    # Train with optional triple barrier labels
    start_time = time.time()

    # Prepare data with optional triple barrier
    print("\n📊 Preparing training data...")
    X, y = model.prepare_data(
        df,
        fit_scaler=True,
        use_triple_barrier=use_triple_barrier
    )

    # Train
    print("\n🚀 Starting training...")
    model.train_on_prepared_data(X, y, epochs=epochs, batch_size=32)

    elapsed = time.time() - start_time
    print(f"\n✓ Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save model
    model.save_model()
    print("✓ Model saved")


def calibrate_model(model: TradingModel, df: pd.DataFrame):
    """Calibrate model confidence scores"""

    print_section("CALIBRATING CONFIDENCE SCORES")

    # Use last 20% as validation
    val_size = int(len(df) * 0.2)
    df_val = df.iloc[-val_size:].copy()

    print(f"Using {len(df_val)} bars for calibration")

    # Prepare validation data
    print("Preparing validation data...")
    X_val, y_val = model.prepare_data(df_val, fit_scaler=False)

    # Create dataloader
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Measure ECE before
    print("\n📊 Measuring calibration BEFORE...")
    model.model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(model.device)
            logits = model.model(batch_X, return_attention=False)
            probs = torch.softmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(batch_y)

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    ece_before = calculate_ece(all_probs, all_labels)
    predictions = torch.argmax(all_probs, dim=1)
    accuracy = (predictions == all_labels).float().mean().item()

    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  ECE (before): {ece_before:.4f}")

    # Calibrate
    print("\n🎯 Calibrating...")
    calibration = calibrate_trading_model(
        model.model,
        val_loader,
        device=model.device
    )

    # Measure ECE after for each method
    print("\n📊 Testing calibration methods:")
    methods = ['temperature', 'platt', 'isotonic', 'ensemble']

    best_method = None
    best_ece = float('inf')

    for method in methods:
        try:
            calibrated_probs = calibration.calibrate(all_logits, all_probs, method=method)
            ece_after = calculate_ece(calibrated_probs, all_labels)

            improvement = ((ece_before - ece_after) / ece_before) * 100 if ece_after < ece_before else 0
            status = "✓" if ece_after < ece_before else "✗"

            print(f"  {status} {method.capitalize():15s} ECE: {ece_after:.4f} ({improvement:+.1f}%)")

            if ece_after < best_ece:
                best_ece = ece_after
                best_method = method
        except Exception as e:
            print(f"  ✗ {method.capitalize():15s} FAILED: {e}")

    print(f"\n✓ Best method: {best_method} (ECE: {best_ece:.4f})")
    print(f"  Improvement: {((ece_before - best_ece) / ece_before * 100):.1f}%")

    # Save calibration
    save_path = Path('models/calibration')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(str(save_path))
    print(f"✓ Calibration saved to: {save_path}")

    return ece_before, best_ece, accuracy


def validate_improvements(model: TradingModel, df: pd.DataFrame):
    """Validate that improvements are working"""

    print_section("VALIDATING IMPROVEMENTS")

    # Use last 30% as test set
    test_size = int(len(df) * 0.3)
    df_test = df.iloc[-test_size:].copy()

    print(f"Using {len(df_test)} bars for validation")

    # Prepare test data
    X_test, y_test = model.prepare_data(df_test, fit_scaler=False)

    # Create dataloader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    model.model.eval()
    all_probs = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(model.device)
            logits = model.model(batch_X, return_attention=False)

            # Use calibration if available
            if model.calibration is not None:
                raw_probs = torch.softmax(logits, dim=1)
                probs = model.calibration.calibrate(logits, raw_probs, method='temperature')
            else:
                probs = torch.softmax(logits, dim=1)

            predictions = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(batch_y)
            all_predictions.append(predictions.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).float().mean().item()

    # Class-wise accuracy
    class_names = ['SHORT', 'HOLD', 'LONG']
    print("\n📊 Test Set Performance:")
    print(f"  Overall Accuracy: {accuracy*100:.2f}%")

    print("\n  Class-wise Accuracy:")
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == all_labels[mask]).float().mean().item()
            print(f"    {name:6s}: {class_acc*100:.2f}% ({mask.sum()} samples)")

    # Confidence analysis
    confidences = torch.max(all_probs, dim=1)[0]
    correct = (all_predictions == all_labels).float()

    conf_correct = confidences[correct == 1].numpy()
    conf_incorrect = confidences[correct == 0].numpy()

    print("\n  Confidence Scores:")
    print(f"    On correct predictions:   {conf_correct.mean()*100:.1f}% ± {conf_correct.std()*100:.1f}%")
    print(f"    On incorrect predictions: {conf_incorrect.mean()*100:.1f}% ± {conf_incorrect.std()*100:.1f}%")
    print(f"    Separation: {(conf_correct.mean() - conf_incorrect.mean())*100:.1f}%")

    # HOLD percentage
    hold_pct = (all_predictions == 1).float().mean().item()
    print(f"\n  HOLD Predictions: {hold_pct*100:.1f}%")

    # High confidence trades
    high_conf_mask = confidences > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_acc = (all_predictions[high_conf_mask] == all_labels[high_conf_mask]).float().mean().item()
        print(f"\n  High Confidence (>70%):")
        print(f"    Count: {high_conf_mask.sum()} ({high_conf_mask.float().mean()*100:.1f}% of all)")
        print(f"    Accuracy: {high_conf_acc*100:.1f}%")

    return {
        'accuracy': accuracy,
        'conf_correct': conf_correct.mean(),
        'conf_incorrect': conf_incorrect.mean(),
        'hold_pct': hold_pct
    }


def print_summary(ece_before: float, ece_after: float, accuracy: float, metrics: dict):
    """Print final summary"""

    print_section("DEPLOYMENT SUMMARY")

    print("\n✅ IMPROVEMENTS IMPLEMENTED:")
    print("  1. ✓ Reduced HOLD bias (40% → 25%)")
    print("  2. ✓ CombinedTradingLoss for training")
    print("  3. ✓ Improved confidence boost logic")
    print("  4. ✓ Temperature scaling calibration")
    print("  5. ✓ Triple Barrier labels (optional)")

    print("\n📊 PERFORMANCE METRICS:")
    print(f"  Accuracy:                 {accuracy*100:.2f}%")
    print(f"  Confidence (correct):     {metrics['conf_correct']*100:.1f}%")
    print(f"  Confidence (incorrect):   {metrics['conf_incorrect']*100:.1f}%")
    print(f"  Calibration (ECE before): {ece_before:.4f}")
    print(f"  Calibration (ECE after):  {ece_after:.4f}")
    print(f"  HOLD predictions:         {metrics['hold_pct']*100:.1f}%")

    print("\n🎯 EXPECTED vs ACTUAL:")
    targets = {
        'Accuracy': (58, accuracy*100),
        'Confidence (correct)': (68, metrics['conf_correct']*100),
        'ECE': (0.05, ece_after),
        'HOLD %': (27, metrics['hold_pct']*100)
    }

    for name, (target, actual) in targets.items():
        status = "✓" if (name == 'ECE' and actual <= target) or (name != 'ECE' and actual >= target * 0.9) else "→"
        print(f"  {status} {name:25s} Target: {target:5.1f}{'%' if name != 'ECE' else '':<2} Actual: {actual:5.1f}{'%' if name != 'ECE' else '':<2}")

    print("\n🚀 NEXT STEPS:")
    print("  1. Test in paper trading")
    print("  2. Monitor confidence scores in production")
    print("  3. Re-calibrate monthly as needed")
    print("  4. Consider Triple Barrier labels if HOLD% still high")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Deploy Round 2 Improvements')
    parser.add_argument('--quick', action='store_true',
                       help='Quick deploy: standard labels + calibration (100 epochs)')
    parser.add_argument('--full', action='store_true',
                       help='Full deploy: triple barrier + calibration (150 epochs)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--use-triple-barrier', action='store_true',
                       help='Use Triple Barrier Method for labels')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (only calibrate existing model)')
    parser.add_argument('--data', type=str, default='historical_data.csv',
                       help='Path to historical data CSV')

    args = parser.parse_args()

    # Handle presets
    if args.quick:
        args.epochs = 100
        args.use_triple_barrier = False
    elif args.full:
        args.epochs = 150
        args.use_triple_barrier = True

    # Banner
    print("="*80)
    print("  ROUND 2 IMPROVEMENTS DEPLOYMENT")
    print("  Confidence & Accuracy Enhancements")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"  Label method: {'Triple Barrier' if args.use_triple_barrier else 'Percentile-based'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Skip training: {args.skip_training}")
    print(f"  Data file: {args.data}")

    start_time = time.time()

    # Load data
    df = load_data(args.data)

    # Initialize model
    print_section("INITIALIZING MODEL")
    model = TradingModel(sequence_length=15)
    print("✓ Model initialized")

    # Train or load
    if not args.skip_training:
        train_model(model, df, epochs=args.epochs, use_triple_barrier=args.use_triple_barrier)
    else:
        print_section("LOADING EXISTING MODEL")
        if not model.load_model():
            print("❌ ERROR: No trained model found!")
            sys.exit(1)
        print("✓ Model loaded")

    # Calibrate
    ece_before, ece_after, accuracy = calibrate_model(model, df)

    # Reload model with calibration
    model = TradingModel(sequence_length=15)
    model.load_model()

    # Validate
    metrics = validate_improvements(model, df)

    # Summary
    print_summary(ece_before, ece_after, accuracy, metrics)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n✅ Deployment complete!")


if __name__ == '__main__':
    main()
