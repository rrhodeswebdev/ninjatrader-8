#!/usr/bin/env python3
"""
Calibrate Trading Model Confidence Scores

This script calibrates a trained model to produce well-calibrated confidence scores.

After training, neural networks are often poorly calibrated:
- "70% confident" might actually be correct 45% of the time
- Calibration fixes this mismatch

Usage:
    uv run python calibrate_model.py

The script will:
1. Load the trained model
2. Load validation data
3. Fit calibration (Temperature Scaling, Platt Scaling, Isotonic Regression)
4. Save calibration parameters
5. Report ECE (Expected Calibration Error) before and after

Expected Results:
- ECE before: 0.10-0.20 (poorly calibrated)
- ECE after: < 0.05 (well calibrated)
- Confidence scores will match actual accuracy
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model import TradingModel
from confidence_calibration_advanced import (
    calibrate_trading_model,
    calculate_ece,
    plot_reliability_diagram
)


def load_validation_data(model: TradingModel, data_path: str = 'historical_data.csv',
                        val_split: float = 0.2):
    """Load and prepare validation data"""
    print("Loading historical data...")
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])

    print(f"Total bars: {len(df)}")

    # Use last 20% as validation
    val_size = int(len(df) * val_split)
    df_val = df.iloc[-val_size:].copy()

    print(f"Validation bars: {len(df_val)}")

    # Prepare data (without fitting scaler - use existing scaler from training)
    print("Preparing validation data...")
    X_val, y_val = model.prepare_data(df_val, fit_scaler=False)

    print(f"Validation sequences: {len(X_val)}")
    print(f"Label distribution: {np.bincount(y_val)}")

    return X_val, y_val


def main():
    print("="*70)
    print("CONFIDENCE CALIBRATION FOR TRADING MODEL")
    print("="*70)

    # Initialize model
    print("\n1. Loading trained model...")
    model = TradingModel(sequence_length=15)

    if not model.load_model():
        print("ERROR: No trained model found!")
        print("Please train a model first using:")
        print("  uv run python main.py --train")
        sys.exit(1)

    print("✓ Model loaded successfully")

    # Load validation data
    print("\n2. Loading validation data...")
    try:
        X_val, y_val = load_validation_data(model)
    except FileNotFoundError:
        print("ERROR: historical_data.csv not found!")
        print("Please ensure historical_data.csv exists in rnn-server/")
        sys.exit(1)

    # Create validation dataloader
    print("\n3. Creating validation dataloader...")
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"✓ Validation batches: {len(val_loader)}")

    # Calculate ECE before calibration
    print("\n4. Measuring calibration BEFORE...")
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
    print(f"  ECE (before calibration): {ece_before:.4f}")

    if ece_before < 0.05:
        print("  → Already well calibrated! (ECE < 0.05)")
    elif ece_before < 0.10:
        print("  → Moderately calibrated (ECE < 0.10)")
    else:
        print("  → Poorly calibrated (ECE > 0.10) - calibration needed!")

    # Calibrate
    print("\n5. Calibrating model...")
    print("  Testing multiple calibration methods...")

    calibration = calibrate_trading_model(
        model.model,
        val_loader,
        device=model.device
    )

    print("✓ Calibration complete")

    # Measure ECE after calibration
    print("\n6. Measuring calibration AFTER...")

    # Test each method
    methods = ['temperature', 'platt', 'isotonic', 'ensemble']

    best_method = None
    best_ece = float('inf')

    for method in methods:
        try:
            calibrated_probs = calibration.calibrate(all_logits, all_probs, method=method)
            ece_after = calculate_ece(calibrated_probs, all_labels)

            print(f"  {method.capitalize():15s} ECE: {ece_after:.4f}", end="")

            if ece_after < ece_before:
                improvement = ((ece_before - ece_after) / ece_before) * 100
                print(f" (↓ {improvement:.1f}% improvement)")
            else:
                print(f" (no improvement)")

            if ece_after < best_ece:
                best_ece = ece_after
                best_method = method
        except Exception as e:
            print(f"  {method.capitalize():15s} FAILED: {e}")

    print(f"\n✓ Best method: {best_method} (ECE: {best_ece:.4f})")

    # Save calibration
    print("\n7. Saving calibration...")
    save_path = Path('models/calibration')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    calibration.save(str(save_path))
    print(f"✓ Calibration saved to: {save_path}")

    # Generate reliability diagram
    print("\n8. Generating reliability diagram...")
    try:
        calibrated_probs = calibration.calibrate(all_logits, all_probs, method=best_method)

        plot_path = Path('models/calibration_reliability_diagram.png')
        plot_reliability_diagram(
            all_probs,
            calibrated_probs,
            all_labels,
            save_path=str(plot_path)
        )
        print(f"✓ Reliability diagram saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not generate reliability diagram: {e}")

    # Summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"Model accuracy:        {accuracy*100:.2f}%")
    print(f"ECE before:            {ece_before:.4f}")
    print(f"ECE after ({best_method}): {best_ece:.4f}")
    print(f"Improvement:           {((ece_before - best_ece) / ece_before * 100):.1f}%")
    print("")

    if best_ece < 0.05:
        print("✓ EXCELLENT: Model is well calibrated (ECE < 0.05)")
        print("  Confidence scores now match actual accuracy!")
    elif best_ece < 0.10:
        print("✓ GOOD: Model is reasonably calibrated (ECE < 0.10)")
        print("  Confidence scores are mostly reliable")
    else:
        print("⚠️  WARNING: Model still poorly calibrated (ECE > 0.10)")
        print("  Consider retraining with better loss function")

    print("")
    print("Next steps:")
    print("1. The model will automatically load calibration on next run")
    print("2. Calibrated predictions will be used in production")
    print("3. Monitor ECE over time to detect drift")
    print("="*70)


if __name__ == '__main__':
    main()
