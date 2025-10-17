#!/usr/bin/env python3
"""
Phase 3 Training Script

Trains all advanced features:
- Adaptive retraining setup
- Regime-specific models
- Meta-labeling trade filter

Usage:
  uv run python train_phase3.py --all
  uv run python train_phase3.py --train-regime-models
  uv run python train_phase3.py --train-meta-labeling
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from model import TradingModel
from adaptive_retraining import AdaptiveRetrainingManager
from regime_models import RegimeModelManager
from meta_labeling import MetaLabeler


def load_data(file_path: str = 'historical_data.csv'):
    """Load historical data"""
    data_file = Path(file_path)

    if not data_file.exists():
        print(f"❌ Error: {data_file} not found")
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


def train_primary_model(df: pd.DataFrame, epochs: int = 100):
    """Train primary trading model with optimized hyperparameters"""
    print("\n" + "="*70)
    print("STEP 1: TRAINING PRIMARY MODEL (OPTIMIZED)")
    print("="*70)

    # PERFORMANCE OPTIMIZATION: Updated hyperparameters
    # - Increased batch size from 32 to 64 (with gradient accumulation = effective 256)
    # - Reduced learning rate from 0.001 to 0.0005 for better convergence
    # - Increased default epochs from 50 to 100
    # - Reduced sequence length from 40 to 15 (better for 1-min data)
    model = TradingModel(sequence_length=15)
    model.train(df, epochs=epochs, batch_size=64, learning_rate=0.0005, validation_split=0.2)
    model.save_model()

    print("\n✅ Primary model training complete")
    return model


def setup_adaptive_retraining(df: pd.DataFrame, model: TradingModel):
    """Initialize adaptive retraining system"""
    print("\n" + "="*70)
    print("STEP 2: SETUP ADAPTIVE RETRAINING")
    print("="*70)

    manager = AdaptiveRetrainingManager(
        model_dir='models/adaptive',
        retrain_interval_days=7,
        min_performance_window=100
    )

    # Do initial "retrain" to set baseline
    print("\nSetting baseline performance...")
    results = manager.retrain_model(df, model, epochs=30, validation_split=0.2)

    print(f"\n✅ Adaptive retraining configured")
    print(f"Baseline performance:")
    print(f"  Accuracy: {results['validation_metrics']['accuracy']:.2%}")
    print(f"  Sharpe: {results['validation_metrics']['sharpe']:.3f}")
    print(f"  Win Rate: {results['validation_metrics']['win_rate']:.2%}")

    return manager


def train_regime_models(df: pd.DataFrame, epochs: int = 50):
    """Train regime-specific models"""
    print("\n" + "="*70)
    print("STEP 3: TRAINING REGIME-SPECIFIC MODELS")
    print("="*70)

    manager = RegimeModelManager(
        model_dir='models/regime_specific',
        min_samples_per_regime=500
    )

    results = manager.train_regime_models(
        df,
        epochs_per_regime=epochs,
        verbose=True
    )

    print(f"\n✅ Regime-specific models training complete")
    print(f"Trained regimes: {', '.join(results['trained_regimes'])}")

    return manager


def train_meta_labeling(df: pd.DataFrame, primary_model: TradingModel, epochs: int = 100):
    """Train meta-labeling trade filter"""
    print("\n" + "="*70)
    print("STEP 4: TRAINING META-LABELING MODEL")
    print("="*70)

    meta_labeler = MetaLabeler(
        primary_model=primary_model,
        model_path='models/meta_label_model.pth',
        accuracy_window=50
    )

    meta_labeler.train(
        df,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        learning_rate=0.001
    )

    print(f"\n✅ Meta-labeling model training complete")

    return meta_labeler


def validate_all_systems(df: pd.DataFrame):
    """Validate all Phase 3 systems"""
    print("\n" + "="*70)
    print("VALIDATION: TESTING ALL SYSTEMS")
    print("="*70)

    # Use last 20% of data for validation
    val_size = int(len(df) * 0.2)
    val_df = df.iloc[-val_size:]

    # Load all models
    print("\n1. Loading models...")
    primary_model = TradingModel()
    if not primary_model.load_model():
        print("❌ Failed to load primary model")
        return

    regime_manager = RegimeModelManager()
    if not regime_manager.load_regime_models():
        print("⚠️  No regime models found")

    meta_labeler = MetaLabeler(primary_model)
    if not meta_labeler.load_model():
        print("⚠️  No meta-labeling model found")

    print("✓ All models loaded")

    # Test regime model performance
    if len(regime_manager.regime_models) > 0:
        print("\n2. Comparing regime model performance...")
        comparison = regime_manager.compare_regime_performance(val_df)

    # Test meta-labeling filter
    print("\n3. Testing meta-labeling filter...")
    filtered_count = 0
    total_signals = 0

    for i in range(len(val_df) - primary_model.sequence_length - 5):
        sequence_data = val_df.iloc[:primary_model.sequence_length + i]

        try:
            signal, confidence = primary_model.predict(sequence_data)

            if signal != 'hold':
                from model import detect_market_regime
                regime = detect_market_regime(sequence_data, lookback=min(100, len(sequence_data)-1))

                should_trade, meta_prob, details = meta_labeler.should_take_trade(
                    sequence_data,
                    signal,
                    confidence,
                    regime,
                    threshold=0.55
                )

                total_signals += 1
                if not should_trade:
                    filtered_count += 1
        except:
            continue

    if total_signals > 0:
        filter_rate = (filtered_count / total_signals) * 100
        print(f"\nMeta-Filter Statistics:")
        print(f"  Total signals: {total_signals}")
        print(f"  Filtered: {filtered_count} ({filter_rate:.1f}%)")
        print(f"  Taken: {total_signals - filtered_count} ({100 - filter_rate:.1f}%)")

        if 25 <= filter_rate <= 45:
            print("  ✅ Filter rate in optimal range (25-45%)")
        elif filter_rate > 45:
            print("  ⚠️  Filtering too many trades (consider lowering threshold)")
        else:
            print("  ⚠️  Not filtering enough (consider raising threshold)")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train Phase 3 advanced features')
    parser.add_argument('--data-file', type=str, default='historical_data.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--all', action='store_true',
                        help='Train all Phase 3 features')
    parser.add_argument('--train-primary', action='store_true',
                        help='Train primary model')
    parser.add_argument('--setup-retraining', action='store_true',
                        help='Setup adaptive retraining')
    parser.add_argument('--train-regime-models', action='store_true',
                        help='Train regime-specific models')
    parser.add_argument('--train-meta-labeling', action='store_true',
                        help='Train meta-labeling filter')
    parser.add_argument('--validate', action='store_true',
                        help='Validate all systems')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100 - optimized)')
    parser.add_argument('--meta-epochs', type=int, default=100,
                        help='Meta-labeling epochs (default: 100)')

    args = parser.parse_args()

    # If --all, enable everything
    if args.all:
        args.train_primary = True
        args.setup_retraining = True
        args.train_regime_models = True
        args.train_meta_labeling = True
        args.validate = True

    # If nothing specified, show help
    if not any([args.train_primary, args.setup_retraining, args.train_regime_models,
                args.train_meta_labeling, args.validate]):
        parser.print_help()
        return

    print("\n" + "="*70)
    print("PHASE 3 TRAINING SCRIPT")
    print("="*70)

    # Load data
    df = load_data(args.data_file)

    if df is None or len(df) < 1000:
        print("\n❌ Error: Not enough data for training (need at least 1000 bars)")
        return

    # Train primary model
    primary_model = None
    if args.train_primary:
        primary_model = train_primary_model(df, args.epochs)
    else:
        # Load existing primary model
        primary_model = TradingModel()
        if not primary_model.load_model():
            print("\n⚠️  No primary model found. Training new one...")
            primary_model = train_primary_model(df, args.epochs)

    # Setup adaptive retraining
    if args.setup_retraining:
        retrain_manager = setup_adaptive_retraining(df, primary_model)

    # Train regime models
    if args.train_regime_models:
        regime_manager = train_regime_models(df, args.epochs)

    # Train meta-labeling
    if args.train_meta_labeling:
        meta_labeler = train_meta_labeling(df, primary_model, args.meta_epochs)

    # Validate
    if args.validate:
        validate_all_systems(df)

    print("\n" + "="*70)
    print("PHASE 3 TRAINING COMPLETE")
    print("="*70)

    print("\nNext steps:")
    print("  1. Review validation results above")
    print("  2. Adjust thresholds if needed")
    print("  3. Test in paper trading before live")
    print("  4. Monitor performance and retrain weekly")

    print("\nTo use in production:")
    print("  - Adaptive retraining: Monitors performance automatically")
    print("  - Regime models: Use RegimeModelManager.predict()")
    print("  - Meta-filter: Use MetaLabeler.should_take_trade()")

    print("\nSee PHASE3_ADVANCED_FEATURES.md for full documentation")
    print()


if __name__ == '__main__':
    main()
