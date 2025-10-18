#!/usr/bin/env python3
"""
Comprehensive Deployment Script for All Improvements

This script orchestrates the deployment of all enhancements:
1. Feature optimization
2. Hyperparameter optimization (optional)
3. Advanced training with curriculum learning
4. Ensemble training
5. Model quantization and export
6. Validation and benchmarking

Usage:
  # Quick deployment (recommended for first run)
  uv run python deploy_improvements.py --quick

  # Full deployment with hyperparameter optimization
  uv run python deploy_improvements.py --full --optimize-hyperparams

  # Just ensemble training (if primary model already trained)
  uv run python deploy_improvements.py --ensemble-only
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime


def load_data(data_path: str = 'historical_data.csv') -> pd.DataFrame:
    """Load historical training data"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("\nGenerating synthetic data for testing...")
        return generate_synthetic_data()

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])

    print(f"✓ Loaded {len(df)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def generate_synthetic_data(n_bars: int = 5000) -> pd.DataFrame:
    """Generate synthetic data for testing"""
    print(f"Generating {n_bars} bars of synthetic data...")

    np.random.seed(42)
    times = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='1min')

    # Generate realistic price movement
    returns = np.random.normal(0.0001, 0.002, n_bars)
    close = 4500 * np.exp(np.cumsum(returns))

    high = close + np.abs(np.random.normal(1, 0.5, n_bars))
    low = close - np.abs(np.random.normal(1, 0.5, n_bars))
    open_prices = np.roll(close, 1)
    open_prices[0] = 4500
    volume = np.random.lognormal(8, 1, n_bars)

    df = pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'bid_volume': volume * np.random.uniform(0.4, 0.6, n_bars),
        'ask_volume': volume * np.random.uniform(0.4, 0.6, n_bars)
    })

    print(f"✓ Generated synthetic data")
    return df


def analyze_features(df: pd.DataFrame, run_analysis: bool = True):
    """Run feature optimization analysis"""
    if not run_analysis:
        print("\nSkipping feature analysis (--skip-feature-analysis)")
        return None

    print("\n" + "="*70)
    print("STEP 1: FEATURE OPTIMIZATION ANALYSIS")
    print("="*70)

    from feature_optimization import FeatureOptimizer
    from model import TradingModel

    # Create temporary model to generate features
    temp_model = TradingModel(sequence_length=15)
    temp_model.update_historical_data(df, None)

    # Prepare features and labels
    try:
        X, y, _ = temp_model.prepare_data(df, lookahead_bars=3, hold_percentage=0.40)

        # Run analysis
        optimizer = FeatureOptimizer(correlation_threshold=0.95, min_importance_score=0.001)
        results = optimizer.analyze_features(X, y, temp_model.feature_names)

        # Save visualization
        optimizer.save_visualization('analysis')

        return results

    except Exception as e:
        print(f"⚠️  Feature analysis failed: {e}")
        return None


def optimize_hyperparameters(df: pd.DataFrame, n_trials: int = 20):
    """Run hyperparameter optimization"""
    print("\n" + "="*70)
    print("STEP 2: HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    from hyperparameter_optimization import HyperparameterOptimizer

    optimizer = HyperparameterOptimizer(
        df,
        n_trials=n_trials,
        validation_split=0.2
    )

    results = optimizer.optimize()
    optimizer.plot_optimization_history()

    return results['best_params']


def train_primary_model(df: pd.DataFrame, hyperparams: dict = None,
                       use_curriculum: bool = True):
    """Train primary model"""
    print("\n" + "="*70)
    print("STEP 3: TRAINING PRIMARY MODEL")
    print("="*70)

    from model import TradingModel

    # Use hyperparams if provided
    if hyperparams:
        print(f"\nUsing optimized hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")

        model = TradingModel(
            sequence_length=hyperparams.get('sequence_length', 15),
            hidden_size=hyperparams.get('hidden_size', 128),
            num_layers=hyperparams.get('num_layers', 2),
            dropout=hyperparams.get('dropout', 0.3)
        )

        epochs = 100
        batch_size = hyperparams.get('batch_size', 64)
        learning_rate = hyperparams.get('learning_rate', 0.0005)
    else:
        # Use defaults
        model = TradingModel(sequence_length=15)
        epochs = 100
        batch_size = 64
        learning_rate = 0.0005

    # Train with or without curriculum learning
    if use_curriculum:
        print("\nUsing curriculum learning...")
        from curriculum_learning import train_with_curriculum

        results = train_with_curriculum(
            model,
            df,
            total_epochs=epochs,
            n_stages=4,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    else:
        print("\nUsing standard training...")
        results = model.train(
            df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

    # Save model
    model.save_model()

    print(f"\n✓ Primary model trained and saved")
    return model


def train_ensemble(df: pd.DataFrame):
    """Train multi-architecture ensemble"""
    print("\n" + "="*70)
    print("STEP 4: TRAINING ENSEMBLE")
    print("="*70)

    from ensemble_advanced import MultiArchitectureEnsemble

    ensemble = MultiArchitectureEnsemble(sequence_length=15)

    results = ensemble.train_all(
        df,
        epochs=80,  # Slightly fewer epochs per model
        batch_size=64,
        learning_rate=0.0005
    )

    print(f"\n✓ Ensemble trained")
    return ensemble, results


def optimize_for_production(model_path: str = 'models/trading_model.pth'):
    """Quantize and export model"""
    print("\n" + "="*70)
    print("STEP 5: PRODUCTION OPTIMIZATION")
    print("="*70)

    from performance_optimization import ModelQuantizer, ONNXExporter
    from model import TradingModel
    import torch

    # Load model
    model = TradingModel(sequence_length=15)
    if not model.load_model(model_path):
        print("❌ Failed to load model")
        return

    print("\n5a. Quantizing model...")
    quantizer = ModelQuantizer(model.model, device=str(model.device))

    # Dynamic quantization (easier, no calibration needed)
    quantized_model = quantizer.quantize_dynamic()

    # Benchmark
    test_input = torch.randn(1, 15, 97).to(model.device)
    benchmark_results = quantizer.benchmark_inference(test_input, n_iterations=100)

    print("\n5b. Exporting to ONNX...")
    exporter = ONNXExporter(model.model, device=str(model.device))

    success = exporter.export(
        input_shape=(1, 15, 97),
        output_path='models/trading_model.onnx'
    )

    if success:
        onnx_benchmark = exporter.benchmark_onnx(
            input_shape=(1, 15, 97),
            n_iterations=100
        )

    print(f"\n✓ Production optimization complete")
    return {
        'quantization': benchmark_results,
        'onnx': onnx_benchmark if success else None
    }


def run_validation(df: pd.DataFrame, ensemble):
    """Validate ensemble performance"""
    print("\n" + "="*70)
    print("STEP 6: VALIDATION")
    print("="*70)

    # Use last 20% for validation
    val_size = int(len(df) * 0.2)
    df_val = df.iloc[-val_size:]

    print(f"Validating on {val_size} bars...")

    results = ensemble.evaluate_on_validation(df_val)

    return results


def generate_deployment_report(results: dict, output_dir: str = 'deployment_results'):
    """Generate comprehensive deployment report"""
    print("\n" + "="*70)
    print("GENERATING DEPLOYMENT REPORT")
    print("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_path / f'deployment_report_{timestamp}.json'

    # Save full results
    with open(report_file, 'w') as f:
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, list, dict, bool, type(None))):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        json.dump(serializable_results, f, indent=2)

    print(f"✓ Report saved to {report_file}")

    # Print summary
    print("\n" + "="*70)
    print("DEPLOYMENT SUMMARY")
    print("="*70)
    print(f"\nTimestamp: {timestamp}")

    if 'feature_analysis' in results and results['feature_analysis']:
        feat_results = results['feature_analysis']
        print(f"\n📊 Feature Analysis:")
        print(f"  Original features: {feat_results['original_feature_count']}")
        print(f"  Recommended features: {feat_results['recommended_feature_count']}")
        print(f"  Reduction: {(1 - feat_results['recommended_feature_count']/feat_results['original_feature_count'])*100:.1f}%")

    if 'hyperparameters' in results and results['hyperparameters']:
        print(f"\n⚙️  Optimized Hyperparameters:")
        for key, value in results['hyperparameters'].items():
            print(f"  {key}: {value}")

    if 'ensemble_results' in results:
        print(f"\n🎯 Ensemble Training:")
        for arch, metrics in results['ensemble_results'].items():
            print(f"  {arch}: Sharpe={metrics['sharpe']:.3f}, Acc={metrics['accuracy']:.2%}")

    if 'production_optimization' in results and results['production_optimization']:
        quant = results['production_optimization'].get('quantization', {})
        if quant:
            print(f"\n⚡ Performance:")
            print(f"  Quantization speedup: {quant.get('speedup', 0):.2f}x")

        onnx = results['production_optimization'].get('onnx', {})
        if onnx:
            print(f"  ONNX speedup: {onnx.get('speedup', 0):.2f}x")

    if 'validation' in results:
        val = results['validation']
        print(f"\n✅ Validation:")
        print(f"  Predictions: {val['n_predictions']}")
        print(f"  Avg Confidence: {val['avg_confidence']:.2%}")
        print(f"  Avg Agreement: {val['avg_agreement_rate']:.2%}")

    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETE")
    print("="*70)

    return report_file


def main():
    parser = argparse.ArgumentParser(description='Deploy all model improvements')
    parser.add_argument('--data-file', type=str, default='historical_data.csv',
                       help='Path to historical data CSV')
    parser.add_argument('--quick', action='store_true',
                       help='Quick deployment (skip hyperparameter optimization)')
    parser.add_argument('--full', action='store_true',
                       help='Full deployment with all optimizations')
    parser.add_argument('--ensemble-only', action='store_true',
                       help='Only train ensemble (assumes primary model exists)')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='Run hyperparameter optimization (slow)')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of hyperparameter optimization trials')
    parser.add_argument('--skip-feature-analysis', action='store_true',
                       help='Skip feature analysis step')
    parser.add_argument('--skip-quantization', action='store_true',
                       help='Skip quantization and ONNX export')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')

    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL IMPROVEMENT DEPLOYMENT")
    print("="*70)
    print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Custom'}")
    print(f"Data: {args.data_file}")
    print("="*70)

    results = {}

    # Load data
    df = load_data(args.data_file)
    if df is None or len(df) < 1000:
        print("❌ Insufficient data")
        return

    # Step 1: Feature Analysis
    if not args.skip_feature_analysis and not args.ensemble_only:
        feature_results = analyze_features(df, run_analysis=True)
        results['feature_analysis'] = feature_results

    # Step 2: Hyperparameter Optimization (optional)
    hyperparams = None
    if args.optimize_hyperparams or (args.full and not args.ensemble_only):
        hyperparams = optimize_hyperparameters(df, n_trials=args.n_trials)
        results['hyperparameters'] = hyperparams

    # Step 3: Train Primary Model
    if not args.ensemble_only:
        primary_model = train_primary_model(
            df,
            hyperparams=hyperparams,
            use_curriculum=not args.no_curriculum
        )
        results['primary_model'] = 'trained'

    # Step 4: Train Ensemble
    ensemble, ensemble_results = train_ensemble(df)
    results['ensemble_results'] = ensemble_results

    # Step 5: Production Optimization
    if not args.skip_quantization:
        prod_results = optimize_for_production()
        results['production_optimization'] = prod_results

    # Step 6: Validation
    validation_results = run_validation(df, ensemble)
    results['validation'] = validation_results

    # Generate report
    elapsed_time = time.time() - start_time
    results['elapsed_time_seconds'] = elapsed_time
    results['elapsed_time_minutes'] = elapsed_time / 60

    report_file = generate_deployment_report(results)

    print(f"\n⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"\n📄 Full report: {report_file}")
    print("\n✅ All improvements deployed successfully!")

    print("\n📋 Next Steps:")
    print("  1. Review the deployment report")
    print("  2. Run walk-forward validation: uv run python walk_forward_validation.py")
    print("  3. Run comprehensive backtest: uv run python run_backtest.py")
    print("  4. Start paper trading to validate live performance")
    print("  5. Monitor with: uv run python monitoring_dashboard.py")


if __name__ == '__main__':
    main()
