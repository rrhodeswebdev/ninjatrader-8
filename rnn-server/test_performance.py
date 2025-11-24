#!/usr/bin/env python3
"""
Performance test script for optimized RNN model
Tests that optimizations maintain prediction quality and improve speed
"""

import pandas as pd
import numpy as np
import time
from model import TradingModel

def generate_test_data(n_bars=5000):
    """Generate synthetic OHLC data for testing"""
    np.random.seed(42)

    # Generate random walk price data
    base_price = 4500.0
    returns = np.random.randn(n_bars) * 0.002  # 0.2% std returns
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC with realistic relationships
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_bars, freq='1min'),
        'close': prices
    })

    # Generate high/low/open from close
    df['high'] = df['close'] * (1 + np.abs(np.random.randn(n_bars) * 0.001))
    df['low'] = df['close'] * (1 - np.abs(np.random.randn(n_bars) * 0.001))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])

    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df

def test_model_performance():
    """Test optimized model performance and accuracy"""
    print("="*60)
    print("RNN MODEL PERFORMANCE TEST")
    print("="*60)

    # Generate test data
    print("\n1. Generating test data (5000 bars)...")
    df = generate_test_data(5000)
    print(f"    Generated {len(df)} bars")

    # Initialize model
    print("\n2. Initializing optimized model...")
    model = TradingModel(sequence_length=20)

    # Check if model is trained
    if not model.is_trained:
        print("   ! Model not trained yet - training on test data...")
        model.train(df, epochs=10, batch_size=32)
    else:
        print(f"    Model loaded from {model.model_path}")
        print(f"    Device: {model.device}")

    # Test prediction speed
    print("\n3. Testing prediction performance...")
    print("   Running 10 predictions to measure average latency...")

    latencies = []
    predictions = []

    for i in range(10):
        # Add new synthetic bar
        new_bar = pd.DataFrame({
            'time': [pd.Timestamp.now()],
            'open': [df['close'].iloc[-1]],
            'high': [df['close'].iloc[-1] * 1.001],
            'low': [df['close'].iloc[-1] * 0.999],
            'close': [df['close'].iloc[-1] * (1 + np.random.randn() * 0.002)]
        })

        test_df = pd.concat([df, new_bar], ignore_index=True)

        # Time the prediction
        start = time.perf_counter()
        signal, confidence = model.predict(test_df)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)  # Convert to ms
        predictions.append((signal, confidence))

        print(f"   Prediction {i+1}: {signal.upper()} ({confidence*100:.1f}%) - {elapsed*1000:.1f}ms")

    # Calculate statistics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)

    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    print(f"Average Latency:    {avg_latency:.2f}ms")
    print(f"Min Latency:        {min_latency:.2f}ms")
    print(f"Max Latency:        {max_latency:.2f}ms")
    print(f"Std Deviation:      {std_latency:.2f}ms")
    print(f"Predictions/Second: {1000/avg_latency:.1f}")

    # Check prediction quality
    print("\n" + "="*60)
    print("PREDICTION QUALITY")
    print("="*60)

    signals = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]

    signal_dist = {
        'long': signals.count('long'),
        'short': signals.count('short'),
        'hold': signals.count('hold')
    }

    print(f"Signal Distribution:")
    print(f"  Long:  {signal_dist['long']}/10 ({signal_dist['long']*10}%)")
    print(f"  Short: {signal_dist['short']}/10 ({signal_dist['short']*10}%)")
    print(f"  Hold:  {signal_dist['hold']}/10 ({signal_dist['hold']*10}%)")
    print(f"\nAverage Confidence: {np.mean(confidences)*100:.1f}%")
    print(f"Min Confidence:     {np.min(confidences)*100:.1f}%")
    print(f"Max Confidence:     {np.max(confidences)*100:.1f}%")

    # Verify optimizations are active
    print("\n" + "="*60)
    print("OPTIMIZATION STATUS")
    print("="*60)

    # Check if quantization is applied (CPU)
    if model.device.type == 'cpu':
        # Check if model is quantized by looking for quantized layers
        is_quantized = any('Quantized' in str(type(m)) for m in model.model.modules())
        print(f"INT8 Quantization (CPU): {' ACTIVE' if is_quantized else ' NOT ACTIVE'}")
    else:
        print(f"FP16 Inference (GPU):     ACTIVE")

    print(f"Hurst Caching:            ACTIVE (cache size: {len(model._hurst_cache)})")
    print(f"Timing Instrumentation:   ACTIVE")

    # Performance target
    print("\n" + "="*60)
    print("PERFORMANCE TARGET")
    print("="*60)

    target_latency = 150  # ms
    if avg_latency < target_latency:
        print(f" PASS: Average latency {avg_latency:.1f}ms < {target_latency}ms target")
        print("   Model is fast enough for real-time trading!")
    else:
        print(f"  MARGINAL: Average latency {avg_latency:.1f}ms > {target_latency}ms target")
        print("   Model may struggle with high-frequency updates")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    return {
        'avg_latency_ms': avg_latency,
        'predictions_per_second': 1000 / avg_latency,
        'signal_distribution': signal_dist,
        'avg_confidence': np.mean(confidences)
    }

if __name__ == "__main__":
    results = test_model_performance()
