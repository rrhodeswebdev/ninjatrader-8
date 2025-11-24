#!/usr/bin/env python3
"""
Check how much historical data the model has
"""

from model import TradingModel

print("="*70)
print("MODEL DATA STATUS CHECK")
print("="*70)

# Initialize model
model = TradingModel(sequence_length=15)

print(f"\n1. Model Configuration:")
print(f"   Sequence length: {model.sequence_length} bars")
print(f"   Is trained: {model.is_trained}")

print(f"\n2. Historical Data:")
if model.historical_data is None or len(model.historical_data) == 0:
    print(f"    NO HISTORICAL DATA")
    print(f"   Bars available: 0")
else:
    print(f"    Historical data loaded")
    print(f"   Bars available: {len(model.historical_data)}")
    print(f"   Data range: {model.historical_data['time'].iloc[0]} to {model.historical_data['time'].iloc[-1]}")

print(f"\n3. Minimum Requirements:")
print(f"   Need at least: {model.sequence_length} bars for predictions")

if model.historical_data is None or len(model.historical_data) < model.sequence_length:
    bars_needed = model.sequence_length - (len(model.historical_data) if model.historical_data is not None else 0)
    print(f"\n{'='*70}")
    print(f"  INSUFFICIENT DATA")
    print(f"{'='*70}")
    print(f"   Current bars: {len(model.historical_data) if model.historical_data is not None else 0}")
    print(f"   Need: {model.sequence_length} bars minimum")
    print(f"   Missing: {bars_needed} more bars")
    print(f"\n SOLUTION:")
    print(f"   Option 1: Send historical data from NinjaTrader")
    print(f"             (Right-click chart  Export  Send to RNN server)")
    print(f"\n   Option 2: Wait for {bars_needed} more real-time bars")
    print(f"             (If using 1-minute bars, wait {bars_needed} minutes)")
    print(f"{'='*70}")
else:
    print(f"\n{'='*70}")
    print(f" SUFFICIENT DATA - Model can make predictions")
    print(f"{'='*70}")

print()
