# Model Not Trained - Quick Fix Guide

## The Problem

You're seeing:
- ✅ Confidence: **0.0** (always zero)
- ✅ Signal: **hold** (always hold)
- ✅ Console: **"WARNING: Model not trained yet!"**
- ✅ Market regime: **volatile** (blocking trades)

## Root Cause

After the pure price action migration (105 → 87 features), the old model checkpoint is **incompatible**. The model needs to be **retrained from scratch**.

Training is triggered automatically when historical data arrives, but it may not have started yet or may have failed.

---

## Quick Fix: Manual Training

### Option 1: Force Training with Existing Data

If you have data accumulated but training hasn't started:

```python
# In Python console or add to main.py temporarily:
from model import TradingModel
import pandas as pd

# Load model
model = TradingModel()

# Check if there's historical data
if hasattr(model, 'historical_data') and len(model.historical_data) > 100:
    print(f"Found {len(model.historical_data)} bars of historical data")
    print("Starting training...")
    model.train(model.historical_data, epochs=100, batch_size=32)
    print("Training complete!")
else:
    print("Not enough historical data yet. Need 100+ bars.")
```

### Option 2: Check Training Status via API

```bash
# Check if model is trained
curl http://localhost:8000/training-status

# Expected response:
{
  "is_training": false,
  "is_trained": false,  # ← This is the problem
  "progress": 0.0
}
```

### Option 3: Restart Server and Monitor

Sometimes training fails silently. Restart and watch for training messages:

```bash
# Stop server (Ctrl+C)
# Restart
cd rnn-server
uv run fastapi dev main.py

# Watch console for:
# "🏋️ TRAINING STARTED..."
# "✅ Training complete!"
```

---

## Why Training Might Not Have Started

### 1. Not Enough Historical Data

Training requires **100+ bars** of data.

**Check:**
```python
# In Python console:
from main import trading_model
print(f"Historical data: {len(trading_model.historical_data) if hasattr(trading_model, 'historical_data') else 0} bars")
```

**Solution:**
- Wait for data to accumulate (NinjaTrader sends 1 bar per minute)
- Or send a historical data request from NinjaTrader

### 2. Training Failed with Error

Check console logs for error messages like:
```
❌ Training error: ...
```

**Common errors after migration:**
- ✅ **FIXED**: "name 'atr' is not defined"
- ✅ **FIXED**: "index 99 is out of bounds"
- ⚠️ **Possible**: Other shape mismatches

**Solution:**
- Check git log - all known errors have been fixed
- If you see a new error, report it

### 3. Training In Progress But Not Complete

Training takes **5-10 minutes**.

**Check:**
```bash
curl http://localhost:8000/training-status
# Look for "is_training": true
```

**Solution:**
- Wait for completion
- Monitor progress via `/training-status` endpoint

---

## Force Training Right Now

### Step 1: Check Current State

Open Python console in rnn-server directory:
```bash
cd rnn-server
uv run python
```

```python
from main import trading_model

print(f"Is trained: {trading_model.is_trained}")
print(f"Has data: {hasattr(trading_model, 'historical_data')}")
if hasattr(trading_model, 'historical_data'):
    print(f"Data length: {len(trading_model.historical_data)}")
```

### Step 2: If Data Exists, Train Now

```python
if hasattr(trading_model, 'historical_data') and len(trading_model.historical_data) >= 100:
    print("Starting training...")
    trading_model.train(trading_model.historical_data, epochs=50, batch_size=32)
    print(f"Training complete! Model trained: {trading_model.is_trained}")
else:
    print("Not enough data. Need to accumulate more bars from NinjaTrader.")
```

### Step 3: Verify Model Works

```python
# Make a test prediction
if trading_model.is_trained and hasattr(trading_model, 'historical_data'):
    signal, confidence = trading_model.predict(trading_model.historical_data)
    print(f"Test prediction: {signal}, confidence: {confidence:.2%}")

    if confidence > 0:
        print("✅ Model is working!")
    else:
        print("⚠️ Model trained but confidence still 0")
```

---

## Alternative: Create Minimal Training Data

If you don't have historical data yet, you can create minimal training data for testing:

```python
import pandas as pd
import numpy as np
from main import trading_model

# Create synthetic data (for testing only!)
n = 500
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
    'open': 5000 + np.random.randn(n).cumsum(),
    'high': 5000 + np.random.randn(n).cumsum() + 5,
    'low': 5000 + np.random.randn(n).cumsum() - 5,
    'close': 5000 + np.random.randn(n).cumsum(),
    'volume': np.random.randint(100, 1000, n),
    'bid_volume': np.random.randint(50, 500, n),
    'ask_volume': np.random.randint(50, 500, n),
    'timeframe': '1m'
})

# Add required columns
df['dailyPnL'] = 0.0
df['dailyGoal'] = 500.0
df['dailyMaxLoss'] = 250.0

print(f"Created {len(df)} bars of synthetic data")
print("Starting training...")

trading_model.train(df, epochs=50, batch_size=32)

print(f"Training complete! Is trained: {trading_model.is_trained}")
```

**⚠️ WARNING:** This is for testing ONLY. The model won't make good predictions with synthetic data. You need real market data for actual trading.

---

## Expected Timeline

### From Cold Start:
1. **0-5 min:** Server starts, waits for historical data request
2. **5-15 min:** NinjaTrader sends historical data → Training triggers
3. **15-25 min:** Training completes (100 epochs)
4. **25+ min:** Model ready, predictions start

### Current State (After 1+ Hour):
- **Should have:** 60+ bars of data
- **Should have:** Training completed
- **Reality:** Model not trained

**This suggests:** Training never started OR failed silently

---

## Immediate Action Plan

### 1. Check Training Status
```bash
curl http://localhost:8000/training-status
```

### 2. If Not Trained, Check for Data
```bash
# In Python console
from main import trading_model
print(f"Data: {len(getattr(trading_model, 'historical_data', []))}")
```

### 3. If Data Exists (>100 bars), Force Train
```bash
# In Python console
from main import trading_model
if hasattr(trading_model, 'historical_data') and len(trading_model.historical_data) >= 100:
    trading_model.train(trading_model.historical_data, epochs=100, batch_size=32)
```

### 4. Restart Server
```bash
# Kill and restart
# Training should trigger automatically on next historical request
```

### 5. Monitor Console
Watch for:
- "🏋️ TRAINING STARTED..."
- "Epoch 1/100..."
- "✅ Training complete!"

---

## After Model is Trained

Once `is_trained = True`:

1. ✅ Confidence will be > 0
2. ✅ Signals will be generated (BUY/SELL/HOLD)
3. ✅ Market regime filter will be more accurate
4. ✅ Trades should start flowing (if regime allows)

**Then you can:**
- Monitor trade quality
- Gradually increase confidence threshold (0.25 → 0.40 → 0.55)
- Validate performance vs baseline

---

## Troubleshooting

### "Training started but never finished"
- Check for errors in console
- Training might have crashed
- Try restarting server

### "Training completes but confidence still 0"
- Check if model saved correctly
- Try making prediction manually in Python console
- May need to debug model.predict() function

### "Not enough historical data"
- Let server run longer
- Send historical request from NinjaTrader
- Or create synthetic data for testing (see above)

---

## Summary

**Problem:** Model not trained → Confidence = 0 → No trades
**Cause:** Old model incompatible after migration to 87 features
**Solution:** Retrain model with accumulated historical data
**Action:** Check training status, force train if needed, restart server

The regime "volatile" issue is **secondary** - main issue is the untrained model.
