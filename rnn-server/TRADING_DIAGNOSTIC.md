# Trading Diagnostic - Why No Trades Are Happening

## Quick Diagnosis Checklist

### 1. Is the Model Trained? ⚠️ **MOST LIKELY ISSUE**

After the pure price action migration, the old model (105 features) is incompatible with the new architecture (87 features). The model needs retraining.

**Check:**
```bash
# Visit in browser or curl:
curl http://localhost:8000/training-status
```

**Expected Response:**
- If `"is_trained": false` → **Model needs training**
- If `"is_training": true` → **Training in progress, wait for completion**
- If `"is_trained": true` → **Model ready, check other filters**

**Solution if not trained:**
The model will automatically train when you send historical data. If it hasn't started:

1. **Send historical request** - NinjaTrader should send historical bars on connection
2. **Wait for training** - Takes 5-10 minutes for 100 epochs
3. **Monitor progress** - Check `/training-status` endpoint

---

### 2. Confidence Threshold Too High?

**Current Setting:** `MIN_CONFIDENCE_THRESHOLD = 0.60` (60%)

This is **VERY HIGH** and will filter out most signals, especially for an untrained or newly-trained model.

**Check main.py line 47:**
```python
MIN_CONFIDENCE_THRESHOLD = 0.60  # 60% - VERY RESTRICTIVE
```

**Recommended Settings:**
- **For new/untrained model:** 0.25 (25%) - Allow model to learn
- **For testing:** 0.40 (40%) - Moderate filtering
- **For production:** 0.55-0.60 (55-60%) - Strict filtering

**Quick Fix:**
```python
# main.py line 47
MIN_CONFIDENCE_THRESHOLD = 0.25  # Lower for testing
```

---

### 3. Market Regime Filter Blocking Trades?

After removing ADX, the market regime filter might be classifying all markets as "choppy" and blocking trades.

**Check Console Output:**
Look for messages like:
```
🚫 MARKET REGIME FILTER: Market regime unfavorable: choppy
```

**If you see this constantly:**
- The regime detection is too strict
- Lower the thresholds (already lowered to 0.3 and 0.35)
- Or temporarily disable the filter to test

**Temporary Disable (main.py):**
```python
# In handle_realtime_request, comment out regime check:
# if skip_trading:
#     return {"signal": "hold", ...}
```

---

### 4. Multi-Timeframe Filter Blocking Trades?

The MTF filter blocks counter-trend trades (e.g., BUY signal when 5m trend is DOWN).

**Check Console Output:**
```
🚫 MULTI-TIMEFRAME FILTER: Trade blocked
   Trend Alignment: Counter-trend BUY in 5m downtrend
```

**If you see this:**
- The filter is working correctly
- Counter-trend trades have negative expected value
- Wait for with-trend signals

**If you want to temporarily disable:**
```python
# In services/request_handler.py, comment out MTF check:
# if not mtf_aligned:
#     return {"signal": "hold", ...}
```

---

### 5. Signal Stability Filter

Prevents over-trading by requiring signal changes to "settle" before triggering.

**Check Console Output:**
```
⚠️  SIGNAL STABILITY CHECK: Signal changed too recently
```

**This is normal** - It prevents whipsaws. Typically blocks 1-2 signals after a change.

---

### 6. Insufficient Data?

The model needs data to make predictions.

**Minimum Requirements:**
- **Primary (1m):** 20 bars minimum
- **Secondary (5m):** 50 bars minimum (for MTF filter)

**Check Console:**
```
⚠️ MTF FILTER: Insufficient 5m data (12 bars) - MTF filter skipped
```

**Solution:**
- Let the server run for 4+ hours to accumulate 5m data
- Or send historical data request with 50+ 5m bars

---

## Diagnostic Steps (In Order)

### Step 1: Check Training Status
```bash
curl http://localhost:8000/training-status
```

**If model not trained:**
- ✅ Expected after migration (old model incompatible)
- ⏳ Wait for automatic training when historical data arrives
- 🔄 Or manually trigger by restarting server (it will request historical data)

### Step 2: Check Console Logs

**Look for these messages:**

**Model Not Trained:**
```
❌ Model not trained yet. Waiting for training to complete...
```
**Solution:** Wait for training

**Confidence Filter:**
```
Signal: BUY, Confidence: 45.3% < 60.0% threshold
(Filtered due to low confidence)
```
**Solution:** Lower MIN_CONFIDENCE_THRESHOLD to 0.25-0.40

**Market Regime Filter:**
```
🚫 MARKET REGIME FILTER: Market regime unfavorable: choppy
   Regime: CHOPPY
```
**Solution:** Lower thresholds or temporarily disable

**Multi-Timeframe Filter:**
```
🚫 MULTI-TIMEFRAME FILTER: Trade blocked
   Trend Alignment: Counter-trend BUY in 5m downtrend
```
**Solution:** Wait for with-trend signals (this filter is working correctly)

### Step 3: Lower Confidence Threshold (Quick Test)

**Edit main.py:**
```python
# Line 47
MIN_CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.60 for testing
```

**Restart server:**
```bash
# Ctrl+C to stop
uv run fastapi dev main.py
```

### Step 4: Monitor for Predictions

**Make a test request:**
```bash
# Send current bar data from NinjaTrader
# Watch console for output
```

**Expected Output (if working):**
```
📊 PREDICTION: BUY signal
   Confidence: 67.3%
   Entry: $5432.50
   Stop: $5427.50
   Target: $5442.50
```

---

## Quick Fixes Summary

### Issue 1: Model Not Trained (MOST LIKELY)
**Symptom:** Server returns `"status": "not_trained"` or `"model_training": true`
**Fix:** Wait for training to complete (5-10 minutes)
**Verify:** Check `/training-status` endpoint

### Issue 2: Confidence Too High
**Symptom:** Predictions show "Filtered due to low confidence"
**Fix:** Lower `MIN_CONFIDENCE_THRESHOLD` in main.py to 0.25
**Verify:** Restart server, watch for signals

### Issue 3: Regime Filter Too Strict
**Symptom:** Console shows "Market regime unfavorable: choppy" constantly
**Fix:** Lower thresholds or disable temporarily
**Verify:** Should see regime change to "trending"

### Issue 4: MTF Filter Working Correctly
**Symptom:** "Multi-timeframe filter: Trade blocked"
**Fix:** **DO NOT DISABLE** - This is preventing bad trades
**Verify:** Wait for with-trend signals

---

## Recommended Action Plan

1. **Check training status** - Visit http://localhost:8000/training-status
   - If `is_trained: false` → **Wait for training**
   - Training takes 5-10 minutes after historical data arrives

2. **Lower confidence threshold** (while testing)
   - Change `MIN_CONFIDENCE_THRESHOLD = 0.25` in main.py
   - Restart server

3. **Monitor console output**
   - Look for prediction logs
   - Look for filter messages
   - Identify which filter is blocking trades

4. **Once model is trained and tested:**
   - Gradually increase confidence threshold (0.40 → 0.55 → 0.60)
   - Monitor win rate and trade quality
   - Adjust based on performance

---

## Expected Timeline

**Fresh Start (No Trained Model):**
- **0-5 min:** Server starts, requests historical data
- **5-15 min:** Model training (100 epochs)
- **15+ min:** Model ready, predictions start
- **4+ hours:** MTF filter activates (50 5m bars accumulated)

**If Model Already Trained:**
- **Immediately:** Predictions start
- **4+ hours:** MTF filter activates

---

## How to Tell What's Blocking Trades

### Model Not Trained
```json
{
  "status": "not_trained",
  "message": "Model not trained yet",
  "signal": "hold",
  "confidence": 0.0
}
```

### Confidence Filter
```
Console: "Filtered due to low confidence < 60%"
```

### Market Regime Filter
```json
{
  "regime_filtered": true,
  "regime": {"regime": "choppy", "should_trade": false}
}
```

### Multi-Timeframe Filter
```json
{
  "mtf_filtered": true,
  "mtf_reasons": ["Counter-trend BUY in 5m downtrend"]
}
```

---

## Current System State (After Migration)

✅ **Architecture:** 87 pure price action features
✅ **Code:** All indicators removed
✅ **Filters:** MTF, regime, confidence, stability all active
⚠️ **Model:** Needs retraining (old model incompatible)
⚠️ **Threshold:** 60% (very high - lower for testing)

**MOST LIKELY ISSUE:** Model not trained yet after migration. Check `/training-status` endpoint.
