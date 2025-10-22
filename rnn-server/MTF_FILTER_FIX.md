# Multi-Timeframe Filter Fix

## Issue

When making predictions, you were seeing:
```
Insufficient 5m data for trend calculation
```

## Root Cause

The MTF filter was being passed `new_bar_secondary` (a single new 5m bar) instead of the full historical 5m data stored in the model.

**Before:**
```python
# ❌ WRONG - only passes single new bar
mtf_aligned, mtf_reasons = check_multi_timeframe_alignment(current_data, new_bar_secondary)
```

The filter needs at least 50 bars of 5m data to:
- Calculate SMA(20) and SMA(50) for trend detection
- Determine trend strength
- Check alignment with 1m signals

## Solution

**After:**
```python
# ✅ CORRECT - passes full historical 5m data from model
secondary_data = model.historical_data_secondary if hasattr(model, 'historical_data_secondary') else None

# Only run MTF filter if we have sufficient 5m data
if secondary_data is not None and len(secondary_data) >= 50:
    mtf_aligned, mtf_reasons = check_multi_timeframe_alignment(current_data, secondary_data)
else:
    # Not enough 5m data yet - allow trade but log warning
    mtf_aligned = True
    mtf_reasons = [f"Insufficient 5m data ({len(secondary_data) if secondary_data is not None else 0} bars) - MTF filter skipped"]
    print(f"⚠️  MTF FILTER: {mtf_reasons[0]}")
```

## How It Works Now

1. **Startup Phase** (first ~50 bars of 5m data):
   - MTF filter is **skipped** (not enough data)
   - You'll see: `⚠️ MTF FILTER: Insufficient 5m data (X bars) - MTF filter skipped`
   - Trades are allowed (other filters still apply)

2. **Normal Operation** (after 50+ bars of 5m data):
   - MTF filter is **active**
   - Uses full historical 5m data from `model.historical_data_secondary`
   - All 4 filters check alignment between 1m and 5m timeframes

## Data Accumulation

The model accumulates 5m bars over time:
- Each 5m bar is added to `model.historical_data_secondary`
- Model keeps last 10,000 bars (configurable in `model.py:4034`)
- MTF filter activates automatically once 50+ bars are available

**Time to activation:**
- 5-minute bars: 50 bars × 5 min = **250 minutes (~4 hours)**
- During market hours, MTF filter will be active after ~4 hours of operation

## What You'll See

### First 4 Hours (Accumulating Data):
```
⚠️  MTF FILTER: Insufficient 5m data (12 bars) - MTF filter skipped
```

### After 4 Hours (Filter Active):
```
🚫 MULTI-TIMEFRAME FILTER: Trade blocked
   Trend Alignment: Counter-trend BUY in 5m downtrend
   Trend Strength: Strong trend: ADX=28.3
   Action: Skipping counter-trend trade
```

Or if trade is allowed (no console output for MTF - proceeds silently).

## Quick Start Recommendation

If you want to test the MTF filter immediately without waiting 4 hours:

1. **Load historical 5m data** when starting the server:
   - Send historical request with 5m bars
   - Model will store them in `historical_data_secondary`
   - MTF filter activates immediately

2. **Monitor console output**:
   - Look for `⚠️ MTF FILTER` warnings during startup
   - Once warning disappears, filter is active

## Testing

To verify the fix is working:

```bash
# Start server
cd rnn-server
uv run fastapi dev main.py

# Make a prediction request
# You should now see either:
# - Warning about insufficient data (if < 50 bars)
# - Filter blocking trades (if >= 50 bars and counter-trend)
# - No MTF output (if >= 50 bars and aligned - trade proceeds)
```

## Files Modified

- ✅ `services/request_handler.py` - Fixed to use `model.historical_data_secondary` instead of `new_bar_secondary`

## No Configuration Changes Needed

The fix is automatic - just restart your server and the MTF filter will:
1. Skip filtering until enough 5m data is accumulated
2. Activate automatically once 50+ bars are available
3. Continue filtering all subsequent predictions
