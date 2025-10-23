# Indicator Removal Plan - Pure Price Action Migration

## Overview

This document details the complete removal of all lagging indicators from the RNN trading model, replacing them with pure price action features.

## What is Being Removed

### Complete Removal List (44 features across both timeframes)

#### 1. Moving Averages (16 features total)
**File:** `core/indicators.py` or wherever calculate_sma/calculate_ema are defined

Remove ALL:
- SMA 9, 21, 50, 100, 200 (5 × 2 timeframes = 10 features)
- EMA 9, 21, 50 (3 × 2 timeframes = 6 features)

**Why:** Moving averages are lagging by definition. They smooth price data, which introduces lag and hides actual price structure.

#### 2. Oscillators (6 features total)
**File:** `core/indicators.py`

Remove ALL:
- RSI (Relative Strength Index) - 2 features (1m, 5m)
- Stochastic %K - 2 features (1m, 5m)
- Stochastic %D - 2 features (1m, 5m)

**Why:** Oscillators are derived from smoothed price data. They don't show actual price structure or order flow.

#### 3. Momentum Indicators (6 features total)
**File:** `core/indicators.py`

Remove ALL:
- MACD - 2 features (1m, 5m)
- MACD Signal - 2 features (1m, 5m)
- MACD Histogram - 2 features (1m, 5m)

**Why:** MACD is based on EMAs (moving averages of moving averages). Double-lagging indicator.

#### 4. Volatility Indicators (8 features total)
**File:** `core/indicators.py`

Remove ALL:
- ATR (Average True Range) - 2 features (1m, 5m)
- Bollinger Upper Band - 2 features (1m, 5m)
- Bollinger Middle Band - 2 features (1m, 5m)
- Bollinger Lower Band - 2 features (1m, 5m)

**Why:** ATR is smoothed/averaged. Bollinger Bands are based on SMA. Pure price range is better.

**Note on ATR:** If currently used ONLY for stop-loss calculation (not as model input), it can stay in risk management code.

#### 5. Volume Indicators (4 features total)
**File:** `core/indicators.py`

Remove:
- OBV (On-Balance Volume) - 2 features (1m, 5m)
- VWAP (Volume Weighted Average Price) - 2 features (1m, 5m)

**Why:** OBV is cumulative (derived). VWAP is an average price. Raw volume and volume pressure are better.

### Total Features Removed: 44 features (76% of current indicators)

---

## What is Being Kept

### Pure Price and Order Flow (12 base features)

#### Raw OHLC Data (8 features)
- ✅ Open (1m, 5m)
- ✅ High (1m, 5m)
- ✅ Low (1m, 5m)
- ✅ Close (1m, 5m)

**Why:** These are actual prices, not derived. Foundation of all price action.

#### Volume (2 features)
- ✅ Volume (1m, 5m)

**Why:** Direct measure of order flow. Not smoothed or averaged.

#### Price Changes (2 features)
- ✅ Returns (percent change bar-to-bar)
- ✅ Log Returns (log of price change)

**Why:** Pure price movement, not smoothed. Shows actual volatility.

---

## What is Being Added

### New Pure Price Action Features (~80 features)

#### 1. Candle Characteristics (11 features)
- body_size - size of candle body
- body_pct - body size as % of price
- upper_shadow - upper wick size
- lower_shadow - lower wick size
- candle_range - total bar range
- range_pct - range as % of price
- is_bullish - bullish/bearish direction
- body_to_range - body fill ratio
- upper_shadow_ratio - upper wick ratio
- lower_shadow_ratio - lower wick ratio
- close_position - where close is in range (0-1)

#### 2. Market Structure (29 features)
**Per Period** (10, 20, 50 bars):
- rolling_high_{period} - recent high
- rolling_low_{period} - recent low
- dist_from_high_{period} - distance to resistance
- dist_from_low_{period} - distance to support
- position_in_range_{period} - where price is in range
- breaking_high_{period} - breakout detection
- breaking_low_{period} - breakdown detection

**Trend Structure:**
- higher_high - uptrend structure
- lower_low - downtrend structure
- higher_low - uptrend structure
- lower_high - downtrend structure
- consecutive_up - consecutive up bars
- consecutive_down - consecutive down bars
- up_streak - count of consecutive up bars
- down_streak - count of consecutive down bars

#### 3. Order Flow (13 features)
- volume_change - volume change bar-to-bar
- volume_ratio_10 - volume vs 10-bar average
- volume_ratio_20 - volume vs 20-bar average
- volume_spike_10 - unusual volume (10-bar)
- volume_spike_20 - unusual volume (20-bar)
- buying_pressure - estimated buy volume
- selling_pressure - estimated sell volume
- net_pressure - buying - selling
- pressure_ratio - buying/selling ratio
- net_pressure_10 - 10-bar net pressure
- net_pressure_20 - 20-bar net pressure

#### 4. Price Rejections (5 features)
- strong_upper_rejection - large upper wick
- strong_lower_rejection - large lower wick
- is_doji - indecision candle
- bullish_pin - bullish pin bar
- bearish_pin - bearish pin bar

#### 5. Price Momentum (12 features)
**Per Period** (3, 5, 10, 20 bars):
- price_change_{period} - % change over period
- up_bars_{period} - count of up bars
- move_strength_{period} - volatility of moves

#### 6. Gaps (4 features)
- gap_up - gap up detection
- gap_up_size - size of gap up
- gap_down - gap down detection
- gap_down_size - size of gap down

#### 7. Fractals (2 features)
- fractal_high - Williams fractal high
- fractal_low - Williams fractal low

#### 8. Order Blocks (2 features)
- bullish_order_block - institutional buy zone
- bearish_order_block - institutional sell zone

#### 9. Fair Value Gaps (4 features)
- bullish_fvg - bullish price imbalance
- bullish_fvg_size - size of imbalance
- bearish_fvg - bearish price imbalance
- bearish_fvg_size - size of imbalance

#### 10. Liquidity Zones (4 features)
- dist_to_buy_stops - distance to stop clusters
- dist_to_sell_stops - distance to stop clusters
- touched_buy_stops - stop hunt detection
- touched_sell_stops - stop hunt detection

### Total New Features: ~80+ features

---

## Files to Modify

### 1. `core/indicators.py` (if exists)
**Action:** DELETE or comment out ALL indicator functions

Functions to remove:
```python
def calculate_sma(...)
def calculate_ema(...)
def calculate_rsi(...)
def calculate_stochastic(...)
def calculate_macd(...)
def calculate_atr(...)
def calculate_bollinger_bands(...)
def calculate_obv(...)
def calculate_vwap(...)
```

**Alternative:** If this file is heavily used, keep it but add deprecation warnings.

### 2. `model.py`
**Action:** Remove all indicator calculations from feature preparation

Search for and remove:
- Lines calculating SMA/EMA
- Lines calculating RSI
- Lines calculating MACD
- Lines calculating ATR
- Lines calculating Bollinger Bands
- Lines calculating OBV/VWAP

Replace with:
```python
from core.price_action_features import prepare_price_action_data

# In feature preparation:
df = prepare_price_action_data(df)
```

### 3. Any data preparation files
**Action:** Replace indicator calculations with price action features

Look for:
- `prepare_features()` functions
- `calculate_indicators()` functions
- `add_technical_indicators()` functions

### 4. Training scripts
**Action:** Update input dimensions

Change:
```python
# OLD
input_size = 58  # OHLCV + 24 indicators × 2 timeframes

# NEW
input_size = 92  # OHLCV + ~80 price action features
```

---

## Migration Checklist

### Phase 1: Backup and Preparation ✅
- [x] Create feature/pure-price-action branch
- [x] Commit all current work
- [x] Create price_action_features.py
- [ ] Document baseline performance

### Phase 2: Code Removal
- [ ] Identify all files using indicators
- [ ] Comment out indicator calculations
- [ ] Remove indicator imports
- [ ] Test that code still runs (without predictions)

### Phase 3: Integration
- [ ] Import price_action_features module
- [ ] Replace indicator calls with price action calls
- [ ] Update feature column lists
- [ ] Test feature generation on sample data

### Phase 4: Model Updates
- [ ] Update input_size parameter
- [ ] Increase model capacity (hidden_size)
- [ ] Add dropout for regularization
- [ ] Test model initialization

### Phase 5: Testing
- [ ] Test data pipeline end-to-end
- [ ] Verify feature shapes match model input
- [ ] Check for NaN/inf values
- [ ] Validate feature scaling

### Phase 6: Retraining
- [ ] Retrain model from scratch
- [ ] Monitor training/validation loss
- [ ] Compare to baseline performance
- [ ] Document results

---

## Expected Benefits

### Performance Improvements
1. **Zero Lag** - No smoothed indicators = instant reaction to price
2. **Better Structure Recognition** - Explicit highs/lows/breakouts
3. **True Order Flow** - Actual buying/selling pressure
4. **Professional Approach** - Trades like institutional traders

### Risk Reduction
1. **No False Signals** - Indicators can diverge from price
2. **Clear Structure** - Obvious support/resistance levels
3. **Better Timing** - No lag means better entries/exits

---

## Rollback Plan

If performance degrades significantly:

1. **Immediate Rollback:**
```bash
git checkout main
cd rnn-server
uv run fastapi dev main.py
```

2. **Selective Rollback:**
- Keep price action features
- Add back ONLY specific indicators that were helpful
- Hybrid approach

3. **Gradual Migration:**
- Keep indicators AND add price action features
- Let model learn which to use
- Remove indicators gradually based on feature importance

---

## Success Criteria

Migration is successful if:

1. ✅ **Model trains without errors**
2. ✅ **Features generate correctly** (no NaN/inf)
3. ✅ **Accuracy ≥ baseline** (or within 5%)
4. ✅ **Sharpe ratio > baseline**
5. ✅ **Win rate ≥ 55%**
6. ✅ **Max drawdown ≤ baseline**

---

## Timeline

- **Day 1:** Removal and integration (TODAY)
- **Days 2-3:** Testing and debugging
- **Days 4-5:** Retraining and validation
- **Days 6-7:** Performance comparison and decision

---

## Next Steps

1. ✅ Created `core/price_action_features.py`
2. **Next:** Find and comment out all indicator calculations in `model.py`
3. **Then:** Integrate price action features
4. **Finally:** Test and retrain

This migration transforms the model from a **technical analyst** to a **price action trader**.
