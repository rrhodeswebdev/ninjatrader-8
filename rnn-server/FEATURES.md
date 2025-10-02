# Enhanced RNN Model Features

## Architecture
- **Model Type**: LSTM-based Recurrent Neural Network
- **Input**: 20-bar sequences with 24 features each
- **Output**: 3-class classification (Short=0, Hold=1, Long=2)
- **Hidden Layers**: 2 LSTM layers with 64 units each
- **Regularization**: Dropout (0.2 in LSTM, 0.3 in FC layers)

## Input Features (24 total)

### **Core Price Data (4 features)**
1. **Open**: Opening price
2. **High**: Highest price
3. **Low**: Lowest price  
4. **Close**: Closing price

### **Market Regime Detection (2 features)**
5. **Hurst Exponent (H)** - [Mottl/hurst library](https://github.com/Mottl/hurst)
   - H < 0.5: Mean-reverting (anti-persistent)
   - H = 0.5: Random walk (no memory)
   - H > 0.5: Trending (persistent)
6. **Hurst Constant (C)** - Additional time series structure info

### **Volatility (1 feature)**
7. **ATR (Average True Range)** - 14-period, measures volatility

### **Price Momentum (2 features)**
8. **Velocity** - Rate of change over 5 bars: `(close[i] - close[i-5]) / 5`
9. **Acceleration** - Change in velocity: `velocity[i] - velocity[i-1]`
   - Detects if price is speeding up or slowing down

### **Range Dynamics (2 features)**
10. **Range Ratio** - Current bar range / Previous bar range
    - `(high[i] - low[i]) / (high[i-1] - low[i-1])`
    - Shows range expansion/contraction
11. **Wick Ratio** - Upper wick / Lower wick
    - Upper wick: `high - max(open, close)`
    - Lower wick: `min(open, close) - low`
    - Indicates buying vs selling pressure

### **Gap Analysis (3 features)**
12. **Gap Up** - `max(0, low[i] - high[i-1])`
13. **Gap Down** - `max(0, low[i-1] - high[i])`
14. **Gap Filled** - Binary: 1 if previous close within current bar range

### **Price Fractals / Market Structure (4 features)**
15. **Swing High** - Binary: 1 if current high > neighbors
16. **Swing Low** - Binary: 1 if current low < neighbors
17. **Bars Since Swing High** - Distance from last swing high
18. **Bars Since Swing Low** - Distance from last swing low

### **Return Distribution (2 features)**
19. **Skewness** - Asymmetry of 20-bar return distribution
    - Positive: Right tail (big up moves)
    - Negative: Left tail (big down moves)
20. **Kurtosis** - Tail thickness (fat tails = abnormal moves)
    - High: Expect big moves
    - Low: Normal distribution

### **Position in Range (1 feature)**
21. **Position in Range** - Where price sits in 20-bar range
    - `(close - rolling_min) / (rolling_max - rolling_min)`
    - 0 = at low, 1 = at high, 0.5 = middle

### **Trend Structure (3 features)**
22. **Higher Highs** - Count of higher highs in last 5 bars
23. **Lower Lows** - Count of lower lows in last 5 bars
24. **Trend Strength** - `higher_highs - lower_lows` (-5 to +5)
    - Positive: Uptrend structure
    - Negative: Downtrend structure

## Feature Engineering Pipeline

```python
1. Raw OHLC bars → DataFrame
2. For each bar:
   - Calculate Hurst (100-bar window)
   - Calculate ATR (14-period)
   - Calculate Velocity & Acceleration
   - Calculate Range & Wick Ratios
   - Detect Gaps (up/down/filled)
   - Identify Swing Points
   - Calculate Return Stats (skew/kurtosis)
   - Compute Position in Range
   - Count Higher Highs/Lower Lows
3. Combine all 24 features
4. StandardScaler normalization
5. Create 20-bar sequences
6. Feed to LSTM
```

## What The Model Learns

The RNN learns complex patterns across all features:

### **Example Pattern 1: Strong Momentum Breakout**
```
Velocity: High (+)
Acceleration: Increasing
Hurst: > 0.5 (trending)
Position in Range: > 0.8 (near high)
Trend Strength: +4 (strong uptrend)
Gap Up: Present
→ Model Predicts: LONG (85% confidence)
```

### **Example Pattern 2: Mean Reversion Setup**
```
Velocity: High (+)
Acceleration: Decreasing (slowing down)
Hurst: < 0.5 (mean-reverting)
Position in Range: > 0.9 (extended)
Wick Ratio: High upper wick (rejection)
Skewness: Positive (overbought)
→ Model Predicts: SHORT (75% confidence)
```

### **Example Pattern 3: Consolidation**
```
Velocity: Near zero
ATR: Decreasing
Range Ratio: < 1 (contracting)
Bars Since Swing: High
Kurtosis: Low (normal distribution)
→ Model Predicts: HOLD (90% confidence)
```

## Performance Optimizations

✅ **torch.compile**: 2x faster inference (Python 3.11)
✅ **Mini-batch training**: Batch size = 32
✅ **Class weighting**: Handles imbalanced data
✅ **Early stopping**: Prevents overfitting (patience=10)
✅ **Learning rate scheduling**: ReduceLROnPlateau
✅ **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
✅ **Adaptive thresholds**: Auto-adjusts to instrument volatility

## Prediction Output

```python
{
  "signal": "long",      # "short", "hold", or "long"
  "confidence": 0.82     # Probability (0-1) for predicted class
}
```

## Advantages Over Traditional Indicators

| Traditional Approach | This Model |
|---------------------|------------|
| Moving averages (lagging) | Velocity/Acceleration (leading) |
| Fixed indicator periods | Adaptive to market regime (Hurst) |
| Single dimension | 24 dimensions of price action |
| Manual rules | Learned patterns from data |
| Static thresholds | Adaptive volatility-based thresholds |
| Miss structure | Captures swings, gaps, fractals |

The model sees price movement as a multi-dimensional pattern, not just a line on a chart!
