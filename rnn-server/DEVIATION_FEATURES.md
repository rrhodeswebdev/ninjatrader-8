# Price Deviation Features - Detailed Documentation

## Overview
Added 23 deviation-based features across 4 different time windows (5, 10, 20, 50 bars) to capture price anomalies, overextension, and volatility changes.

## Feature Categories (23 total)

### **1. Mean Deviation (4 features)**
Measures how far price deviates from the average

```python
mean_dev = (close - mean(window)) / mean(window)
```

| Feature | Window | What It Detects |
|---------|--------|-----------------|
| `mean_dev_5` | 5 bars | Short-term deviation (scalping) |
| `mean_dev_10` | 10 bars | Quick mean reversion signals |
| `mean_dev_20` | 20 bars | Standard deviation from trend |
| `mean_dev_50` | 50 bars | Long-term overextension |

**Trading Signal:**
- Positive (>0): Price above average (potentially overbought)
- Negative (<0): Price below average (potentially oversold)
- Near zero: Price at equilibrium

---

### **2. Median Deviation (4 features)**
More robust to outliers than mean

```python
median_dev = (close - median(window)) / median(window)
```

| Feature | Window | Advantage Over Mean |
|---------|--------|---------------------|
| `median_dev_5` | 5 bars | Ignores single bar spikes |
| `median_dev_10` | 10 bars | Better for choppy markets |
| `median_dev_20` | 20 bars | Filters false breakouts |
| `median_dev_50` | 50 bars | True trend deviation |

**Why Median?**
- Mean can be skewed by 1-2 large bars
- Median shows where "most" prices were
- Better for detecting genuine moves

---

### **3. Standard Deviation / Volatility (4 features)**
Measures price dispersion (volatility)

```python
std_dev = std(close prices in window)
```

| Feature | Window | Use Case |
|---------|--------|----------|
| `std_dev_5` | 5 bars | Immediate volatility |
| `std_dev_10` | 10 bars | Short-term vol changes |
| `std_dev_20` | 20 bars | ATR-like volatility |
| `std_dev_50` | 50 bars | Market regime volatility |

**Trading Signal:**
- Increasing: Volatility expanding → Breakout/trend
- Decreasing: Volatility contracting → Consolidation → Coiling for move
- High: Wide stops needed
- Low: Tight stops possible

---

### **4. Z-Score (4 features)**
Standardized deviation (how many std devs from mean)

```python
z_score = (close - mean) / std
```

| Feature | Window | Interpretation |
|---------|--------|----------------|
| `z_score_5` | 5 bars | Extreme short-term moves |
| `z_score_10` | 10 bars | Quick reversions |
| `z_score_20` | 20 bars | Standard overbought/oversold |
| `z_score_50` | 50 bars | Major extremes |

**Trading Thresholds:**
- `|z| > 2`: Significant deviation (95% confidence)
- `|z| > 3`: Extreme deviation (99.7% confidence) → High probability reversal
- `|z| < 1`: Normal price action

**Example:**
```
z_score_20 = +2.5
→ Price is 2.5 std devs ABOVE 20-bar mean
→ Statistically rare (only 1.2% probability)
→ Likely to revert → SHORT signal
```

---

### **5. Bollinger Band Width (4 features)**
Measures range of volatility envelope

```python
bb_width = (4 * std) / mean  # ±2 std devs = 95% of prices
```

| Feature | Window | What It Shows |
|---------|--------|---------------|
| `bb_width_5` | 5 bars | Immediate squeeze/expansion |
| `bb_width_10` | 10 bars | Volatility breakout signals |
| `bb_width_20` | 20 bars | Classic BB squeeze |
| `bb_width_50` | 50 bars | Major regime changes |

**Trading Patterns:**
- **Squeeze** (bb_width decreasing): Consolidation → Breakout coming
- **Expansion** (bb_width increasing): Trend in motion
- **Wide then narrow**: Trend exhaustion

**Bollinger Squeeze Example:**
```
bb_width_20 = 0.02 (narrowest in 50 bars)
→ Volatility extremely compressed
→ Major breakout imminent
→ Wait for direction, then follow
```

---

### **6. Volatility Acceleration (1 feature)**
Rate of change of volatility

```python
vol_acceleration = std_dev_20[i] - std_dev_20[i-1]
```

**Trading Signal:**
- Positive: Volatility expanding (trend accelerating)
- Negative: Volatility contracting (trend dying)
- Zero: Stable volatility

**Divergence Trading:**
```
Price: Making new highs
Vol Acceleration: Negative (volatility decreasing)
→ Bearish divergence → Reversal signal
```

---

### **7. High/Low Deviation (2 features)**
Distance from recent extremes

```python
high_deviation = (close - recent_high) / recent_high
low_deviation = (close - recent_low) / recent_low
```

| Feature | What It Measures |
|---------|------------------|
| `high_deviation` | How far below recent high (always ≤0) |
| `low_deviation` | How far above recent low (always ≥0) |

**Trading Applications:**

**Breakout Detection:**
```
high_deviation = 0 (at recent high)
→ Potential breakout
→ Wait for confirmation

high_deviation = -0.001 (new high!)
→ Confirmed breakout → LONG
```

**Support/Resistance:**
```
low_deviation = 0.05 (5% above recent low)
→ Price bounced strongly from support
→ LONG signal
```

---

## Multi-Timeframe Deviation Analysis

The model can detect complex patterns across timeframes:

### **Pattern 1: Mean Reversion Setup**
```
mean_dev_5:  +0.03 (3% above 5-bar mean)
mean_dev_20: +0.01 (1% above 20-bar mean)
z_score_20:  +2.3  (extreme short term)
bb_width_20: 0.015 (normal)
→ SHORT signal (overextended short-term)
```

### **Pattern 2: Volatility Breakout**
```
bb_width_5:  0.005 (squeeze)
bb_width_10: 0.008 (squeeze)
bb_width_20: 0.010 (squeeze)
vol_accel:   +0.002 (starting to expand)
high_dev:    -0.001 (at high)
→ Breakout LONG setup
```

### **Pattern 3: Trend Exhaustion**
```
mean_dev_20:  +0.05 (extended)
z_score_20:   +2.8  (extreme)
vol_accel:    -0.003 (vol decreasing)
std_dev_20:   Declining
→ Trend losing steam → Exit/Reverse
```

### **Pattern 4: Volatility Compression**
```
bb_width_5:   0.003 (tight)
bb_width_10:  0.004 (tight)
bb_width_20:  0.006 (tight)
bb_width_50:  0.015 (normal)
vol_accel:    -0.001 (still contracting)
→ Major move coming → Wait for direction
```

---

## Why These Features Work

### **1. Multiple Timeframes**
- 5-bar: Scalping/noise
- 10-bar: Swing trading
- 20-bar: Position trading
- 50-bar: Trend trading

### **2. Redundancy Reduces False Signals**
- If only z_score_5 is extreme → Could be noise
- If z_score_5, z_score_10, z_score_20 ALL extreme → Real signal

### **3. Volatility Tells The Story**
- High vol + High deviation → Strong trend
- Low vol + High deviation → Exhaustion
- Low vol + Low deviation → Compression

### **4. No Indicators Needed**
- All features derived purely from price
- No lagging moving averages
- No arbitrary parameters (except windows)

---

## Feature Importance

**Most Predictive (Expected):**
1. `z_score_20` - Standard statistical deviation
2. `bb_width_20` - Volatility regime
3. `vol_acceleration` - Momentum of volatility
4. `mean_dev_20` - Core overextension
5. `high_deviation` / `low_deviation` - Breakout confirmation

**Supporting Features:**
- Multiple windows provide confirmation
- Median features filter noise
- Std dev features show regime

---

## Total Feature Count: 47

| Category | Count |
|----------|-------|
| OHLC | 4 |
| Hurst | 2 |
| ATR | 1 |
| Price Features | 18 |
| **Deviation Features** | **23** |
| **TOTAL** | **47** |

**Deviation Breakdown:**
- Mean Deviation: 4
- Median Deviation: 4
- Std Deviation: 4
- Z-Score: 4
- BB Width: 4
- Vol Acceleration: 1
- High/Low Deviation: 2

The RNN now has a complete statistical view of price behavior across all timeframes! 📊
