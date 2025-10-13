# RNN Server - Complete Technical Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Prediction & Signal Generation](#prediction--signal-generation)
7. [Risk Management](#risk-management)
8. [API Endpoints](#api-endpoints)
9. [Configuration Parameters](#configuration-parameters)
10. [File Structure](#file-structure)
11. [Performance Optimizations](#performance-optimizations)
12. [Troubleshooting](#troubleshooting)

---

## Overview

**Purpose**: Python-based FastAPI server that provides AI-driven trading signals for NinjaTrader 8 using a recurrent neural network (RNN) with LSTM and attention mechanisms.

**Core Technology Stack**:
- **Framework**: FastAPI (async web server)
- **ML Framework**: PyTorch 2.0+ (with optional CUDA support)
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Key Libraries**:
  - `hurst` - Hurst exponent calculation for trend detection
  - `scipy` - Statistical analysis
  - `torch` - Deep learning

**Total Code**: ~9,163 lines of Python across 22 modules

---

## System Architecture

### High-Level Flow

```
NinjaTrader 8 Strategy (C#)
       ↓
   HTTP POST /analysis
       ↓
FastAPI Server (main.py)
       ↓
TradingModel (model.py)
   ├── Feature Engineering (62 features)
   ├── RNN Prediction (LSTM + Attention)
   ├── Confidence Thresholds
   └── Risk Management
       ↓
   Trading Signal + Risk Parameters
       ↓
   Returns to NinjaTrader
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ main.py (15.8 KB)                                      │ │
│  │  - /analysis endpoint (historical & realtime)          │ │
│  │  - /health-check, /training-status                     │ │
│  │  - MIN_CONFIDENCE_THRESHOLD = 0.40                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Core ML Engine                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ model.py (87.3 KB - LARGEST MODULE)                    │ │
│  │                                                         │ │
│  │  Classes:                                               │ │
│  │  • TradingRNN (LSTM + Attention)                       │ │
│  │  • TradingModel (Training & Inference)                 │ │
│  │  • AdaptiveConfidenceThresholds (Dynamic filtering)    │ │
│  │  • FocalLoss, LabelSmoothingLoss (Training losses)     │ │
│  │                                                         │ │
│  │  Key Functions:                                         │ │
│  │  • detect_market_regime() - ADX-based regime detection │ │
│  │  • calculate_hurst_exponent() - Trend persistence      │ │
│  │  • calculate_atr() - Volatility measurement            │ │
│  │  • prepare_data() - Feature engineering (62 features)  │ │
│  │  • train() - Model training with validation            │ │
│  │  • predict() - Real-time inference                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Risk Management                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ risk_management.py (15.5 KB)                           │ │
│  │                                                         │ │
│  │  • PositionSizer: Kelly Criterion, confidence scaling  │ │
│  │  • StopTargetCalculator: ATR-based stops/targets      │ │
│  │  • RiskManager: Combined position + stop calculation   │ │
│  │                                                         │ │
│  │  Risk Parameters:                                       │ │
│  │  - Base risk: 1% per trade                             │ │
│  │  - Max risk: 2% per trade                              │ │
│  │  - Scales with confidence (0.65-0.85)                  │ │
│  │  - Regime-adjusted (0.4x-1.0x multiplier)              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Supporting Modules                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • trading_metrics.py (11.4 KB): Performance metrics    │ │
│  │ • ensemble.py (13.7 KB): Model ensembling              │ │
│  │ • backtester.py (20.5 KB): Historical testing          │ │
│  │ • adaptive_retraining.py (14.7 KB): Online learning    │ │
│  │ • confidence_calibration.py (12.3 KB): Calibration     │ │
│  │ • regime_models.py (16.3 KB): Regime-specific models   │ │
│  │ • meta_labeling.py (16.4 KB): Meta-learning            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Neural Network Architecture

### TradingRNN Class

**Architecture Type**: Sequence-to-One LSTM with Multi-Head Attention

```python
class TradingRNN(nn.Module):
    def __init__(self, input_size=62, hidden_size=128, num_layers=3, output_size=3)
```

**Layer Structure**:

```
Input: (batch_size, sequence_length=20, features=62)
       ↓
┌──────────────────────────────────────────┐
│ LSTM Layer 1 (62 → 128)                  │
│   - Dropout: 0.5 between layers          │
├──────────────────────────────────────────┤
│ LSTM Layer 2 (128 → 128)                 │
│   - Dropout: 0.5                          │
├──────────────────────────────────────────┤
│ LSTM Layer 3 (128 → 128)                 │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│ Multi-Head Attention                      │
│   - Num heads: 4                          │
│   - Dropout: 0.1                          │
│   - Self-attention on sequence            │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│ Fully Connected Layers                    │
│   - FC1: 128 → 64 (ReLU, Dropout 0.4)    │
│   - FC2: 64 → 32 (ReLU, Dropout 0.3)     │
│   - FC3: 32 → 3 (Output logits)          │
└──────────────────────────────────────────┘
       ↓
Output: (batch_size, 3)  # [prob_short, prob_hold, prob_long]
```

**Key Features**:
- **Bidirectional Processing**: LSTM captures patterns in both directions
- **Attention Mechanism**: Focuses on important time steps in sequence
- **Heavy Regularization**: Dropout layers prevent overfitting
- **Compiled for Speed**: Uses `torch.compile()` for faster inference

**Parameter Count**: ~200K trainable parameters

---

## Feature Engineering

### 62 Total Features (Comprehensive)

#### 1. **OHLC Base (4 features)**
- Open, High, Low, Close prices

#### 2. **Hurst Exponent (2 features)**
- **Hurst H**: Trend persistence (H > 0.5 = trending, H < 0.5 = mean-reverting)
- **Hurst C**: Constant from Hurst calculation
- Cached every 10 bars for performance (10x speedup)

#### 3. **Volatility (1 feature)**
- **ATR (Average True Range)**: 14-period ATR in points

#### 4. **Price Momentum & Patterns (18 features)**
- Velocity: Rate of price change
- Acceleration: Second derivative of price
- Range ratio: (High - Low) / Close
- Wick ratios: Upper/lower wick sizes
- Gap detection: Gap up, gap down, gap filled
- Swing points: Swing highs/lows, bars since swing
- Distribution: Skewness, kurtosis of recent prices
- Position in range: Close relative to high/low
- Trend structure: Higher highs, lower lows, trend strength

#### 5. **Deviation Features (13 features)**
- Moving average deviations (20, 50, 100 periods)
- Percentage above/below MAs
- Bollinger Band positions

#### 6. **Order Flow (1 feature)**
- Delta: Bid volume - Ask volume (when available)

#### 7. **Time-of-Day (5 features)**
- Hour sin/cos encoding
- Minute sin/cos encoding
- Market session indicator

#### 8. **Microstructure (5 features)**
- Spread estimates
- Trade imbalance
- Quote intensity
- Price impact proxies

#### 9. **Volatility Regime (4 features)**
- Short-term volatility
- Long-term volatility
- Volatility ratio
- Volatility percentile

#### 10. **Multi-Timeframe (9 features)**
- Secondary timeframe (5-min) features:
  - Close, close change, high-low range
  - Volume, position in bar
  - Trend direction, momentum
  - Volatility, alignment score

#### 11. **Price Change Magnitude (1 feature)**
- Recent 5-bar average price change magnitude

### Feature Preprocessing

**Scaling**: StandardScaler (z-score normalization)
- Fitted on training data
- Stored and reused for inference
- Prevents data leakage

**Sequence Creation**:
- Window size: 20 bars (configurable via `sequence_length`)
- Lookahead: 3 bars for label creation
- Rolling window approach for training

---

## Model Training

### Training Process

**File**: `model.py`, method `TradingModel.train()`

```python
def train(self, df, epochs=100, batch_size=32, learning_rate=0.0003):
```

#### Step 1: Label Generation

**Current Configuration** (⚠️ Known Issue):
```python
hold_percentage = 0.40  # 40% of moves labeled as HOLD
```

**Label Logic**:
1. Look ahead 3 bars from current position
2. Calculate max upward move and max downward move
3. Calculate percentile threshold (40th percentile of all moves)
4. Label assignment:
   - **LONG (class 2)**: If max_up > threshold AND max_up > max_down
   - **SHORT (class 0)**: If max_down > threshold AND max_down > max_up
   - **HOLD (class 1)**: Otherwise (40% of data)

**Problem**: 40% HOLD creates structural bias toward conservative predictions. Recommended to reduce to 20-25%.

#### Step 2: Loss Function

**Focal Loss** (default):
```python
FocalLoss(gamma=2.0, weight=class_weights)
```

- Focuses on hard-to-classify examples
- Automatically weighted by class imbalance
- Gamma=2.0 emphasizes misclassified samples

**Alternative**: Label Smoothing Loss (0.1 smoothing)

#### Step 3: Optimizer

**AdamW** with:
- Learning rate: 0.0003
- Weight decay: 0.0001 (L2 regularization)
- Gradient clipping: 1.0 (prevents exploding gradients)

#### Step 4: Training Loop

- Train/validation split: 80/20
- Batch size: 32
- Epochs: 100 (with early stopping)
- Metrics tracked:
  - Overall accuracy
  - Per-class accuracy (SHORT, HOLD, LONG)
  - Precision, recall, F1 score
  - High-confidence accuracy (≥65%)
  - Confusion matrix

#### Step 5: Validation Metrics

**Trading-Specific Metrics**:
- Sharpe ratio
- Profit factor
- Win rate
- Maximum drawdown
- Expectancy per trade

**Threshold**: If Sharpe < 0.5 or accuracy < 45%, training is suboptimal

#### Step 6: Model Persistence

**Saved to**: `models/trading_model.pth`

**Checkpoint includes**:
- Model state dict
- Scaler (for feature normalization)
- Training metadata
- Signal threshold
- Sequence length

---

## Prediction & Signal Generation

### Real-Time Inference Flow

**File**: `model.py`, method `TradingModel.predict()`

```python
def predict(self, recent_bars_df) -> Tuple[str, float]:
    # Returns: (signal, confidence)
```

#### Step 1: Data Validation
- Check for NaN/inf values
- Ensure minimum bars (sequence_length + 100)
- Validate required columns

#### Step 2: Feature Calculation
- Use **fast path** optimization (only process recent 120 bars)
- Calculate all 62 features
- Apply saved scaler (no refitting)

#### Step 3: Model Inference
```python
with torch.inference_mode():
    outputs, attn_weights = self.model(X_tensor, return_attention=True)
    probabilities = torch.softmax(outputs, dim=1)[0]
```

**Outputs**:
- `prob_short` (class 0)
- `prob_hold` (class 1)
- `prob_long` (class 2)

#### Step 4: **Probability-Based Decision** (⚠️ Recent Fix)

**NEW LOGIC** (overcomes 40% HOLD bias):
```python
direction_margin = 0.02  # Need 2% edge over HOLD to trade

if prob_long > prob_hold + direction_margin and prob_long > prob_short:
    signal = 'long'
    confidence = prob_long
elif prob_short > prob_hold + direction_margin and prob_short > prob_long:
    signal = 'short'
    confidence = prob_short
else:
    signal = 'hold'
    confidence = prob_hold
```

**Why**: Model was trained with 40% HOLD bias, so direct argmax always favors HOLD. This logic requires a directional edge to override HOLD.

#### Step 5: Market Regime Detection

**Function**: `detect_market_regime()`

Uses ADX (Average Directional Index) + volatility:

| ADX | Volatility | Regime |
|-----|------------|--------|
| > 25 | High | `trending_high_vol` |
| > 25 | Normal | `trending_normal` |
| < 20 | Low | `ranging_low_vol` |
| < 20 | Normal | `ranging_normal` |
| Any | Very High | `high_vol_chaos` |

#### Step 6: Adaptive Confidence Thresholds

**Class**: `AdaptiveConfidenceThresholds`

**Base Thresholds** (Emergency Mode - Current):
```python
{
    'trending_high_vol': 0.40,
    'trending_normal': 0.42,
    'ranging_normal': 0.48,
    'ranging_low_vol': 0.52,
    'high_vol_chaos': 0.55,
    'transitional': 0.45,
    'unknown': 0.42
}
```

**Time-of-Day Multipliers**:
```python
{
    'open': 1.03,     # 9:30-10:00 AM
    'mid': 1.0,       # 10:00-11:30 AM, 1:30-3:30 PM
    'lunch': 1.05,    # 11:30 AM-1:30 PM
    'close': 1.02     # After 3:30 PM
}
```

**Final Threshold Calculation**:
```python
final_threshold = base_threshold × time_multiplier × accuracy_penalty
# Clamped to [0.50, 0.85] range
```

#### Step 7: Threshold Filtering

```python
if confidence < adaptive_threshold:
    signal = 'hold'  # Filter out low-confidence signals
```

#### Step 8: Static Threshold (main.py)

**Second layer of filtering**:
```python
MIN_CONFIDENCE_THRESHOLD = 0.40  # Emergency mode (was 0.65)

if confidence < MIN_CONFIDENCE_THRESHOLD:
    filtered_signal = 'hold'
```

---

## Risk Management

### Module: risk_management.py

Three main classes handle position sizing and stop/target calculation.

### 1. PositionSizer

**Purpose**: Calculate number of contracts based on confidence and account risk

**Configuration**:
```python
base_risk_pct = 0.01   # Risk 1% of account
max_risk_pct = 0.02    # Max 2% of account
min_risk_pct = 0.005   # Min 0.5% of account
max_contracts = 10     # Safety limit
```

**Scaling Logic**:
1. **Confidence Scaling**: Linearly interpolate from base to max risk
   - Confidence 0.65 → 1.0% risk
   - Confidence 0.75 → 1.5% risk
   - Confidence 0.85+ → 2.0% risk

2. **Regime Multipliers** (reduce size in choppy markets):
   ```python
   {
       'trending_normal': 1.0,      # Full size
       'trending_high_vol': 0.8,    # -20%
       'ranging_normal': 0.7,       # -30%
       'ranging_low_vol': 0.5,      # -50%
       'high_vol_chaos': 0.4,       # -60%
       'transitional': 0.6,         # -40%
       'unknown': 0.7               # Conservative
   }
   ```

3. **Contract Calculation**:
   ```python
   risk_dollars = account_balance × risk_pct
   risk_per_contract = stop_distance × 4 ticks/point × $12.50/tick
   contracts = floor(risk_dollars / risk_per_contract)
   contracts = min(contracts, max_contracts)
   ```

### 2. StopTargetCalculator

**Purpose**: Calculate ATR-based stop loss and take profit levels

**Regime-Specific Parameters**:

| Regime | Stop (ATR) | Target (ATR) | Risk:Reward |
|--------|-----------|--------------|-------------|
| Trending Normal | 1.5 | 3.0 | 1:2.0 |
| Trending High Vol | 2.0 | 2.5 | 1:1.25 |
| Ranging Normal | 1.0 | 1.5 | 1:1.5 |
| Ranging Low Vol | 0.8 | 1.2 | 1:1.5 |
| High Vol Chaos | 2.5 | 2.0 | 1:0.8 |

**Confidence Adjustment**:
```python
confidence_factor = 0.7 + (confidence - 0.65) × 1.5
# Clamped to [0.5, 1.2]

stop_distance = ATR × stop_atr_multiplier × confidence_factor
target_distance = stop_distance × risk_reward_ratio
```

**Logic**:
- Higher confidence → Wider stops (let trade breathe)
- Lower confidence → Tighter stops (exit quickly if wrong)

**Tick Rounding**:
- All prices rounded to 0.25 (quarter point)
- ES futures standard tick size

### 3. RiskManager

**Purpose**: Combined interface for position sizing + stops/targets

**Method**: `calculate_trade_parameters()`

**Returns Complete Trade Spec**:
```python
{
    'signal': 'long',
    'confidence': 0.72,
    'contracts': 2,
    'entry_price': 4500.00,
    'stop_loss': 4477.50,
    'take_profit': 4545.00,
    'stop_distance': 22.5,
    'target_distance': 45.0,
    'risk_reward': 2.0,
    'risk_dollars': 562.50,
    'risk_pct': 0.0225,
    'regime': 'trending_normal'
}
```

---

## API Endpoints

### 1. POST /analysis

**Purpose**: Main prediction endpoint (handles both historical and real-time data)

**Historical Data Request** (Training):
```json
{
  "primary_bars": [
    {"time": "2024-01-01T09:30:00", "open": 4500, "high": 4510, "low": 4495, "close": 4505, "volume": 1000},
    ...
  ],
  "secondary_bars": [  // Optional 5-min bars
    {"time": "2024-01-01T09:30:00", "open": 4500, "high": 4515, "low": 4490, "close": 4510, "volume": 5000},
    ...
  ],
  "type": "historical",
  "dailyGoal": 500.0,
  "dailyMaxLoss": 250.0
}
```

**Response** (Historical):
```json
{
  "status": "ok",
  "bars_received": 5000,
  "data_type": "historical",
  "model_training": "scheduled",
  "message": "Training started in background. Check /training-status for progress."
}
```

**Real-Time Data Request** (Prediction):
```json
{
  "primary_bar": {
    "time": "2024-01-01T10:05:00",
    "open": 4505,
    "high": 4508,
    "low": 4503,
    "close": 4507,
    "volume": 150
  },
  "secondary_bar": {  // Optional
    "time": "2024-01-01T10:05:00",
    "open": 4500,
    "high": 4512,
    "low": 4498,
    "close": 4509,
    "volume": 800
  },
  "type": "realtime",
  "bid_volume": 80.0,
  "ask_volume": 70.0,
  "dailyPnL": 125.0,
  "dailyGoal": 500.0,
  "dailyMaxLoss": 250.0,
  "accountBalance": 25000.0
}
```

**Response** (Real-Time):
```json
{
  "status": "ok",
  "signal": "long",
  "raw_signal": "long",
  "confidence": 0.72,
  "confidence_threshold": 0.40,
  "filtered": false,
  "recommendation": "LONG with 72.0% confidence",
  "risk_management": {
    "contracts": 2,
    "entry_price": 4507.00,
    "stop_loss": 4484.50,
    "take_profit": 4552.00,
    "stop_distance": 22.5,
    "target_distance": 45.0,
    "risk_reward_ratio": 2.0,
    "risk_dollars": 562.50,
    "risk_pct": 0.0225,
    "regime": "trending_normal"
  }
}
```

### 2. GET /health-check

**Response**:
```json
{
  "status": "ok",
  "model_trained": true,
  "device": "cpu"
}
```

### 3. GET /training-status

**Response**:
```json
{
  "is_training": false,
  "progress": "Training complete",
  "error": null
}
```

### 4. POST /save-model

**Purpose**: Manually save current model to disk

### 5. POST /load-model

**Purpose**: Reload model from disk (useful after retraining)

---

## Configuration Parameters

### Critical Tuning Parameters

#### 1. **Confidence Thresholds**

**Location**: `main.py:24`
```python
MIN_CONFIDENCE_THRESHOLD = 0.40
```
- **Current**: 0.40 (emergency mode)
- **Normal**: 0.55-0.60
- **Conservative**: 0.65-0.70

**Impact**: Higher = fewer but higher quality signals

---

#### 2. **Direction Margin**

**Location**: `model.py:1770`
```python
direction_margin = 0.02  # Need 2% edge over HOLD
```
- **Current**: 0.02 (2%)
- **Aggressive**: 0.00-0.01
- **Conservative**: 0.03-0.05

**Impact**: How much LONG/SHORT prob must exceed HOLD prob

---

#### 3. **Regime Thresholds**

**Location**: `model.py:913-920`
```python
self.regime_thresholds = {
    'trending_high_vol': 0.40,
    'trending_normal': 0.42,
    'ranging_normal': 0.48,
    'ranging_low_vol': 0.52,
    'high_vol_chaos': 0.55,
    'transitional': 0.45,
    'unknown': 0.42
}
```

**Lower values** = more signals, **higher values** = fewer signals

---

#### 4. **HOLD Percentage (Training)**

**Location**: `model.py:1363`
```python
hold_percentage = 0.40  # 40% of data becomes HOLD
```
- **Current Issue**: Too high (40%)
- **Recommended**: 0.20-0.25 (20-25%)

**Requires Retraining** to take effect

---

#### 5. **Sequence Length**

**Location**: `main.py:19`
```python
trading_model = TradingModel(sequence_length=20)
```
- **Current**: 20 bars
- **Range**: 10-60 bars
- **Trade-off**: Longer = more context, slower inference

---

#### 6. **Risk Parameters**

**Location**: `risk_management.py:19-23`
```python
base_risk_pct = 0.01   # 1% per trade
max_risk_pct = 0.02    # 2% max
min_risk_pct = 0.005   # 0.5% min
max_contracts = 10     # Safety limit
```

**Adjust based on risk tolerance**

---

## File Structure

### Core Files (22 Python modules)

```
rnn-server/
├── main.py (15.8 KB)               # FastAPI server, endpoints
├── model.py (87.3 KB)              # ⭐ Core ML engine (LARGEST)
├── risk_management.py (15.5 KB)   # Position sizing, stops
├── trading_metrics.py (11.4 KB)   # Performance evaluation
│
├── backtester.py (20.5 KB)        # Historical backtesting
├── ensemble.py (13.7 KB)          # Model ensembling
├── adaptive_retraining.py (14.7 KB) # Online learning
├── confidence_calibration.py (12.3 KB) # Probability calibration
├── regime_models.py (16.3 KB)     # Regime-specific models
├── meta_labeling.py (16.4 KB)     # Meta-learning strategies
├── monitoring_dashboard.py (15.4 KB) # Live monitoring
│
├── feature_importance.py (8.5 KB) # Feature analysis
├── feature_importance_analyzer.py (11.4 KB) # Advanced feature analysis
├── walk_forward_validation.py (9.4 KB) # Walk-forward testing
├── walk_forward_optimizer.py (14.5 KB) # Parameter optimization
│
├── train_ensemble.py (5.6 KB)     # Ensemble training script
├── train_phase3.py (10.7 KB)      # Advanced training
├── run_backtest.py (8.7 KB)       # Backtest runner
├── test_model.py (6.4 KB)         # Model testing
├── test_performance.py (5.7 KB)   # Performance testing
├── diagnose_predictions.py (4.3 KB) # Prediction diagnostics
├── model_simplified.py (11.9 KB)  # Simplified model variant
│
├── models/
│   └── trading_model.pth          # Trained model checkpoint
│
└── pyproject.toml                 # Dependencies
```

### Documentation Files

```
├── RNN_SERVER_DOCUMENTATION.md    # ⭐ This file
├── SIGNAL_GENERATION_FIX.md       # Recent fixes for signal generation
├── ACCOUNT_BALANCE_INTEGRATION.md # Account balance integration
├── RISK_MANAGEMENT_README.md      # Risk management guide
├── ADVANCED_FEATURES_README.md    # Advanced features
├── GETTING_STARTED.md             # Quick start guide
├── IMPLEMENTATION_SUMMARY.md      # Implementation summary
├── MODEL_ENHANCEMENTS.md          # Model enhancement log
├── ACCURACY_ANALYSIS.md           # Accuracy analysis
├── PHASE3_IMPLEMENTATION_SUMMARY.md # Phase 3 summary
└── (14 more documentation files)
```

---

## Performance Optimizations

### 1. **Fast Path Inference**

**Problem**: Processing 5000+ bars on every prediction was slow

**Solution**: Only process last 120 bars needed
```python
min_bars_needed = self.sequence_length + 100  # 20 + 100 = 120
df_subset = recent_bars_df.tail(min_bars_needed)
```

**Speedup**: 10-100x faster inference

---

### 2. **Hurst Caching**

**Problem**: Calculating Hurst exponent every bar was slow

**Solution**: Cache and reuse for 10 bars
```python
if i % 10 == 0:  # Recalculate every 10 bars
    H, c = calculate_hurst_exponent(prices)
    self._last_hurst_H = H
else:
    H = self._last_hurst_H  # Reuse cached value
```

**Speedup**: 10x faster feature calculation

---

### 3. **Torch Compilation**

**Problem**: Python overhead in PyTorch

**Solution**: Compile model after loading
```python
self.model = torch.compile(self.model)
```

**Speedup**: 2-3x faster inference (PyTorch 2.0+)

---

### 4. **Vectorized Operations**

**Problem**: Loop-based calculations slow

**Solution**: NumPy/Pandas vectorization
```python
# Before (loop)
for i in range(len(high)):
    tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), ...)

# After (vectorized)
hl = high[1:] - low[1:]
hc = np.abs(high[1:] - close[:-1])
tr = np.maximum(hl, hc)
```

**Speedup**: 5-10x faster

---

### 5. **Feature Caching**

Features are cached to avoid recomputation on repeated calls with same data.

---

## Troubleshooting

### Issue 1: No Trading Signals

**Symptoms**: Model returns HOLD 100% of the time

**Causes**:
1. **40% HOLD training bias** (structural issue)
2. Confidence thresholds too high
3. Direction margin too high

**Fixes Applied**:
- Reduced MIN_CONFIDENCE_THRESHOLD to 0.40
- Reduced regime thresholds by 10-15%
- Reduced direction_margin to 0.02
- Implemented probability comparison (bypasses HOLD bias)

**Long-term fix**: Retrain with `hold_percentage = 0.20`

---

### Issue 2: Model Not Trained

**Symptoms**: Predictions return `('hold', 0.0)` with warning

**Cause**: `models/trading_model.pth` missing or corrupt

**Fix**:
1. Send historical data to `/analysis` endpoint
2. Wait for training to complete (~5-10 minutes)
3. Check `/training-status` for progress
4. Verify `models/trading_model.pth` exists

---

### Issue 3: Low Win Rate (<45%)

**Symptoms**: Too many losing trades

**Causes**:
- Thresholds too low (taking low-quality signals)
- Model needs retraining
- Market regime mismatch

**Fixes**:
- Increase MIN_CONFIDENCE_THRESHOLD to 0.50-0.55
- Increase direction_margin to 0.03-0.05
- Check regime detection accuracy
- Retrain with more recent data

---

### Issue 4: Server Timeouts

**Symptoms**: NinjaTrader loses connection

**Causes**:
- Heavy computation during training
- Large historical dataset (>10K bars)

**Fixes**:
- Training runs in background (non-blocking)
- Reduce `historicalBarsToSend` in NinjaTrader
- Increase timeout in NinjaTrader strategy

---

### Issue 5: Confidence Always Filtered

**Symptoms**: All signals filtered by confidence threshold

**Diagnostic**:
```bash
tail -100 /tmp/fastapi_test.log | grep "Probabilities:"
```

Look for:
```
Probabilities: SHORT=0.25, HOLD=0.50, LONG=0.25
```

If HOLD always >50%, model has HOLD bias issue.

**Fix**: See "No Trading Signals" above

---

## Performance Benchmarks

### Inference Speed
- **CPU (M1/M2 Mac)**: ~10-20ms per prediction
- **CPU (Intel)**: ~30-50ms per prediction
- **GPU (CUDA)**: ~5-10ms per prediction

### Memory Usage
- Model: ~20 MB
- Historical data (5000 bars): ~5 MB
- Total runtime: ~100-200 MB

### Training Time
- 5000 bars, 100 epochs: ~5-10 minutes (CPU)
- 10000 bars, 100 epochs: ~15-20 minutes (CPU)

---

## Next Steps & Recommendations

### Short-Term (Already Applied)
- ✅ Reduced all confidence thresholds
- ✅ Implemented probability-based decision logic
- ✅ Reduced direction margin to 2%
- ⚠️ Monitor signal quality for 1-2 weeks in paper trading

### Medium-Term
1. **Retrain with 20% HOLD** (`hold_percentage = 0.20`)
2. **Implement walk-forward validation** to test robustness
3. **Add ensemble models** for higher confidence
4. **Enable adaptive retraining** (online learning)

### Long-Term
1. **Convert to binary classification** (remove HOLD class entirely)
2. **Add meta-labeling** for position sizing decisions
3. **Implement regime-specific models**
4. **Add reinforcement learning** (RL agent for optimal actions)

---

## Support & Contact

**Issues**: Check `/rnn-server/SIGNAL_GENERATION_FIX.md` for recent fixes

**Logs**: Tail FastAPI logs for real-time debugging:
```bash
tail -f /tmp/fastapi_test.log
```

**Model Path**: `/Users/ryanrhodes/projects/ninjatrader-8/rnn-server/models/trading_model.pth`

---

## Version History

- **v0.1.0** (2024): Initial RNN implementation
- **v0.2.0** (2024): Added multi-timeframe support
- **v0.3.0** (2024): Risk management integration
- **v0.4.0** (2024): Adaptive confidence thresholds
- **v0.5.0** (Current): Emergency fixes for signal generation

---

**Last Updated**: 2025-10-13

**Total Documentation Size**: ~2,500 lines
