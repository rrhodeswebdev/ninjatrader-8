import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from pathlib import Path
import json
import math
from hurst import compute_Hc
from scipy import stats
import time
from functools import wraps
from trading_metrics import evaluate_trading_performance, calculate_sharpe_ratio
from core.indicators import calculate_adx

# ROUND 2 IMPROVEMENTS: Import advanced modules
try:
    from advanced_loss_functions import CombinedTradingLoss, LabelSmoothingCrossEntropy
    from improved_label_generation import calculate_triple_barrier_labels, calculate_regime_adaptive_labels
    from confidence_calibration_advanced import EnsembleCalibration, TemperatureScaling
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced modules not available: {e}")
    print("   Run with standard functionality. Install advanced modules for improvements.")
    ADVANCED_MODULES_AVAILABLE = False

# PERFORMANCE OPTIMIZATION: Add timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 0.01:  # Only log if > 10ms
            print(f"⚡ {func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper


def augment_time_series(X_sequence, augmentation_prob=0.3):
    """
    Data augmentation for time series trading data
    Applied during training to reduce overfitting

    Augmentation types:
    - Jitter: Add small random noise
    - Scale: Scale magnitude slightly
    - Magnitude warp: Warp magnitude of random features
    """
    if np.random.random() > augmentation_prob:
        return X_sequence

    aug_type = np.random.choice(['jitter', 'scale', 'magnitude_warp'])

    if aug_type == 'jitter':
        # Add small random noise (0.5% of std)
        noise = np.random.normal(0, 0.005, X_sequence.shape)
        return X_sequence + noise

    elif aug_type == 'scale':
        # Scale magnitude slightly (95-105%)
        scale = np.random.uniform(0.98, 1.02)
        return X_sequence * scale

    elif aug_type == 'magnitude_warp':
        # Warp magnitude of random features (10-20% of features)
        n_features = X_sequence.shape[1]
        n_warp = max(1, n_features // 10)
        warp_features = np.random.choice(n_features, size=n_warp, replace=False)
        warped = X_sequence.copy()
        warped[:, warp_features] *= np.random.uniform(0.95, 1.05, size=(warped.shape[0], n_warp))
        return warped

    return X_sequence

# ============================================================================
# INDICATOR FUNCTIONS - COMMENTED OUT FOR PURE PRICE ACTION MIGRATION
# These lagging indicators are being replaced with pure price action features
# See: core/price_action_features.py and PRICE_ACTION_MIGRATION_SUMMARY.md
# ============================================================================

# def calculate_adx(high, low, close, period=14):
#     """
#     Calculate Average Directional Index (ADX) for trend strength detection
#     ADX > 25: Strong trend
#     ADX < 20: Ranging/weak trend
#     """
#     n = len(high)
#     if n < period + 1:
#         return np.zeros(n)
#
#     # Calculate +DM and -DM
#     plus_dm = np.zeros(n)
#     minus_dm = np.zeros(n)
#
#     for i in range(1, n):
#         high_diff = high[i] - high[i-1]
#         low_diff = low[i-1] - low[i]
#
#         if high_diff > low_diff and high_diff > 0:
#             plus_dm[i] = high_diff
#         if low_diff > high_diff and low_diff > 0:
#             minus_dm[i] = low_diff
#
#     # Calculate True Range
#     tr = np.zeros(n)
#     for i in range(1, n):
#         tr[i] = max(high[i] - low[i],
#                    abs(high[i] - close[i-1]),
#                    abs(low[i] - close[i-1]))
#
#     # Smooth with EMA
#     import pandas as pd
#     plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, adjust=False).mean() * 100
#     minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, adjust=False).mean() * 100
#
#     # Calculate DX
#     dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100
#
#     # ADX is smoothed DX
#     adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
#
#     return adx


def detect_market_regime(df, lookback=100):
    """
    Detect current market regime for adaptive strategy
    Returns: regime string for backward compatibility
    Use detect_market_regime_enhanced() for full details
    """
    result = detect_market_regime_enhanced(df, lookback)
    return result['regime']


def detect_market_regime_enhanced(df, lookback=100):
    """
    Enhanced regime detection that includes trend direction and strength

    Returns:
        dict with:
            - regime: str (trending_normal, ranging_normal, etc.)
            - trend_direction: str (bullish, bearish, neutral)
            - trend_strength: float (ADX value)
            - vol_ratio: float (current volatility / historical volatility)
    """
    # ADX calculation requires period*2 bars (14*2=28), plus some extra for EMAs
    min_bars_required = 60

    if len(df) < min_bars_required:
        return {
            'regime': 'unknown',
            'trend_direction': 'neutral',
            'trend_strength': 0.0,
            'vol_ratio': 1.0
        }

    recent_data = df.tail(min(lookback, len(df)))

    # Ensure we have enough data for ADX calculation
    if len(recent_data) < min_bars_required:
        return {
            'regime': 'unknown',
            'trend_direction': 'neutral',
            'trend_strength': 0.0,
            'vol_ratio': 1.0
        }

    # Calculate ADX for trend detection
    adx = calculate_adx(recent_data['high'].values,
                       recent_data['low'].values,
                       recent_data['close'].values,
                       period=14)
    current_adx = adx[-1] if len(adx) > 0 else 0

    # NEW: Calculate trend direction using multiple timeframes
    ema_20 = recent_data['close'].ewm(span=20).mean().iloc[-1]
    ema_50 = recent_data['close'].ewm(span=50).mean().iloc[-1]
    current_price = recent_data['close'].iloc[-1]

    # Determine trend direction
    if ema_20 > ema_50 and current_price > ema_20:
        trend_direction = 'bullish'
    elif ema_20 < ema_50 and current_price < ema_20:
        trend_direction = 'bearish'
    else:
        trend_direction = 'neutral'

    # Calculate volatility regime
    returns = recent_data['close'].pct_change().dropna()
    current_vol = returns.tail(20).std()
    hist_vol = returns.std()
    vol_ratio = current_vol / (hist_vol + 1e-8)

    # Classify regime
    if current_adx > 25:
        if vol_ratio > 1.5:
            regime = 'trending_high_vol'
        else:
            regime = 'trending_normal'
    elif current_adx < 20:
        if vol_ratio < 0.7:
            regime = 'ranging_low_vol'
        else:
            regime = 'ranging_normal'
    else:
        if vol_ratio > 1.5:
            regime = 'high_vol_chaos'
        else:
            regime = 'transitional'

    return {
        'regime': regime,
        'trend_direction': trend_direction,
        'trend_strength': current_adx,
        'vol_ratio': vol_ratio
    }


def calculate_trend_alignment_feature(df, lookback=50):
    """
    Calculate how aligned the current price action is with the trend
    Used as a feature to help the model understand counter-trend scenarios

    Returns:
        pandas Series with values from -1 (strong counter-trend) to +1 (strong with-trend)
    """
    if len(df) < lookback:
        return pd.Series(0.0, index=df.index)

    # Calculate EMAs
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()

    # Trend strength: normalized difference between EMAs
    trend_strength = (ema_20 - ema_50) / (df['close'] + 1e-8)

    # Price position relative to EMA20
    price_position = (df['close'] - ema_20) / (df['close'] + 1e-8)

    # Combine into alignment score
    # Positive when price and trend aligned, negative when counter-trend
    alignment = trend_strength * np.sign(price_position) * 100  # Scale to reasonable range

    return alignment


def calculate_hurst_exponent(prices, min_window=10):
    """
    Calculate Hurst exponent for time series using the Mottl/hurst library
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trending (persistent)

    Returns both H and C (Hurst exponent and constant from fit)
    """
    if len(prices) < min_window:
        return 0.5, 1.0  # Default to random walk if insufficient data

    try:
        # compute_Hc returns (H, c, data) where:
        # H = Hurst exponent
        # c = Constant from the fit
        # data = (x, y) values used for fitting
        H, c, _ = compute_Hc(prices, kind='price', simplified=True)

        # Clamp H between 0 and 1 for safety
        H = max(0.0, min(1.0, H))

        return H, c
    except Exception as e:
        # Fallback to default if computation fails
        print(f"Warning: Hurst calculation failed: {e}")
        return 0.5, 1.0


# def calculate_atr(high, low, close, period=14):
#     """
#     Calculate Average True Range (VECTORIZED VERSION)
#     Measures market volatility
#     """
#     n = len(high)
#     if n < period + 1:
#         return np.zeros(n)
#
#     # PERFORMANCE OPTIMIZATION: Vectorized True Range calculation
#     hl = high[1:] - low[1:]
#     hc = np.abs(high[1:] - close[:-1])
#     lc = np.abs(low[1:] - close[:-1])
#     tr = np.maximum(np.maximum(hl, hc), lc)
#
#     # PERFORMANCE OPTIMIZATION: Use pandas rolling mean (faster than loop)
#     import pandas as pd
#     tr_series = pd.Series(tr)
#     atr_values = tr_series.rolling(window=period, min_periods=1).mean().values
#
#     # Add zero for first value (no previous close) - ensure correct length
#     atr = np.zeros(n)
#     atr[1:] = atr_values
#
#     return atr


# def calculate_rsi(close, period=14):
#     """
#     Calculate Relative Strength Index (RSI)
#     RSI > 70: Overbought
#     RSI < 30: Oversold
#     """
#     n = len(close)
#     if n < period + 1:
#         return np.zeros(n)
#
#     # Calculate price changes
#     delta = np.diff(close)
#
#     # Separate gains and losses
#     gain = np.where(delta > 0, delta, 0)
#     loss = np.where(delta < 0, -delta, 0)
#
#     # Calculate average gain and loss using EMA
#     import pandas as pd
#     avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
#     avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
#
#     # Calculate RS and RSI
#     rs = avg_gain / (avg_loss + 1e-8)
#     rsi = 100 - (100 / (1 + rs))
#
#     # Add zero for first value (no previous close)
#     rsi_full = np.zeros(n)
#     rsi_full[1:] = rsi
#
#     return rsi_full


# def calculate_rsi_divergence(close, rsi, lookback=10):
#     """
#     Detect RSI divergence (price vs RSI direction mismatch) - VECTORIZED VERSION
#     +1: Bullish divergence (price down, RSI up)
#     -1: Bearish divergence (price up, RSI down)
#     0: No divergence
#
#     PERFORMANCE OPTIMIZATION: Fully vectorized for ~10x speedup
#     """
#     n = len(close)
#     divergence = np.zeros(n)
#
#     # PERFORMANCE OPTIMIZATION: Vectorized sliding window calculation
#     price_changes = close[lookback:] - close[:-lookback]
#     rsi_changes = rsi[lookback:] - rsi[:-lookback]
#
#     # Vectorized divergence detection
#     # Bullish divergence: price_change < 0 and rsi_change > 0
#     # Bearish divergence: price_change > 0 and rsi_change < 0
#     divergence[lookback:] = np.where(
#         (price_changes < 0) & (rsi_changes > 0), 1,
#         np.where((price_changes > 0) & (rsi_changes < 0), -1, 0)
#     )
#
#     return divergence


# def calculate_macd(close, fast=12, slow=26, signal=9):
#     """
#     Calculate MACD (Moving Average Convergence Divergence)
#     Returns: macd_line, signal_line, histogram
#     """
#     n = len(close)
#     if n < slow:
#         return np.zeros(n), np.zeros(n), np.zeros(n)
#
#     import pandas as pd
#     close_series = pd.Series(close)
#
#     # Calculate EMAs
#     ema_fast = close_series.ewm(span=fast, adjust=False).mean().values
#     ema_slow = close_series.ewm(span=fast, adjust=False).mean().values
#
#     # MACD line
#     macd_line = ema_fast - ema_slow
#
#     # Signal line (EMA of MACD)
#     macd_series = pd.Series(macd_line)
#     signal_line = macd_series.ewm(span=signal, adjust=False).mean().values
#
#     # Histogram (difference)
#     histogram = macd_line - signal_line
#
#     return macd_line, signal_line, histogram


# def calculate_vwma_deviation(close, volume, period=20):
#     """
#     Calculate Volume-Weighted Moving Average deviation
#     Better than SMA for futures markets with volume information
#     """
#     n = len(close)
#     if n < period:
#         return np.zeros(n)
#
#     import pandas as pd
#     close_series = pd.Series(close)
#     volume_series = pd.Series(volume)
#
#     # Volume-weighted MA
#     vwma = (close_series * volume_series).rolling(period).sum() / volume_series.rolling(period).sum()
#     vwma = vwma.bfill().values  # Fixed: Use bfill() instead of deprecated fillna(method='bfill')
#
#     # Deviation as percentage
#     deviation = (close - vwma) / (vwma + 1e-8)
#
#     return deviation


# def calculate_garman_klass_volatility(open_prices, high, low, close, period=20):
#     """
#     Calculate Garman-Klass volatility estimator
#     More efficient than Parkinson volatility
#     """
#     n = len(close)
#     if n < 2:
#         return np.zeros(n)
#
#     # Garman-Klass formula
#     hl = np.log(high / (low + 1e-8)) ** 2
#     co = np.log(close / (open_prices + 1e-8)) ** 2
#     gk_vol = 0.5 * hl - (2 * np.log(2) - 1) * co
#
#     # Rolling average
#     import pandas as pd
#     gk_vol_series = pd.Series(gk_vol)
#     gk_vol_smooth = gk_vol_series.rolling(period, min_periods=1).mean().values
#
#     return gk_vol_smooth


# def calculate_price_impact(close, volume):
#     """
#     Calculate price impact per unit volume
#     Measures market liquidity and institutional activity
#     """
#     n = len(close)
#
#     # Price impact = abs(price change) / volume
#     price_change = np.abs(np.diff(close))
#     price_impact = np.zeros(n)
#     price_impact[1:] = price_change / (volume[1:] + 1)
#
#     return price_impact


# def calculate_volume_weighted_price_change(close, volume, period=5):
#     """
#     Volume-weighted price change
#     Emphasizes price moves with high volume
#     """
#     n = len(close)
#     vwpc = np.zeros(n)
#
#     import pandas as pd
#     close_series = pd.Series(close)
#     volume_series = pd.Series(volume)
#
#     price_change = close_series.diff()
#     avg_volume = volume_series.rolling(period, min_periods=1).mean()
#
#     vwpc_values = (price_change * volume_series) / (avg_volume + 1)
#     vwpc = vwpc_values.fillna(0).values
#
#     return vwpc


def align_secondary_to_primary(df_primary, df_secondary):
    """
    Align secondary timeframe data to primary timeframe
    For each primary bar, find the corresponding secondary bar value

    Args:
        df_primary: Primary timeframe dataframe (e.g., 1-min)
        df_secondary: Secondary timeframe dataframe (e.g., 5-min)

    Returns:
        Dictionary with secondary features aligned to primary timeframe
    """
    if df_secondary is None:
        # Return zeros if no secondary data
        n_bars = len(df_primary)
        return {
            'tf2_close': np.zeros(n_bars),
            'tf2_close_change': np.zeros(n_bars),
            'tf2_high_low_range': np.zeros(n_bars),
            'tf2_volume': np.zeros(n_bars),
            'tf2_position_in_bar': np.zeros(n_bars),
            'tf2_trend_direction': np.zeros(n_bars),
            'tf2_momentum': np.zeros(n_bars),
            'tf2_volatility': np.zeros(n_bars),
            'tf2_alignment_score': np.zeros(n_bars),
            'tf2_delta': np.zeros(n_bars)
        }

    if len(df_secondary) == 0:
        # Return zeros if empty secondary data
        n_bars = len(df_primary)
        return {
            'tf2_close': np.zeros(n_bars),
            'tf2_close_change': np.zeros(n_bars),
            'tf2_high_low_range': np.zeros(n_bars),
            'tf2_volume': np.zeros(n_bars),
            'tf2_position_in_bar': np.zeros(n_bars),
            'tf2_trend_direction': np.zeros(n_bars),
            'tf2_momentum': np.zeros(n_bars),
            'tf2_volatility': np.zeros(n_bars),
            'tf2_alignment_score': np.zeros(n_bars),
            'tf2_delta': np.zeros(n_bars)
        }

    n_bars = len(df_primary)
    aligned_features = {
        'tf2_close': np.zeros(n_bars),
        'tf2_close_change': np.zeros(n_bars),
        'tf2_high_low_range': np.zeros(n_bars),
        'tf2_volume': np.zeros(n_bars),
        'tf2_position_in_bar': np.zeros(n_bars),
        'tf2_trend_direction': np.zeros(n_bars),
        'tf2_momentum': np.zeros(n_bars),
        'tf2_volatility': np.zeros(n_bars),
        'tf2_alignment_score': np.zeros(n_bars),
        'tf2_delta': np.zeros(n_bars)
    }

    # Convert to numpy for faster access
    primary_times = df_primary['time'].values
    primary_close = df_primary['close'].values
    secondary_times = df_secondary['time'].values
    secondary_close = df_secondary['close'].values
    secondary_high = df_secondary['high'].values
    secondary_low = df_secondary['low'].values
    secondary_volume = df_secondary['volume'].values if 'volume' in df_secondary.columns else np.zeros(len(df_secondary))

    # Calculate 5-min EMA for trend detection (using 20-period EMA on 5-min = ~100 min)
    tf2_ema = np.zeros(len(df_secondary))
    alpha = 2.0 / (20 + 1)  # 20-period EMA
    tf2_ema[0] = secondary_close[0]
    for i in range(1, len(df_secondary)):
        tf2_ema[i] = alpha * secondary_close[i] + (1 - alpha) * tf2_ema[i-1]

    # For each primary bar, find the latest secondary bar at or before it
    secondary_idx = 0
    for i in range(n_bars):
        primary_time = primary_times[i]

        # Find the secondary bar that contains this primary time
        while secondary_idx < len(secondary_times) - 1 and secondary_times[secondary_idx + 1] <= primary_time:
            secondary_idx += 1

        if secondary_idx < len(secondary_times):
            # Extract secondary features
            aligned_features['tf2_close'][i] = secondary_close[secondary_idx]
            aligned_features['tf2_high_low_range'][i] = secondary_high[secondary_idx] - secondary_low[secondary_idx]
            aligned_features['tf2_volume'][i] = secondary_volume[secondary_idx]

            # Position within secondary bar (0 = at low, 1 = at high)
            range_size = secondary_high[secondary_idx] - secondary_low[secondary_idx]
            if range_size > 0:
                aligned_features['tf2_position_in_bar'][i] = (primary_close[i] - secondary_low[secondary_idx]) / range_size

            # Secondary close price change
            if secondary_idx > 0:
                aligned_features['tf2_close_change'][i] = (secondary_close[secondary_idx] - secondary_close[secondary_idx - 1]) / secondary_close[secondary_idx - 1]

            # NEW FEATURES:
            # 1. Trend direction (EMA slope normalized)
            if secondary_idx >= 2:
                ema_slope = (tf2_ema[secondary_idx] - tf2_ema[secondary_idx-2]) / (tf2_ema[secondary_idx-2] + 1e-8)
                aligned_features['tf2_trend_direction'][i] = np.tanh(ema_slope * 100)  # Normalize to -1 to 1

            # 2. Momentum (rate of change over 3 bars = 15 min)
            if secondary_idx >= 3:
                roc = (secondary_close[secondary_idx] - secondary_close[secondary_idx-3]) / (secondary_close[secondary_idx-3] + 1e-8)
                aligned_features['tf2_momentum'][i] = np.tanh(roc * 50)  # Normalize

            # 3. Volatility (ATR-like on 5-min)
            if secondary_idx >= 5:
                tr_values = []
                for j in range(secondary_idx-4, secondary_idx+1):
                    if j > 0:
                        tr = max(secondary_high[j] - secondary_low[j],
                                abs(secondary_high[j] - secondary_close[j-1]),
                                abs(secondary_low[j] - secondary_close[j-1]))
                        tr_values.append(tr)
                if len(tr_values) > 0:
                    aligned_features['tf2_volatility'][i] = np.mean(tr_values) / (secondary_close[secondary_idx] + 1e-8)

            # 4. Alignment score (how aligned is 1-min with 5-min trend)
            # Calculate primary 1-min trend over recent bars
            if i >= 5 and secondary_idx >= 1:
                primary_trend_short = (primary_close[i] - primary_close[i-5]) / (primary_close[i-5] + 1e-8)
                secondary_trend_short = (secondary_close[secondary_idx] - secondary_close[secondary_idx-1]) / (secondary_close[secondary_idx-1] + 1e-8)

                # Trend alignment scoring:
                # +1: Both trending same direction (both > 0 or both < 0)
                # -1: Trending opposite directions
                # 0: One or both are flat (< threshold)
                threshold = 0.0001  # 0.01% threshold for "flat"

                if abs(primary_trend_short) < threshold or abs(secondary_trend_short) < threshold:
                    alignment = 0.0  # One timeframe is flat
                else:
                    # Both trending - check if same direction
                    alignment = np.sign(primary_trend_short) * np.sign(secondary_trend_short)

                aligned_features['tf2_alignment_score'][i] = alignment

            # 5. Secondary delta (if available)
            if 'bid_volume' in df_secondary.columns and 'ask_volume' in df_secondary.columns:
                bid_vol = df_secondary['bid_volume'].values[secondary_idx] if secondary_idx < len(df_secondary) else 0
                ask_vol = df_secondary['ask_volume'].values[secondary_idx] if secondary_idx < len(df_secondary) else 0
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    aligned_features['tf2_delta'][i] = (ask_vol - bid_vol) / total_vol

    return aligned_features


def calculate_order_flow_features(df):
    """
    Calculate order flow and volume-based features
    Returns dictionary of feature arrays
    """
    n_bars = len(df)
    features = {}

    # Extract volume data
    volume = df['volume'].values if 'volume' in df.columns else np.zeros(n_bars)
    bid_volume = df['bid_volume'].values if 'bid_volume' in df.columns else np.zeros(n_bars)
    ask_volume = df['ask_volume'].values if 'ask_volume' in df.columns else np.zeros(n_bars)

    # REMOVED: 50/50 fallback for bid/ask volume
    # This created fake data that taught the model incorrect patterns
    # Order flow features are now excluded when bid/ask data is unavailable

    # REMOVED FEATURES (always 0 during training - no bid/ask data in historical):
    # - delta (buy - sell volume)
    # - cumulative_delta
    # - volume_imbalance
    # - aggressive_buy_ratio
    # - delta_divergence
    # - cumulative_delta_slope

    # ONLY KEEP: Volume relative to moving average (uses total volume, always available)
    volume_ratio = np.ones(n_bars)
    window = 20
    for i in range(window, n_bars):
        avg_volume = np.mean(volume[i-window:i])
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / (avg_volume + 1e-8)
    features['volume_ratio'] = volume_ratio

    return features


def calculate_price_features(df):
    """
    Calculate advanced price-based features (no typical indicators)
    Returns dictionary of feature arrays
    """
    ohlc = df[['open', 'high', 'low', 'close']].values
    n_bars = len(ohlc)

    # Initialize feature dictionary
    features = {}

    # 1. Price Momentum (Velocity & Acceleration)
    velocity = np.zeros(n_bars)
    acceleration = np.zeros(n_bars)
    lookback = 5
    for i in range(lookback, n_bars):
        velocity[i] = (ohlc[i, 3] - ohlc[i-lookback, 3]) / lookback
        if i > lookback:
            acceleration[i] = velocity[i] - velocity[i-1]
    features['velocity'] = velocity
    features['acceleration'] = acceleration

    # 2. Range Dynamics (Range Ratio & Wick Ratio)
    range_ratio = np.ones(n_bars)
    wick_ratio = np.zeros(n_bars)
    for i in range(1, n_bars):
        current_range = ohlc[i, 1] - ohlc[i, 2]  # high - low
        prev_range = ohlc[i-1, 1] - ohlc[i-1, 2]
        range_ratio[i] = current_range / (prev_range + 1e-8)

        # Wick calculations
        upper_wick = ohlc[i, 1] - max(ohlc[i, 0], ohlc[i, 3])  # high - max(open, close)
        lower_wick = min(ohlc[i, 0], ohlc[i, 3]) - ohlc[i, 2]  # min(open, close) - low
        wick_ratio[i] = upper_wick / (lower_wick + 1e-8)
    features['range_ratio'] = range_ratio
    features['wick_ratio'] = wick_ratio

    # 3. Gap Analysis
    gap_up = np.zeros(n_bars)
    gap_down = np.zeros(n_bars)
    gap_filled = np.zeros(n_bars)
    for i in range(1, n_bars):
        gap_up[i] = max(0, ohlc[i, 2] - ohlc[i-1, 1])  # low - prev high
        gap_down[i] = max(0, ohlc[i-1, 2] - ohlc[i, 1])  # prev low - high
        # Check if gap filled (prev close within current range)
        if ohlc[i, 2] <= ohlc[i-1, 3] <= ohlc[i, 1]:
            gap_filled[i] = 1
    features['gap_up'] = gap_up
    features['gap_down'] = gap_down
    features['gap_filled'] = gap_filled

    # 4. Price Fractals (Swing Highs/Lows)
    swing_high = np.zeros(n_bars)
    swing_low = np.zeros(n_bars)
    bars_since_swing_high = np.zeros(n_bars)
    bars_since_swing_low = np.zeros(n_bars)

    last_swing_high_idx = 0
    last_swing_low_idx = 0

    for i in range(2, n_bars - 1):
        # Swing high: higher than neighbors
        if ohlc[i, 1] > ohlc[i-1, 1] and ohlc[i, 1] > ohlc[i+1, 1]:
            swing_high[i] = 1
            last_swing_high_idx = i
        # Swing low: lower than neighbors
        if ohlc[i, 2] < ohlc[i-1, 2] and ohlc[i, 2] < ohlc[i+1, 2]:
            swing_low[i] = 1
            last_swing_low_idx = i

        bars_since_swing_high[i] = i - last_swing_high_idx
        bars_since_swing_low[i] = i - last_swing_low_idx

    features['swing_high'] = swing_high
    features['swing_low'] = swing_low
    features['bars_since_swing_high'] = bars_since_swing_high
    features['bars_since_swing_low'] = bars_since_swing_low

    # 5. Return Distribution (Skewness & Kurtosis)
    # FIX: Look-ahead bias - exclude current bar from calculation
    skewness = np.zeros(n_bars)
    kurtosis_vals = np.zeros(n_bars)
    window = 20

    returns = np.diff(ohlc[:, 3]) / ohlc[:-1, 3]
    for i in range(window, n_bars):
        # FIXED: Use returns up to i-1 to exclude current bar (no look-ahead bias)
        recent_returns = returns[max(0, i-window-1):i-1]
        if len(recent_returns) > 3:  # Need minimum for stats
            skewness[i] = stats.skew(recent_returns)
            kurtosis_vals[i] = stats.kurtosis(recent_returns)
    features['skewness'] = skewness
    features['kurtosis'] = kurtosis_vals

    # 6. Rolling Min/Max Distance (Position in Range)
    # FIX: Look-ahead bias - exclude current bar from range calculation
    position_in_range = np.zeros(n_bars)
    window = 20
    for i in range(window, n_bars):
        # FIXED: Calculate range using only past data (i-window:i, not i+1)
        rolling_max = np.max(ohlc[i-window:i, 3])
        rolling_min = np.min(ohlc[i-window:i, 3])
        range_size = rolling_max - rolling_min
        if range_size > 1e-8:
            position_in_range[i] = (ohlc[i, 3] - rolling_min) / range_size
    features['position_in_range'] = position_in_range

    # 7. Trend Structure (Higher Highs / Lower Lows count)
    higher_highs = np.zeros(n_bars)
    lower_lows = np.zeros(n_bars)
    trend_strength = np.zeros(n_bars)
    window = 5

    for i in range(window, n_bars):
        hh_count = sum([1 for j in range(i-window+1, i+1) if ohlc[j, 1] > ohlc[j-1, 1]])
        ll_count = sum([1 for j in range(i-window+1, i+1) if ohlc[j, 2] < ohlc[j-1, 2]])
        higher_highs[i] = hh_count
        lower_lows[i] = ll_count
        trend_strength[i] = hh_count - ll_count  # -5 to +5

    features['higher_highs'] = higher_highs
    features['lower_lows'] = lower_lows
    features['trend_strength'] = trend_strength

    # 8. Price Deviation Features
    close_prices = ohlc[:, 3]

    # REDUCED: Only keep windows 20 and 50 (removed 5 and 10 for redundancy)
    # This reduces deviation features from 23 to 13
    windows = [20, 50]

    for window in windows:
        # Deviation from mean
        mean_dev = np.zeros(n_bars)
        # Deviation from median
        median_dev = np.zeros(n_bars)
        # Standard deviation (volatility measure)
        std_dev = np.zeros(n_bars)
        # Z-score (standardized deviation)
        z_score = np.zeros(n_bars)
        # Bollinger width equivalent (range of ±2 std devs)
        bb_width = np.zeros(n_bars)

        for i in range(window, n_bars):
            window_prices = close_prices[i-window:i]

            # Mean and deviation from mean
            mean_price = np.mean(window_prices)
            mean_dev[i] = (close_prices[i] - mean_price) / (mean_price + 1e-8)

            # Median and deviation from median
            median_price = np.median(window_prices)
            median_dev[i] = (close_prices[i] - median_price) / (median_price + 1e-8)

            # Standard deviation
            std = np.std(window_prices)
            std_dev[i] = std

            # Z-score (how many std devs away from mean)
            if std > 1e-8:
                z_score[i] = (close_prices[i] - mean_price) / std

            # Bollinger Band Width (4 std devs / mean price)
            bb_width[i] = (4 * std) / (mean_price + 1e-8)

        # Store with window size in name
        features[f'mean_dev_{window}'] = mean_dev
        features[f'median_dev_{window}'] = median_dev
        features[f'std_dev_{window}'] = std_dev
        features[f'z_score_{window}'] = z_score
        features[f'bb_width_{window}'] = bb_width

    # Additional: Rate of change of standard deviation (volatility acceleration)
    vol_accel_20 = np.zeros(n_bars)
    for i in range(21, n_bars):
        current_vol = features['std_dev_20'][i]
        prev_vol = features['std_dev_20'][i-1]
        vol_accel_20[i] = current_vol - prev_vol
    features['vol_acceleration'] = vol_accel_20

    # Distance from recent extremes as percentage
    high_dev = np.zeros(n_bars)
    low_dev = np.zeros(n_bars)
    for i in range(20, n_bars):
        recent_high = np.max(ohlc[i-20:i+1, 1])
        recent_low = np.min(ohlc[i-20:i+1, 2])
        high_dev[i] = (close_prices[i] - recent_high) / (recent_high + 1e-8)
        low_dev[i] = (close_prices[i] - recent_low) / (recent_low + 1e-8)
    features['high_deviation'] = high_dev
    features['low_deviation'] = low_dev

    return features


def detect_candlestick_patterns(df):
    """
    Detect key candlestick patterns with context-aware scoring
    Returns dictionary of pattern detection arrays
    """
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    n = len(df)

    patterns = {
        'bullish_engulfing': np.zeros(n),
        'bearish_engulfing': np.zeros(n),
        'hammer': np.zeros(n),
        'shooting_star': np.zeros(n),
        'doji': np.zeros(n),
        'inside_bar': np.zeros(n),
        'outside_bar': np.zeros(n),
        'pin_bar_bull': np.zeros(n),
        'pin_bar_bear': np.zeros(n)
    }

    for i in range(1, n):
        body_size = abs(c[i] - o[i])
        total_range = h[i] - l[i]
        prev_body_size = abs(c[i-1] - o[i-1])

        # Avoid division by zero
        if total_range < 1e-8:
            continue

        # Bullish Engulfing
        if (c[i-1] < o[i-1] and  # Previous red candle
            c[i] > o[i] and      # Current green candle
            o[i] < c[i-1] and    # Opens below prev close
            c[i] > o[i-1]):      # Closes above prev open
            patterns['bullish_engulfing'][i] = 1.0

        # Bearish Engulfing
        if (c[i-1] > o[i-1] and  # Previous green
            c[i] < o[i] and      # Current red
            o[i] > c[i-1] and    # Opens above prev close
            c[i] < o[i-1]):      # Closes below prev open
            patterns['bearish_engulfing'][i] = 1.0

        # Hammer (bullish reversal)
        lower_wick = min(o[i], c[i]) - l[i]
        upper_wick = h[i] - max(o[i], c[i])
        if (lower_wick > 2 * body_size and
            upper_wick < 0.1 * total_range and
            body_size > 0.01 * total_range):  # Must have some body
            patterns['hammer'][i] = 1.0

        # Shooting Star (bearish reversal)
        if (upper_wick > 2 * body_size and
            lower_wick < 0.1 * total_range and
            body_size > 0.01 * total_range):
            patterns['shooting_star'][i] = 1.0

        # Doji (indecision)
        if body_size < 0.1 * total_range:
            patterns['doji'][i] = 1.0

        # Inside Bar (consolidation)
        if h[i] < h[i-1] and l[i] > l[i-1]:
            patterns['inside_bar'][i] = 1.0

        # Outside Bar (breakout)
        if h[i] > h[i-1] and l[i] < l[i-1]:
            patterns['outside_bar'][i] = 1.0

        # Pin Bar patterns
        if lower_wick > 0.6 * total_range and body_size < 0.3 * total_range:
            patterns['pin_bar_bull'][i] = 1.0
        if upper_wick > 0.6 * total_range and body_size < 0.3 * total_range:
            patterns['pin_bar_bear'][i] = 1.0

    return patterns


def detect_support_resistance_levels(df, lookback=200, num_levels=5):
    """
    Detect S/R levels using volume-weighted price clustering
    Returns list of price levels and their strengths
    """
    if len(df) < lookback:
        lookback = len(df)

    closes = df['close'].values[-lookback:]
    volumes = df['volume'].values[-lookback:] if 'volume' in df.columns else np.ones(lookback)

    # Create price bins (0.1% increments for futures)
    price_range = closes.max() - closes.min()
    if price_range < 1e-8:
        return [], []

    bin_size = price_range * 0.001  # 0.1% bins
    bins = np.arange(closes.min(), closes.max() + bin_size, bin_size)

    # Volume-weighted histogram
    level_volumes = {}
    for price, volume in zip(closes, volumes):
        bin_idx = int((price - bins[0]) / bin_size)
        if bin_idx < 0 or bin_idx >= len(bins):
            continue
        if bin_idx not in level_volumes:
            level_volumes[bin_idx] = 0
        level_volumes[bin_idx] += volume

    # Find top N levels by volume
    if len(level_volumes) == 0:
        return [], []

    top_levels = sorted(level_volumes.items(),
                       key=lambda x: x[1], reverse=True)[:num_levels]

    sr_prices = [bins[idx] for idx, _ in top_levels]
    sr_strengths = [vol for _, vol in top_levels]

    # Normalize strengths
    max_strength = max(sr_strengths) if sr_strengths else 1.0
    sr_strengths = [s / max_strength for s in sr_strengths]

    return sr_prices, sr_strengths


def calculate_sr_features(df):
    """
    Calculate features based on S/R levels
    Returns dictionary of S/R-related features
    """
    n_bars = len(df)
    features = {
        'dist_to_nearest_sr': np.zeros(n_bars),
        'nearest_sr_strength': np.zeros(n_bars),
        'is_near_sr': np.zeros(n_bars),
        'above_sr': np.zeros(n_bars)
    }

    # Need at least 50 bars to detect meaningful S/R levels
    for i in range(50, n_bars):
        current_price = df['close'].iloc[i]

        # Get S/R levels based on history up to current bar
        df_history = df.iloc[:i]
        sr_prices, sr_strengths = detect_support_resistance_levels(df_history)

        if len(sr_prices) == 0:
            continue

        # Distance to nearest support/resistance
        distances = [abs(current_price - sr) / current_price for sr in sr_prices]
        nearest_idx = np.argmin(distances)

        features['dist_to_nearest_sr'][i] = distances[nearest_idx]
        features['nearest_sr_strength'][i] = sr_strengths[nearest_idx]
        features['is_near_sr'][i] = 1.0 if distances[nearest_idx] < 0.002 else 0.0  # Within 0.2%
        features['above_sr'][i] = 1.0 if current_price > sr_prices[nearest_idx] else 0.0

    return features


def calculate_volume_profile(df, lookback=100, num_bins=20):
    """
    Calculate volume profile features
    Returns POC, VAH, VAL and related features
    """
    n_bars = len(df)
    features = {
        'dist_to_poc': np.zeros(n_bars),
        'volume_at_price': np.zeros(n_bars),
        'above_vah': np.zeros(n_bars),
        'below_val': np.zeros(n_bars),
        'in_value_area': np.zeros(n_bars)
    }

    for i in range(lookback, n_bars):
        recent = df.iloc[i-lookback:i]

        # Create price bins
        price_min = recent['low'].min()
        price_max = recent['high'].max()

        if price_max - price_min < 1e-8:
            continue

        bins = np.linspace(price_min, price_max, num_bins + 1)

        # Accumulate volume at each price level
        volume_at_price = np.zeros(num_bins)

        for _, bar in recent.iterrows():
            # Distribute bar volume across its range
            bar_min_bin = np.digitize(bar['low'], bins) - 1
            bar_max_bin = np.digitize(bar['high'], bins) - 1

            bar_min_bin = max(0, min(bar_min_bin, num_bins - 1))
            bar_max_bin = max(0, min(bar_max_bin, num_bins - 1))

            bins_in_bar = max(1, bar_max_bin - bar_min_bin + 1)
            vol_per_bin = bar['volume'] / bins_in_bar if 'volume' in bar else 1.0 / bins_in_bar

            for b in range(bar_min_bin, bar_max_bin + 1):
                if b < num_bins:
                    volume_at_price[b] += vol_per_bin

        # Find POC (Point of Control) - price with highest volume
        poc_bin = np.argmax(volume_at_price)
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area (70% of volume)
        total_volume = volume_at_price.sum()
        if total_volume < 1e-8:
            continue

        target_volume = total_volume * 0.70

        # Find VAH and VAL
        sorted_bins = np.argsort(volume_at_price)[::-1]
        accumulated_vol = 0
        value_area_bins = []

        for bin_idx in sorted_bins:
            accumulated_vol += volume_at_price[bin_idx]
            value_area_bins.append(bin_idx)
            if accumulated_vol >= target_volume:
                break

        if len(value_area_bins) == 0:
            continue

        vah = bins[max(value_area_bins) + 1]  # Value Area High
        val = bins[min(value_area_bins)]      # Value Area Low

        current_price = df['close'].iloc[i]

        features['dist_to_poc'][i] = (current_price - poc_price) / current_price

        # Volume at current price level
        current_bin = np.digitize(current_price, bins) - 1
        current_bin = max(0, min(current_bin, num_bins - 1))
        features['volume_at_price'][i] = volume_at_price[current_bin] / total_volume if total_volume > 0 else 0

        features['above_vah'][i] = 1.0 if current_price > vah else 0.0
        features['below_val'][i] = 1.0 if current_price < val else 0.0
        features['in_value_area'][i] = 1.0 if val <= current_price <= vah else 0.0

    return features


def calculate_time_features(df):
    """
    Calculate time-of-day features for intraday trading
    Returns dictionary of feature arrays
    """
    n_bars = len(df)
    features = {}

    # Extract time information
    times = pd.to_datetime(df['time'])

    # 1. Hour of day (normalized 0-1, market hours typically 9:30-16:00 ET)
    hour_of_day = times.dt.hour + times.dt.minute / 60.0
    # Normalize assuming 9.5 hours (9:30am) to 16.0 hours (4:00pm)
    normalized_hour = (hour_of_day - 9.5) / 6.5  # 6.5 hour trading day
    normalized_hour = np.clip(normalized_hour, 0, 1)  # Clip to 0-1 range
    features['hour_of_day'] = normalized_hour.values

    # 2. Minutes into session (0-390 for typical 6.5 hour day)
    minutes_into_session = (hour_of_day - 9.5) * 60
    minutes_into_session = np.clip(minutes_into_session, 0, 390)
    features['minutes_into_session'] = minutes_into_session.values / 390.0  # Normalize

    # 3. Distance to market close (important for EOD behavior)
    minutes_to_close = 390 - minutes_into_session
    features['minutes_to_close'] = minutes_to_close / 390.0  # Normalize

    # 4. Is it opening/closing period? (binary flags for volatility periods)
    is_opening_period = (minutes_into_session <= 30).astype(float).values  # First 30 min
    is_closing_period = (minutes_to_close <= 30).astype(float).values  # Last 30 min
    features['is_opening_period'] = is_opening_period
    features['is_closing_period'] = is_closing_period

    return features


def calculate_microstructure_features(df):
    """
    Calculate microstructure and tape reading features
    Returns dictionary of feature arrays
    """
    n_bars = len(df)
    features = {}

    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # 1. Volume surge detection (current volume / 20-bar average)
    volume_surge = np.ones(n_bars)
    for i in range(20, n_bars):
        avg_volume = np.mean(volume[i-20:i])
        if avg_volume > 0:
            volume_surge[i] = volume[i] / avg_volume
    features['volume_surge'] = np.clip(volume_surge, 0, 5)  # Clip outliers

    # 2. Price acceleration (2nd derivative)
    price_acceleration = np.zeros(n_bars)
    for i in range(2, n_bars):
        velocity_current = close[i] - close[i-1]
        velocity_prev = close[i-1] - close[i-2]
        price_acceleration[i] = velocity_current - velocity_prev
    features['price_acceleration'] = price_acceleration / (close + 1e-8)  # Normalize

    # 3. Effective spread proxy (high-low range as % of close)
    effective_spread = (high - low) / (close + 1e-8)
    features['effective_spread'] = effective_spread

    # 4. Large print detection (volume spikes)
    large_print = np.zeros(n_bars)
    for i in range(20, n_bars):
        avg_volume = np.mean(volume[i-20:i])
        if volume[i] > 3 * avg_volume:
            large_print[i] = 1.0
    features['large_print'] = large_print

    # 5. VWAP deviation
    vwap_dev = np.zeros(n_bars)
    cumulative_pv = 0
    cumulative_v = 0
    for i in range(n_bars):
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        cumulative_pv += typical_price * volume[i]
        cumulative_v += volume[i]
        if cumulative_v > 0:
            vwap = cumulative_pv / cumulative_v
            vwap_dev[i] = (close[i] - vwap) / vwap
    features['vwap_deviation'] = vwap_dev

    # 6. VPIN (Volume-Synchronized Probability of Informed Trading)
    # Detects toxic order flow / informed trading
    vpin = np.zeros(n_bars)
    bucket_size = max(1, int(np.mean(volume[volume > 0])))  # Average bar volume
    for i in range(50, n_bars):  # Need history for bucketing
        # Simplified VPIN: uses price direction as proxy for buy/sell classification
        recent_volume = volume[i-50:i+1]
        recent_close = close[i-50:i+1]

        # Classify bars as buy or sell based on price movement
        price_changes = np.diff(recent_close)
        buy_volume = np.sum(recent_volume[1:][price_changes > 0])
        sell_volume = np.sum(recent_volume[1:][price_changes < 0])
        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            vpin[i] = abs(buy_volume - sell_volume) / total_volume
    features['vpin'] = np.clip(vpin, 0, 1)

    # 7. Roll's Spread Estimator (effective spread measure)
    # Estimates bid-ask spread from price series
    roll_spread = np.zeros(n_bars)
    for i in range(20, n_bars):
        # Roll (1984): spread = 2 * sqrt(-cov(Δp_t, Δp_t-1))
        price_changes = np.diff(close[i-20:i+1])
        if len(price_changes) > 1:
            cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            if cov < 0:
                roll_spread[i] = 2 * np.sqrt(-cov) / close[i]  # Normalize by price
    features['roll_spread'] = roll_spread

    # 8. Tick Imbalance (Lee-Ready algorithm approximation)
    # Classifies trades as buyer or seller initiated
    tick_imbalance = np.zeros(n_bars)
    cumulative_tick_imbalance = np.zeros(n_bars)
    cum_imbalance = 0

    for i in range(1, n_bars):
        # Tick rule: if price up from last trade -> buy, down -> sell
        price_change = close[i] - close[i-1]

        if price_change > 0:
            tick = 1  # Buy
        elif price_change < 0:
            tick = -1  # Sell
        else:
            tick = 0  # No change (use previous tick, but we simplify to 0)

        tick_imbalance[i] = tick
        cum_imbalance += tick
        cumulative_tick_imbalance[i] = cum_imbalance

    # Normalize cumulative imbalance by rolling window
    tick_imbalance_normalized = np.zeros(n_bars)
    for i in range(20, n_bars):
        window_imbalance = cumulative_tick_imbalance[i] - cumulative_tick_imbalance[i-20]
        tick_imbalance_normalized[i] = window_imbalance / 20.0  # Average per bar

    features['tick_imbalance'] = tick_imbalance_normalized

    return features


def calculate_volatility_regime_features(df):
    """
    Calculate volatility regime and market state features
    Returns dictionary of feature arrays
    """
    n_bars = len(df)
    features = {}

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values

    # 1. Volatility regime (current vol / 100-bar historical vol)
    # FIX: Look-ahead bias - exclude current bar
    volatility_regime = np.ones(n_bars)
    for i in range(100, n_bars):
        # Current volatility (20-bar) - FIXED: up to i-1
        # np.diff produces n-1 elements, so we need to match the denominator
        close_recent = close[i-20:i]
        returns_recent = np.diff(close_recent) / close_recent[:-1]
        current_vol = np.std(returns_recent) if len(returns_recent) > 1 else 0

        # Historical volatility (100-bar) - FIXED: up to i-1
        close_hist = close[i-100:i]
        returns_hist = np.diff(close_hist) / close_hist[:-1]
        hist_vol = np.std(returns_hist) if len(returns_hist) > 1 else 1e-8

        volatility_regime[i] = current_vol / (hist_vol + 1e-8)
    features['volatility_regime'] = np.clip(volatility_regime, 0, 5)

    # 2. Parkinson volatility (high-low range estimator)
    # FIX: Look-ahead bias - exclude current bar
    parkinson_vol = np.zeros(n_bars)
    for i in range(20, n_bars):
        # FIXED: up to i, not i+1
        hl_ratios = np.log(high[i-20:i] / low[i-20:i])
        parkinson_vol[i] = np.sqrt(np.sum(hl_ratios**2) / (4 * 20 * np.log(2)))
    features['parkinson_volatility'] = parkinson_vol

    # 3. Volume regime (current volume / historical average)
    # FIX: Look-ahead bias - exclude current bar
    volume_regime = np.ones(n_bars)
    for i in range(100, n_bars):
        # FIXED: up to i, not i+1
        current_avg_volume = np.mean(volume[i-20:i])
        hist_avg_volume = np.mean(volume[i-100:i])
        volume_regime[i] = current_avg_volume / (hist_avg_volume + 1e-8)
    features['volume_regime'] = np.clip(volume_regime, 0, 5)

    # 4. Trending vs Ranging (simplified ADX)
    # Calculate directional movement
    trending_score = np.zeros(n_bars)
    for i in range(14, n_bars):
        # Positive directional movement
        plus_dm = np.maximum(high[i] - high[i-1], 0)
        # Negative directional movement
        minus_dm = np.maximum(low[i-1] - low[i], 0)

        # Average over 14 periods
        avg_plus_dm = np.mean([np.maximum(high[j] - high[j-1], 0) for j in range(i-13, i+1)])
        avg_minus_dm = np.mean([np.maximum(low[j-1] - low[j], 0) for j in range(i-13, i+1)])

        # Trend strength (higher = more trending)
        total_dm = avg_plus_dm + avg_minus_dm
        if total_dm > 1e-8:
            trending_score[i] = abs(avg_plus_dm - avg_minus_dm) / total_dm
    features['trending_score'] = trending_score

    return features


class FocalLoss(nn.Module):
    """
    Focal Loss - focuses training on hard examples
    Reduces weight of easy examples, increases weight of hard-to-classify examples
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
        return focal_loss


class TrendAwareTradingLoss(nn.Module):
    """
    PRIORITY 2: Loss function that penalizes counter-trend trades more heavily
    and rewards with-trend trades
    """
    def __init__(self, base_weight=1.0, trend_penalty_weight=0.5):
        super().__init__()
        self.base_weight = base_weight
        self.trend_penalty_weight = trend_penalty_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, trend_features=None):
        # Base cross-entropy loss
        base_loss = self.ce_loss(logits, targets)

        if trend_features is None:
            return base_loss.mean()

        # trend_features should be (batch_size, 2) with [hurst, trend_strength]
        hurst = trend_features[:, 0]
        trend_strength = trend_features[:, 1]

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        # Calculate trend alignment penalty
        # 0=short, 1=hold, 2=long
        # Penalty when: (trending_up AND pred=short) OR (trending_down AND pred=long)
        trending_up = (hurst > 0.55) & (trend_strength > 2)
        trending_down = (hurst > 0.55) & (trend_strength < -2)

        counter_trend_penalty = torch.zeros_like(base_loss)

        # Penalize counter-trend trades
        counter_trend_penalty[trending_up & (preds == 0)] = self.trend_penalty_weight  # SHORT in uptrend
        counter_trend_penalty[trending_down & (preds == 2)] = self.trend_penalty_weight  # LONG in downtrend

        # Reward with-trend trades (reduce loss)
        counter_trend_penalty[trending_up & (preds == 2)] = -self.trend_penalty_weight * 0.5  # LONG in uptrend
        counter_trend_penalty[trending_down & (preds == 0)] = -self.trend_penalty_weight * 0.5  # SHORT in downtrend

        # Combined loss
        total_loss = base_loss + counter_trend_penalty

        return total_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss - prevents overconfident predictions
    Smoothing distributes some probability mass to non-target classes
    Improves calibration and generalization
    """
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)

        # Create smooth target distribution
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute loss
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Apply class weights if provided
        if self.weight is not None:
            weight_tensor = self.weight[target]
            loss = loss * weight_tensor

        return loss.mean()


class ProfitWeightedLoss(nn.Module):
    """
    Profit-Weighted Loss - weights predictions by their profit potential
    Predictions for big moves are more important than small moves
    """
    def __init__(self, weight=None):
        super(ProfitWeightedLoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, targets, price_changes):
        # Base cross entropy loss
        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.weight)

        # Weight by absolute price change magnitude
        # Bigger moves = more important to get right
        move_weights = torch.abs(price_changes)
        move_weights = move_weights / (move_weights.mean() + 1e-8)  # Normalize

        # Apply weights
        weighted_loss = (ce_loss * move_weights).mean()
        return weighted_loss


def simulate_trade(df, entry_idx, sl_price, tp_price, max_bars, is_long):
    """
    Simulate a single trade from entry to exit
    Returns dict with pnl, bars_held, and exit_reason
    """
    if entry_idx >= len(df) - 1:
        return {'pnl': 0.0, 'bars_held': 0, 'exit': 'INVALID'}

    entry_price = df['close'].iloc[entry_idx]

    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        bar_high = df['high'].iloc[i]
        bar_low = df['low'].iloc[i]

        if is_long:
            # Check SL first (conservative - worst case)
            if bar_low <= sl_price:
                return {'pnl': (sl_price - entry_price), 'bars_held': i - entry_idx, 'exit': 'SL'}
            # Then check TP
            if bar_high >= tp_price:
                return {'pnl': (tp_price - entry_price), 'bars_held': i - entry_idx, 'exit': 'TP'}
        else:
            # SHORT trade
            if bar_high >= sl_price:
                return {'pnl': (entry_price - sl_price), 'bars_held': i - entry_idx, 'exit': 'SL'}
            if bar_low <= tp_price:
                return {'pnl': (entry_price - tp_price), 'bars_held': i - entry_idx, 'exit': 'TP'}

    # Time stop - exit at market
    exit_price = df['close'].iloc[min(entry_idx + max_bars, len(df) - 1)]
    pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return {'pnl': pnl, 'bars_held': max_bars, 'exit': 'TIME'}


def generate_label_with_simulation(df, entry_idx, sl_pct=0.005, tp_pct=0.01, max_bars=10,
                                   commission=2.50, slippage_ticks=0.25, tick_value=5.0):
    """
    Generate trade label by simulating actual trades with SL/TP
    Returns: label (0=SHORT, 1=HOLD, 2=LONG), expected_return, bars_held
    """
    if entry_idx >= len(df) - max_bars:
        return 1, 0.0, 0  # HOLD - not enough future data

    entry_price = df['close'].iloc[entry_idx]

    # Try LONG trade
    long_sl = entry_price * (1 - sl_pct)
    long_tp = entry_price * (1 + tp_pct)
    long_result = simulate_trade(df, entry_idx, long_sl, long_tp, max_bars, is_long=True)

    # Try SHORT trade
    short_sl = entry_price * (1 + sl_pct)
    short_tp = entry_price * (1 - tp_pct)
    short_result = simulate_trade(df, entry_idx, short_sl, short_tp, max_bars, is_long=False)

    # Calculate expected returns (accounting for costs)
    slippage_cost = 2 * slippage_ticks * tick_value  # Round trip

    long_pnl = (long_result['pnl'] * tick_value / 0.25) - commission - slippage_cost  # Convert to dollars
    short_pnl = (short_result['pnl'] * tick_value / 0.25) - commission - slippage_cost

    # Decision logic
    min_edge = 10.0  # Need at least $10 edge to trade

    # More balanced decision logic - ensure both directions get fair chance
    if long_pnl > min_edge and long_pnl > short_pnl:
        return 2, long_pnl, long_result['bars_held']  # LONG
    elif short_pnl > min_edge and short_pnl > long_pnl:
        return 0, short_pnl, short_result['bars_held']  # SHORT
    else:
        return 1, 0.0, 0  # HOLD - no clear edge


def calculate_realtime_order_flow(df):
    """
    Enhanced order flow features for real-time trading
    ONLY use when bid/ask data is available from live market
    """
    bid_vol = df['bid_volume'].values if 'bid_volume' in df.columns else np.zeros(len(df))
    ask_vol = df['ask_volume'].values if 'ask_volume' in df.columns else np.zeros(len(df))

    # Check if we have real bid/ask data (not zeros)
    if bid_vol.sum() == 0 and ask_vol.sum() == 0:
        # Return zero features if no bid/ask data
        n = len(df)
        return {
            'delta': np.zeros(n),
            'cumulative_delta': np.zeros(n),
            'delta_divergence': np.zeros(n),
            'aggressive_buy_ratio': np.zeros(n),
            'order_flow_imbalance': np.zeros(n),
            'cum_delta_momentum': np.zeros(n),
            'cum_delta_roc': np.zeros(n),  # Added for new features
            'delta_acceleration': np.zeros(n)  # Added for new features
        }

    n = len(df)
    features = {}

    # 1. Delta (buy pressure - sell pressure)
    delta = ask_vol - bid_vol
    total_vol = ask_vol + bid_vol
    features['delta'] = delta / (total_vol + 1e-8)

    # 2. Cumulative Delta (running total)
    cum_delta = np.cumsum(delta)
    features['cumulative_delta'] = cum_delta

    # 3. Delta Divergence (price up but delta down = bearish)
    price_direction = np.zeros(n)
    delta_direction = np.zeros(n)

    for i in range(1, n):
        price_direction[i] = np.sign(df['close'].iloc[i] - df['close'].iloc[i-1])
        delta_direction[i] = np.sign(delta[i] - delta[i-1])

    divergence = (price_direction != delta_direction).astype(float)
    features['delta_divergence'] = divergence

    # 4. Aggressive Buy Ratio
    features['aggressive_buy_ratio'] = ask_vol / (total_vol + 1e-8)

    # 5. Order Flow Imbalance
    imbalance = (ask_vol - bid_vol) / (total_vol + 1e-8)
    features['order_flow_imbalance'] = imbalance

    # 6. Cumulative Delta Momentum (deviation from moving average)
    window = 20
    cum_delta_ma = pd.Series(cum_delta).rolling(window, min_periods=1).mean().values
    features['cum_delta_momentum'] = cum_delta - cum_delta_ma

    # 7. Cumulative Delta Rate of Change (momentum acceleration)
    cum_delta_roc = np.zeros(n)
    for i in range(1, n):
        if abs(cum_delta[i-1]) > 1e-8:
            cum_delta_roc[i] = (cum_delta[i] - cum_delta[i-1]) / (abs(cum_delta[i-1]) + 1)
    features['cum_delta_roc'] = cum_delta_roc

    # 8. Delta Acceleration (second derivative)
    delta_accel = np.zeros(n)
    for i in range(2, n):
        delta_accel[i] = delta[i] - 2*delta[i-1] + delta[i-2]
    features['delta_acceleration'] = delta_accel

    return features


class TradingRNN(nn.Module):
    """
    Enhanced LSTM-based RNN with attention mechanism for predicting trade signals

    Features (86 total with all enhancements):
    - OHLC (4) + Hurst (2) + ATR (1) = 7
    - Price Momentum (2) + Price Patterns (15) + Deviation Features (13) = 30
    - Order Flow (1) = 1
    - Time-of-Day (5) + Microstructure (5) + Volatility Regime (4) = 14
    - Multi-Timeframe (9) = 9
    - Price Change Magnitude (1) = 1
    Total: 62 features (removed 25 redundant/zero-value features from original 87)

    Architecture improvements:
    - Increased hidden size: 64 → 128
    - Increased layers: 2 → 3
    - Added self-attention mechanism
    - Better regularization
    """
    def __init__(self, input_size=86, hidden_size=128, num_layers=3, output_size=3):
        super(TradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with dropout between layers (increased for better regularization)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # Self-attention mechanism (allows model to focus on important time steps)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layers with increased capacity and stronger regularization
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.4)  # Increased from 0.3
        self.dropout2 = nn.Dropout(0.3)  # Increased from 0.2
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, return_attention=False):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Apply self-attention to lstm output
        # This lets the model learn which time steps are most important
        attended_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Add residual connection and layer norm
        attended_out = self.layer_norm(lstm_out + attended_out)

        # Take the last output from the sequence
        last_output = attended_out[:, -1, :]

        # Pass through fully connected layers with dropout
        out = self.relu(self.fc1(last_output))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)  # No activation - softmax applied in loss function

        if return_attention:
            return out, attn_weights
        return out


class ImprovedTradingRNN(nn.Module):
    """
    Improved LSTM-based RNN with optimized architecture

    Key improvements over TradingRNN:
    - Reduced dropout from 0.5 to 0.3 (less aggressive regularization)
    - Added batch normalization for training stability
    - Added learnable positional encoding for better time awareness
    - Deeper FC layers (4 layers instead of 3)
    - Optimized for 87 input features (pure price action) and sequence_length=15
    """
    def __init__(self, input_size=87, hidden_size=128, num_layers=2, output_size=3, sequence_length=15):
        super(ImprovedTradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length  # Use passed parameter instead of hardcoded value

        # IMPROVED: Reduced dropout from 0.5 to 0.3
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

        # IMPROVED: Add learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.sequence_length, hidden_size) * 0.02)

        # Self-attention mechanism (keep from original - this works well)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # IMPROVED: Deeper FC layers with batch normalization
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_size)

        # IMPROVED: Reduced dropout to 0.25 (was 0.4 and 0.3)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, return_attention=False):
        # x shape: (batch, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # IMPROVED: Add positional encoding (dynamically adjust if sequence length differs)
        if seq_len == self.sequence_length:
            lstm_out = lstm_out + self.positional_encoding
        else:
            # Handle variable sequence lengths during inference
            pos_enc = self.positional_encoding[:, :seq_len, :]
            lstm_out = lstm_out + pos_enc

        # Apply self-attention
        attended_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Add residual connection and layer norm
        attended_out = self.layer_norm(lstm_out + attended_out)

        # Take the last output from the sequence
        last_output = attended_out[:, -1, :]

        # IMPROVED: Pass through deeper FC layers with batch norm
        out = self.relu(self.bn1(self.fc1(last_output)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out)  # No activation - softmax applied in loss function

        if return_attention:
            return out, attn_weights
        return out


class SimplifiedTradingRNN(nn.Module):
    """
    PERFORMANCE OPTIMIZATION: Simplified architecture for faster training

    Key optimizations:
    - Reduced FC layers from 4 to 3 (128 → 64 → 3)
    - Reduced dropout from 0.25 to 0.2
    - ~20% fewer parameters than ImprovedTradingRNN
    - Faster forward pass, less overfitting risk
    """
    def __init__(self, input_size=97, hidden_size=128, num_layers=2, output_size=3):
        super(SimplifiedTradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = 15

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.sequence_length, hidden_size) * 0.02)

        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # SIMPLIFIED: Only 3 FC layers instead of 4
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_size)

        # Reduced dropout
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Add positional encoding
        if seq_len == self.sequence_length:
            lstm_out = lstm_out + self.positional_encoding
        else:
            pos_enc = self.positional_encoding[:, :seq_len, :]
            lstm_out = lstm_out + pos_enc

        # Apply self-attention
        attended_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Add residual connection and layer norm
        attended_out = self.layer_norm(lstm_out + attended_out)

        # Take the last output
        last_output = attended_out[:, -1, :]

        # SIMPLIFIED: Pass through 2 FC layers instead of 4
        out = self.relu(self.bn1(self.fc1(last_output)))
        out = self.dropout(out)
        out = self.fc2(out)

        if return_attention:
            return out, attn_weights
        return out


class GRUTradingModel(nn.Module):
    """
    PERFORMANCE OPTIMIZATION: GRU-based architecture

    Key advantages:
    - GRU has 20-30% fewer parameters than LSTM
    - Faster training (simpler gating mechanism)
    - Often performs similarly to LSTM on financial data
    - Better for scenarios with limited training data
    """
    def __init__(self, input_size=97, hidden_size=128, num_layers=2, output_size=3):
        super(GRUTradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = 15

        # GRU instead of LSTM (faster, fewer parameters)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.sequence_length, hidden_size) * 0.02)

        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # FC layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape

        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Add positional encoding
        if seq_len == self.sequence_length:
            gru_out = gru_out + self.positional_encoding
        else:
            pos_enc = self.positional_encoding[:, :seq_len, :]
            gru_out = gru_out + pos_enc

        # Apply self-attention
        attended_out, attn_weights = self.attention(gru_out, gru_out, gru_out)

        # Add residual connection and layer norm
        attended_out = self.layer_norm(gru_out + attended_out)

        # Take the last output
        last_output = attended_out[:, -1, :]

        # FC layers
        out = self.relu(self.bn1(self.fc1(last_output)))
        out = self.dropout(out)
        out = self.fc2(out)

        if return_attention:
            return out, attn_weights
        return out


class HybridTradingModel(nn.Module):
    """
    Hybrid LSTM+Transformer architecture for improved pattern recognition
    Combines LSTM for sequential processing with Transformer for long-range dependencies
    """
    def __init__(self, input_size=85, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM branch for temporal features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)

        # Transformer encoder for pattern recognition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Attention to combine LSTM and Transformer outputs
        self.fusion_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )

        # Price level memory (learned embedding of S/R levels)
        self.level_memory = nn.Parameter(torch.randn(20, hidden_size))

        # Final classification layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 for LSTM+Transformer concat
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # LSTM path
        lstm_out, _ = self.lstm(x)

        # Transformer path
        transformer_out = self.transformer(x)

        # Combine using attention
        combined, _ = self.fusion_attention(
            lstm_out, transformer_out, transformer_out
        )
        combined = self.layer_norm(combined)

        # Take last time step
        last_hidden = combined[:, -1, :]

        # Also attend to price level memory
        level_context, _ = self.fusion_attention(
            last_hidden.unsqueeze(1),
            self.level_memory.unsqueeze(0).expand(x.size(0), -1, -1),
            self.level_memory.unsqueeze(0).expand(x.size(0), -1, -1)
        )
        level_context = level_context.squeeze(1)

        # Concatenate both contexts
        features = torch.cat([last_hidden, level_context], dim=1)

        # Classification
        out = F.relu(self.fc1(features))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out


class MultiTaskTradingModel(nn.Module):
    """
    Multi-task learning model that predicts:
    1. Direction (SHORT/HOLD/LONG)
    2. Magnitude (expected price change)
    3. Time to target (bars until TP/SL hit)
    """
    def __init__(self, input_size=85, hidden_size=128):
        super().__init__()

        # Shared feature extractor (LSTM with attention)
        self.lstm = nn.LSTM(input_size, hidden_size, 3,
                           batch_first=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Task-specific heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # SHORT/HOLD/LONG
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Predicted price change %
        )

        self.time_to_target_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Expected bars to target
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.layer_norm(lstm_out + attended)  # Residual connection
        features = attended[:, -1, :]

        direction = self.direction_head(features)
        magnitude = self.magnitude_head(features)
        time_to_target = self.time_to_target_head(features)

        return direction, magnitude, time_to_target


class AdaptiveConfidenceThresholds:
    """
    Adaptive confidence thresholds based on market regime and time-of-day
    """
    def __init__(self):
        # Base thresholds by market regime (AGGRESSIVELY REDUCED - Emergency Mode)
        # Model was trained with 40% HOLD bias, so we need very low thresholds
        self.regime_thresholds = {
            'trending_high_vol': 0.40,    # Easier to predict (was 0.50 -> now 0.40)
            'trending_normal': 0.42,      # (was 0.52 -> now 0.42)
            'ranging_normal': 0.48,       # Harder to predict (was 0.60 -> now 0.48)
            'ranging_low_vol': 0.52,      # Very noisy (was 0.63 -> now 0.52)
            'high_vol_chaos': 0.55,       # Extremely difficult (was 0.68 -> now 0.55)
            'transitional': 0.45,         # (was 0.58 -> now 0.45)
            'unknown': 0.42,              # Default (was 0.55 -> now 0.42)
        }

        # Time-of-day multipliers (REDUCED for more signals)
        self.time_multipliers = {
            'open': 1.03,      # 3% higher threshold (was 1.10)
            'mid': 1.0,        # Normal
            'lunch': 1.05,     # 5% higher (was 1.15)
            'close': 1.02,     # 2% higher (was 1.05)
        }

    def get_time_period(self, timestamp):
        """Determine time period of trading day"""
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        hour = timestamp.hour
        minute = timestamp.minute
        time_decimal = hour + minute / 60.0

        # Market hours: 9:30 - 16:00 ET
        if time_decimal < 10.0:  # Before 10am
            return 'open'
        elif time_decimal < 11.5:  # 10am - 11:30am
            return 'mid'
        elif time_decimal < 13.5:  # 11:30am - 1:30pm
            return 'lunch'
        elif time_decimal < 15.5:  # 1:30pm - 3:30pm
            return 'mid'
        else:  # After 3:30pm
            return 'close'

    def get_threshold(self, regime, timestamp, recent_accuracy=None):
        """
        Calculate adaptive confidence threshold

        Args:
            regime: Market regime string
            timestamp: Current bar timestamp
            recent_accuracy: Optional recent model accuracy (0-1)

        Returns:
            Confidence threshold (0-1)
        """
        # Get base threshold for regime
        base_threshold = self.regime_thresholds.get(regime, 0.65)

        # Get time-of-day multiplier
        time_period = self.get_time_period(timestamp)
        time_multiplier = self.time_multipliers.get(time_period, 1.0)

        # Adjust for recent accuracy if provided
        if recent_accuracy is not None:
            if recent_accuracy < 0.45:  # Model struggling
                accuracy_penalty = 1.2
            elif recent_accuracy > 0.60:  # Model performing well
                accuracy_penalty = 0.9
            else:
                accuracy_penalty = 1.0
        else:
            accuracy_penalty = 1.0

        # Calculate final threshold
        final_threshold = base_threshold * time_multiplier * accuracy_penalty

        # Clamp to reasonable range
        return min(0.85, max(0.50, final_threshold))


class TradingModel:
    """
    Wrapper class for training and prediction with state management
    """
    def __init__(self, sequence_length=15, model_path='models/trading_model.pth'):
        # PHASE 2 UPDATE: Changed input_size from 105 → 87 (removed 18 indicator features)
        # Pure price action features only (no ATR, RSI, MACD, etc.)
        # sequence_length=15, num_layers=2 for better generalization
        self.sequence_length = sequence_length
        self.model = ImprovedTradingRNN(
            input_size=87,
            hidden_size=128,
            num_layers=2,
            output_size=3,
            sequence_length=sequence_length  # Pass sequence_length to model architecture
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compiled = False  # Track compilation state

        # Adaptive confidence thresholds
        self.adaptive_thresholds = AdaptiveConfidenceThresholds()

        # Track recent predictions for accuracy calculation
        self.recent_predictions = []
        self.max_recent_predictions = 50

        # ROUND 2 IMPROVEMENT: Confidence calibration support
        self.calibration = None
        if ADVANCED_MODULES_AVAILABLE:
            try:
                self.calibration = EnsembleCalibration()
                calib_path = Path('models/calibration')
                if calib_path.exists():
                    self.calibration.load(str(calib_path))
                    print("✓ Loaded confidence calibration")
            except Exception as e:
                print(f"⚠️  Could not load calibration: {e}")
                self.calibration = None

        print(f"Using device: {self.device}")
        print(f"Model: ImprovedTradingRNN with {self.model.count_parameters():,} parameters")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Input features: 105 (including 6 trend boost features)")
        print(f"Calibration: {'Enabled' if self.calibration else 'Disabled'}")
        self.model.to(self.device)

        # DO NOT auto-load existing model - model should always start as untrained
        # User must explicitly train the model after server starts
        # if self.model_path.exists():
        #     self.load_model()

        print(f"\n⚠️  MODEL INITIALIZATION: is_trained = {self.is_trained}")
        print(f"   Model file exists: {self.model_path.exists()}")
        print(f"   Auto-load: DISABLED (model must be trained explicitly)\n")

        # PERFORMANCE OPTIMIZATION: torch.compile() disabled due to compatibility issues with LSTM
        # if hasattr(torch, 'compile') and not self.compiled:
        #     try:
        #         print("Compiling model with torch.compile()...")
        #         self.model = torch.compile(self.model, mode='reduce-overhead')
        #         self.compiled = True
        #         print("✓ Model compiled successfully (expect 10-20% speedup)")
        #     except Exception as e:
        #         print(f"⚠️  torch.compile() failed: {e}")
        #         print("   Continuing with uncompiled model...")

        # Historical data storage (multi-timeframe support)
        self.historical_data = None
        self.historical_data_secondary = None  # Secondary timeframe (e.g., 5-min)

        # PERFORMANCE OPTIMIZATION: Cache computed features to avoid recomputation
        self._feature_cache = None
        self._feature_cache_key = None
        self._cache_length = 0

        # PERFORMANCE OPTIMIZATION: Hurst calculation cache
        self._hurst_cache = []
        self._last_hurst_H = 0.5
        self._last_hurst_C = 1.0

    def _validate_data(self, df):
        """Validate input data for NaN, inf, and required columns"""
        required_cols = ['open', 'high', 'low', 'close']

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        if df[required_cols].isnull().any().any():
            raise ValueError("Input data contains NaN values")

        if np.isinf(df[required_cols].values).any():
            raise ValueError("Input data contains infinite values")

        if len(df) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} bars, got {len(df)}")

        # Add daily P&L columns if they don't exist (for backward compatibility)
        if 'dailyPnL' not in df.columns:
            df['dailyPnL'] = 0.0
        if 'dailyGoal' not in df.columns:
            df['dailyGoal'] = 500.0
        if 'dailyMaxLoss' not in df.columns:
            df['dailyMaxLoss'] = 250.0

        # Add volume columns if they don't exist (for backward compatibility)
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        if 'bid_volume' not in df.columns:
            df['bid_volume'] = 0.0
        if 'ask_volume' not in df.columns:
            df['ask_volume'] = 0.0

    @timing_decorator
    def prepare_data(self, df, df_secondary=None, fit_scaler=False, adaptive_threshold=True, use_triple_barrier=False):
        """
        Prepare data for training with enhanced features including multi-timeframe (OPTIMIZED VERSION)
        Creates sequences and labels based on future price movement

        Args:
            df: Primary timeframe DataFrame with OHLC data
            df_secondary: Optional secondary timeframe DataFrame (e.g., 5-min)
            fit_scaler: If True, fit the scaler. If False, only transform (use False for inference)
            adaptive_threshold: If True, calculate threshold based on data volatility
            use_triple_barrier: If True, use Triple Barrier Method for labels (ROUND 2 IMPROVEMENT)
        """
        self._validate_data(df)

        # Extract OHLC features
        ohlc = df[['open', 'high', 'low', 'close']].values

        # Calculate adaptive threshold based on price volatility
        if adaptive_threshold or not hasattr(self, 'signal_threshold'):
            price_changes = np.diff(ohlc[:, 3]) / ohlc[:-1, 3] * 100  # % changes
            volatility = np.std(price_changes)

            # Threshold = 0.4x volatility (REDUCED from 0.5x for more signals)
            # This ensures ~55-65% of data is not HOLD (was 60-70%)
            self.signal_threshold = max(0.01, volatility * 0.4)

            # Only log during training
            if fit_scaler:
                print(f"Data volatility: {volatility:.4f}%")
                print(f"Adaptive threshold set to: {self.signal_threshold:.4f}%")

        # PERFORMANCE OPTIMIZATION: Calculate Hurst exponent with caching
        # Only recalculate every 10 bars instead of every bar (massive speedup)
        hurst_H_values = []
        hurst_C_values = []

        for i in range(len(df)):
            if i < 100:  # Need minimum 100 bars for hurst library
                H, c = 0.5, 1.0
            else:
                # Only recalculate every 10 bars (10x speedup for real-time)
                if i % 10 == 0 or i == len(df) - 1:  # Always calc on last bar
                    prices = df['close'].iloc[i-99:i+1].values
                    H, c = calculate_hurst_exponent(prices)
                    self._last_hurst_H = H
                    self._last_hurst_C = c
                else:
                    # Reuse last calculated value
                    H, c = self._last_hurst_H, self._last_hurst_C

            hurst_H_values.append(H)
            hurst_C_values.append(c)

        # Log Hurst statistics for the dataset (only during training, not inference)
        if fit_scaler:  # Only log during training
            hurst_valid = [h for h in hurst_H_values if h != 0.5]
            if len(hurst_valid) > 0:
                avg_hurst = np.mean(hurst_valid)
                print(f"Hurst exponent statistics:")
                print(f"  Mean H: {avg_hurst:.4f}")
                print(f"  Min H: {min(hurst_valid):.4f}")
                print(f"  Max H: {max(hurst_valid):.4f}")
                if avg_hurst > 0.5:
                    print(f"  → Overall TRENDING (persistent) market")
                elif avg_hurst < 0.5:
                    print(f"  → Overall MEAN-REVERTING (anti-persistent) market")
                else:
                    print(f"  → Overall RANDOM WALK market")

        # ============================================================================
        # REMOVED: All lagging indicator calculations for pure price action migration
        # ============================================================================
        # OLD CODE (commented out):
        # atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values)
        # rsi = calculate_rsi(df['close'].values)
        # rsi_divergence = calculate_rsi_divergence(df['close'].values, rsi)
        # macd_line, macd_signal, macd_histogram = calculate_macd(df['close'].values)
        # vwma_dev = calculate_vwma_deviation(df['close'].values, volume)
        # gk_volatility = calculate_garman_klass_volatility(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        # price_impact = calculate_price_impact(df['close'].values, volume)
        # vwpc = calculate_volume_weighted_price_change(df['close'].values, volume)

        # NEW: Calculate all PURE PRICE ACTION features in one call
        from core.price_action_features import prepare_price_action_data
        df_with_pa_features = prepare_price_action_data(df)

        # Calculate all advanced price features (market structure, order flow, patterns)
        price_features = calculate_price_features(df)

        # Calculate candlestick patterns
        candlestick_patterns = detect_candlestick_patterns(df)

        # Calculate support/resistance features
        sr_features = calculate_sr_features(df)

        # Calculate volume profile features
        volume_profile_features = calculate_volume_profile(df)

        # Calculate order flow features
        order_flow_features = calculate_order_flow_features(df)

        # Calculate enhanced real-time order flow (only if bid/ask data available)
        realtime_order_flow = calculate_realtime_order_flow(df)

        # Calculate time-of-day features
        time_features = calculate_time_features(df)

        # Calculate microstructure features
        microstructure_features = calculate_microstructure_features(df)

        # Calculate volatility regime features
        volatility_regime_features = calculate_volatility_regime_features(df)

        # Calculate and align secondary timeframe features (if available)
        df_secondary_to_use = df_secondary if df_secondary is not None else self.historical_data_secondary
        secondary_features = align_secondary_to_primary(df, df_secondary_to_use)

        # Combine all features (62 total):
        # OHLC (4) + Hurst (2) + ATR (1) + Price Features (18) + Deviation Features (23) = 48
        # Wait, let me recount: 4+2+1+18+23 = 48, but we said 47...
        # Actually: Original 24 + Deviation 23 = 47 ✓

        # Debug: Check array sizes before stacking
        n_bars = len(ohlc)
        assert len(hurst_H_values) == n_bars, f"Hurst H mismatch: {len(hurst_H_values)} vs {n_bars}"
        assert len(hurst_C_values) == n_bars, f"Hurst C mismatch: {len(hurst_C_values)} vs {n_bars}"
        # REMOVED: assert len(atr) == n_bars (ATR removed in Phase 2)

        # Calculate price change magnitude (recent volatility indicator)
        price_change_magnitude = np.zeros(n_bars)
        for i in range(5, n_bars):
            recent_changes = np.abs(np.diff(ohlc[i-5:i+1, 3]) / ohlc[i-5:i, 3])
            price_change_magnitude[i] = np.mean(recent_changes)

        # REMOVED: Feature Interaction Terms (indicators-based)
        # vol_volume_interaction = volatility_regime_features['volatility_regime'] * volatility_regime_features['volume_regime']
        # trend_tf_interaction = price_features['trend_strength'] * secondary_features['tf2_trend_direction']
        # explosive_signal = price_features['position_in_range'] * price_features['std_dev_20']

        # REMOVED: Lagged Features for indicators
        # rsi_lag1 = np.roll(rsi, 1)
        # rsi_lag2 = np.roll(rsi, 2)
        # macd_hist_lag1 = np.roll(macd_histogram, 1)
        # macd_hist_lag2 = np.roll(macd_histogram, 2)

        # KEPT: Lagged Features for PURE PRICE ACTION
        # Velocity lag 1 and 2
        velocity_lag1 = np.roll(price_features['velocity'], 1)
        velocity_lag1[0] = price_features['velocity'][0]
        velocity_lag2 = np.roll(price_features['velocity'], 2)
        velocity_lag2[:2] = price_features['velocity'][0]

        # Cumulative delta lag 1 and 2 (when available)
        cum_delta_lag1 = np.roll(realtime_order_flow['cumulative_delta'], 1)
        cum_delta_lag1[0] = realtime_order_flow['cumulative_delta'][0]
        cum_delta_lag2 = np.roll(realtime_order_flow['cumulative_delta'], 2)
        cum_delta_lag2[:2] = realtime_order_flow['cumulative_delta'][0]

        # REMOVED: Daily P&L features (always 0 during training - 3 features removed)
        # REMOVED: 6 order flow features (always 0 during training)
        # REMOVED: 10 redundant deviation features (windows 5 and 10)
        # REMOVED: tf2_delta (always 0)
        # Total removed: 23 features (87 → 64)

        # Debug: Validate all feature array lengths before stacking
        feature_arrays = {
            'ohlc': ohlc,
            'hurst_H': hurst_H_values,
            'hurst_C': hurst_C_values,
            # REMOVED: 'atr': atr (indicator removed in Phase 2)
            'velocity': price_features['velocity'],
            'acceleration': price_features['acceleration'],
            'range_ratio': price_features['range_ratio'],
            'wick_ratio': price_features['wick_ratio'],
            'gap_up': price_features['gap_up'],
            'gap_down': price_features['gap_down'],
            'gap_filled': price_features['gap_filled'],
            'swing_high': price_features['swing_high'],
            'swing_low': price_features['swing_low'],
            'bars_since_swing_high': price_features['bars_since_swing_high'],
            'bars_since_swing_low': price_features['bars_since_swing_low'],
            'skewness': price_features['skewness'],
            'kurtosis': price_features['kurtosis'],
            'position_in_range': price_features['position_in_range'],
            'higher_highs': price_features['higher_highs'],
            'lower_lows': price_features['lower_lows'],
            'trend_strength': price_features['trend_strength'],
            'mean_dev_20': price_features['mean_dev_20'],
            'mean_dev_50': price_features['mean_dev_50'],
            'median_dev_20': price_features['median_dev_20'],
            'median_dev_50': price_features['median_dev_50'],
            'std_dev_20': price_features['std_dev_20'],
            'std_dev_50': price_features['std_dev_50'],
            'z_score_20': price_features['z_score_20'],
            'z_score_50': price_features['z_score_50'],
            'bb_width_20': price_features['bb_width_20'],
            'bb_width_50': price_features['bb_width_50'],
            'vol_acceleration': price_features['vol_acceleration'],
            'high_deviation': price_features['high_deviation'],
            'low_deviation': price_features['low_deviation'],
            'volume_ratio': order_flow_features['volume_ratio'],
            'hour_of_day': time_features['hour_of_day'],
            'minutes_into_session': time_features['minutes_into_session'],
            'minutes_to_close': time_features['minutes_to_close'],
            'is_opening_period': time_features['is_opening_period'],
            'is_closing_period': time_features['is_closing_period'],
            'volume_surge': microstructure_features['volume_surge'],
            'price_accel_micro': microstructure_features['price_acceleration'],
            'effective_spread': microstructure_features['effective_spread'],
            'large_print': microstructure_features['large_print'],
            'vwap_deviation': microstructure_features['vwap_deviation'],
            'volatility_regime': volatility_regime_features['volatility_regime'],
            'parkinson_volatility': volatility_regime_features['parkinson_volatility'],
            'volume_regime': volatility_regime_features['volume_regime'],
            'trending_score': volatility_regime_features['trending_score'],
            'tf2_close': secondary_features['tf2_close'],
            'tf2_close_change': secondary_features['tf2_close_change'],
            'tf2_high_low_range': secondary_features['tf2_high_low_range'],
            'tf2_volume': secondary_features['tf2_volume'],
            'tf2_position_in_bar': secondary_features['tf2_position_in_bar'],
            'tf2_trend_direction': secondary_features['tf2_trend_direction'],
            'tf2_momentum': secondary_features['tf2_momentum'],
            'tf2_volatility': secondary_features['tf2_volatility'],
            'tf2_alignment_score': secondary_features['tf2_alignment_score'],
            'price_change_magnitude': price_change_magnitude
        }

        # Check for length mismatches
        for name, arr in feature_arrays.items():
            # Handle both lists and numpy arrays
            if isinstance(arr, list):
                arr_len = len(arr)
            elif hasattr(arr, 'shape'):
                arr_len = arr.shape[0] if len(arr.shape) >= 1 else 1
            else:
                arr_len = len(arr) if hasattr(arr, '__len__') else 0

            if arr_len != n_bars:
                raise ValueError(f"Feature '{name}' has length {arr_len}, expected {n_bars}")

        # ============================================================================
        # PURE PRICE ACTION FEATURES - REBUILT FROM SCRATCH
        # Removed all lagging indicators (ATR, RSI, MACD, etc.)
        # ============================================================================

        features = np.column_stack([
            # Core features: OHLC + Hurst = 6 (removed ATR)
            ohlc,                                    # 4
            hurst_H_values,                          # 1
            hurst_C_values,                          # 1
            # REMOVED: atr (lagging indicator)
            # Price momentum: 2
            price_features['velocity'],              # 1
            price_features['acceleration'],          # 1
            # Price patterns: 12
            price_features['range_ratio'],           # 1
            price_features['wick_ratio'],            # 1
            price_features['gap_up'],                # 1
            price_features['gap_down'],              # 1
            price_features['gap_filled'],            # 1
            price_features['swing_high'],            # 1
            price_features['swing_low'],             # 1
            price_features['bars_since_swing_high'], # 1
            price_features['bars_since_swing_low'],  # 1
            price_features['skewness'],              # 1
            price_features['kurtosis'],              # 1
            price_features['position_in_range'],     # 1
            price_features['higher_highs'],          # 1
            price_features['lower_lows'],            # 1
            price_features['trend_strength'],        # 1
            # Deviation features - REDUCED: only window 20 (5 metrics, removed 50-period = 5 features removed)
            price_features['mean_dev_20'],           # 1
            # REMOVED: mean_dev_50 (redundant)
            price_features['median_dev_20'],         # 1
            # REMOVED: median_dev_50 (redundant)
            price_features['std_dev_20'],            # 1
            # REMOVED: std_dev_50 (redundant)
            price_features['z_score_20'],            # 1
            # REMOVED: z_score_50 (redundant)
            price_features['bb_width_20'],           # 1
            # REMOVED: bb_width_50 (redundant)
            # Additional deviation features: 3
            price_features['vol_acceleration'],      # 1
            price_features['high_deviation'],        # 1
            price_features['low_deviation'],         # 1
            # Order flow - REDUCED: only volume_ratio (1 feature, removed 6)
            order_flow_features['volume_ratio'],     # 1
            # Time-of-day features: REDUCED from 5 to 3 (removed minutes_into_session, minutes_to_close)
            time_features['hour_of_day'],            # 1
            # REMOVED: minutes_into_session (redundant with hour_of_day)
            # REMOVED: minutes_to_close (inverse of minutes_into_session)
            time_features['is_opening_period'],      # 1
            time_features['is_closing_period'],      # 1
            # Microstructure features: 5
            microstructure_features['volume_surge'], # 1
            microstructure_features['price_acceleration'], # 1
            microstructure_features['effective_spread'], # 1
            microstructure_features['large_print'],  # 1
            microstructure_features['vwap_deviation'], # 1
            # Volatility regime features: 4
            volatility_regime_features['volatility_regime'], # 1
            volatility_regime_features['parkinson_volatility'], # 1
            volatility_regime_features['volume_regime'], # 1
            volatility_regime_features['trending_score'], # 1
            # Multi-timeframe features: 9 (tf2_delta removed earlier)
            secondary_features['tf2_close'],         # 1
            secondary_features['tf2_close_change'],  # 1
            secondary_features['tf2_high_low_range'], # 1
            secondary_features['tf2_volume'],        # 1
            secondary_features['tf2_position_in_bar'], # 1
            secondary_features['tf2_trend_direction'], # 1
            secondary_features['tf2_momentum'],      # 1
            secondary_features['tf2_volatility'],    # 1
            secondary_features['tf2_alignment_score'], # 1
            # Candlestick patterns: REDUCED from 9 to 7 (removed pin_bar_bull, pin_bar_bear)
            candlestick_patterns['bullish_engulfing'], # 1
            candlestick_patterns['bearish_engulfing'], # 1
            candlestick_patterns['hammer'],            # 1
            candlestick_patterns['shooting_star'],     # 1
            candlestick_patterns['doji'],              # 1
            candlestick_patterns['inside_bar'],        # 1
            candlestick_patterns['outside_bar'],       # 1
            # REMOVED: pin_bar_bull (redundant with hammer)
            # REMOVED: pin_bar_bear (redundant with shooting_star)
            # Support/Resistance features: 4
            sr_features['dist_to_nearest_sr'],         # 1
            sr_features['nearest_sr_strength'],        # 1
            sr_features['is_near_sr'],                 # 1
            sr_features['above_sr'],                   # 1
            # Volume Profile features: 5
            volume_profile_features['dist_to_poc'],    # 1
            volume_profile_features['volume_at_price'], # 1
            volume_profile_features['above_vah'],      # 1
            volume_profile_features['below_val'],      # 1
            volume_profile_features['in_value_area'],  # 1
            # Real-time Order Flow features: 8 (added 2 new: cum_delta_roc, delta_acceleration)
            realtime_order_flow['delta'],              # 1
            realtime_order_flow['cumulative_delta'],   # 1
            realtime_order_flow['delta_divergence'],   # 1
            realtime_order_flow['aggressive_buy_ratio'], # 1
            realtime_order_flow['order_flow_imbalance'], # 1
            realtime_order_flow['cum_delta_momentum'], # 1
            realtime_order_flow['cum_delta_roc'],      # 1 (NEW)
            realtime_order_flow['delta_acceleration'], # 1 (NEW)
            # Price change magnitude: 1
            price_change_magnitude,                    # 1
            # REMOVED: RSI indicators (2 features) - lagging
            # REMOVED: MACD indicators (3 features) - double lagging
            # REMOVED: VWMA deviation, GK volatility, price impact, VWPC (4 features) - all lagging/derived
            # REMOVED: Feature interactions using indicators (3 features)
            # REMOVED: RSI and MACD lagged features (4 features)
            #
            # KEPT: Pure price action lagged features
            velocity_lag1,                             # 1
            velocity_lag2,                             # 1
            cum_delta_lag1,                            # 1
            cum_delta_lag2,                            # 1
            # TREND BOOST - Pure price action trend features (no indicators)
            np.array(hurst_H_values) * 2,              # 1 - Hurst emphasized (pure)
            price_features['trend_strength'] * 1.5,    # 1 - Trend strength emphasized (pure)
            secondary_features['tf2_trend_direction'] * 1.5,  # 1 - Multi-timeframe trend (pure)
            # REMOVED: trending_score (was ADX-based)
            # Trend momentum (new feature: trend acceleration)
            np.gradient(price_features['trend_strength']),  # 1 - Is trend strengthening?
            # Hurst * trend_strength interaction (strong when both align)
            (np.array(hurst_H_values) - 0.5) * price_features['trend_strength']  # 1
        ])

        # Only log during training
        if fit_scaler:
            # UPDATED FEATURE COUNT (Pure Price Action Migration)
            # Original: 105 features
            # Removed indicators: ATR(1) + RSI(2) + MACD(3) + Others(4) + Interactions(3) + Lagged(4) + TrendingScore(1) = 18 removed
            # New total: 105 - 18 = 87 features (PURE PRICE ACTION)
            print(f"\\n{'='*80}")
            print(f"PURE PRICE ACTION FEATURES (Indicators Removed)")
            print(f"{'='*80}")
            print(f"Total features: {features.shape[1]}")
            print(f"Expected: 87 (was 105 before removing indicators)")
            print(f"Removed: ATR, RSI, MACD, VWMA, GK Vol, Price Impact, VWPC, Indicator Interactions, Indicator Lags, ADX-based features")
            print(f"{'='*80}\\n")

        # Scale the features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # IMPORTANT: Check for NaN/inf values after scaling and replace with 0
        # This can happen with division by zero or when scaler hasn't been fitted
        nan_mask = np.isnan(features_scaled) | np.isinf(features_scaled)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            print(f"WARNING: Found {nan_count} NaN/inf values in scaled features, replacing with 0")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences with improved labeling
        X, y = [], []
        lookahead_bars = 3  # Look ahead 3 bars to reduce noise

        # ROUND 2 IMPROVEMENT: Option to use Triple Barrier Method for labels
        if use_triple_barrier and ADVANCED_MODULES_AVAILABLE:
            if fit_scaler:
                print(f"\n{'='*50}")
                print("USING TRIPLE BARRIER METHOD FOR LABELS")
                print(f"{'='*50}")

            # Generate labels using Triple Barrier Method
            labels_full = calculate_triple_barrier_labels(
                df,
                profit_taking_multiple=2.0,
                stop_loss_multiple=1.0,
                max_holding_period=10,
                min_return_threshold=0.0015
            )

            # Create sequences with pre-calculated labels
            for i in range(len(features_scaled) - self.sequence_length):
                if i + self.sequence_length < len(labels_full):
                    X.append(features_scaled[i:i + self.sequence_length])
                    y.append(labels_full[i + self.sequence_length - 1])

        else:
            # Original percentile-based labeling
            # First pass: Calculate all price changes for percentile-based thresholding
            all_price_changes = []
            for i in range(len(features_scaled) - self.sequence_length - lookahead_bars + 1):
                current_close = ohlc[i + self.sequence_length - 1, 3]

                # Use maximum move in next 3 bars (captures best opportunity)
                future_slice = ohlc[i + self.sequence_length:i + self.sequence_length + lookahead_bars]
                future_highs = future_slice[:, 1]
                future_lows = future_slice[:, 2]

                max_up_move = (np.max(future_highs) - current_close) / current_close * 100
                max_down_move = (current_close - np.min(future_lows)) / current_close * 100

                # Store the dominant move
                if max_up_move > max_down_move:
                    all_price_changes.append(max_up_move)
                else:
                    all_price_changes.append(-max_down_move)

            all_price_changes = np.array(all_price_changes)
            abs_changes = np.abs(all_price_changes)

            # Percentile-based threshold: 25% smallest moves become HOLD (IMPROVED from 0.40)
            # Lower HOLD percentage = more directional signals = higher confidence
            hold_percentage = 0.25
            percentile_threshold = np.percentile(abs_changes, hold_percentage * 100)

            if fit_scaler:
                print(f"\n{'='*50}")
                print("LABEL GENERATION STATISTICS")
                print(f"{'='*50}")
                print(f"Lookahead: {lookahead_bars} bars")
                print(f"Target HOLD percentage: {hold_percentage*100:.0f}%")
                print(f"Percentile threshold: {percentile_threshold:.4f}%")
                print(f"Price change range: [{np.min(all_price_changes):.4f}%, {np.max(all_price_changes):.4f}%]")

            # Second pass: Create sequences and labels using the threshold
            # PRIORITY 1: TREND-AWARE LABEL GENERATION
            trend_adjusted_long = 0
            trend_adjusted_short = 0
            trend_adjusted_hold = 0

            for i in range(len(features_scaled) - self.sequence_length - lookahead_bars + 1):
                X.append(features_scaled[i:i + self.sequence_length])

                current_close = ohlc[i + self.sequence_length - 1, 3]

                # PRIORITY 1: Get current trend context
                current_hurst = hurst_H_values[i + self.sequence_length - 1]
                # Get trend_strength from price_features (it's feature index around 688-696)
                # Since we don't have easy access to price_features here, calculate it
                idx = i + self.sequence_length - 1
                if idx >= 5:
                    # Simple trend strength: count higher highs vs lower lows in last 5 bars
                    recent_highs = ohlc[idx-4:idx+1, 1]
                    recent_lows = ohlc[idx-4:idx+1, 2]
                    hh_count = sum([1 for j in range(1, len(recent_highs)) if recent_highs[j] > recent_highs[j-1]])
                    ll_count = sum([1 for j in range(1, len(recent_lows)) if recent_lows[j] < recent_lows[j-1]])
                    current_trend = hh_count - ll_count  # -4 to +4
                else:
                    current_trend = 0

                # Use maximum move in next 3 bars
                future_slice = ohlc[i + self.sequence_length:i + self.sequence_length + lookahead_bars]
                future_highs = future_slice[:, 1]
                future_lows = future_slice[:, 2]

                max_up_move = (np.max(future_highs) - current_close) / current_close * 100
                max_down_move = (current_close - np.min(future_lows)) / current_close * 100

                # PRIORITY 1: Adjust threshold based on trend strength
                # In strong trends (H > 0.55 or trend_strength > 2), lower threshold for WITH-trend signals
                # In weak trends, keep threshold higher
                is_trending_up = (current_hurst > 0.55 and current_trend > 2)
                is_trending_down = (current_hurst > 0.55 and current_trend < -2)

                threshold = percentile_threshold
                if is_trending_up:
                    # Lower threshold for LONG signals in uptrends (easier to trigger)
                    up_threshold = threshold * 0.7
                    down_threshold = threshold * 1.3  # Higher threshold for SHORT (counter-trend)
                elif is_trending_down:
                    # Lower threshold for SHORT signals in downtrends
                    down_threshold = threshold * 0.7
                    up_threshold = threshold * 1.3  # Higher threshold for LONG (counter-trend)
                else:
                    up_threshold = down_threshold = threshold

                # Create label with trend-adjusted thresholds
                if max_up_move > up_threshold and max_up_move > max_down_move:
                    y.append(2)  # Long
                    if is_trending_up:
                        trend_adjusted_long += 1
                elif max_down_move > down_threshold and max_down_move > max_up_move:
                    y.append(0)  # Short
                    if is_trending_down:
                        trend_adjusted_short += 1
                else:
                    y.append(1)  # Hold
                    if is_trending_up or is_trending_down:
                        trend_adjusted_hold += 1

            if fit_scaler:
                print(f"\nTREND-AWARE LABEL ADJUSTMENTS:")
                print(f"  Labels adjusted in uptrends (easier LONG): {trend_adjusted_long}")
                print(f"  Labels adjusted in downtrends (easier SHORT): {trend_adjusted_short}")
                print(f"  Labels kept as HOLD in trends: {trend_adjusted_hold}")

        X = np.array(X)
        y = np.array(y)

        # Print class distribution
        if fit_scaler:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\nCLASS DISTRIBUTION:")
            print(f"  SHORT (0): {counts[0]:4d} ({counts[0]/len(y)*100:5.1f}%)")
            print(f"  HOLD  (1): {counts[1]:4d} ({counts[1]/len(y)*100:5.1f}%)")
            print(f"  LONG  (2): {counts[2]:4d} ({counts[2]/len(y)*100:5.1f}%)")
            print(f"{'='*50}\n")

            # PRIORITY 6: Diagnostic - Analyze labels by trend context
            print(f"{'='*50}")
            print("PRIORITY 6: LABEL DISTRIBUTION BY TREND CONTEXT")
            print(f"{'='*50}")

            # Categorize each label by its trend context
            uptrend_labels = []
            downtrend_labels = []
            ranging_labels = []

            for i in range(len(y)):
                idx = i + self.sequence_length - 1
                if idx < len(hurst_H_values):
                    current_hurst = hurst_H_values[idx]

                    # Calculate trend strength
                    if idx >= 5:
                        recent_highs = ohlc[idx-4:idx+1, 1]
                        recent_lows = ohlc[idx-4:idx+1, 2]
                        hh_count = sum([1 for j in range(1, len(recent_highs)) if recent_highs[j] > recent_highs[j-1]])
                        ll_count = sum([1 for j in range(1, len(recent_lows)) if recent_lows[j] < recent_lows[j-1]])
                        current_trend = hh_count - ll_count
                    else:
                        current_trend = 0

                    # Categorize by trend
                    if current_hurst > 0.55 and current_trend > 2:
                        uptrend_labels.append(y[i])
                    elif current_hurst > 0.55 and current_trend < -2:
                        downtrend_labels.append(y[i])
                    else:
                        ranging_labels.append(y[i])

            # Print distributions
            if len(uptrend_labels) > 0:
                uptrend_counts = np.bincount(uptrend_labels, minlength=3)
                print(f"\nUPTREND LABELS ({len(uptrend_labels)} samples):")
                print(f"  SHORT: {uptrend_counts[0]:4d} ({uptrend_counts[0]/len(uptrend_labels)*100:5.1f}%)")
                print(f"  HOLD:  {uptrend_counts[1]:4d} ({uptrend_counts[1]/len(uptrend_labels)*100:5.1f}%)")
                print(f"  LONG:  {uptrend_counts[2]:4d} ({uptrend_counts[2]/len(uptrend_labels)*100:5.1f}%)")
                if uptrend_counts[2] > uptrend_counts[0]:
                    print(f"  ✓ MORE LONG than SHORT (good for uptrend)")
                else:
                    print(f"  ⚠️  MORE SHORT than LONG (BAD for uptrend)")

            if len(downtrend_labels) > 0:
                downtrend_counts = np.bincount(downtrend_labels, minlength=3)
                print(f"\nDOWNTREND LABELS ({len(downtrend_labels)} samples):")
                print(f"  SHORT: {downtrend_counts[0]:4d} ({downtrend_counts[0]/len(downtrend_labels)*100:5.1f}%)")
                print(f"  HOLD:  {downtrend_counts[1]:4d} ({downtrend_counts[1]/len(downtrend_labels)*100:5.1f}%)")
                print(f"  LONG:  {downtrend_counts[2]:4d} ({downtrend_counts[2]/len(downtrend_labels)*100:5.1f}%)")
                if downtrend_counts[0] > downtrend_counts[2]:
                    print(f"  ✓ MORE SHORT than LONG (good for downtrend)")
                else:
                    print(f"  ⚠️  MORE LONG than SHORT (BAD for downtrend)")

            if len(ranging_labels) > 0:
                ranging_counts = np.bincount(ranging_labels, minlength=3)
                print(f"\nRANGING/WEAK TREND LABELS ({len(ranging_labels)} samples):")
                print(f"  SHORT: {ranging_counts[0]:4d} ({ranging_counts[0]/len(ranging_labels)*100:5.1f}%)")
                print(f"  HOLD:  {ranging_counts[1]:4d} ({ranging_counts[1]/len(ranging_labels)*100:5.1f}%)")
                print(f"  LONG:  {ranging_counts[2]:4d} ({ranging_counts[2]/len(ranging_labels)*100:5.1f}%)")

            print(f"{'='*50}\n")

        return X, y

    def train(self, df, epochs=100, learning_rate=0.001, batch_size=32, validation_split=0.2):
        """
        Train the model on historical data with validation split and early stopping
        """
        print(f"\n{'='*50}")
        print("TRAINING RNN MODEL")
        print(f"{'='*50}")

        # Extract OHLC for later use in metrics calculation
        ohlc = df[['open', 'high', 'low', 'close']].values

        # Prepare data with scaler fitting (multi-timeframe)
        X, y = self.prepare_data(df, df_secondary=self.historical_data_secondary, fit_scaler=True)
        print(f"Total samples: {len(X)}")
        print(f"Sequence length: {self.sequence_length}")

        if len(X) < 20:
            print("WARNING: Not enough data to train effectively!")
            return

        # Time-based validation split (critical for time series)
        # Train on first 80%, validate on last 20%
        split_idx = int(len(X) * (1 - validation_split))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        print(f"\n{'='*50}")
        print("TIME-BASED VALIDATION SPLIT")
        print(f"{'='*50}")
        print(f"Training samples: {len(X_train)} (first {(1-validation_split)*100:.0f}% of data)")
        print(f"Validation samples: {len(X_val)} (last {validation_split*100:.0f}% of data)")

        # Show class distribution for both sets
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        val_unique, val_counts = np.unique(y_val, return_counts=True)

        print(f"\nTraining set distribution:")
        print(f"  SHORT: {train_counts[0]:4d} ({train_counts[0]/len(y_train)*100:5.1f}%)")
        print(f"  HOLD:  {train_counts[1]:4d} ({train_counts[1]/len(y_train)*100:5.1f}%)")
        print(f"  LONG:  {train_counts[2]:4d} ({train_counts[2]/len(y_train)*100:5.1f}%)")

        print(f"\nValidation set distribution:")
        print(f"  SHORT: {val_counts[0]:4d} ({val_counts[0]/len(y_val)*100:5.1f}%)")
        print(f"  HOLD:  {val_counts[1]:4d} ({val_counts[1]/len(y_val)*100:5.1f}%)")
        print(f"  LONG:  {val_counts[2]:4d} ({val_counts[2]/len(y_val)*100:5.1f}%)")
        print(f"{'='*50}\n")

        # IMPROVED: Create DataLoader with data augmentation
        # Note: Augmentation is applied on-the-fly during training for better diversity
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        # PERFORMANCE OPTIMIZATION: Parallel data loading for 20-30% speedup
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,           # Parallel data loading
            pin_memory=True,         # Faster GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )

        # Validation tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Calculate class weights to handle imbalance
        # Using more aggressive weighting to prevent minority class suppression
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)

        # Method 1: Standard inverse frequency (current)
        standard_weights = torch.FloatTensor([
            total_samples / (3 * max(count, 1)) for count in class_counts
        ])

        # Method 2: Square root of inverse frequency (more aggressive for rare classes)
        sqrt_weights = torch.FloatTensor([
            np.sqrt(total_samples / (3 * max(count, 1))) for count in class_counts
        ])

        # Method 3: Effective number of samples (best for extreme imbalance)
        beta = 0.9999
        effective_nums = 1.0 - np.power(beta, class_counts)
        effective_nums = np.where(effective_nums == 0, 1, effective_nums)
        ens_weights = torch.FloatTensor((1.0 - beta) / effective_nums)

        # Use ENS weights (most aggressive) - change to standard_weights or sqrt_weights if needed
        class_weights = ens_weights.to(self.device)

        print(f"\nClass counts: Short={class_counts[0]}, Hold={class_counts[1]}, Long={class_counts[2]}")
        print(f"Class weights (ENS): Short={class_weights[0]:.4f}, Hold={class_weights[1]:.4f}, Long={class_weights[2]:.4f}")
        print(f"Weight ratios - Short/Hold: {class_weights[0]/class_weights[1]:.2f}x, Short/Long: {class_weights[0]/class_weights[2]:.2f}x")

        # PRIORITY 2: Use TrendAwareTradingLoss for trend-following optimization
        # This penalizes counter-trend trades and rewards with-trend trades
        criterion = TrendAwareTradingLoss(
            base_weight=1.0,
            trend_penalty_weight=0.5  # 50% penalty for counter-trend
        )
        print("✓ Using TrendAwareTradingLoss (PRIORITY 2: trend-aware optimization)")

        # Store trend features from X_train for use in loss calculation
        # We need hurst (feature index 4) and trend_strength (feature index varies, we'll extract it)
        # Since features are scaled, we need to extract from the unscaled data
        # For now, we'll calculate trend features on-the-fly in the training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # PERFORMANCE OPTIMIZATION: Better scheduler for faster convergence
        # CosineAnnealingWarmRestarts provides better learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.001
        )

        # IMPROVED: Early stopping on trading Sharpe ratio (not just validation loss)
        best_val_sharpe = -np.inf
        best_val_loss = float('inf')
        patience = 15  # Increased patience for Sharpe-based stopping
        patience_counter = 0

        # PERFORMANCE OPTIMIZATION: Mixed precision training for 30-50% speedup
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler() if self.device.type == 'cuda' else None
        use_amp = self.device.type == 'cuda'

        # PERFORMANCE OPTIMIZATION: Gradient accumulation for larger effective batch size
        accumulation_steps = 4  # Effective batch size = batch_size * 4

        if use_amp:
            print(f"✓ Mixed precision training enabled (expect 30-50% speedup)")
        print(f"✓ Gradient accumulation: effective batch size = {batch_size * accumulation_steps}")

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # IMPROVED: Apply data augmentation (30% probability)
                batch_X_np = batch_X.numpy()
                batch_X_aug = np.array([augment_time_series(seq) for seq in batch_X_np])
                batch_X = torch.FloatTensor(batch_X_aug).to(self.device)
                batch_y = batch_y.to(self.device)

                # PRIORITY 2: Extract trend features for TrendAwareTradingLoss
                # UPDATED: Feature indices changed after indicator removal (105 → 87 features)
                # Feature indices: hurst_H is at index 4 (after OHLC), hurst_emphasized is at index 82
                # trend_strength_emphasized is at index 83
                # We use the last position in each sequence as the current trend
                batch_trend_features = torch.stack([
                    batch_X[:, -1, 82],  # hurst_H_emphasized (already multiplied by 2, so divide)
                    batch_X[:, -1, 83]   # trend_strength_emphasized
                ], dim=1)
                # Adjust hurst back to original scale (was multiplied by 2)
                batch_trend_features[:, 0] = batch_trend_features[:, 0] / 2.0

                # PERFORMANCE OPTIMIZATION: Mixed precision forward pass
                with autocast(enabled=use_amp):
                    outputs = self.model(batch_X, return_attention=False)
                    loss = criterion(outputs, batch_y, trend_features=batch_trend_features)
                    loss = loss / accumulation_steps  # Scale loss for accumulation

                # PERFORMANCE OPTIMIZATION: Mixed precision backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # PERFORMANCE OPTIMIZATION: Gradient accumulation - only step every N batches
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if use_amp:
                        # Gradient clipping with mixed precision
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping without mixed precision
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor, return_attention=False)  # Don't need attention during validation
                val_loss = criterion(val_outputs, y_val_tensor)

                # Analyze raw logits to detect class suppression
                val_logits_mean = val_outputs.mean(dim=0).cpu().numpy()
                val_logits_std = val_outputs.std(dim=0).cpu().numpy()

                # Calculate metrics
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, val_predictions, average='weighted', zero_division=0
                )

                # HIGH-CONFIDENCE ACCURACY (critical metric for trading)
                val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                val_confidences = np.max(val_probs, axis=1)
                high_conf_threshold = 0.65

                high_conf_mask = val_confidences >= high_conf_threshold
                high_conf_preds = val_predictions[high_conf_mask]
                high_conf_labels = y_val[high_conf_mask]

                if len(high_conf_preds) > 0:
                    high_conf_accuracy = accuracy_score(high_conf_labels, high_conf_preds)
                    high_conf_dist = np.bincount(high_conf_preds, minlength=3)
                else:
                    high_conf_accuracy = 0.0
                    high_conf_dist = np.array([0, 0, 0])

            self.model.train()

            # IMPROVED: Calculate Sharpe ratio for early stopping (compute every epoch, not just every 10)
            val_start_idx = len(X) - len(X_val)
            val_price_changes = []
            for i in range(val_start_idx, min(len(ohlc) - self.sequence_length, val_start_idx + len(X_val))):
                if i + self.sequence_length < len(ohlc):
                    current_close = ohlc[i + self.sequence_length - 1, 3]
                    future_close = ohlc[i + self.sequence_length, 3]
                    price_change = (future_close - current_close) / current_close
                    val_price_changes.append(price_change)

            if len(val_price_changes) > 0:
                val_price_changes = np.array(val_price_changes[:len(val_predictions)])
                daily_pnl_config = {'dailyGoal': 500.0, 'dailyMaxLoss': 250.0}
                trading_metrics = evaluate_trading_performance(val_predictions, y_val[:len(val_price_changes)],
                                                              val_price_changes, daily_pnl_config)
                current_sharpe = trading_metrics['sharpe_ratio']
            else:
                current_sharpe = -np.inf

            # PERFORMANCE OPTIMIZATION: Step scheduler per epoch (CosineAnnealing)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                # Show prediction distribution
                pred_dist = np.bincount(val_predictions, minlength=3)

                # Calculate trading metrics (simulated returns based on predictions)
                # Get the corresponding price changes for validation set
                val_start_idx = len(X) - len(X_val)
                val_price_changes = []
                for i in range(val_start_idx, len(ohlc) - self.sequence_length):
                    current_close = ohlc[i + self.sequence_length - 1, 3]
                    future_close = ohlc[i + self.sequence_length, 3]
                    price_change = (future_close - current_close) / current_close
                    val_price_changes.append(price_change)

                val_price_changes = np.array(val_price_changes)

                # Pass daily P&L limits for realistic simulation
                daily_pnl_config = {
                    'dailyGoal': 500.0,  # Default daily goal
                    'dailyMaxLoss': 250.0  # Default max loss
                }
                trading_metrics = evaluate_trading_performance(val_predictions, y_val, val_price_changes, daily_pnl_config)

                # Per-class accuracy
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_val, val_predictions, labels=[0, 1, 2])
                per_class_acc = cm.diagonal() / cm.sum(axis=1)

                print(f"\n{'='*50}")
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"{'='*50}")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
                print(f"  Overall Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
                print(f"  Per-Class Accuracy: SHORT={per_class_acc[0]:.3f}, HOLD={per_class_acc[1]:.3f}, LONG={per_class_acc[2]:.3f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print(f"\n  Raw Logits (mean±std):")
                print(f"    SHORT: {val_logits_mean[0]:+.3f}±{val_logits_std[0]:.3f}")
                print(f"    HOLD:  {val_logits_mean[1]:+.3f}±{val_logits_std[1]:.3f}")
                print(f"    LONG:  {val_logits_mean[2]:+.3f}±{val_logits_std[2]:.3f}")
                print(f"\n  All Predictions: Short={pred_dist[0]}, Hold={pred_dist[1]}, Long={pred_dist[2]}")
                if pred_dist[0] == 0:
                    print(f"  ⚠️  WARNING: Model is NOT predicting SHORT at all! Check logits above.")
                print(f"  High-Confidence (>={high_conf_threshold*100:.0f}%):")
                print(f"    Count: {len(high_conf_preds)} ({len(high_conf_preds)/len(val_predictions)*100:.1f}% of predictions)")
                print(f"    Accuracy: {high_conf_accuracy:.4f} ({high_conf_accuracy*100:.1f}%)")
                print(f"    Distribution: Short={high_conf_dist[0]}, Hold={high_conf_dist[1]}, Long={high_conf_dist[2]}")
                print(f"\n  Trading Metrics (simulated with daily limits):")
                print(f"    Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
                print(f"    Win Rate: {trading_metrics['win_rate']*100:.2f}%")
                print(f"    Profit Factor: {trading_metrics['profit_factor']:.4f}")
                print(f"    Expectancy: {trading_metrics['expectancy']:.6f}")
                print(f"    Max Drawdown: {trading_metrics['max_drawdown']*100:.2f}%")
                if trading_metrics['trades_stopped_by_goal'] > 0:
                    print(f"    Stopped by goal: {trading_metrics['trades_stopped_by_goal']} times")
                if trading_metrics['trades_stopped_by_loss'] > 0:
                    print(f"    Stopped by loss: {trading_metrics['trades_stopped_by_loss']} times")

            # IMPROVED: Early stopping based on Sharpe ratio (trading performance, not just loss)
            if current_sharpe > best_val_sharpe:
                best_val_sharpe = current_sharpe
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
                if (epoch + 1) % 10 == 0:
                    print(f"  ✓ New best Sharpe ratio: {best_val_sharpe:.4f} (saved model)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n{'='*50}")
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    print(f"Best validation Sharpe ratio: {best_val_sharpe:.4f}")
                    print(f"{'='*50}")
                    break

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor, return_attention=False)
            val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()

            # Calculate final trading metrics
            val_start_idx = len(X) - len(X_val)
            val_price_changes = []
            for i in range(val_start_idx, len(ohlc) - self.sequence_length):
                current_close = ohlc[i + self.sequence_length - 1, 3]
                future_close = ohlc[i + self.sequence_length, 3]
                price_change = (future_close - current_close) / current_close
                val_price_changes.append(price_change)

            val_price_changes = np.array(val_price_changes)

            # Pass daily P&L limits for realistic simulation
            daily_pnl_config = {
                'dailyGoal': 500.0,  # Default daily goal
                'dailyMaxLoss': 250.0  # Default max loss
            }
            final_metrics = evaluate_trading_performance(val_predictions, y_val, val_price_changes, daily_pnl_config)

            print(f"\n{'='*50}")
            print("FINAL VALIDATION METRICS")
            print(f"{'='*50}")
            print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")

            # Create confusion matrix with all labels to ensure correct shape
            cm = confusion_matrix(y_val, val_predictions, labels=[0, 1, 2])

            print("\nConfusion Matrix:")
            print("              Predicted")
            print("              Short  Hold  Long")
            print(f"Actual Short   {cm[0][0]:5}  {cm[0][1]:5}  {cm[0][2]:5}")
            print(f"       Hold    {cm[1][0]:5}  {cm[1][1]:5}  {cm[1][2]:5}")
            print(f"       Long    {cm[2][0]:5}  {cm[2][1]:5}  {cm[2][2]:5}")

            # Print class distribution
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            unique_pred, counts_pred = np.unique(val_predictions, return_counts=True)
            print(f"\nValidation set distribution: {dict(zip(unique_val, counts_val))}")
            print(f"Predictions distribution: {dict(zip(unique_pred, counts_pred))}")

            # Print comprehensive trading metrics
            print(f"\n{'='*50}")
            print("TRADING PERFORMANCE METRICS (with daily limits)")
            print(f"{'='*50}")
            print(f"Sharpe Ratio:        {final_metrics['sharpe_ratio']:>8.4f}")
            print(f"Sortino Ratio:       {final_metrics['sortino_ratio']:>8.4f}")
            print(f"Profit Factor:       {final_metrics['profit_factor']:>8.4f}")
            print(f"Win Rate:            {final_metrics['win_rate']*100:>7.2f}%")
            print(f"Expectancy:          {final_metrics['expectancy']:>8.6f}")
            print(f"Max Drawdown:        {final_metrics['max_drawdown']*100:>7.2f}%")
            print(f"Total Return:        {final_metrics['total_return']*100:>7.2f}%")
            print(f"Number of Trades:    {final_metrics['num_trades']:>8d}")
            print(f"\nDaily Limit Events:")
            print(f"Stopped by goal:     {final_metrics['trades_stopped_by_goal']:>8d} times")
            print(f"Stopped by loss:     {final_metrics['trades_stopped_by_loss']:>8d} times")
            print(f"{'='*50}")

        self.is_trained = True

        # Compile model disabled due to LSTM compatibility issues
        # if not self.compiled:
        #     try:
        #         self.model = torch.compile(self.model)
        #         self.compiled = True
        #         print("Model compiled for optimized inference")
        #     except Exception as e:
        #         print(f"Model compilation not available: {e}")

        print(f"\n{'='*50}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*50}\n")

    def progressive_train(self, df, max_epochs=150, initial_lr=0.001):
        """
        PERFORMANCE OPTIMIZATION: Progressive training with increasing complexity

        Trains the model in phases:
        1. Phase 1: Short sequences (10 bars) with higher LR
        2. Phase 2: Full sequences (15 bars) with medium LR
        3. Phase 3: Fine-tuning with low LR

        Args:
            df: Training data
            max_epochs: Total epochs across all phases
            initial_lr: Starting learning rate
        """
        print(f"\n{'='*70}")
        print("PROGRESSIVE TRAINING")
        print(f"{'='*70}")

        # Save original sequence length
        original_seq_length = self.sequence_length

        # Phase 1: Train with shorter sequences
        print(f"\n{'='*50}")
        print("PHASE 1: Short sequences (10 bars)")
        print(f"{'='*50}")
        self.sequence_length = 10
        epochs_phase1 = max_epochs // 3
        self.train(df, epochs=epochs_phase1, learning_rate=initial_lr, batch_size=64)

        # Phase 2: Increase to full sequence length
        print(f"\n{'='*50}")
        print("PHASE 2: Full sequences ({original_seq_length} bars)")
        print(f"{'='*50}")
        self.sequence_length = original_seq_length
        epochs_phase2 = max_epochs // 3
        self.train(df, epochs=epochs_phase2, learning_rate=initial_lr * 0.5, batch_size=64)

        # Phase 3: Fine-tune with lower learning rate
        print(f"\n{'='*50}")
        print("PHASE 3: Fine-tuning")
        print(f"{'='*50}")
        epochs_phase3 = max_epochs - epochs_phase1 - epochs_phase2
        self.train(df, epochs=epochs_phase3, learning_rate=initial_lr * 0.1, batch_size=64)

        print(f"\n{'='*70}")
        print("PROGRESSIVE TRAINING COMPLETE")
        print(f"{'='*70}\n")

    def precompute_features(self, df, df_secondary=None, save_path='features_cache.npz'):
        """
        PERFORMANCE OPTIMIZATION: Pre-compute features and save to disk

        Computes all features once and saves them for faster re-training.
        Useful when experimenting with hyperparameters or model architectures.

        Args:
            df: Primary timeframe data
            df_secondary: Secondary timeframe data (optional)
            save_path: Path to save pre-computed features

        Returns:
            Tuple of (X, y) sequences
        """
        print(f"\n{'='*50}")
        print("PRE-COMPUTING FEATURES")
        print(f"{'='*50}")

        import time
        start_time = time.time()

        # Prepare data (this computes all features)
        X, y = self.prepare_data(df, df_secondary=df_secondary, fit_scaler=True)

        elapsed = time.time() - start_time
        print(f"\n✓ Feature computation complete in {elapsed:.2f}s")
        print(f"  Sequences: {len(X)}")
        print(f"  Features per sequence: {X.shape[1]} timesteps × {X.shape[2]} features")

        # Save to compressed numpy format
        save_path = Path(save_path)
        np.savez_compressed(
            save_path,
            X=X,
            y=y,
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            sequence_length=self.sequence_length
        )

        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Features saved to {save_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Speedup: ~{elapsed/10:.1f}x faster when loading from cache")

        return X, y

    def train_from_cache(self, cache_path, epochs=100, learning_rate=0.001, batch_size=64, validation_split=0.2):
        """
        PERFORMANCE OPTIMIZATION: Train from pre-computed features

        Loads pre-computed features from disk and trains directly,
        skipping expensive feature computation.

        Args:
            cache_path: Path to cached features (.npz file)
            epochs: Training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print(f"\n{'='*50}")
        print("TRAINING FROM CACHED FEATURES")
        print(f"{'='*50}")

        # Load cached data
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {cache_path}")

        data = np.load(cache_path)
        X = data['X']
        y = data['y']

        # Restore scaler
        self.scaler.mean_ = data['scaler_mean']
        self.scaler.scale_ = data['scaler_scale']
        self.sequence_length = int(data['sequence_length'])

        print(f"✓ Loaded {len(X)} sequences from cache")
        print(f"  Feature dimensions: {X.shape}")
        print(f"  Skipping expensive feature computation...")

        # Time-based validation split
        split_idx = int(len(X) * (1 - validation_split))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Training setup (similar to regular train method but without data prep)
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)

        beta = 0.9999
        effective_nums = 1.0 - np.power(beta, class_counts)
        effective_nums = np.where(effective_nums == 0, 1, effective_nums)
        ens_weights = torch.FloatTensor((1.0 - beta) / effective_nums)
        class_weights = ens_weights.to(self.device)

        # ROUND 2 IMPROVEMENT: Use CombinedTradingLoss
        if ADVANCED_MODULES_AVAILABLE:
            criterion = CombinedTradingLoss(
                smoothing=0.1,
                confidence_weight=0.15,
                directional_weight=1.5,
                use_sharpe=False
            )
        else:
            criterion = FocalLoss(gamma=2.0, weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.001
        )

        # Mixed precision setup
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler() if self.device.type == 'cuda' else None
        use_amp = self.device.type == 'cuda'
        accumulation_steps = 4

        if use_amp:
            print(f"✓ Mixed precision training enabled")
        print(f"✓ Gradient accumulation: effective batch size = {batch_size * accumulation_steps}\n")

        # Training loop (simplified version)
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                with autocast(enabled=use_amp):
                    outputs = self.model(batch_X, return_attention=False)
                    loss = criterion(outputs, batch_y) / accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                batch_count += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor, return_attention=False)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, val_predictions)
            self.model.train()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_train_loss = epoch_loss / batch_count
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")

        self.is_trained = True
        print(f"\n✓ Training from cache complete!\n")

    @timing_decorator
    def predict(self, recent_bars_df):
        """
        Predict trade signal for new bar (OPTIMIZED VERSION with fast path)
        Returns: (signal, confidence)
            signal: 'long', 'short', or 'hold'
            confidence: float between 0 and 1
        """
        if not self.is_trained:
            print("WARNING: Model not trained yet!")
            return 'hold', 0.0

        # PERFORMANCE OPTIMIZATION: Use cached Hurst value (updated every 10 bars)
        current_hurst_H = self._last_hurst_H
        current_hurst_C = self._last_hurst_C

        # CRITICAL PERFORMANCE FIX: Only process the LAST sequence_length + some buffer bars
        # Instead of processing ALL historical data every time
        min_bars_needed = self.sequence_length + 100  # +100 for Hurst calculation

        if len(recent_bars_df) > min_bars_needed:
            # Use only the most recent bars needed for prediction
            df_subset = recent_bars_df.tail(min_bars_needed).reset_index(drop=True)
            print(f"⚡ Fast path: Using {len(df_subset)} recent bars instead of {len(recent_bars_df)}")
        else:
            df_subset = recent_bars_df

        # Prepare secondary timeframe data (if available)
        df_secondary_subset = None
        if self.historical_data_secondary is not None and len(self.historical_data_secondary) > 0:
            # Use same subset strategy for secondary data
            if len(self.historical_data_secondary) > min_bars_needed:
                df_secondary_subset = self.historical_data_secondary.tail(min_bars_needed).reset_index(drop=True)
            else:
                df_secondary_subset = self.historical_data_secondary

        # Validate and prepare data (without fitting scaler)
        try:
            X, _ = self.prepare_data(df_subset, df_secondary=df_secondary_subset, fit_scaler=False)

            if len(X) == 0:
                print(f"WARNING: Need at least {self.sequence_length + 1} bars for prediction")
                return 'hold', 0.0

            # Take the last sequence for prediction
            last_sequence = X[-1:]

        except Exception as e:
            print(f"ERROR preparing prediction data: {e}")
            return 'hold', 0.0

        # PERFORMANCE OPTIMIZATION: Use FP16 for faster inference on GPU
        X_tensor = torch.FloatTensor(last_sequence)
        if self.device.type == 'cuda':
            X_tensor = X_tensor.half().to(self.device)  # FP16 on GPU
        else:
            X_tensor = X_tensor.to(self.device)  # FP32 on CPU

        # Predict with inference mode (faster than no_grad)
        self.model.eval()
        with torch.inference_mode():  # Faster than no_grad()
            outputs, attn_weights = self.model(X_tensor, return_attention=True)
            if self.device.type == 'cuda':
                outputs = outputs.float()  # Convert back to FP32 for softmax

            # ROUND 2 IMPROVEMENT: Apply confidence calibration if available
            if self.calibration is not None:
                try:
                    # Calibrate probabilities using temperature scaling
                    logits = outputs
                    raw_probs = torch.softmax(logits, dim=1)
                    calibrated_probs = self.calibration.calibrate(
                        logits,
                        raw_probs,
                        method='temperature'
                    )
                    probabilities = calibrated_probs[0]
                except Exception as e:
                    print(f"⚠️  Calibration failed, using raw probabilities: {e}")
                    probabilities = torch.softmax(outputs, dim=1)[0]
            else:
                probabilities = torch.softmax(outputs, dim=1)[0]

            # IMPROVED: Compare LONG vs SHORT probabilities directly (ignore HOLD bias)
            # This overcomes model trained with 40% HOLD
            prob_short = probabilities[0].item()
            prob_hold = probabilities[1].item()
            prob_long = probabilities[2].item()

            # Check for NaN in model outputs (indicates untrained model or bad input)
            if math.isnan(prob_short) or math.isnan(prob_hold) or math.isnan(prob_long):
                print(f"ERROR: Model output contains NaN - model may not be trained or input has NaN values")
                print(f"  Probabilities: SHORT={prob_short}, HOLD={prob_hold}, LONG={prob_long}")
                print(f"  Model trained: {self.is_trained}")
                print(f"  Input shape: {X_tensor.shape}")
                # Check if input contains NaN
                if torch.isnan(X_tensor).any():
                    print(f"  WARNING: Input tensor contains NaN values!")
                return 'hold', 0.0

            # NEW APPROACH: Use simple argmax but normalize directional conviction
            # This increases signal frequency while maintaining quality via confidence threshold
            # Calculate directional edge: how much stronger is the best direction vs the other?
            directional_edge = abs(prob_long - prob_short)

            # Determine signal by simple comparison (no direction margin)
            if prob_long > prob_short and prob_long > prob_hold:
                predicted_class = 2  # Long
                confidence = prob_long
            elif prob_short > prob_long and prob_short > prob_hold:
                predicted_class = 0  # Short
                confidence = prob_short
            else:
                predicted_class = 1  # Hold
                confidence = prob_hold

            # DEBUG: Log raw confidence before boost
            raw_confidence = confidence
            print(f"🔍 Raw confidence (before boost): {raw_confidence:.3f}")

            # SAFETY CHECK: Catch zero confidence immediately
            if raw_confidence == 0.0:
                print(f"❌ ZERO CONFIDENCE DETECTED!")
                print(f"   prob_long={prob_long:.4f}, prob_short={prob_short:.4f}, prob_hold={prob_hold:.4f}")
                print(f"   predicted_class={predicted_class}")
                # Force to max probability
                confidence = max(prob_long, prob_short, prob_hold)
                raw_confidence = confidence
                print(f"   FIXED: Using max probability as confidence: {confidence:.4f}")

            # IMPROVED: Multi-level confidence boost based on directional conviction AND trend alignment
            # Rewards clear directional signals with higher confidence
            # FIXED: Reduced boost multipliers to prevent 100% confidence
            if predicted_class in [0, 2]:  # Only boost directional signals (LONG/SHORT)
                # Base boost from directional edge
                if directional_edge > 0.20:  # Very strong conviction (20%+ edge)
                    confidence = min(0.95, confidence * 1.15)  # 15% boost, cap at 95%
                elif directional_edge > 0.15:  # Strong conviction (15-20% edge)
                    confidence = min(0.90, confidence * 1.10)  # 10% boost, cap at 90%
                elif directional_edge > 0.10:  # Moderate conviction (10-15% edge)
                    confidence = min(0.85, confidence * 1.05)  # 5% boost, cap at 85%
                # Weak edge (<10%): no boost

                # NEW: Additional boost for trend-aligned trades (PRIORITY 4)
                if len(recent_bars_df) >= 100:
                    current_hurst = calculate_hurst_exponent(recent_bars_df['close'].tail(100).values)[0]

                    # Calculate short-term trend (last 20 bars)
                    recent_closes = recent_bars_df['close'].tail(20).values
                    trend_slope = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

                    # Check if signal aligns with trend
                    is_uptrend = (current_hurst > 0.55 and trend_slope > 0.001)
                    is_downtrend = (current_hurst > 0.55 and trend_slope < -0.001)

                    # Boost with-trend signals
                    if (is_uptrend and predicted_class == 2) or (is_downtrend and predicted_class == 0):
                        confidence = min(0.98, confidence * 1.10)  # 10% additional boost
                        print(f"📈 TREND BOOST: Signal aligns with {('UP' if is_uptrend else 'DOWN')}TREND (H={current_hurst:.3f}, slope={trend_slope*100:.3f}%)")
                    # Reduce counter-trend confidence
                    elif (is_uptrend and predicted_class == 0) or (is_downtrend and predicted_class == 2):
                        confidence = confidence * 0.85  # 15% penalty
                        print(f"⚠️  COUNTER-TREND: Signal against trend, confidence reduced (H={current_hurst:.3f}, slope={trend_slope*100:.3f}%)")

            if raw_confidence > 0:
                boost_pct = (confidence/raw_confidence - 1)*100
            else:
                boost_pct = 0
            print(f"🔍 Final confidence (after boost): {confidence:.3f} (boost: {boost_pct:.1f}%)")

            # Handle NaN/inf in confidence (should not happen after checks above)
            if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
                print(f"WARNING: Invalid confidence value after calculation: {confidence}, using max probability as fallback")
                confidence = max(prob_short, prob_hold, prob_long)  # Use max probability instead of 0
                print(f"  Fallback confidence: {confidence:.3f}")

            # Log all probabilities for debugging
            print(f"Probabilities: SHORT={prob_short:.3f}, HOLD={prob_hold:.3f}, LONG={prob_long:.3f}")
            print(f"Directional Edge: {directional_edge:.3f} ({'Strong' if directional_edge > 0.10 else 'Weak'})")

            # Extract and log attention weights (which bars model focuses on)
            # Attention weights shape: (batch, num_heads, seq_len, seq_len)
            # We want the attention pattern from the last position to all positions

            # Debug: Check actual shape
            # print(f"DEBUG: attn_weights shape: {attn_weights.shape}")

            # Handle different attention weight shapes
            if len(attn_weights.shape) == 4:  # (batch, num_heads, seq_len, seq_len)
                avg_attn = attn_weights[0].mean(dim=0)  # Average across attention heads: (seq_len, seq_len)
                last_bar_attn = avg_attn[-1, :].cpu().numpy()  # Attention from last bar to all bars
            elif len(attn_weights.shape) == 3:  # (batch, seq_len, seq_len) - already averaged
                last_bar_attn = attn_weights[0, -1, :].cpu().numpy()
            else:
                # Fallback: create uniform attention
                seq_len = last_sequence.shape[1]
                last_bar_attn = np.ones(seq_len) / seq_len

            # Find top 5 most attended bars
            top_k = min(5, len(last_bar_attn))
            top_indices = np.argsort(last_bar_attn)[-top_k:][::-1]
            top_weights = last_bar_attn[top_indices]

        # Map class to signal: 0=short, 1=hold, 2=long
        signal_map = {0: 'short', 1: 'hold', 2: 'long'}
        signal = signal_map[predicted_class]

        print(f"DEBUG: About to detect market regime...")
        # Detect market regime
        try:
            regime = detect_market_regime(recent_bars_df, lookback=100)
            print(f"DEBUG: Regime detected: {regime}")
        except Exception as e:
            print(f"ERROR detecting regime: {e}")
            regime = "unknown"

        # Calculate recent accuracy if we have enough predictions
        recent_accuracy = None
        if len(self.recent_predictions) >= 20:
            correct = sum([1 for pred, actual in self.recent_predictions if pred == actual])
            recent_accuracy = correct / len(self.recent_predictions)

        # Get adaptive confidence threshold
        print(f"DEBUG: About to get adaptive threshold...")
        try:
            current_timestamp = recent_bars_df['time'].iloc[-1]
            adaptive_threshold = self.adaptive_thresholds.get_threshold(
                regime, current_timestamp, recent_accuracy
            )
            print(f"DEBUG: Adaptive threshold: {adaptive_threshold}")
        except Exception as e:
            print(f"ERROR getting adaptive threshold: {e}")
            adaptive_threshold = 0.65

        # Log Hurst values, regime, and P&L context with prediction
        print(f"DEBUG: About to log prediction context...")
        try:
            current_pnl = recent_bars_df['dailyPnL'].iloc[-1] if 'dailyPnL' in recent_bars_df.columns else 0.0
            current_goal = recent_bars_df['dailyGoal'].iloc[-1] if 'dailyGoal' in recent_bars_df.columns else 0.0
            current_max_loss = recent_bars_df['dailyMaxLoss'].iloc[-1] if 'dailyMaxLoss' in recent_bars_df.columns else 0.0

            print(f"\n--- Prediction Context ---")
            print(f"Market Regime: {regime.upper() if isinstance(regime, str) else 'UNKNOWN'}")
            print(f"Adaptive Threshold: {adaptive_threshold:.2%} (vs fixed 65%)")
            if recent_accuracy is not None:
                print(f"Recent Accuracy: {recent_accuracy:.2%} (last {len(self.recent_predictions)} predictions)")

            # Log attention: which bars the model focused on
            print(f"\nAttention Focus (top {top_k} bars):")
            for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
                bars_ago = len(last_bar_attn) - 1 - idx
                print(f"  {i+1}. Bar -{bars_ago:2d} (weight: {weight:.3f})")
        except Exception as e:
            print(f"ERROR logging prediction context: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nCurrent Hurst H: {current_hurst_H:.4f} ", end="")
        if current_hurst_H > 0.5:
            print("(TRENDING)")
        elif current_hurst_H < 0.5:
            print("(MEAN-REVERTING)")
        else:
            print("(RANDOM WALK)")
        print(f"Current Hurst C: {current_hurst_C:.4f}")
        print(f"Daily P&L: ${current_pnl:.2f} (Goal: ${current_goal:.2f}, Max Loss: -${current_max_loss:.2f})")
        if current_goal > 0:
            print(f"P&L Progress: {(current_pnl / current_goal * 100):.1f}% of goal")
        if current_max_loss > 0 and current_pnl < 0:
            print(f"Risk Used: {(abs(current_pnl) / current_max_loss * 100):.1f}% of max loss")
        print(f"Predicted Signal: {signal.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        # Apply adaptive threshold filtering
        if confidence < adaptive_threshold:
            print(f"⚠️  Confidence below adaptive threshold ({confidence*100:.1f}% < {adaptive_threshold*100:.1f}%)")
            signal = 'hold'

        print(f"Final Signal: {signal.upper()}")
        print(f"-------------------------\n")

        return signal, confidence

    def detect_early_exit(self, recent_bars_df, current_position: str, entry_price: float):
        """
        Detect early exit conditions before waiting for HOLD signal

        Args:
            recent_bars_df: Recent bar data
            current_position: 'long' or 'short'
            entry_price: Entry price of current position

        Returns:
            Dictionary with:
                - should_exit: bool
                - reason: str (why to exit)
                - urgency: str ('immediate', 'normal', 'none')
        """
        if current_position not in ['long', 'short']:
            return {'should_exit': False, 'reason': 'No position', 'urgency': 'none'}

        current_bar = recent_bars_df.iloc[-1]
        current_price = current_bar['close']

        # Calculate position P&L
        if current_position == 'long':
            pnl_points = current_price - entry_price
        else:  # short
            pnl_points = entry_price - current_price

        # Get probabilities for the current bar
        signal, confidence = self.predict(recent_bars_df)

        # Get model probabilities directly (need to call model again for internal probs)
        try:
            X, _ = self.prepare_data(recent_bars_df.tail(self.sequence_length + 100), fit_scaler=False)
            if len(X) > 0:
                last_sequence = X[-1:]
                X_tensor = torch.FloatTensor(last_sequence).to(self.device)
                self.model.eval()
                with torch.inference_mode():
                    outputs, _ = self.model(X_tensor, return_attention=True)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    prob_short = probabilities[0].item()
                    prob_hold = probabilities[1].item()
                    prob_long = probabilities[2].item()
            else:
                prob_short, prob_hold, prob_long = 0.33, 0.34, 0.33
        except:
            prob_short, prob_hold, prob_long = 0.33, 0.34, 0.33

        # EXIT CONDITION 1: Opposite direction signal with high confidence
        # INCREASED from 0.35 to 0.65 to hold winning trades longer
        if current_position == 'long' and signal == 'short' and confidence > 0.65:
            return {
                'should_exit': True,
                'reason': f'Strong SHORT signal ({confidence*100:.1f}%) against LONG position',
                'urgency': 'immediate'
            }
        elif current_position == 'short' and signal == 'long' and confidence > 0.65:
            return {
                'should_exit': True,
                'reason': f'Strong LONG signal ({confidence*100:.1f}%) against SHORT position',
                'urgency': 'immediate'
            }

        # EXIT CONDITION 2: Confidence collapse (model uncertainty)
        # If we're in a position and HOLD probability spikes above 70%
        # INCREASED from 0.50 to 0.70 to reduce premature exits
        if prob_hold > 0.70:
            return {
                'should_exit': True,
                'reason': f'High HOLD probability ({prob_hold*100:.1f}%) - model uncertain',
                'urgency': 'normal'
            }

        # EXIT CONDITION 3: Directional probability reversal
        # Long position but SHORT probability exceeds LONG probability
        # INCREASED from 0.05 to 0.15 to filter out noise and hold longer
        if current_position == 'long' and prob_short > prob_long + 0.15:
            return {
                'should_exit': True,
                'reason': f'Directional reversal: SHORT prob ({prob_short*100:.1f}%) > LONG prob ({prob_long*100:.1f}%)',
                'urgency': 'normal'
            }
        elif current_position == 'short' and prob_long > prob_short + 0.15:
            return {
                'should_exit': True,
                'reason': f'Directional reversal: LONG prob ({prob_long*100:.1f}%) > SHORT prob ({prob_short*100:.1f}%)',
                'urgency': 'normal'
            }

        # EXIT CONDITION 4: Momentum loss (DISABLED to hold winners longer)
        # This was exiting winners during normal pullbacks
        # Let stop loss and take profit handle exits instead
        # DISABLED: Removed momentum loss check to hold trending trades longer
        if False:  # Disabled
            last_5_closes = recent_bars_df['close'].tail(5).values
            if current_position == 'long':
                # Check for 5 consecutive lower closes
                if (last_5_closes[-1] < last_5_closes[-2] < last_5_closes[-3] <
                    last_5_closes[-4] < last_5_closes[-5]):
                    return {
                        'should_exit': True,
                        'reason': 'Momentum loss: 5 consecutive lower closes in LONG',
                        'urgency': 'normal'
                    }
            else:  # short
                # Check for 5 consecutive higher closes
                if (last_5_closes[-1] > last_5_closes[-2] > last_5_closes[-3] >
                    last_5_closes[-4] > last_5_closes[-5]):
                    return {
                        'should_exit': True,
                        'reason': 'Momentum loss: 5 consecutive higher closes in SHORT',
                        'urgency': 'normal'
                    }

        # No exit conditions met
        return {'should_exit': False, 'reason': 'Position looks good', 'urgency': 'none'}

    def predict_with_risk_params(
        self,
        recent_bars_df,
        account_balance: float = 25000.0,
        tick_value: float = 12.50
    ):
        """
        Predict trade signal with complete risk management parameters

        Returns: Dictionary with signal, confidence, and risk management params
        """
        from risk_management import RiskManager

        # Get basic prediction
        signal, confidence = self.predict(recent_bars_df)

        if signal == 'hold':
            return {
                'signal': 'hold',
                'confidence': confidence,
                'contracts': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'reason': 'Hold signal - no trade'
            }

        # Get current market data
        current_bar = recent_bars_df.iloc[-1]
        entry_price = current_bar['close']

        # Get volatility estimate (pure price action - no ATR indicator)
        # Calculate average true range using pure candle range (no EMA smoothing)
        if len(recent_bars_df) >= 14:
            high = recent_bars_df['high'].values
            low = recent_bars_df['low'].values
            close = recent_bars_df['close'].values

            # Calculate true range for last 14 bars
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )

            # Simple average (no EMA - pure price action)
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else 15.0
        else:
            atr = 15.0  # Default fallback

        # DEBUG: Log ATR and data size
        print(f"\n🔍 RISK CALCULATION DEBUG:")
        print(f"   Historical bars: {len(recent_bars_df)}")
        print(f"   ATR: {atr:.2f} points")
        print(f"   Entry price: ${entry_price:.2f}")

        # Get regime info using PURE PRICE ACTION (not indicator-based)
        from core.market_regime import calculate_market_regime
        regime_info = calculate_market_regime(recent_bars_df, lookback=min(20, len(recent_bars_df)-1), use_adx=False)
        regime = regime_info['regime']

        # Extract metrics for logging
        trend_strength = regime_info['metrics']['trend_strength']
        directional_consistency = regime_info['metrics']['directional_consistency']

        print(f"   Regime: {regime}")
        print(f"   Trend strength: {trend_strength:.2f}, Directional consistency: {directional_consistency:.2f}")

        # Calculate trade parameters using risk manager with enhanced regime info
        risk_mgr = RiskManager()
        trade_params = risk_mgr.calculate_trade_parameters(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            atr=atr,
            regime=regime,
            account_balance=account_balance,
            tick_value=tick_value,
            regime_info=regime_info  # Pass enhanced regime info
        )

        return trade_params

    def save_model(self, path=None):
        """Save model state, scaler, and configuration"""
        if path is None:
            path = self.model_path
        else:
            path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
            'scaler_var': self.scaler.var_ if hasattr(self.scaler, 'var_') else None,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained,
            'signal_threshold': self.signal_threshold if hasattr(self, 'signal_threshold') else 0.05,
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load model state, scaler, and configuration"""
        if path is None:
            path = self.model_path
        else:
            path = Path(path)

        if not path.exists():
            print(f"No model found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Load model state
            # Handle torch.compile() prefixed keys (_orig_mod.)
            state_dict = checkpoint['model_state_dict']

            # Check if state_dict has _orig_mod prefix (saved after torch.compile)
            has_compile_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())

            if has_compile_prefix:
                # Remove _orig_mod. prefix from all keys
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('_orig_mod.'):
                        new_key = key.replace('_orig_mod.', '')
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
                print("Detected torch.compile() prefixed model, stripping _orig_mod. prefix...")

            self.model.load_state_dict(state_dict)

            # Load scaler state
            if checkpoint['scaler_mean'] is not None:
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
                self.scaler.var_ = checkpoint['scaler_var']
                self.scaler.n_features_in_ = len(checkpoint['scaler_mean'])
                self.scaler.n_samples_seen_ = 1  # Dummy value

            # Load configuration
            self.sequence_length = checkpoint['sequence_length']
            self.is_trained = checkpoint['is_trained']
            self.signal_threshold = checkpoint.get('signal_threshold', 0.05)

            print(f"Model loaded from {path}")
            print(f"Signal threshold: {self.signal_threshold:.4f}%")

            # Compile model disabled due to LSTM compatibility issues
            # if not self.compiled:
            #     try:
            #         self.model = torch.compile(self.model)
            #         self.compiled = True
            #         print("Model compiled for optimized inference")
            #     except Exception as e:
            #         print(f"Model compilation not available: {e}")

            # PERFORMANCE OPTIMIZATION: Quantization disabled to allow retraining
            # Quantization prevents backpropagation, so it's incompatible with the
            # background training task in FastAPI. Can be re-enabled if training is removed.
            # if self.device.type == 'cpu' and self.is_trained:
            #     try:
            #         # Store original model for potential retraining
            #         self._original_model = self.model
            #         self.model = torch.quantization.quantize_dynamic(
            #             self.model,
            #             {torch.nn.LSTM, torch.nn.Linear},  # Quantize LSTM and Linear layers
            #             dtype=torch.qint8
            #         )
            #         print("✓ Model quantized to INT8 for faster CPU inference")
            #     except Exception as e:
            #         print(f"Note: Quantization not applied: {e}")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def update_historical_data(self, df, df_secondary=None):
        """Update historical data storage (multi-timeframe support)"""
        # Update primary timeframe
        if self.historical_data is None:
            self.historical_data = df.copy()
        else:
            self.historical_data = pd.concat([self.historical_data, df], ignore_index=True)

        # Keep only recent data to prevent memory issues (e.g., last 50,000 bars)
        if len(self.historical_data) > 50000:
            self.historical_data = self.historical_data.tail(50000).reset_index(drop=True)

        # Update secondary timeframe if provided
        if df_secondary is not None:
            if self.historical_data_secondary is None:
                self.historical_data_secondary = df_secondary.copy()
            else:
                self.historical_data_secondary = pd.concat([self.historical_data_secondary, df_secondary], ignore_index=True)

            # Keep only recent secondary data
            if len(self.historical_data_secondary) > 10000:
                self.historical_data_secondary = self.historical_data_secondary.tail(10000).reset_index(drop=True)

        return self.historical_data

# Global model instance
trading_model = TradingModel(sequence_length=20)
