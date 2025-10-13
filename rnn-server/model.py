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

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX) for trend strength detection
    ADX > 25: Strong trend
    ADX < 20: Ranging/weak trend
    """
    n = len(high)
    if n < period + 1:
        return np.zeros(n)

    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

    # Calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                   abs(high[i] - close[i-1]),
                   abs(low[i] - close[i-1]))

    # Smooth with EMA
    import pandas as pd
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, adjust=False).mean() * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, adjust=False).mean() * 100

    # Calculate DX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100

    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

    return adx


def detect_market_regime(df, lookback=100):
    """
    Detect current market regime for adaptive strategy
    Returns: 'trending', 'ranging', 'high_vol', or 'low_vol'
    """
    if len(df) < lookback:
        return 'unknown'

    recent_data = df.tail(lookback)

    # Calculate ADX for trend detection
    adx = calculate_adx(recent_data['high'].values,
                       recent_data['low'].values,
                       recent_data['close'].values,
                       period=14)
    current_adx = adx[-1] if len(adx) > 0 else 0

    # Calculate volatility regime
    returns = recent_data['close'].pct_change().dropna()
    current_vol = returns.tail(20).std()
    hist_vol = returns.std()
    vol_ratio = current_vol / (hist_vol + 1e-8)

    # Classify regime
    if current_adx > 25:
        if vol_ratio > 1.5:
            return 'trending_high_vol'
        else:
            return 'trending_normal'
    elif current_adx < 20:
        if vol_ratio < 0.7:
            return 'ranging_low_vol'
        else:
            return 'ranging_normal'
    else:
        if vol_ratio > 1.5:
            return 'high_vol_chaos'
        else:
            return 'transitional'


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


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (VECTORIZED VERSION)
    Measures market volatility
    """
    n = len(high)
    if n < period + 1:
        return np.zeros(n)

    # PERFORMANCE OPTIMIZATION: Vectorized True Range calculation
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(hl, hc), lc)

    # PERFORMANCE OPTIMIZATION: Use pandas rolling mean (faster than loop)
    import pandas as pd
    tr_series = pd.Series(tr)
    atr_values = tr_series.rolling(window=period, min_periods=1).mean().values

    # Add zero for first value (no previous close) - ensure correct length
    atr = np.zeros(n)
    atr[1:] = atr_values

    return atr


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
            if i >= 5:
                primary_trend = (primary_close[i] - primary_close[i-5]) / (primary_close[i-5] + 1e-8)
                secondary_trend = aligned_features['tf2_close_change'][i]
                # Alignment: -1 (counter-trend) to +1 (aligned)
                if abs(primary_trend) > 1e-8 and abs(secondary_trend) > 1e-8:
                    alignment = np.sign(primary_trend) * np.sign(secondary_trend)
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


class TradingRNN(nn.Module):
    """
    Enhanced LSTM-based RNN with attention mechanism for predicting trade signals

    Features (62 total after feature reduction):
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
    def __init__(self, input_size=62, hidden_size=128, num_layers=3, output_size=3):
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
    def __init__(self, sequence_length=40, model_path='models/trading_model.pth'):
        self.model = TradingRNN(input_size=62, hidden_size=128, num_layers=3, output_size=3)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.is_trained = False
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compiled = False  # Track compilation state

        # Adaptive confidence thresholds
        self.adaptive_thresholds = AdaptiveConfidenceThresholds()

        # Track recent predictions for accuracy calculation
        self.recent_predictions = []
        self.max_recent_predictions = 50

        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Don't compile yet - wait until after potential model loading
        # Compilation happens after first training or successful load

        # Try to load existing model
        if self.model_path.exists():
            self.load_model()

        # Historical data storage (multi-timeframe support)
        self.historical_data = None
        self.historical_data_secondary = None  # Secondary timeframe (e.g., 5-min)

        # PERFORMANCE OPTIMIZATION: Cache computed features to avoid recomputation
        self._feature_cache = None
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
    def prepare_data(self, df, df_secondary=None, fit_scaler=False, adaptive_threshold=True):
        """
        Prepare data for training with enhanced features including multi-timeframe (OPTIMIZED VERSION)
        Creates sequences and labels based on future price movement

        Args:
            df: Primary timeframe DataFrame with OHLC data
            df_secondary: Optional secondary timeframe DataFrame (e.g., 5-min)
            fit_scaler: If True, fit the scaler. If False, only transform (use False for inference)
            adaptive_threshold: If True, calculate threshold based on data volatility
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

        # Calculate ATR
        atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values)

        # Calculate all advanced price features
        price_features = calculate_price_features(df)

        # Calculate order flow features
        order_flow_features = calculate_order_flow_features(df)

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
        assert len(atr) == n_bars, f"ATR mismatch: {len(atr)} vs {n_bars}"

        # Calculate price change magnitude (recent volatility indicator)
        price_change_magnitude = np.zeros(n_bars)
        for i in range(5, n_bars):
            recent_changes = np.abs(np.diff(ohlc[i-5:i+1, 3]) / ohlc[i-5:i, 3])
            price_change_magnitude[i] = np.mean(recent_changes)

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
            'atr': atr,
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

        features = np.column_stack([
            # Core features: OHLC + Hurst + ATR = 7
            ohlc,                                    # 4
            hurst_H_values,                          # 1
            hurst_C_values,                          # 1
            atr,                                     # 1
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
            # Deviation features - REDUCED: only windows 20 & 50 (2 windows × 5 metrics = 10)
            price_features['mean_dev_20'],           # 1
            price_features['mean_dev_50'],           # 1
            price_features['median_dev_20'],         # 1
            price_features['median_dev_50'],         # 1
            price_features['std_dev_20'],            # 1
            price_features['std_dev_50'],            # 1
            price_features['z_score_20'],            # 1
            price_features['z_score_50'],            # 1
            price_features['bb_width_20'],           # 1
            price_features['bb_width_50'],           # 1
            # Additional deviation features: 3
            price_features['vol_acceleration'],      # 1
            price_features['high_deviation'],        # 1
            price_features['low_deviation'],         # 1
            # Order flow - REDUCED: only volume_ratio (1 feature, removed 6)
            order_flow_features['volume_ratio'],     # 1
            # Time-of-day features: 5
            time_features['hour_of_day'],            # 1
            time_features['minutes_into_session'],   # 1
            time_features['minutes_to_close'],       # 1
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
            # Multi-timeframe features - REDUCED: 9 (removed tf2_delta)
            secondary_features['tf2_close'],         # 1
            secondary_features['tf2_close_change'],  # 1
            secondary_features['tf2_high_low_range'], # 1
            secondary_features['tf2_volume'],        # 1
            secondary_features['tf2_position_in_bar'], # 1
            secondary_features['tf2_trend_direction'], # 1
            secondary_features['tf2_momentum'],      # 1
            secondary_features['tf2_volatility'],    # 1
            secondary_features['tf2_alignment_score'], # 1
            # Price change magnitude: 1
            price_change_magnitude                   # 1
        ])

        # Only log during training
        if fit_scaler:
            print(f"Total features: {features.shape[1]} (OHLC:4 + Hurst:2 + ATR:1 + Price:2 + Patterns:15 + Deviation:13 + OrderFlow:1 + TimeOfDay:5 + Microstructure:5 + VolRegime:4 + MultiTF:9 + PriceChangeMag:1 = 62)")

        # Scale the features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # Create sequences with improved labeling
        X, y = [], []
        lookahead_bars = 3  # Look ahead 3 bars to reduce noise

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

        # Percentile-based threshold: 40% smallest moves become HOLD
        hold_percentage = 0.40
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
        for i in range(len(features_scaled) - self.sequence_length - lookahead_bars + 1):
            X.append(features_scaled[i:i + self.sequence_length])

            current_close = ohlc[i + self.sequence_length - 1, 3]

            # Use maximum move in next 3 bars
            future_slice = ohlc[i + self.sequence_length:i + self.sequence_length + lookahead_bars]
            future_highs = future_slice[:, 1]
            future_lows = future_slice[:, 2]

            max_up_move = (np.max(future_highs) - current_close) / current_close * 100
            max_down_move = (current_close - np.min(future_lows)) / current_close * 100

            # Create label based on dominant move and threshold
            if max_up_move > percentile_threshold and max_up_move > max_down_move:
                y.append(2)  # Long
            elif max_down_move > percentile_threshold and max_down_move > max_up_move:
                y.append(0)  # Short
            else:
                y.append(1)  # Hold

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

        # Create DataLoader for mini-batch training
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Validation tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)

        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)
        class_weights = torch.FloatTensor([
            total_samples / (3 * max(count, 1)) for count in class_counts
        ]).to(self.device)

        print(f"Class weights: Short={class_weights[0]:.2f}, Hold={class_weights[1]:.2f}, Long={class_weights[2]:.2f}")

        # Training setup with Focal Loss (better for imbalanced classes and hard examples)
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Early stopping setup
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X, return_attention=False)  # Don't need attention during training
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor, return_attention=False)  # Don't need attention during validation
                val_loss = criterion(val_outputs, y_val_tensor)

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

            # Learning rate scheduling
            scheduler.step(val_loss)

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
                print(f"\n  All Predictions: Short={pred_dist[0]}, Hold={pred_dist[1]}, Long={pred_dist[2]}")
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

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
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

        # Compile model for faster inference after training (PyTorch 2.0+)
        if not self.compiled:
            try:
                self.model = torch.compile(self.model)
                self.compiled = True
                print("Model compiled for optimized inference")
            except Exception as e:
                print(f"Model compilation not available: {e}")

        print(f"\n{'='*50}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*50}\n")

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

            probabilities = torch.softmax(outputs, dim=1)[0]

            # IMPROVED: Compare LONG vs SHORT probabilities directly (ignore HOLD bias)
            # This overcomes model trained with 40% HOLD
            prob_short = probabilities[0].item()
            prob_hold = probabilities[1].item()
            prob_long = probabilities[2].item()

            # Determine signal by comparing directional probabilities
            # Only trade if directional conviction exceeds HOLD by a margin
            direction_margin = 0.02  # Need 2% edge over hold to trade (reduced from 5%)

            if prob_long > prob_hold + direction_margin and prob_long > prob_short:
                predicted_class = 2  # Long
                confidence = prob_long
            elif prob_short > prob_hold + direction_margin and prob_short > prob_long:
                predicted_class = 0  # Short
                confidence = prob_short
            else:
                predicted_class = 1  # Hold
                confidence = prob_hold

            # Handle NaN/inf in confidence
            if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
                print(f"WARNING: Invalid confidence value: {confidence}, using 0.0")
                confidence = 0.0

            # Log all probabilities for debugging
            print(f"Probabilities: SHORT={prob_short:.3f}, HOLD={prob_hold:.3f}, LONG={prob_long:.3f}")

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

        # Detect market regime
        regime = detect_market_regime(recent_bars_df, lookback=100)

        # Calculate recent accuracy if we have enough predictions
        recent_accuracy = None
        if len(self.recent_predictions) >= 20:
            correct = sum([1 for pred, actual in self.recent_predictions if pred == actual])
            recent_accuracy = correct / len(self.recent_predictions)

        # Get adaptive confidence threshold
        current_timestamp = recent_bars_df['time'].iloc[-1]
        adaptive_threshold = self.adaptive_thresholds.get_threshold(
            regime, current_timestamp, recent_accuracy
        )

        # Log Hurst values, regime, and P&L context with prediction
        current_pnl = recent_bars_df['dailyPnL'].iloc[-1] if 'dailyPnL' in recent_bars_df.columns else 0.0
        current_goal = recent_bars_df['dailyGoal'].iloc[-1] if 'dailyGoal' in recent_bars_df.columns else 0.0
        current_max_loss = recent_bars_df['dailyMaxLoss'].iloc[-1] if 'dailyMaxLoss' in recent_bars_df.columns else 0.0

        print(f"\n--- Prediction Context ---")
        print(f"Market Regime: {regime.upper()}")
        print(f"Adaptive Threshold: {adaptive_threshold:.2%} (vs fixed 65%)")
        if recent_accuracy is not None:
            print(f"Recent Accuracy: {recent_accuracy:.2%} (last {len(self.recent_predictions)} predictions)")

        # Log attention: which bars the model focused on
        print(f"\nAttention Focus (top {top_k} bars):")
        for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
            bars_ago = len(last_bar_attn) - 1 - idx
            print(f"  {i+1}. Bar -{bars_ago:2d} (weight: {weight:.3f})")

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

        # Get ATR (calculate if not in dataframe)
        if 'atr' in recent_bars_df.columns:
            atr = recent_bars_df['atr'].iloc[-1]
        else:
            # Calculate ATR on the fly
            atr_values = calculate_atr(
                recent_bars_df['high'].values,
                recent_bars_df['low'].values,
                recent_bars_df['close'].values
            )
            atr = atr_values[-1] if len(atr_values) > 0 else 15.0  # Default to 15 points

        # Get regime
        regime = detect_market_regime(recent_bars_df, lookback=min(100, len(recent_bars_df)-1))

        # Calculate trade parameters using risk manager
        risk_mgr = RiskManager()
        trade_params = risk_mgr.calculate_trade_parameters(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            atr=atr,
            regime=regime,
            account_balance=account_balance,
            tick_value=tick_value
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
            self.model.load_state_dict(checkpoint['model_state_dict'])

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

            # Compile model for faster inference after successful load
            if not self.compiled:
                try:
                    self.model = torch.compile(self.model)
                    self.compiled = True
                    print("Model compiled for optimized inference")
                except Exception as e:
                    print(f"Model compilation not available: {e}")

            # PERFORMANCE OPTIMIZATION: Apply dynamic quantization for CPU inference
            # Only quantize if model is trained (has parameters)
            if self.device.type == 'cpu' and self.is_trained:
                try:
                    # Store original model for potential retraining
                    self._original_model = self.model
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.LSTM, torch.nn.Linear},  # Quantize LSTM and Linear layers
                        dtype=torch.qint8
                    )
                    print("✓ Model quantized to INT8 for faster CPU inference")
                except Exception as e:
                    print(f"Note: Quantization not applied: {e}")

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
