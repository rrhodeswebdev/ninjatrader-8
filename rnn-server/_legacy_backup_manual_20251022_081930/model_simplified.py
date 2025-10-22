"""
SIMPLIFIED TRADING MODEL - 25 Core Features
===========================================

This module contains a streamlined version with only the most predictive features.
Reduces overfitting and improves generalization.

Core Features (25 total):
1. OHLC (4): open, high, low, close
2. Hurst Exponent (1): H value only (removed C)
3. ATR (1): Average True Range
4. Price Momentum (2): velocity, acceleration
5. Key Price Patterns (3): range_ratio, position_in_range, trend_strength
6. Volatility (2): std_dev_20, volatility_regime
7. Volume (1): volume_ratio
8. Time Features (3): hour_of_day, is_opening_period, is_closing_period
9. Microstructure (2): vwap_deviation, volume_surge
10. Multi-Timeframe (5): tf2_close_change, tf2_trend_direction, tf2_momentum, tf2_volatility, tf2_alignment_score
11. Market Regime (1): regime indicator

REMOVED (37 features):
- Hurst C value (redundant)
- 12 redundant price patterns (gaps, wicks, swings, etc.)
- 13 deviation features (kept only std_dev_20)
- 3 microstructure features (effective_spread, large_print, price_acceleration)
- 2 volatility features (parkinson, volume_regime)
- 2 time features (minutes_into_session, minutes_to_close)
- 4 secondary timeframe features (tf2_close, tf2_high_low_range, tf2_volume, tf2_position_in_bar)
- Skewness, kurtosis (noisy)
"""

import numpy as np
import pandas as pd
from scipy import stats
from hurst import compute_Hc


def calculate_hurst_exponent_simple(prices, min_window=10):
    """Simplified Hurst calculation - returns only H"""
    if len(prices) < min_window:
        return 0.5

    try:
        H, _, _ = compute_Hc(prices, kind='price', simplified=True)
        return max(0.0, min(1.0, H))
    except:
        return 0.5


def calculate_atr_simple(high, low, close, period=14):
    """Calculate ATR (vectorized)"""
    n = len(high)
    if n < period + 1:
        return np.zeros(n)

    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(hl, hc), lc)

    atr_values = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    atr = np.zeros(n)
    atr[1:] = atr_values

    return atr


def detect_market_regime_simple(close, high, low, lookback=100):
    """
    Simple regime detection: trending vs ranging
    Returns: 'trending', 'ranging', or 'unknown'
    """
    if len(close) < lookback:
        return 'unknown'

    recent_close = close[-lookback:]
    recent_high = high[-lookback:]
    recent_low = low[-lookback:]

    # Simple ADX calculation
    plus_dm = []
    minus_dm = []
    for i in range(1, len(recent_high)):
        high_diff = recent_high[i] - recent_high[i-1]
        low_diff = recent_low[i-1] - recent_low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)

        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)

    avg_plus = np.mean(plus_dm[-14:]) if len(plus_dm) >= 14 else 0
    avg_minus = np.mean(minus_dm[-14:]) if len(minus_dm) >= 14 else 0

    total = avg_plus + avg_minus
    if total > 0:
        adx_score = abs(avg_plus - avg_minus) / total
        return 'trending' if adx_score > 0.25 else 'ranging'

    return 'unknown'


def prepare_simplified_features(df, df_secondary=None):
    """
    Prepare simplified 25-feature dataset

    Args:
        df: Primary timeframe DataFrame
        df_secondary: Optional secondary timeframe

    Returns:
        numpy array of shape (n_bars, 25)
    """
    n_bars = len(df)
    ohlc = df[['open', 'high', 'low', 'close']].values
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.zeros(n_bars)

    # 1. OHLC (4 features)
    features_list = [ohlc]

    # 2. Hurst Exponent (1 feature) - simplified
    hurst_H = []
    for i in range(n_bars):
        if i < 100:
            H = 0.5
        elif i % 10 == 0 or i == n_bars - 1:
            prices = close[max(0, i-99):i+1]
            H = calculate_hurst_exponent_simple(prices)
        else:
            H = hurst_H[-1] if hurst_H else 0.5
        hurst_H.append(H)
    features_list.append(np.array(hurst_H).reshape(-1, 1))

    # 3. ATR (1 feature)
    atr = calculate_atr_simple(high, low, close)
    features_list.append(atr.reshape(-1, 1))

    # 4. Price Momentum (2 features)
    velocity = np.zeros(n_bars)
    acceleration = np.zeros(n_bars)
    for i in range(5, n_bars):
        velocity[i] = (close[i] - close[i-5]) / 5
        if i > 5:
            acceleration[i] = velocity[i] - velocity[i-1]
    features_list.append(velocity.reshape(-1, 1))
    features_list.append(acceleration.reshape(-1, 1))

    # 5. Key Price Patterns (3 features)
    # 5a. Range ratio
    range_ratio = np.ones(n_bars)
    for i in range(1, n_bars):
        current_range = high[i] - low[i]
        prev_range = high[i-1] - low[i-1]
        range_ratio[i] = current_range / (prev_range + 1e-8)
    features_list.append(range_ratio.reshape(-1, 1))

    # 5b. Position in range (NO LOOK-AHEAD)
    position_in_range = np.zeros(n_bars)
    for i in range(20, n_bars):
        rolling_max = np.max(close[i-20:i])  # Exclude current bar
        rolling_min = np.min(close[i-20:i])
        range_size = rolling_max - rolling_min
        if range_size > 1e-8:
            position_in_range[i] = (close[i] - rolling_min) / range_size
    features_list.append(position_in_range.reshape(-1, 1))

    # 5c. Trend strength
    trend_strength = np.zeros(n_bars)
    for i in range(5, n_bars):
        hh_count = sum([1 for j in range(i-4, i+1) if high[j] > high[j-1]])
        ll_count = sum([1 for j in range(i-4, i+1) if low[j] < low[j-1]])
        trend_strength[i] = hh_count - ll_count
    features_list.append(trend_strength.reshape(-1, 1))

    # 6. Volatility (2 features)
    # 6a. 20-period standard deviation (NO LOOK-AHEAD)
    std_dev_20 = np.zeros(n_bars)
    for i in range(20, n_bars):
        window_prices = close[i-20:i]  # Exclude current
        std_dev_20[i] = np.std(window_prices)
    features_list.append(std_dev_20.reshape(-1, 1))

    # 6b. Volatility regime
    volatility_regime = np.ones(n_bars)
    for i in range(100, n_bars):
        returns_recent = np.diff(close[i-20:i]) / close[i-21:i-1]
        current_vol = np.std(returns_recent) if len(returns_recent) > 1 else 0
        returns_hist = np.diff(close[i-100:i]) / close[i-101:i-1]
        hist_vol = np.std(returns_hist) if len(returns_hist) > 1 else 1e-8
        volatility_regime[i] = np.clip(current_vol / hist_vol, 0, 5)
    features_list.append(volatility_regime.reshape(-1, 1))

    # 7. Volume (1 feature)
    volume_ratio = np.ones(n_bars)
    for i in range(20, n_bars):
        avg_volume = np.mean(volume[i-20:i])
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / avg_volume
    features_list.append(volume_ratio.reshape(-1, 1))

    # 8. Time Features (3 features)
    times = pd.to_datetime(df['time'])
    hour_of_day = times.dt.hour + times.dt.minute / 60.0
    normalized_hour = np.clip((hour_of_day - 9.5) / 6.5, 0, 1)
    features_list.append(normalized_hour.values.reshape(-1, 1))

    minutes_into_session = (hour_of_day - 9.5) * 60
    minutes_into_session = np.clip(minutes_into_session, 0, 390)
    is_opening = (minutes_into_session <= 30).astype(float)
    is_closing = ((390 - minutes_into_session) <= 30).astype(float)
    features_list.append(is_opening.values.reshape(-1, 1))
    features_list.append(is_closing.values.reshape(-1, 1))

    # 9. Microstructure (2 features)
    # 9a. VWAP deviation
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
    features_list.append(vwap_dev.reshape(-1, 1))

    # 9b. Volume surge
    volume_surge = np.ones(n_bars)
    for i in range(20, n_bars):
        avg_volume = np.mean(volume[i-20:i])
        if avg_volume > 0:
            volume_surge[i] = np.clip(volume[i] / avg_volume, 0, 5)
    features_list.append(volume_surge.reshape(-1, 1))

    # 10. Multi-Timeframe (5 features) - simplified alignment
    if df_secondary is not None and len(df_secondary) > 0:
        tf2_close_change = np.zeros(n_bars)
        tf2_trend = np.zeros(n_bars)
        tf2_momentum = np.zeros(n_bars)
        tf2_volatility = np.zeros(n_bars)
        tf2_alignment = np.zeros(n_bars)

        # Simple alignment logic (production code would be more sophisticated)
        primary_times = df['time'].values
        secondary_times = df_secondary['time'].values
        secondary_close = df_secondary['close'].values

        secondary_idx = 0
        for i in range(n_bars):
            primary_time = primary_times[i]
            while secondary_idx < len(secondary_times) - 1 and secondary_times[secondary_idx + 1] <= primary_time:
                secondary_idx += 1

            if secondary_idx < len(secondary_times) and secondary_idx > 0:
                tf2_close_change[i] = (secondary_close[secondary_idx] - secondary_close[secondary_idx-1]) / secondary_close[secondary_idx-1]

                if secondary_idx >= 2:
                    ema_slope = (secondary_close[secondary_idx] - secondary_close[secondary_idx-2]) / secondary_close[secondary_idx-2]
                    tf2_trend[i] = np.tanh(ema_slope * 100)

                if secondary_idx >= 3:
                    roc = (secondary_close[secondary_idx] - secondary_close[secondary_idx-3]) / secondary_close[secondary_idx-3]
                    tf2_momentum[i] = np.tanh(roc * 50)

                if secondary_idx >= 5:
                    recent_vol = np.std(secondary_close[max(0, secondary_idx-5):secondary_idx+1])
                    tf2_volatility[i] = recent_vol / (secondary_close[secondary_idx] + 1e-8)

                if i >= 5:
                    primary_trend = (close[i] - close[i-5]) / close[i-5]
                    if abs(primary_trend) > 1e-8 and abs(tf2_close_change[i]) > 1e-8:
                        tf2_alignment[i] = np.sign(primary_trend) * np.sign(tf2_close_change[i])

        features_list.extend([
            tf2_close_change.reshape(-1, 1),
            tf2_trend.reshape(-1, 1),
            tf2_momentum.reshape(-1, 1),
            tf2_volatility.reshape(-1, 1),
            tf2_alignment.reshape(-1, 1)
        ])
    else:
        # No secondary timeframe - use zeros
        features_list.extend([np.zeros((n_bars, 1)) for _ in range(5)])

    # 11. Market Regime (1 feature)
    regime = detect_market_regime_simple(close, high, low, lookback=min(100, len(close)-1))
    regime_numeric = 1.0 if regime == 'trending' else -1.0 if regime == 'ranging' else 0.0
    regime_feature = np.full((n_bars, 1), regime_numeric)
    features_list.append(regime_feature)

    # Stack all features
    features = np.column_stack([f if f.ndim == 2 else f.reshape(-1, 1) for f in features_list])

    assert features.shape[1] == 25, f"Expected 25 features, got {features.shape[1]}"

    return features


if __name__ == '__main__':
    print("Simplified Model - 25 Core Features")
    print("="*50)
    print("\nFeature breakdown:")
    print("  1. OHLC: 4")
    print("  2. Hurst H: 1")
    print("  3. ATR: 1")
    print("  4. Momentum: 2")
    print("  5. Price Patterns: 3")
    print("  6. Volatility: 2")
    print("  7. Volume: 1")
    print("  8. Time: 3")
    print("  9. Microstructure: 2")
    print(" 10. Multi-TF: 5")
    print(" 11. Regime: 1")
    print(" " + "-"*48)
    print(" Total: 25 features")
