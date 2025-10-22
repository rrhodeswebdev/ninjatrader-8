"""Pure functions for technical indicators and calculations.

These functions are extracted from model.py and refactored to be pure:
- Deterministic output
- No side effects
- No state mutation
"""

import numpy as np
from typing import Tuple
import math


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX) for trend strength detection.

    Pure function with no side effects.

    ADX > 25: Strong trend
    ADX < 20: Ranging/weak trend

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period (default 14)

    Returns:
        Array of ADX values
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
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Smooth +DM, -DM, and TR using Wilder's smoothing
    smooth_plus_dm = np.zeros(n)
    smooth_minus_dm = np.zeros(n)
    smooth_tr = np.zeros(n)

    # First values
    smooth_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period+1])
    smooth_tr[period] = np.sum(tr[1:period+1])

    # Subsequent values
    for i in range(period+1, n):
        smooth_plus_dm[i] = smooth_plus_dm[i-1] - (smooth_plus_dm[i-1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i-1] - (smooth_minus_dm[i-1] / period) + minus_dm[i]
        smooth_tr[i] = smooth_tr[i-1] - (smooth_tr[i-1] / period) + tr[i]

    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    for i in range(period, n):
        if smooth_tr[i] != 0:
            plus_di[i] = (smooth_plus_dm[i] / smooth_tr[i]) * 100
            minus_di[i] = (smooth_minus_dm[i] / smooth_tr[i]) * 100

    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100

    # Calculate ADX (smoothed DX)
    adx = np.zeros(n)
    adx[period*2-1] = np.mean(dx[period:period*2])

    for i in range(period*2, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).

    Pure function for volatility measurement.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period (default 14)

    Returns:
        Array of ATR values
    """
    n = len(high)
    if n < 2:
        return np.zeros(n)

    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR (smoothed TR)
    atr = np.zeros(n)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).

    Pure function for momentum measurement.

    Args:
        close: Array of close prices
        period: RSI period (default 14)

    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)

    # Calculate price changes
    deltas = np.diff(close)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial averages
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Calculate smoothed averages
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period

    # Calculate RS and RSI
    rsi = np.zeros(n)
    rsi[:period] = 50.0  # Neutral value for initial period

    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Pure function for trend and momentum.

    Args:
        close: Array of close prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    n = len(close)

    if n < slow:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    # Calculate EMAs
    ema_fast = exponential_moving_average(close, fast)
    ema_slow = exponential_moving_average(close, slow)

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = exponential_moving_average(macd_line, signal)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def exponential_moving_average(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    Pure function for smoothed averages.

    Args:
        values: Array of values
        period: EMA period

    Returns:
        Array of EMA values
    """
    n = len(values)
    if n < period:
        return np.zeros(n)

    ema = np.zeros(n)
    multiplier = 2 / (period + 1)

    # Start with SMA
    ema[period-1] = np.mean(values[:period])

    # Calculate EMA
    for i in range(period, n):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema


def simple_moving_average(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).

    Pure function for simple averages.

    Args:
        values: Array of values
        period: SMA period

    Returns:
        Array of SMA values
    """
    n = len(values)
    if n < period:
        return np.zeros(n)

    sma = np.zeros(n)

    for i in range(period-1, n):
        sma[i] = np.mean(values[i-period+1:i+1])

    return sma


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Pure function for volatility bands.

    Args:
        close: Array of close prices
        period: Period for moving average (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = simple_moving_average(close, period)

    n = len(close)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)

    for i in range(period-1, n):
        std = np.std(close[i-period+1:i+1])
        upper_band[i] = middle_band[i] + (num_std * std)
        lower_band[i] = middle_band[i] - (num_std * std)

    return upper_band, middle_band, lower_band


def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator.

    Pure function for momentum.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: Period (default 14)

    Returns:
        Tuple of (%K, %D)
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)

    k_values = np.zeros(n)

    for i in range(period-1, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])

        if highest_high - lowest_low != 0:
            k_values[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        else:
            k_values[i] = 50.0

    # %D is 3-period SMA of %K
    d_values = simple_moving_average(k_values, 3)

    return k_values, d_values


def augment_time_series(X_sequence: np.ndarray, augmentation_prob: float = 0.3) -> np.ndarray:
    """
    Data augmentation for time series trading data.

    Note: This function uses randomness, so it's not pure in the strictest sense,
    but it's deterministic given the same random seed.

    Augmentation types:
    - Jitter: Add small random noise
    - Scale: Scale magnitude slightly
    - Magnitude warp: Warp magnitude of random features

    Args:
        X_sequence: Input sequence
        augmentation_prob: Probability of augmentation (0-1)

    Returns:
        Augmented sequence
    """
    if np.random.random() > augmentation_prob:
        return X_sequence.copy()

    aug_type = np.random.choice(['jitter', 'scale', 'magnitude_warp'])

    if aug_type == 'jitter':
        noise = np.random.normal(0, 0.005, X_sequence.shape)
        return X_sequence + noise

    elif aug_type == 'scale':
        scale = np.random.uniform(0.98, 1.02)
        return X_sequence * scale

    elif aug_type == 'magnitude_warp':
        n_features = X_sequence.shape[1]
        n_warp = max(1, n_features // 10)
        warp_features = np.random.choice(n_features, size=n_warp, replace=False)
        warped = X_sequence.copy()
        warped[:, warp_features] *= np.random.uniform(0.95, 1.05, size=(warped.shape[0], n_warp))
        return warped

    return X_sequence.copy()


def normalize_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize values to mean=0, std=1.

    Pure function for normalization.

    Args:
        values: Array of values

    Returns:
        Normalized values
    """
    if len(values) == 0:
        return values

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return np.zeros_like(values)

    return (values - mean) / std


def clamp_values(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clamp array values to range [min_val, max_val].

    Pure function for clamping.

    Args:
        values: Array of values
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped values
    """
    return np.clip(values, min_val, max_val)
