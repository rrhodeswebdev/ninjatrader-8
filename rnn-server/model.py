import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from pathlib import Path
import json
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
            'tf2_position_in_bar': np.zeros(n_bars)
        }

    if len(df_secondary) == 0:
        # Return zeros if empty secondary data
        n_bars = len(df_primary)
        return {
            'tf2_close': np.zeros(n_bars),
            'tf2_close_change': np.zeros(n_bars),
            'tf2_high_low_range': np.zeros(n_bars),
            'tf2_volume': np.zeros(n_bars),
            'tf2_position_in_bar': np.zeros(n_bars)
        }

    n_bars = len(df_primary)
    aligned_features = {
        'tf2_close': np.zeros(n_bars),
        'tf2_close_change': np.zeros(n_bars),
        'tf2_high_low_range': np.zeros(n_bars),
        'tf2_volume': np.zeros(n_bars),
        'tf2_position_in_bar': np.zeros(n_bars)
    }

    # Convert to numpy for faster access
    primary_times = df_primary['time'].values
    secondary_times = df_secondary['time'].values
    secondary_close = df_secondary['close'].values
    secondary_high = df_secondary['high'].values
    secondary_low = df_secondary['low'].values
    secondary_volume = df_secondary['volume'].values if 'volume' in df_secondary.columns else np.zeros(len(df_secondary))

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
                aligned_features['tf2_position_in_bar'][i] = (df_primary['close'].iloc[i] - secondary_low[secondary_idx]) / range_size

            # Secondary close price change
            if secondary_idx > 0:
                aligned_features['tf2_close_change'][i] = (secondary_close[secondary_idx] - secondary_close[secondary_idx - 1]) / secondary_close[secondary_idx - 1]

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

    # 1. Delta (buy volume - sell volume)
    delta = ask_volume - bid_volume
    features['delta'] = delta

    # 2. Cumulative Delta (running sum of delta)
    cumulative_delta = np.cumsum(delta)
    features['cumulative_delta'] = cumulative_delta

    # 3. Volume Imbalance Ratio (ask / bid)
    volume_imbalance = np.zeros(n_bars)
    for i in range(n_bars):
        if bid_volume[i] > 0:
            volume_imbalance[i] = ask_volume[i] / (bid_volume[i] + 1e-8)
        else:
            volume_imbalance[i] = 1.0
    features['volume_imbalance'] = volume_imbalance

    # 4. Aggressive Buy Ratio (ask volume / total volume)
    aggressive_buy_ratio = np.zeros(n_bars)
    for i in range(n_bars):
        total = ask_volume[i] + bid_volume[i]
        if total > 0:
            aggressive_buy_ratio[i] = ask_volume[i] / total
        else:
            aggressive_buy_ratio[i] = 0.5
    features['aggressive_buy_ratio'] = aggressive_buy_ratio

    # 5. Volume relative to moving average
    volume_ratio = np.ones(n_bars)
    window = 20
    for i in range(window, n_bars):
        avg_volume = np.mean(volume[i-window:i])
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / (avg_volume + 1e-8)
    features['volume_ratio'] = volume_ratio

    # 6. Delta Divergence (price up but delta down, or vice versa)
    delta_divergence = np.zeros(n_bars)
    close_prices = df['close'].values
    for i in range(1, n_bars):
        price_change = close_prices[i] - close_prices[i-1]
        delta_change = delta[i] - delta[i-1]

        # Divergence: price and delta move in opposite directions
        if price_change > 0 and delta_change < 0:
            delta_divergence[i] = -1  # Bearish divergence
        elif price_change < 0 and delta_change > 0:
            delta_divergence[i] = 1   # Bullish divergence
        else:
            delta_divergence[i] = 0
    features['delta_divergence'] = delta_divergence

    # 7. Cumulative Delta Slope (rate of change)
    cumulative_delta_slope = np.zeros(n_bars)
    for i in range(5, n_bars):
        cumulative_delta_slope[i] = cumulative_delta[i] - cumulative_delta[i-5]
    features['cumulative_delta_slope'] = cumulative_delta_slope

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
    skewness = np.zeros(n_bars)
    kurtosis_vals = np.zeros(n_bars)
    window = 20

    returns = np.diff(ohlc[:, 3]) / ohlc[:-1, 3]
    for i in range(window, n_bars):
        recent_returns = returns[i-window:i]
        if len(recent_returns) > 3:  # Need minimum for stats
            skewness[i] = stats.skew(recent_returns)
            kurtosis_vals[i] = stats.kurtosis(recent_returns)
    features['skewness'] = skewness
    features['kurtosis'] = kurtosis_vals

    # 6. Rolling Min/Max Distance (Position in Range)
    position_in_range = np.zeros(n_bars)
    window = 20
    for i in range(window, n_bars):
        rolling_max = np.max(ohlc[i-window:i+1, 3])
        rolling_min = np.min(ohlc[i-window:i+1, 3])
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

    # Multiple window sizes for different timeframes
    windows = [5, 10, 20, 50]

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


class TradingRNN(nn.Module):
    """
    LSTM-based RNN for predicting trade signals with 3-class output
    Features (57 total):
    - OHLC (4)
    - Hurst H & C (2)
    - ATR (1)
    - Velocity & Acceleration (2)
    - Range Ratio & Wick Ratio (2)
    - Gap Up/Down/Filled (3)
    - Swing High/Low & Bars Since (4)
    - Skewness & Kurtosis (2)
    - Position in Range (1)
    - Higher Highs/Lower Lows/Trend Strength (3)
    - Deviation Features:
      * Mean Deviation (4 windows: 5,10,20,50)
      * Median Deviation (4 windows)
      * Std Deviation (4 windows)
      * Z-Score (4 windows)
      * BB Width (4 windows)
      * Vol Acceleration (1)
      * High/Low Deviation (2)
      Total: 23 deviation features
    - Daily P&L Features:
      * P&L % of Goal (1)
      * P&L % of Max Loss (1)
      * Risk Headroom (1)
      Total: 3 P&L features
    - Order Flow Features:
      * Delta (1)
      * Cumulative Delta (1)
      * Volume Imbalance (1)
      * Aggressive Buy Ratio (1)
      * Volume Ratio (1)
      * Delta Divergence (1)
      * Cumulative Delta Slope (1)
      Total: 7 order flow features
    - Multi-Timeframe Features (5-min):
      * TF2 Close (1)
      * TF2 Close Change (1)
      * TF2 High-Low Range (1)
      * TF2 Volume (1)
      * TF2 Position in Bar (1)
      Total: 5 multi-timeframe features
    """
    def __init__(self, input_size=62, hidden_size=64, num_layers=2, output_size=3):
        super(TradingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with dropout between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers with dropout
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)  # No activation - softmax applied in loss function

        return out

class TradingModel:
    """
    Wrapper class for training and prediction with state management
    """
    def __init__(self, sequence_length=20, model_path='models/trading_model.pth'):
        self.model = TradingRNN(input_size=62, hidden_size=64, num_layers=2, output_size=3)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.is_trained = False
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compiled = False  # Track compilation state

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

            # Threshold = 0.5x volatility (adjust multiplier as needed)
            # This ensures ~60-70% of data is not HOLD
            self.signal_threshold = max(0.01, volatility * 0.5)

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

        # Extract daily P&L features (normalized)
        daily_pnl = df['dailyPnL'].values
        daily_goal = df['dailyGoal'].values
        daily_max_loss = df['dailyMaxLoss'].values

        # Normalize P&L features
        # 1. P&L as percentage of goal (-1 to +1 range, where 1 = goal reached)
        pnl_pct_of_goal = np.zeros(n_bars)
        for i in range(n_bars):
            if daily_goal[i] > 0:
                pnl_pct_of_goal[i] = daily_pnl[i] / daily_goal[i]
            else:
                pnl_pct_of_goal[i] = 0.0

        # 2. P&L as percentage of max loss (-1 to +1 range, where -1 = max loss hit)
        pnl_pct_of_loss = np.zeros(n_bars)
        for i in range(n_bars):
            if daily_max_loss[i] > 0:
                pnl_pct_of_loss[i] = daily_pnl[i] / daily_max_loss[i]
            else:
                pnl_pct_of_loss[i] = 0.0

        # 3. Risk headroom: how close to max loss (0 = at max loss, 1 = no loss)
        risk_headroom = np.ones(n_bars)
        for i in range(n_bars):
            if daily_max_loss[i] > 0:
                # If P&L is negative, calculate how much headroom remains
                if daily_pnl[i] < 0:
                    risk_headroom[i] = max(0.0, 1.0 + (daily_pnl[i] / daily_max_loss[i]))
                else:
                    risk_headroom[i] = 1.0  # Full headroom when profitable
            else:
                risk_headroom[i] = 1.0

        features = np.column_stack([
            ohlc,                                    # 4
            hurst_H_values,                          # 1
            hurst_C_values,                          # 1
            atr,                                     # 1
            price_features['velocity'],              # 1
            price_features['acceleration'],          # 1
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
            # Deviation features - 4 windows × 5 metrics = 20
            price_features['mean_dev_5'],            # 1
            price_features['mean_dev_10'],           # 1
            price_features['mean_dev_20'],           # 1
            price_features['mean_dev_50'],           # 1
            price_features['median_dev_5'],          # 1
            price_features['median_dev_10'],         # 1
            price_features['median_dev_20'],         # 1
            price_features['median_dev_50'],         # 1
            price_features['std_dev_5'],             # 1
            price_features['std_dev_10'],            # 1
            price_features['std_dev_20'],            # 1
            price_features['std_dev_50'],            # 1
            price_features['z_score_5'],             # 1
            price_features['z_score_10'],            # 1
            price_features['z_score_20'],            # 1
            price_features['z_score_50'],            # 1
            price_features['bb_width_5'],            # 1
            price_features['bb_width_10'],           # 1
            price_features['bb_width_20'],           # 1
            price_features['bb_width_50'],           # 1
            # Additional deviation features: 3
            price_features['vol_acceleration'],      # 1
            price_features['high_deviation'],        # 1
            price_features['low_deviation'],         # 1
            # Daily P&L features: 3
            pnl_pct_of_goal,                         # 1
            pnl_pct_of_loss,                         # 1
            risk_headroom,                           # 1
            # Order flow features: 7
            order_flow_features['delta'],            # 1
            order_flow_features['cumulative_delta'], # 1
            order_flow_features['volume_imbalance'], # 1
            order_flow_features['aggressive_buy_ratio'], # 1
            order_flow_features['volume_ratio'],     # 1
            order_flow_features['delta_divergence'], # 1
            order_flow_features['cumulative_delta_slope'], # 1
            # Multi-timeframe features: 5
            secondary_features['tf2_close'],         # 1
            secondary_features['tf2_close_change'],  # 1
            secondary_features['tf2_high_low_range'], # 1
            secondary_features['tf2_volume'],        # 1
            secondary_features['tf2_position_in_bar'] # 1
        ])

        # Only log during training
        if fit_scaler:
            print(f"Total features: {features.shape[1]} (OHLC:4 + Hurst:2 + ATR:1 + Price:18 + Deviation:23 + DailyPnL:3 + OrderFlow:7 + MultiTF:5)")

        # Scale the features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])

            # 3-class label: 0=short, 1=hold, 2=long
            # Look ahead 1 bar to see price movement
            current_close = ohlc[i + self.sequence_length - 1, 3]
            future_close = ohlc[i + self.sequence_length, 3]

            price_change_pct = (future_close - current_close) / current_close * 100

            # Use adaptive threshold
            if price_change_pct > self.signal_threshold:
                y.append(2)  # Long
            elif price_change_pct < -self.signal_threshold:
                y.append(0)  # Short
            else:
                y.append(1)  # Hold

        return np.array(X), np.array(y)

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

        # Check if we can stratify (need at least 2 samples per class)
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        can_stratify = min_samples >= 2 and len(X) >= 10

        print(f"Class distribution: {dict(zip(unique, counts))}")

        # Train/validation split
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            print("Using stratified split")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            print("Using non-stratified split (insufficient samples per class)")

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

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

        # Training setup with weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
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
                outputs = self.model(batch_X)
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
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

                # Calculate metrics
                val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_accuracy = accuracy_score(y_val, val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, val_predictions, average='weighted', zero_division=0
                )

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

                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
                print(f"  Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"  Predictions: Short={pred_dist[0]}, Hold={pred_dist[1]}, Long={pred_dist[2]}")
                print(f"  Trading Metrics (with daily limits):")
                print(f"    Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
                print(f"    Win Rate: {trading_metrics['win_rate']*100:.2f}%")
                print(f"    Profit Factor: {trading_metrics['profit_factor']:.4f}")
                print(f"    Expectancy: {trading_metrics['expectancy']:.6f}")
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
            val_outputs = self.model(X_val_tensor)
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
            outputs = self.model(X_tensor)
            if self.device.type == 'cuda':
                outputs = outputs.float()  # Convert back to FP32 for softmax

            probabilities = torch.softmax(outputs, dim=1)[0]

            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # Map class to signal: 0=short, 1=hold, 2=long
        signal_map = {0: 'short', 1: 'hold', 2: 'long'}
        signal = signal_map[predicted_class]

        # Log Hurst values and P&L context with prediction
        current_pnl = recent_bars_df['dailyPnL'].iloc[-1] if 'dailyPnL' in recent_bars_df.columns else 0.0
        current_goal = recent_bars_df['dailyGoal'].iloc[-1] if 'dailyGoal' in recent_bars_df.columns else 0.0
        current_max_loss = recent_bars_df['dailyMaxLoss'].iloc[-1] if 'dailyMaxLoss' in recent_bars_df.columns else 0.0

        print(f"\n--- Prediction Context ---")
        print(f"Current Hurst H: {current_hurst_H:.4f} ", end="")
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
        print(f"-------------------------\n")

        return signal, confidence

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
