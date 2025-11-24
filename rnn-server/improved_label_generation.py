"""
Improved Label Generation Strategies

Addresses the 40% HOLD bias problem and creates better training labels.

Key improvements:
1. Dynamic hold percentage based on market regime
2. Multi-horizon labeling (short, medium, long-term)
3. Triple-barrier method (from Advances in Financial ML)
4. Profit-based labeling instead of movement-based
5. Meta-labeling for trade filtering
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


def calculate_triple_barrier_labels(df: pd.DataFrame,
                                    profit_taking_multiple: float = 2.0,
                                    stop_loss_multiple: float = 1.0,
                                    max_holding_period: int = 10,
                                    min_return_threshold: float = 0.0015) -> np.ndarray:
    """
    Triple Barrier Method for Label Generation

    From "Advances in Financial Machine Learning" by Marcos Lpez de Prado

    Creates labels based on which barrier is hit first:
    - Upper barrier (profit target): Label as LONG/SHORT (depending on direction)
    - Lower barrier (stop loss): Label as opposite direction
    - Time barrier (max holding): Label as HOLD

    Args:
        df: DataFrame with OHLC data
        profit_taking_multiple: Profit target as multiple of volatility
        stop_loss_multiple: Stop loss as multiple of volatility
        max_holding_period: Maximum bars to hold
        min_return_threshold: Minimum return to consider (filters noise)

    Returns:
        labels: Array of labels (0=SHORT, 1=HOLD, 2=LONG)
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    n = len(close)
    labels = np.ones(n, dtype=int)  # Default to HOLD

    # Calculate volatility (using ATR or recent std dev)
    returns = np.diff(close) / close[:-1]
    volatility = pd.Series(returns).rolling(20).std().values
    volatility = np.concatenate([[volatility[20]], volatility])  # Pad first value

    for i in range(n - max_holding_period):
        vol = volatility[i] if not np.isnan(volatility[i]) else 0.01

        # Define barriers
        entry_price = close[i]
        upper_barrier = entry_price * (1 + profit_taking_multiple * vol)
        lower_barrier = entry_price * (1 - stop_loss_multiple * vol)

        # Look ahead for barrier hits
        for j in range(1, max_holding_period + 1):
            if i + j >= n:
                break

            future_high = high[i + j]
            future_low = low[i + j]

            # Check if upper barrier hit (profit target)
            if future_high >= upper_barrier:
                return_pct = (upper_barrier - entry_price) / entry_price
                if abs(return_pct) > min_return_threshold:
                    labels[i] = 2  # LONG
                break

            # Check if lower barrier hit (stop loss)
            elif future_low <= lower_barrier:
                return_pct = (entry_price - lower_barrier) / entry_price
                if abs(return_pct) > min_return_threshold:
                    labels[i] = 0  # SHORT
                break

        # If no barrier hit within max_holding_period, stays as HOLD

    return labels


def calculate_regime_adaptive_labels(df: pd.DataFrame,
                                     lookahead_bars: int = 3,
                                     base_hold_pct: float = 0.30) -> np.ndarray:
    """
    Regime-Adaptive Label Generation

    Adjusts hold percentage based on market regime:
    - Trending markets: Lower hold percentage (more signals)
    - Ranging markets: Higher hold percentage (fewer signals)
    - High volatility: Higher hold percentage (avoid noise)

    Args:
        df: DataFrame with OHLC data
        lookahead_bars: Bars to look ahead
        base_hold_pct: Base hold percentage (will be adjusted)

    Returns:
        labels: Array of labels (0=SHORT, 1=HOLD, 2=LONG)
    """
    from model import detect_market_regime, calculate_adx

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    n = len(close)
    labels = np.zeros(n, dtype=int)

    # Calculate market regime for each window
    window_size = 100

    for i in range(window_size, n - lookahead_bars):
        # Get regime for this window
        window_df = df.iloc[max(0, i-window_size):i+1]
        regime = detect_market_regime(window_df, lookback=min(window_size, len(window_df)-1))

        # Adjust hold percentage based on regime
        regime_hold_pct = {
            'trending_normal': base_hold_pct * 0.7,      # 21% hold (more signals)
            'trending_high_vol': base_hold_pct * 0.85,   # 25.5% hold
            'ranging_normal': base_hold_pct * 1.3,       # 39% hold (fewer signals)
            'ranging_low_vol': base_hold_pct * 1.5,      # 45% hold
            'high_vol_chaos': base_hold_pct * 1.8,       # 54% hold (avoid noise)
            'transitional': base_hold_pct * 1.1,         # 33% hold
            'unknown': base_hold_pct                      # 30% hold
        }

        hold_pct = regime_hold_pct.get(regime, base_hold_pct)

        # Calculate future move
        future_slice = close[i+1:i+1+lookahead_bars]
        if len(future_slice) < lookahead_bars:
            continue

        max_up = np.max((future_slice - close[i]) / close[i])
        max_down = np.max((close[i] - future_slice) / close[i])

        # Adaptive threshold
        abs_change = max(max_up, max_down)

        # Use regime-specific hold percentage
        local_changes = np.abs(np.diff(close[max(0, i-100):i+1]) / close[max(0, i-100):i])
        percentile_threshold = np.percentile(local_changes, hold_pct * 100)

        # Label
        if abs_change < percentile_threshold:
            labels[i] = 1  # HOLD
        elif max_up > max_down:
            labels[i] = 2  # LONG
        else:
            labels[i] = 0  # SHORT

    return labels


def calculate_profit_based_labels(df: pd.DataFrame,
                                  lookahead_bars: int = 5,
                                  min_profit_threshold: float = 0.002,
                                  commission: float = 0.0001) -> np.ndarray:
    """
    Profit-Based Label Generation

    Labels based on expected profit after costs, not just price movement.

    Only generates LONG/SHORT signals if expected profit > threshold.
    Otherwise labeled as HOLD.

    Args:
        df: DataFrame with OHLC data
        lookahead_bars: Bars to simulate holding
        min_profit_threshold: Minimum profit (0.002 = 0.2% = 2 ticks on ES)
        commission: Round-trip commission (0.0001 = 0.01% = $5 RT on ES)

    Returns:
        labels: Array of labels (0=SHORT, 1=HOLD, 2=LONG)
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    n = len(close)
    labels = np.ones(n, dtype=int)  # Default to HOLD

    for i in range(n - lookahead_bars):
        entry_price = close[i]

        # Simulate long trade
        future_prices = close[i+1:i+1+lookahead_bars]
        max_profit_long = np.max(future_prices - entry_price) / entry_price - commission

        # Simulate short trade
        max_profit_short = np.max(entry_price - future_prices) / entry_price - commission

        # Label based on best profit
        if max_profit_long > min_profit_threshold and max_profit_long > max_profit_short:
            labels[i] = 2  # LONG
        elif max_profit_short > min_profit_threshold and max_profit_short > max_profit_long:
            labels[i] = 0  # SHORT
        else:
            labels[i] = 1  # HOLD (not profitable enough)

    return labels


def calculate_multi_horizon_labels(df: pd.DataFrame,
                                   short_horizon: int = 3,
                                   medium_horizon: int = 5,
                                   long_horizon: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Horizon Label Generation

    Creates labels for different time horizons and combines them.

    Uses voting:
    - If 2+ horizons agree on direction: Use that direction
    - If all disagree: HOLD

    Args:
        df: DataFrame with OHLC data
        short_horizon: Short-term lookahead
        medium_horizon: Medium-term lookahead
        long_horizon: Long-term lookahead

    Returns:
        labels: Combined labels
        horizon_labels: Individual horizon labels (for analysis)
    """
    close = df['close'].values
    n = len(close)

    # Calculate labels for each horizon
    horizon_labels = {}

    for name, horizon in [('short', short_horizon),
                          ('medium', medium_horizon),
                          ('long', long_horizon)]:

        labels = np.ones(n, dtype=int)

        for i in range(n - horizon):
            future_slice = close[i+1:i+1+horizon]
            if len(future_slice) < horizon:
                continue

            # Simple up/down/sideways
            entry = close[i]
            exit_price = future_slice[-1]  # Final price at horizon

            change_pct = (exit_price - entry) / entry

            if change_pct > 0.0015:  # >0.15% = LONG
                labels[i] = 2
            elif change_pct < -0.0015:  # <-0.15% = SHORT
                labels[i] = 0
            else:
                labels[i] = 1  # HOLD

        horizon_labels[name] = labels

    # Voting: Combine all horizons
    combined_labels = np.ones(n, dtype=int)

    for i in range(n):
        votes = [horizon_labels['short'][i],
                horizon_labels['medium'][i],
                horizon_labels['long'][i]]

        # Count votes
        long_votes = votes.count(2)
        short_votes = votes.count(0)

        # Majority voting (need 2+ agreeing)
        if long_votes >= 2:
            combined_labels[i] = 2
        elif short_votes >= 2:
            combined_labels[i] = 0
        else:
            combined_labels[i] = 1  # No consensus = HOLD

    return combined_labels, horizon_labels


def calculate_meta_labels(df: pd.DataFrame,
                          primary_labels: np.ndarray,
                          lookahead_bars: int = 5) -> np.ndarray:
    """
    Meta-Labeling

    Secondary model that predicts:
    "Should we take this trade (predicted by primary model)?"

    Binary classification:
    - 1: Trade is profitable (take it)
    - 0: Trade is unprofitable (skip it)

    Use this to filter primary model signals.

    Args:
        df: DataFrame with OHLC data
        primary_labels: Labels from primary model (0=SHORT, 1=HOLD, 2=LONG)
        lookahead_bars: Bars to evaluate trade outcome

    Returns:
        meta_labels: Binary labels (0=skip, 1=take) for non-HOLD predictions
    """
    close = df['close'].values
    n = len(close)

    meta_labels = np.zeros(n, dtype=int)

    for i in range(n - lookahead_bars):
        primary_signal = primary_labels[i]

        if primary_signal == 1:  # HOLD - no meta-label needed
            continue

        # Evaluate if this trade would be profitable
        entry_price = close[i]
        future_prices = close[i+1:i+1+lookahead_bars]

        if primary_signal == 2:  # LONG trade
            # Check if long would be profitable
            max_price = np.max(future_prices)
            profit = (max_price - entry_price) / entry_price

            meta_labels[i] = 1 if profit > 0.002 else 0  # Profitable = take it

        elif primary_signal == 0:  # SHORT trade
            # Check if short would be profitable
            min_price = np.min(future_prices)
            profit = (entry_price - min_price) / entry_price

            meta_labels[i] = 1 if profit > 0.002 else 0  # Profitable = take it

    return meta_labels


def analyze_label_distribution(labels: np.ndarray, label_names: list = None) -> dict:
    """Analyze and print label distribution"""
    if label_names is None:
        label_names = ['SHORT', 'HOLD', 'LONG']

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print("\n" + "="*70)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*70)

    for label, count in zip(unique, counts):
        name = label_names[label] if label < len(label_names) else f"Class {label}"
        pct = count / total * 100
        print(f"{name:10s}: {count:6d} ({pct:5.1f}%)")

    print("="*70)

    return {label_names[i]: counts[i] / total for i, counts in enumerate(counts) if i < len(label_names)}


if __name__ == '__main__':
    print("Improved Label Generation Strategies")
    print("="*70)

    # Demo with synthetic data
    n = 1000
    times = pd.date_range('2024-01-01', periods=n, freq='1min')
    close = 4500 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(100, 1000, n)

    df = pd.DataFrame({
        'time': times,
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    print("\nTesting label generation methods on synthetic data:")
    print(f"Data: {len(df)} bars")

    # 1. Triple Barrier
    print("\n1. Triple Barrier Method:")
    labels_triple = calculate_triple_barrier_labels(df)
    analyze_label_distribution(labels_triple)

    # 2. Regime Adaptive
    print("\n2. Regime-Adaptive Labels:")
    labels_regime = calculate_regime_adaptive_labels(df, base_hold_pct=0.30)
    analyze_label_distribution(labels_regime)

    # 3. Profit-Based
    print("\n3. Profit-Based Labels:")
    labels_profit = calculate_profit_based_labels(df)
    analyze_label_distribution(labels_profit)

    # 4. Multi-Horizon
    print("\n4. Multi-Horizon Labels:")
    labels_multi, horizons = calculate_multi_horizon_labels(df)
    analyze_label_distribution(labels_multi)

    print("\n All label generation methods working correctly")
