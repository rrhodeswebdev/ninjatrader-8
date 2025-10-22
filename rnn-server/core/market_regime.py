"""Market regime detection to adjust trading behavior.

This module classifies market conditions (trending, choppy, volatile) and
adjusts confidence thresholds accordingly to reduce over-trading in unfavorable
conditions.
"""

import pandas as pd
import numpy as np
from typing import Literal, Dict, Any
from dataclasses import dataclass


@dataclass
class RegimeMetrics:
    """Metrics for market regime classification."""
    volatility: float
    mean_return: float
    trend_strength: float
    directional_consistency: float
    regime: Literal["trending", "choppy", "volatile", "unknown"]


def calculate_market_regime(
    bars: pd.DataFrame,
    lookback: int = 20,
    historical_lookback: int = 100,
    trend_strength_threshold: float = 0.3,  # LOWERED: Was 0.4, more lenient without ADX
    directional_consistency_threshold: float = 0.35,  # LOWERED: Was 0.4, more lenient without ADX
    volatility_percentile_threshold: float = 0.75,
    use_adx: bool = False,  # CHANGED: Default to False (pure price action)
    adx_threshold: float = 25.0
) -> Dict[str, Any]:
    """
    Classify market regime to adjust confidence thresholds.

    This function analyzes recent price action to determine if the market is:
    - Trending: Strong directional move, good for trading
    - Choppy: Sideways/range-bound, causes whipsaws
    - Volatile: High volatility, excessive noise

    Uses multiple indicators for robust classification:
    - Short-term metrics (20 bars): Recent trend strength and direction
    - Long-term context (100 bars): Historical volatility baseline
    - ADX indicator: Confirms trend strength
    - Volatility percentile: Compares current vs historical volatility

    Args:
        bars: DataFrame with OHLCV data
        lookback: Number of bars for recent analysis (default 20)
        historical_lookback: Number of bars for historical baseline (default 100)
        trend_strength_threshold: Minimum trend strength for "trending" (default 0.4)
        directional_consistency_threshold: Minimum consistency for "trending" (default 0.4)
        volatility_percentile_threshold: Percentile threshold for "volatile" (default 0.75)
        use_adx: Whether to use ADX for trend confirmation (default True)
        adx_threshold: Minimum ADX for trending confirmation (default 25.0)

    Returns:
        Dictionary with:
            - regime: "trending" | "choppy" | "volatile" | "unknown"
            - confidence_multiplier: 1.0-1.5 (higher = require more confidence)
            - should_trade: bool (whether to trade in this regime)
            - metrics: detailed metrics for analysis

    Examples:
        >>> regime = calculate_market_regime(bars_df)
        >>> if regime["should_trade"]:
        ...     adjusted_threshold = base_threshold * regime["confidence_multiplier"]
    """
    min_bars_required = max(lookback, 14)  # Need at least 14 for ADX

    if len(bars) < min_bars_required:
        return {
            "regime": "unknown",
            "confidence_multiplier": 1.5,  # Conservative until we have data
            "should_trade": False,
            "reason": f"Insufficient data ({len(bars)}/{min_bars_required} bars)",
            "metrics": {
                "trend_strength": 0.0,
                "directional_consistency": 0.0,
                "volatility": 0.0,
                "volatility_percentile": 0.0,
                "adx": 0.0,
                "bars_analyzed": len(bars)
            }
        }

    # Analyze recent bars for current regime
    recent_bars = bars.tail(lookback)
    returns = recent_bars['close'].pct_change().dropna()

    if len(returns) == 0:
        return {
            "regime": "unknown",
            "confidence_multiplier": 1.5,
            "should_trade": False,
            "reason": "No valid returns calculated",
            "metrics": {
                "trend_strength": 0.0,
                "directional_consistency": 0.0,
                "volatility": 0.0,
                "volatility_percentile": 0.0,
                "adx": 0.0,
                "bars_analyzed": 0
            }
        }

    # Calculate current volatility
    current_volatility = returns.std()
    mean_return = returns.mean()

    # Calculate trend strength (Sharpe-like ratio)
    trend_strength = abs(mean_return) / current_volatility if current_volatility > 0 else 0

    # Calculate directional consistency (trending vs choppy)
    positive_bars = (returns > 0).sum()
    negative_bars = (returns < 0).sum()
    total_bars = len(returns)
    directional_consistency = abs(positive_bars - negative_bars) / total_bars if total_bars > 0 else 0

    # Calculate historical volatility baseline for context
    historical_bars = min(historical_lookback, len(bars))
    if historical_bars > lookback:
        historical_returns = bars['close'].tail(historical_bars).pct_change().dropna()
        historical_volatility = historical_returns.rolling(window=lookback).std().dropna()

        if len(historical_volatility) > 0:
            # Calculate percentile of current volatility
            volatility_percentile = (historical_volatility < current_volatility).sum() / len(historical_volatility)
        else:
            volatility_percentile = 0.5
    else:
        volatility_percentile = 0.5  # No baseline, assume neutral

    # Calculate ADX for trend strength confirmation
    adx_value = 0.0
    if use_adx and 'adx' in bars.columns and len(bars) > 0:
        adx_value = bars['adx'].iloc[-1]
    elif use_adx:
        # Calculate ADX if not present
        from core.indicators import calculate_adx
        high = bars['high'].values
        low = bars['low'].values
        close = bars['close'].values

        if len(high) >= 14:
            adx_values = calculate_adx(high, low, close, period=14)
            if len(adx_values) > 0:
                adx_value = adx_values[-1]

    # Compile metrics
    metrics = {
        "trend_strength": float(trend_strength),
        "directional_consistency": float(directional_consistency),
        "volatility": float(current_volatility),
        "volatility_percentile": float(volatility_percentile),
        "adx": float(adx_value),
        "mean_return": float(mean_return),
        "positive_bars": int(positive_bars),
        "negative_bars": int(negative_bars),
        "bars_analyzed": len(returns),
        "historical_bars_used": historical_bars
    }

    # Multi-factor regime classification

    # Strong trending conditions:
    # 1. High trend strength AND directional consistency
    # 2. ADX confirms trend (if enabled)
    # 3. Volatility not extreme
    is_trending = (
        trend_strength > trend_strength_threshold and
        directional_consistency > directional_consistency_threshold and
        (not use_adx or adx_value > adx_threshold) and
        volatility_percentile < 0.85  # Not in extreme volatility
    )

    if is_trending:
        return {
            "regime": "trending",
            "confidence_multiplier": 1.0,  # Normal threshold
            "should_trade": True,
            "reason": f"Strong trend detected (strength={trend_strength:.2f}, consistency={directional_consistency:.2f}, ADX={adx_value:.1f})",
            "metrics": metrics
        }

    # High volatility conditions (relative to history):
    # Current volatility in top quartile of historical volatility
    if volatility_percentile > volatility_percentile_threshold:
        return {
            "regime": "volatile",
            "confidence_multiplier": 1.3,  # Require 30% higher confidence
            "should_trade": False,  # Skip volatile markets
            "reason": f"High volatility detected (percentile={volatility_percentile:.1%}, current={current_volatility:.4f})",
            "metrics": metrics
        }

    # Choppy/sideways conditions (default):
    # Not trending, not extremely volatile, but lacks clear direction
    return {
        "regime": "choppy",
        "confidence_multiplier": 1.5,  # Require 50% higher confidence
        "should_trade": False,  # Skip choppy markets (causes whipsaws)
        "reason": f"Choppy/sideways market (strength={trend_strength:.2f}, consistency={directional_consistency:.2f}, ADX={adx_value:.1f})",
        "metrics": metrics
    }


def get_adjusted_confidence_threshold(
    base_threshold: float,
    regime: Dict[str, Any]
) -> float:
    """
    Adjust confidence threshold based on market regime.

    In unfavorable regimes (choppy, volatile), requires higher confidence
    to generate signals, reducing over-trading.

    Args:
        base_threshold: Base confidence threshold (e.g., 0.60)
        regime: Regime dictionary from calculate_market_regime()

    Returns:
        Adjusted confidence threshold

    Examples:
        >>> base = 0.60
        >>> regime = {"regime": "choppy", "confidence_multiplier": 1.5}
        >>> adjusted = get_adjusted_confidence_threshold(base, regime)
        >>> # Returns 0.90 (60% * 1.5)
    """
    return min(base_threshold * regime["confidence_multiplier"], 0.95)


def should_skip_trading(regime: Dict[str, Any]) -> tuple[bool, str]:
    """
    Determine if trading should be skipped based on regime.

    Args:
        regime: Regime dictionary from calculate_market_regime()

    Returns:
        Tuple of (should_skip: bool, reason: str)

    Examples:
        >>> regime = {"regime": "choppy", "should_trade": False}
        >>> should_skip, reason = should_skip_trading(regime)
        >>> # Returns (True, "Market regime unfavorable: choppy")
    """
    if not regime.get("should_trade", False):
        return True, f"Market regime unfavorable: {regime['regime']}"
    return False, ""


def calculate_atr_regime(bars: pd.DataFrame, lookback: int = 14) -> Dict[str, Any]:
    """
    Calculate ATR-based regime (alternative to return-based regime).

    Uses Average True Range to classify volatility regimes.

    Args:
        bars: DataFrame with OHLCV data
        lookback: ATR period (default 14)

    Returns:
        Dictionary with regime classification

    Examples:
        >>> regime = calculate_atr_regime(bars_df)
        >>> if regime["atr_percentile"] > 0.8:
        ...     print("High volatility regime")
    """
    if len(bars) < lookback + 1:
        return {
            "regime": "unknown",
            "atr": 0.0,
            "atr_percentile": 0.5,
            "confidence_multiplier": 1.5
        }

    # Calculate True Range
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values

    tr = np.zeros(len(bars))
    tr[0] = high[0] - low[0]

    for i in range(1, len(bars)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR
    atr = np.zeros(len(bars))
    atr[lookback-1] = np.mean(tr[:lookback])

    for i in range(lookback, len(bars)):
        atr[i] = (atr[i-1] * (lookback-1) + tr[i]) / lookback

    current_atr = atr[-1]

    # Calculate ATR percentile (relative to recent ATR values)
    recent_atr = atr[atr > 0][-50:]  # Last 50 non-zero ATR values
    if len(recent_atr) > 0:
        atr_percentile = (recent_atr < current_atr).sum() / len(recent_atr)
    else:
        atr_percentile = 0.5

    # Classify based on ATR percentile
    if atr_percentile > 0.8:
        regime = "volatile"
        multiplier = 1.4
        should_trade = False
    elif atr_percentile > 0.6:
        regime = "active"
        multiplier = 1.1
        should_trade = True
    else:
        regime = "calm"
        multiplier = 1.0
        should_trade = True

    return {
        "regime": regime,
        "atr": float(current_atr),
        "atr_percentile": float(atr_percentile),
        "confidence_multiplier": multiplier,
        "should_trade": should_trade,
        "reason": f"ATR-based regime: {regime} (percentile={atr_percentile:.2f})"
    }


def combine_regimes(
    return_regime: Dict[str, Any],
    atr_regime: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine return-based and ATR-based regime classifications.

    Uses the more conservative classification (higher multiplier).

    Args:
        return_regime: Regime from calculate_market_regime()
        atr_regime: Regime from calculate_atr_regime()

    Returns:
        Combined regime dictionary

    Examples:
        >>> return_reg = calculate_market_regime(bars)
        >>> atr_reg = calculate_atr_regime(bars)
        >>> combined = combine_regimes(return_reg, atr_reg)
    """
    # Use the more conservative multiplier
    multiplier = max(
        return_regime.get("confidence_multiplier", 1.0),
        atr_regime.get("confidence_multiplier", 1.0)
    )

    # Trade only if both regimes allow it
    should_trade = (
        return_regime.get("should_trade", False) and
        atr_regime.get("should_trade", False)
    )

    # Combine regime names
    if return_regime["regime"] == "trending" and atr_regime["regime"] == "calm":
        regime = "ideal"  # Best conditions
    elif return_regime["regime"] == "volatile" or atr_regime["regime"] == "volatile":
        regime = "volatile"
    elif return_regime["regime"] == "choppy":
        regime = "choppy"
    else:
        regime = "mixed"

    return {
        "regime": regime,
        "confidence_multiplier": multiplier,
        "should_trade": should_trade,
        "reason": f"Combined: {return_regime['regime']} + {atr_regime['regime']}",
        "return_regime": return_regime,
        "atr_regime": atr_regime
    }


def get_regime_description(regime: Dict[str, Any]) -> str:
    """
    Get human-readable description of market regime.

    Args:
        regime: Regime dictionary

    Returns:
        Descriptive string

    Examples:
        >>> regime = calculate_market_regime(bars)
        >>> desc = get_regime_description(regime)
        >>> print(desc)
        "Trending market - favorable for trading"
    """
    regime_type = regime.get("regime", "unknown")
    should_trade = regime.get("should_trade", False)

    descriptions = {
        "trending": "Strong trend - favorable for trading",
        "choppy": "Choppy/sideways market - expect whipsaws",
        "volatile": "High volatility - excessive noise",
        "ideal": "Ideal conditions - trending with calm volatility",
        "mixed": "Mixed conditions - trade cautiously",
        "unknown": "Insufficient data for classification"
    }

    base_desc = descriptions.get(regime_type, "Unknown regime")

    if not should_trade:
        base_desc += " (SKIPPING TRADES)"

    return base_desc
