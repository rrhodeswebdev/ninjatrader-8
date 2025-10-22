"""
Price Action Patterns Module

Quantified price action pattern recognition based on market structure,
support/resistance, and classical chart patterns.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats


class PriceActionPatterns:
    """Quantified price action pattern detection"""

    @staticmethod
    def swing_failure(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, window: int = 10) -> int:
        """
        Detect swing failure patterns (failed breakouts/breakdowns)

        A bullish swing failure occurs when price makes a new low but
        quickly reverses, trapping sellers. Vice versa for bearish.

        Args:
            highs: High prices
            lows: Low prices
            closes: Closing prices
            window: Lookback period

        Returns:
            1 (bullish failure), -1 (bearish failure), 0 (none)
        """
        if len(highs) < window or len(lows) < window:
            return 0

        recent_high = np.max(highs[-window:-1])  # Exclude current
        recent_low = np.min(lows[-window:-1])

        current_high = highs[-1]
        current_low = lows[-1]
        current_close = closes[-1]

        # Bullish: New low made but closed above recent high (failed breakdown)
        if current_low < recent_low and current_close > recent_high:
            return 1

        # Bearish: New high made but closed below recent low (failed breakout)
        elif current_high > recent_high and current_close < recent_low:
            return -1

        return 0

    @staticmethod
    def support_resistance_strength(price: np.ndarray,
                                   touches_threshold: int = 3,
                                   tolerance_pct: float = 0.001) -> Tuple[float, float]:
        """
        Quantify support/resistance levels based on price touches

        Args:
            price: Price series
            touches_threshold: Minimum touches to qualify as S/R
            tolerance_pct: Price tolerance for level matching (0.1% default)

        Returns:
            (support_distance, resistance_distance) as % of current price
        """
        if len(price) < 20:
            return 0.0, 0.0

        current = price[-1]

        # Round prices to significant levels (0.25 tick increments for ES)
        tick_size = 0.25
        rounded_prices = np.round(price / tick_size) * tick_size

        # Count touches at each level
        unique, counts = np.unique(rounded_prices, return_counts=True)

        # Find strong levels (multiple touches)
        strong_levels = unique[counts >= touches_threshold]

        if len(strong_levels) == 0:
            return 0.0, 0.0

        # Nearest support (below current price)
        support_levels = strong_levels[strong_levels <= current]
        if len(support_levels) > 0:
            nearest_support = np.max(support_levels)
            support_dist = (current - nearest_support) / current if current > 0 else 0
        else:
            support_dist = 0.0

        # Nearest resistance (above current price)
        resistance_levels = strong_levels[strong_levels >= current]
        if len(resistance_levels) > 0:
            nearest_resistance = np.min(resistance_levels)
            resistance_dist = (nearest_resistance - current) / current if current > 0 else 0
        else:
            resistance_dist = 0.0

        return float(support_dist), float(resistance_dist)

    @staticmethod
    def trend_strength(prices: np.ndarray, window: int = 20) -> float:
        """
        ADX-like trend strength calculation

        Combines slope and R-squared to measure trend quality.
        Range: 0 (no trend) to 1 (strong trend)

        Args:
            prices: Price series
            window: Analysis period

        Returns:
            Trend strength (0-1)
        """
        if len(prices) < window:
            return 0.0

        y = prices[-window:]
        x = np.arange(window)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Normalize slope by price volatility
        volatility = np.std(y)
        if volatility == 0:
            return 0.0

        normalized_slope = abs(slope * window / volatility)

        # Combine slope strength with R-squared (goodness of fit)
        r_squared = r_value ** 2

        # Trend strength = slope magnitude * how well data fits trend line
        trend_strength = min(normalized_slope * r_squared, 1.0)

        return float(trend_strength)

    @staticmethod
    def breakout_strength(highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray, volume: np.ndarray,
                         window: int = 20) -> float:
        """
        Measure breakout/breakdown strength

        Strong breakouts have:
        - Clear distance from prior range
        - High volume confirmation
        - Decisive close

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volume: Volume series
            window: Lookback for range

        Returns:
            Breakout score (positive = bullish, negative = bearish, 0 = range)
        """
        if len(closes) < window + 1:
            return 0.0

        # Prior range (excluding current bar)
        prior_high = np.max(highs[-window-1:-1])
        prior_low = np.min(lows[-window-1:-1])
        prior_range = prior_high - prior_low

        if prior_range == 0:
            return 0.0

        current_close = closes[-1]

        # Distance from range
        if current_close > prior_high:
            # Bullish breakout
            distance = (current_close - prior_high) / prior_range

            # Volume confirmation
            avg_volume = np.mean(volume[-window-1:-1])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Strength = distance * volume confirmation
            strength = distance * min(volume_ratio, 2.0)  # Cap volume ratio
            return float(min(strength, 1.0))

        elif current_close < prior_low:
            # Bearish breakdown
            distance = (prior_low - current_close) / prior_range

            avg_volume = np.mean(volume[-window-1:-1])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            strength = distance * min(volume_ratio, 2.0)
            return float(-min(strength, 1.0))

        else:
            # Inside range
            return 0.0

    @staticmethod
    def higher_highs_higher_lows(highs: np.ndarray, lows: np.ndarray,
                                 window: int = 5) -> int:
        """
        Detect uptrend structure (HH/HL) or downtrend (LH/LL)

        Args:
            highs: High prices
            lows: Low prices
            window: Number of swings to analyze

        Returns:
            1 (uptrend), -1 (downtrend), 0 (no clear trend)
        """
        if len(highs) < window * 2 or len(lows) < window * 2:
            return 0

        # Find swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(window, len(highs) - window):
            # Swing high: higher than surrounding bars
            if highs[i] == np.max(highs[i-window:i+window+1]):
                swing_highs.append(highs[i])

            # Swing low: lower than surrounding bars
            if lows[i] == np.min(lows[i-window:i+window+1]):
                swing_lows.append(lows[i])

        # Need at least 2 swings to determine trend
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0

        # Check for higher highs and higher lows (uptrend)
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]

        higher_highs = recent_highs[-1] > recent_highs[-2]
        higher_lows = recent_lows[-1] > recent_lows[-2]
        lower_highs = recent_highs[-1] < recent_highs[-2]
        lower_lows = recent_lows[-1] < recent_lows[-2]

        if higher_highs and higher_lows:
            return 1  # Uptrend
        elif lower_highs and lower_lows:
            return -1  # Downtrend
        else:
            return 0  # Choppy/ranging

    @staticmethod
    def reversal_candle_pattern(opens: np.ndarray, highs: np.ndarray,
                                lows: np.ndarray, closes: np.ndarray,
                                volume: Optional[np.ndarray] = None) -> int:
        """
        Detect reversal candlestick patterns

        Includes: hammer, shooting star, engulfing patterns

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            volume: Volume (optional, for confirmation)

        Returns:
            1 (bullish reversal), -1 (bearish reversal), 0 (none)
        """
        if len(closes) < 2:
            return 0

        # Current candle
        o_curr = opens[-1]
        h_curr = highs[-1]
        l_curr = lows[-1]
        c_curr = closes[-1]

        # Previous candle
        o_prev = opens[-2]
        c_prev = closes[-2]

        # Body and shadow sizes
        body = abs(c_curr - o_curr)
        upper_shadow = h_curr - max(o_curr, c_curr)
        lower_shadow = min(o_curr, c_curr) - l_curr
        total_range = h_curr - l_curr

        if total_range == 0:
            return 0

        # Hammer (bullish reversal at bottom)
        # - Small body at top
        # - Long lower shadow (2x body)
        # - Minimal upper shadow
        if (lower_shadow > body * 2 and
            upper_shadow < body * 0.3 and
            body / total_range < 0.3):
            return 1

        # Shooting star (bearish reversal at top)
        # - Small body at bottom
        # - Long upper shadow (2x body)
        # - Minimal lower shadow
        if (upper_shadow > body * 2 and
            lower_shadow < body * 0.3 and
            body / total_range < 0.3):
            return -1

        # Bullish engulfing
        # - Previous candle bearish
        # - Current candle bullish
        # - Current body engulfs previous body
        prev_bearish = c_prev < o_prev
        curr_bullish = c_curr > o_curr

        if (prev_bearish and curr_bullish and
            c_curr > o_prev and o_curr < c_prev):
            # Volume confirmation if available
            if volume is not None and len(volume) >= 2:
                if volume[-1] > volume[-2]:
                    return 1
            else:
                return 1

        # Bearish engulfing
        prev_bullish = c_prev > o_prev
        curr_bearish = c_curr < o_curr

        if (prev_bullish and curr_bearish and
            c_curr < o_prev and o_curr > c_prev):
            if volume is not None and len(volume) >= 2:
                if volume[-1] > volume[-2]:
                    return -1
            else:
                return -1

        return 0

    @staticmethod
    def consolidation_detection(highs: np.ndarray, lows: np.ndarray,
                               window: int = 10,
                               compression_threshold: float = 0.3) -> bool:
        """
        Detect price consolidation/compression

        Consolidations often precede strong moves (continuation or reversal)

        Args:
            highs: High prices
            lows: Low prices
            window: Consolidation period
            compression_threshold: How tight the range is vs historical

        Returns:
            True if consolidating, False otherwise
        """
        if len(highs) < window + 50:
            return False

        # Recent range
        recent_range = np.max(highs[-window:]) - np.min(lows[-window:])

        # Historical average range
        historical_ranges = []
        for i in range(len(highs) - window - 50, len(highs) - window):
            if i >= 0:
                range_val = np.max(highs[i:i+window]) - np.min(lows[i:i+window])
                historical_ranges.append(range_val)

        avg_historical_range = np.mean(historical_ranges) if historical_ranges else recent_range

        if avg_historical_range == 0:
            return False

        # Consolidating if recent range is much smaller than historical
        compression_ratio = recent_range / avg_historical_range

        return compression_ratio < compression_threshold

    @staticmethod
    def gap_analysis(opens: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, closes: np.ndarray) -> Tuple[float, str]:
        """
        Analyze price gaps and their implications

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            (gap_size, gap_type) where gap_size is % and type is UP/DOWN/NONE
        """
        if len(closes) < 2 or len(opens) < 1:
            return 0.0, "NONE"

        prev_close = closes[-2]
        curr_open = opens[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]

        # Gap up: current open > previous close
        if curr_open > prev_close:
            gap_size = (curr_open - prev_close) / prev_close
            return float(gap_size), "UP"

        # Gap down: current open < previous close
        elif curr_open < prev_close:
            gap_size = (prev_close - curr_open) / prev_close
            return float(gap_size), "DOWN"

        return 0.0, "NONE"
