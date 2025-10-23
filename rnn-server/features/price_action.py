"""
Price Action Pattern Recognition - Complete Implementation

Implements all price action patterns from the quant analysis:
- Engulfing patterns with context
- Pin bars (rejection candles)
- Inside/Outside bars
- Momentum waves
- Doji patterns
- Hammer/Shooting star
- Three bar patterns
"""

import pandas as pd
import numpy as np
from typing import Dict


class PriceActionPatterns:
    """
    Advanced candlestick and price pattern detection
    NO indicators - pure price action only
    """

    def __init__(self):
        self.min_body_ratio = 0.1  # Minimum body size as % of range
        self.pin_wick_ratio = 2.0  # Wick must be 2x body for pin bar

    def detect_engulfing_patterns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Bullish/bearish engulfing with context

        Returns:
            bullish_engulfing: Boolean
            bearish_engulfing: Boolean
            engulfing_strength: Body size ratio
            engulfing_at_structure: Near support/resistance
        """
        n = len(df)

        bullish_engulfing = np.zeros(n, dtype=bool)
        bearish_engulfing = np.zeros(n, dtype=bool)
        engulfing_strength = np.zeros(n)
        engulfing_at_structure = np.zeros(n, dtype=bool)

        # Need at least 2 bars
        for i in range(1, n):
            prev_body = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            curr_body = abs(df['close'].iloc[i] - df['open'].iloc[i])

            # Bullish engulfing: current bullish candle engulfs previous bearish
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Current is bullish
                df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Previous is bearish
                df['open'].iloc[i] <= df['close'].iloc[i-1] and  # Opens at/below prev close
                df['close'].iloc[i] >= df['open'].iloc[i-1]):  # Closes at/above prev open

                bullish_engulfing[i] = True
                engulfing_strength[i] = curr_body / (prev_body + 1e-10)

                # Check if near support (within 1% of recent low)
                if i >= 20:
                    recent_low = df['low'].iloc[i-20:i].min()
                    if abs(df['low'].iloc[i] - recent_low) / df['close'].iloc[i] < 0.01:
                        engulfing_at_structure[i] = True

            # Bearish engulfing: current bearish candle engulfs previous bullish
            elif (df['close'].iloc[i] < df['open'].iloc[i] and  # Current is bearish
                  df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Previous is bullish
                  df['open'].iloc[i] >= df['close'].iloc[i-1] and  # Opens at/above prev close
                  df['close'].iloc[i] <= df['open'].iloc[i-1]):  # Closes at/below prev open

                bearish_engulfing[i] = True
                engulfing_strength[i] = curr_body / (prev_body + 1e-10)

                # Check if near resistance (within 1% of recent high)
                if i >= 20:
                    recent_high = df['high'].iloc[i-20:i].max()
                    if abs(df['high'].iloc[i] - recent_high) / df['close'].iloc[i] < 0.01:
                        engulfing_at_structure[i] = True

        return {
            'bullish_engulfing': bullish_engulfing.astype(float),
            'bearish_engulfing': bearish_engulfing.astype(float),
            'engulfing_strength': engulfing_strength,
            'engulfing_at_structure': engulfing_at_structure.astype(float),
        }

    def detect_pin_bars(self, df: pd.DataFrame, wick_ratio: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Pin bar (rejection candles) detection

        Returns:
            bullish_pin: Long lower wick rejection
            bearish_pin: Long upper wick rejection
            pin_wick_size: Wick length in percentage
            pin_body_size: Body length in percentage
            pin_at_key_level: Near support/resistance
        """
        n = len(df)

        bullish_pin = np.zeros(n, dtype=bool)
        bearish_pin = np.zeros(n, dtype=bool)
        pin_wick_size = np.zeros(n)
        pin_body_size = np.zeros(n)
        pin_at_key_level = np.zeros(n, dtype=bool)

        for i in range(n):
            # Calculate candle components
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]

            if total_range == 0:
                continue

            upper_wick = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            lower_wick = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]

            body_ratio = body / total_range

            # Bullish pin bar: long lower wick, small body
            if (lower_wick > body * wick_ratio and
                body_ratio < 0.3 and
                upper_wick < lower_wick * 0.5):  # Upper wick must be small

                bullish_pin[i] = True
                pin_wick_size[i] = (lower_wick / df['close'].iloc[i]) * 100
                pin_body_size[i] = (body / df['close'].iloc[i]) * 100

                # Check if at support
                if i >= 20:
                    recent_low = df['low'].iloc[i-20:i].min()
                    if abs(df['low'].iloc[i] - recent_low) / df['close'].iloc[i] < 0.01:
                        pin_at_key_level[i] = True

            # Bearish pin bar: long upper wick, small body
            elif (upper_wick > body * wick_ratio and
                  body_ratio < 0.3 and
                  lower_wick < upper_wick * 0.5):  # Lower wick must be small

                bearish_pin[i] = True
                pin_wick_size[i] = (upper_wick / df['close'].iloc[i]) * 100
                pin_body_size[i] = (body / df['close'].iloc[i]) * 100

                # Check if at resistance
                if i >= 20:
                    recent_high = df['high'].iloc[i-20:i].max()
                    if abs(df['high'].iloc[i] - recent_high) / df['close'].iloc[i] < 0.01:
                        pin_at_key_level[i] = True

        return {
            'bullish_pin': bullish_pin.astype(float),
            'bearish_pin': bearish_pin.astype(float),
            'pin_wick_size': pin_wick_size,
            'pin_body_size': pin_body_size,
            'pin_at_key_level': pin_at_key_level.astype(float),
        }

    def detect_inside_outside_bars(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Inside bars (consolidation) and outside bars (volatility expansion)

        Returns:
            inside_bar: Current bar within previous
            outside_bar: Current bar engulfs previous
            consecutive_inside: Number of inside bars
            outside_bar_volume: Volume confirmation
        """
        n = len(df)

        inside_bar = np.zeros(n, dtype=bool)
        outside_bar = np.zeros(n, dtype=bool)
        consecutive_inside = np.zeros(n)
        outside_bar_volume = np.zeros(n)

        inside_count = 0

        for i in range(1, n):
            # Inside bar: high and low are within previous bar's range
            if (df['high'].iloc[i] <= df['high'].iloc[i-1] and
                df['low'].iloc[i] >= df['low'].iloc[i-1]):

                inside_bar[i] = True
                inside_count += 1
                consecutive_inside[i] = inside_count
            else:
                inside_count = 0

            # Outside bar: high and low exceed previous bar's range
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-1]):

                outside_bar[i] = True

                # Check volume confirmation (outside bar should have high volume)
                if i >= 20:
                    avg_volume = df['volume'].iloc[i-20:i].mean()
                    outside_bar_volume[i] = df['volume'].iloc[i] / (avg_volume + 1e-10)

        return {
            'inside_bar': inside_bar.astype(float),
            'outside_bar': outside_bar.astype(float),
            'consecutive_inside': consecutive_inside,
            'outside_bar_volume': outside_bar_volume,
        }

    def calculate_momentum_waves(self, df: pd.DataFrame, period: int = 5) -> Dict[str, np.ndarray]:
        """
        Price momentum without indicators

        Returns:
            bullish_momentum: Consecutive higher closes
            bearish_momentum: Consecutive lower closes
            momentum_strength: Average price change
            momentum_acceleration: Rate of change of momentum
        """
        n = len(df)

        bullish_momentum = np.zeros(n)
        bearish_momentum = np.zeros(n)
        momentum_strength = np.zeros(n)
        momentum_acceleration = np.zeros(n)

        for i in range(period, n):
            window = df['close'].iloc[i-period:i+1]

            # Count consecutive higher/lower closes
            bull_count = 0
            bear_count = 0

            for j in range(1, len(window)):
                if window.iloc[j] > window.iloc[j-1]:
                    bull_count += 1
                    bear_count = 0
                elif window.iloc[j] < window.iloc[j-1]:
                    bear_count += 1
                    bull_count = 0

            bullish_momentum[i] = bull_count
            bearish_momentum[i] = bear_count

            # Momentum strength: average price change over period
            price_changes = window.diff().dropna()
            momentum_strength[i] = abs(price_changes.mean()) / window.iloc[0] * 100

            # Momentum acceleration: is momentum increasing or decreasing
            if i >= period * 2:
                prev_momentum = momentum_strength[i-period]
                curr_momentum = momentum_strength[i]
                momentum_acceleration[i] = curr_momentum - prev_momentum

        return {
            'bullish_momentum': bullish_momentum,
            'bearish_momentum': bearish_momentum,
            'momentum_strength': momentum_strength,
            'momentum_acceleration': momentum_acceleration,
        }

    def detect_doji(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Doji pattern detection (indecision candles)

        Returns:
            is_doji: Standard doji
            dragonfly_doji: Long lower wick, no upper wick
            gravestone_doji: Long upper wick, no lower wick
            doji_at_extreme: Doji at recent high/low
        """
        n = len(df)

        is_doji = np.zeros(n, dtype=bool)
        dragonfly_doji = np.zeros(n, dtype=bool)
        gravestone_doji = np.zeros(n, dtype=bool)
        doji_at_extreme = np.zeros(n, dtype=bool)

        for i in range(n):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]

            if total_range == 0:
                continue

            body_ratio = body / total_range

            upper_wick = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            lower_wick = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]

            # Standard doji: very small body
            if body_ratio < 0.1:
                is_doji[i] = True

                # Dragonfly doji: long lower wick, minimal upper wick
                if lower_wick > total_range * 0.6 and upper_wick < total_range * 0.1:
                    dragonfly_doji[i] = True

                # Gravestone doji: long upper wick, minimal lower wick
                elif upper_wick > total_range * 0.6 and lower_wick < total_range * 0.1:
                    gravestone_doji[i] = True

                # Check if at extreme (recent high or low)
                if i >= 20:
                    recent_high = df['high'].iloc[i-20:i].max()
                    recent_low = df['low'].iloc[i-20:i].min()

                    if (abs(df['high'].iloc[i] - recent_high) / df['close'].iloc[i] < 0.005 or
                        abs(df['low'].iloc[i] - recent_low) / df['close'].iloc[i] < 0.005):
                        doji_at_extreme[i] = True

        return {
            'is_doji': is_doji.astype(float),
            'dragonfly_doji': dragonfly_doji.astype(float),
            'gravestone_doji': gravestone_doji.astype(float),
            'doji_at_extreme': doji_at_extreme.astype(float),
        }

    def detect_hammer_shooting_star(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Hammer and shooting star patterns

        Returns:
            hammer: Bullish reversal (long lower wick)
            shooting_star: Bearish reversal (long upper wick)
            hammer_strength: Size of rejection wick
            pattern_at_trend_extreme: Pattern at end of trend
        """
        n = len(df)

        hammer = np.zeros(n, dtype=bool)
        shooting_star = np.zeros(n, dtype=bool)
        hammer_strength = np.zeros(n)
        pattern_at_trend_extreme = np.zeros(n, dtype=bool)

        for i in range(n):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]

            if total_range == 0:
                continue

            upper_wick = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            lower_wick = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]

            body_ratio = body / total_range

            # Hammer: long lower wick, small body, small upper wick
            if (lower_wick > total_range * 0.6 and
                body_ratio < 0.3 and
                upper_wick < total_range * 0.15):

                hammer[i] = True
                hammer_strength[i] = (lower_wick / df['close'].iloc[i]) * 100

                # Check if at downtrend extreme
                if i >= 20:
                    recent_low = df['low'].iloc[i-20:i].min()
                    if abs(df['low'].iloc[i] - recent_low) / df['close'].iloc[i] < 0.01:
                        pattern_at_trend_extreme[i] = True

            # Shooting star: long upper wick, small body, small lower wick
            elif (upper_wick > total_range * 0.6 and
                  body_ratio < 0.3 and
                  lower_wick < total_range * 0.15):

                shooting_star[i] = True
                hammer_strength[i] = (upper_wick / df['close'].iloc[i]) * 100

                # Check if at uptrend extreme
                if i >= 20:
                    recent_high = df['high'].iloc[i-20:i].max()
                    if abs(df['high'].iloc[i] - recent_high) / df['close'].iloc[i] < 0.01:
                        pattern_at_trend_extreme[i] = True

        return {
            'hammer': hammer.astype(float),
            'shooting_star': shooting_star.astype(float),
            'hammer_strength': hammer_strength,
            'pattern_at_trend_extreme': pattern_at_trend_extreme.astype(float),
        }

    def detect_three_bar_patterns(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Three bar patterns: Morning star, Evening star, Three white soldiers, Three black crows

        Returns:
            morning_star: Bullish reversal pattern
            evening_star: Bearish reversal pattern
            three_white_soldiers: Strong bullish continuation
            three_black_crows: Strong bearish continuation
        """
        n = len(df)

        morning_star = np.zeros(n, dtype=bool)
        evening_star = np.zeros(n, dtype=bool)
        three_white_soldiers = np.zeros(n, dtype=bool)
        three_black_crows = np.zeros(n, dtype=bool)

        for i in range(2, n):
            # Get three bars
            bar1_bull = df['close'].iloc[i-2] > df['open'].iloc[i-2]
            bar2_bull = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            bar3_bull = df['close'].iloc[i] > df['open'].iloc[i]

            bar1_body = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
            bar2_body = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            bar3_body = abs(df['close'].iloc[i] - df['open'].iloc[i])

            # Morning star: bearish, small body, bullish
            if (not bar1_bull and  # First bar is bearish
                bar2_body < bar1_body * 0.3 and  # Second bar has small body
                bar3_bull and  # Third bar is bullish
                bar3_body > bar1_body * 0.5):  # Third bar has decent body
                morning_star[i] = True

            # Evening star: bullish, small body, bearish
            elif (bar1_bull and  # First bar is bullish
                  bar2_body < bar1_body * 0.3 and  # Second bar has small body
                  not bar3_bull and  # Third bar is bearish
                  bar3_body > bar1_body * 0.5):  # Third bar has decent body
                evening_star[i] = True

            # Three white soldiers: three consecutive bullish candles
            if (bar1_bull and bar2_bull and bar3_bull and
                df['close'].iloc[i] > df['close'].iloc[i-1] and
                df['close'].iloc[i-1] > df['close'].iloc[i-2]):
                three_white_soldiers[i] = True

            # Three black crows: three consecutive bearish candles
            elif (not bar1_bull and not bar2_bull and not bar3_bull and
                  df['close'].iloc[i] < df['close'].iloc[i-1] and
                  df['close'].iloc[i-1] < df['close'].iloc[i-2]):
                three_black_crows[i] = True

        return {
            'morning_star': morning_star.astype(float),
            'evening_star': evening_star.astype(float),
            'three_white_soldiers': three_white_soldiers.astype(float),
            'three_black_crows': three_black_crows.astype(float),
        }

    def calculate_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all price action patterns and add to dataframe

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all pattern features added
        """
        df = df.copy()

        # Detect all patterns
        patterns = {
            **self.detect_engulfing_patterns(df),
            **self.detect_pin_bars(df),
            **self.detect_inside_outside_bars(df),
            **self.calculate_momentum_waves(df),
            **self.detect_doji(df),
            **self.detect_hammer_shooting_star(df),
            **self.detect_three_bar_patterns(df),
        }

        # Add all patterns to dataframe
        for name, values in patterns.items():
            df[f'pa_{name}'] = values

        return df
