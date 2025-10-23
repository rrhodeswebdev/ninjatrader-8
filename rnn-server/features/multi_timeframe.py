"""
Multi-Timeframe Features - Complete Implementation

Implements multi-timeframe analysis from the quant analysis:
- HTF (Higher Timeframe) alignment
- Multi-timeframe trend confirmation
- Multi-timeframe support/resistance
- Timeframe confluence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MultiTimeframeFeatures:
    """
    Align multiple timeframes for higher probability setups
    """

    def __init__(self):
        # Standard timeframe multipliers
        self.timeframe_multipliers = {
            '5x': 5,    # If base is 1m, this is 5m
            '15x': 15,  # If base is 1m, this is 15m
            '60x': 60,  # If base is 1m, this is 1h
            '240x': 240, # If base is 1m, this is 4h
        }

    def resample_to_higher_timeframe(self, df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
        """
        Resample data to higher timeframe

        Args:
            df: DataFrame with time index and OHLCV data
            multiplier: How many base periods to aggregate

        Returns:
            Resampled DataFrame
        """
        # Ensure time is index
        if 'time' in df.columns:
            df = df.set_index('time')

        # Resample rules
        resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }

        # Add bid/ask volume if present
        if 'bid_volume' in df.columns:
            resample_rules['bid_volume'] = 'sum'
        if 'ask_volume' in df.columns:
            resample_rules['ask_volume'] = 'sum'

        # Resample
        resampled = df.resample(f'{multiplier}T').agg(resample_rules)

        # Drop NaN rows
        resampled = resampled.dropna()

        return resampled

    def calculate_htf_trend(self, df: pd.DataFrame, lookback: int = 20) -> str:
        """
        Determine trend on a given timeframe

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if len(df) < lookback:
            return 'NEUTRAL'

        # Use simple trend determination
        recent_highs = df['high'].iloc[-lookback:]
        recent_lows = df['low'].iloc[-lookback:]

        # Check for higher highs and higher lows (uptrend)
        higher_highs = sum(recent_highs.iloc[i] > recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
        higher_lows = sum(recent_lows.iloc[i] > recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))

        # Check for lower highs and lower lows (downtrend)
        lower_highs = sum(recent_highs.iloc[i] < recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
        lower_lows = sum(recent_lows.iloc[i] < recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))

        # Price change over period
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]

        # Determine trend
        if higher_highs > lower_highs and higher_lows > lower_lows and price_change > 0.01:
            return 'BULLISH'
        elif lower_highs > higher_highs and lower_lows > higher_lows and price_change < -0.01:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def calculate_htf_alignment(self, df: pd.DataFrame, df_higher_tf: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Check if higher timeframes align with trade direction

        Args:
            df: Base timeframe DataFrame
            df_higher_tf: Dict of higher timeframe DataFrames {'5x': df_5m, '15x': df_15m, ...}

        Returns:
            htf_bullish_alignment: All HTF bullish
            htf_bearish_alignment: All HTF bearish
            htf_conflict: Mixed signals
            strongest_htf_trend: Dominant timeframe direction
            htf_trend_strength: How many TFs agree
        """
        n = len(df)

        htf_bullish_alignment = np.zeros(n, dtype=bool)
        htf_bearish_alignment = np.zeros(n, dtype=bool)
        htf_conflict = np.zeros(n, dtype=bool)
        htf_trend_strength = np.zeros(n)

        # Determine current trend for each timeframe
        trends = {}

        for tf_name, tf_df in df_higher_tf.items():
            if len(tf_df) > 0:
                trends[tf_name] = self.calculate_htf_trend(tf_df)

        # For each bar, check alignment
        for i in range(n):
            if not trends:
                continue

            bullish_count = sum(1 for trend in trends.values() if trend == 'BULLISH')
            bearish_count = sum(1 for trend in trends.values() if trend == 'BEARISH')
            neutral_count = sum(1 for trend in trends.values() if trend == 'NEUTRAL')

            total_tfs = len(trends)

            # All bullish
            if bullish_count == total_tfs:
                htf_bullish_alignment[i] = True
                htf_trend_strength[i] = 1.0

            # All bearish
            elif bearish_count == total_tfs:
                htf_bearish_alignment[i] = True
                htf_trend_strength[i] = -1.0

            # Majority bullish
            elif bullish_count > bearish_count and bullish_count > neutral_count:
                htf_trend_strength[i] = bullish_count / total_tfs

            # Majority bearish
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                htf_trend_strength[i] = -bearish_count / total_tfs

            # Conflict
            elif bullish_count == bearish_count and bullish_count > 0:
                htf_conflict[i] = True
                htf_trend_strength[i] = 0.0

        return {
            'htf_bullish_alignment': htf_bullish_alignment.astype(float),
            'htf_bearish_alignment': htf_bearish_alignment.astype(float),
            'htf_conflict': htf_conflict.astype(float),
            'htf_trend_strength': htf_trend_strength,
        }

    def calculate_mtf_structure(self, df: pd.DataFrame, df_higher_tf: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Multi-timeframe market structure state

        Returns:
            mtf_support_level: Strongest multi-TF support
            mtf_resistance_level: Strongest multi-TF resistance
            mtf_structure_strength: Confluence across timeframes
            near_mtf_support: Price near MTF support
            near_mtf_resistance: Price near MTF resistance
        """
        n = len(df)

        mtf_support_level = np.zeros(n)
        mtf_resistance_level = np.zeros(n)
        mtf_structure_strength = np.zeros(n)
        near_mtf_support = np.zeros(n, dtype=bool)
        near_mtf_resistance = np.zeros(n, dtype=bool)

        for i in range(n):
            current_price = df['close'].iloc[i]

            # Collect support/resistance levels from all timeframes
            support_levels = []
            resistance_levels = []

            # Add base timeframe levels
            if i >= 20:
                support_levels.append(df['low'].iloc[i-20:i].min())
                resistance_levels.append(df['high'].iloc[i-20:i].max())

            # Add higher timeframe levels
            for tf_name, tf_df in df_higher_tf.items():
                if len(tf_df) >= 10:
                    support_levels.append(tf_df['low'].iloc[-10:].min())
                    resistance_levels.append(tf_df['high'].iloc[-10:].max())

            if support_levels and resistance_levels:
                # Find closest support/resistance
                support_levels = np.array(support_levels)
                resistance_levels = np.array(resistance_levels)

                # Filter to levels below/above current price
                supports_below = support_levels[support_levels < current_price]
                resistances_above = resistance_levels[resistance_levels > current_price]

                if len(supports_below) > 0:
                    # Find support with most confluence (multiple TFs near same level)
                    mtf_support_level[i] = self._find_confluence_level(supports_below)

                    # Check if near support (within 0.5%)
                    if abs(current_price - mtf_support_level[i]) / current_price < 0.005:
                        near_mtf_support[i] = True

                if len(resistances_above) > 0:
                    # Find resistance with most confluence
                    mtf_resistance_level[i] = self._find_confluence_level(resistances_above)

                    # Check if near resistance (within 0.5%)
                    if abs(current_price - mtf_resistance_level[i]) / current_price < 0.005:
                        near_mtf_resistance[i] = True

                # Structure strength: how many timeframes have levels near current support/resistance
                mtf_structure_strength[i] = len(support_levels) + len(resistance_levels)

        return {
            'mtf_support_level': mtf_support_level,
            'mtf_resistance_level': mtf_resistance_level,
            'mtf_structure_strength': mtf_structure_strength,
            'near_mtf_support': near_mtf_support.astype(float),
            'near_mtf_resistance': near_mtf_resistance.astype(float),
        }

    def _find_confluence_level(self, levels: np.ndarray, tolerance: float = 0.01) -> float:
        """
        Find price level with most confluence (multiple levels clustered together)

        Args:
            levels: Array of price levels
            tolerance: How close levels must be to cluster (default 1%)

        Returns:
            Price level with most confluence
        """
        if len(levels) == 0:
            return 0.0

        # Sort levels
        sorted_levels = np.sort(levels)

        # Find clusters
        max_cluster_size = 0
        best_level = sorted_levels[0]

        for level in sorted_levels:
            # Count how many levels are within tolerance
            cluster_size = np.sum(np.abs(sorted_levels - level) / level < tolerance)

            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                best_level = level

        return best_level

    def calculate_htf_momentum(self, df_higher_tf: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate momentum on higher timeframes

        Returns:
            Dictionary of momentum values for each timeframe
        """
        momentum = {}

        for tf_name, tf_df in df_higher_tf.items():
            if len(tf_df) >= 10:
                # Calculate price change over last 10 bars
                price_change = (tf_df['close'].iloc[-1] - tf_df['close'].iloc[-10]) / tf_df['close'].iloc[-10]
                momentum[tf_name] = price_change
            else:
                momentum[tf_name] = 0.0

        return momentum

    def calculate_all_mtf_features(
        self,
        df: pd.DataFrame,
        include_timeframes: List[str] = ['5x', '15x', '60x']
    ) -> pd.DataFrame:
        """
        Calculate all multi-timeframe features

        Args:
            df: Base timeframe DataFrame with time index
            include_timeframes: Which higher timeframes to include

        Returns:
            DataFrame with all MTF features added
        """
        df = df.copy()

        # Ensure time is in index for resampling
        if 'time' in df.columns and df.index.name != 'time':
            df = df.set_index('time')

        # Resample to higher timeframes
        df_higher_tf = {}
        for tf_name in include_timeframes:
            if tf_name in self.timeframe_multipliers:
                multiplier = self.timeframe_multipliers[tf_name]
                df_higher_tf[tf_name] = self.resample_to_higher_timeframe(df, multiplier)

        # Calculate HTF alignment
        htf_features = self.calculate_htf_alignment(df, df_higher_tf)

        # Calculate MTF structure
        mtf_features = self.calculate_mtf_structure(df, df_higher_tf)

        # Calculate HTF momentum
        htf_momentum = self.calculate_htf_momentum(df_higher_tf)

        # Add features to dataframe
        for name, values in htf_features.items():
            df[f'mtf_{name}'] = values

        for name, values in mtf_features.items():
            df[f'mtf_{name}'] = values

        # Add momentum features (broadcast to all rows)
        for tf_name, momentum_val in htf_momentum.items():
            df[f'mtf_momentum_{tf_name}'] = momentum_val

        # Reset index if needed
        if df.index.name == 'time':
            df = df.reset_index()

        return df
