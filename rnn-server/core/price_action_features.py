"""
Pure Price Action Feature Engineering
No lagging indicators, no smoothed values - only raw price structure and order flow

This module replaces ALL technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger, etc.)
with pure price action features that institutional traders actually use:
- Candle patterns and structure
- Market structure (highs, lows, breakouts)
- Order flow (volume pressure)
- Price rejections and liquidity zones
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class PriceActionFeatures:
    """
    Calculate pure price action features for trading model
    Focus: Price structure, market structure, order flow

    NO lagging indicators - only raw price movement analysis
    """

    def __init__(
        self,
        lookback_short: int = 10,
        lookback_medium: int = 20,
        lookback_long: int = 50
    ):
        """
        Initialize price action feature calculator.

        Args:
            lookback_short: Short-term lookback (default 10 bars)
            lookback_medium: Medium-term lookback (default 20 bars)
            lookback_long: Long-term lookback (default 50 bars)
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all pure price action features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all price action features added
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate returns first (needed by other features)
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 1. Candle characteristics
        df = self._calculate_candle_features(df)

        # 2. Market structure
        df = self._calculate_market_structure(df)

        # 3. Order flow
        df = self._calculate_order_flow(df)

        # 4. Price rejections
        df = self._calculate_rejections(df)

        # 5. Price momentum (pure)
        df = self._calculate_price_momentum(df)

        # 6. Gaps
        df = self._calculate_gaps(df)

        # 7. Fractals
        df = self._calculate_fractals(df)

        # 8. Order blocks
        df = self._calculate_order_blocks(df)

        # 9. Fair value gaps
        df = self._calculate_fvg(df)

        # 10. Liquidity zones
        df = self._calculate_liquidity_zones(df)

        return df

    def _calculate_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candle pattern features - pure price action."""

        # Body size (absolute and percentage)
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body_size'] / df['open'] * 100

        # Upper and lower shadows/wicks
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # Total candle range
        df['candle_range'] = df['high'] - df['low']
        df['range_pct'] = df['candle_range'] / df['open'] * 100

        # Direction
        df['is_bullish'] = (df['close'] > df['open']).astype(float)

        # Ratios (show candle conviction)
        df['body_to_range'] = df['body_size'] / (df['candle_range'] + 1e-10)
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['candle_range'] + 1e-10)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['candle_range'] + 1e-10)

        # Close position in range (0 = at low, 1 = at high)
        df['close_position'] = (df['close'] - df['low']) / (df['candle_range'] + 1e-10)

        return df

    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market structure features - highs, lows, trends."""

        # Rolling highs/lows at different timeframes
        for period in [self.lookback_short, self.lookback_medium, self.lookback_long]:
            df[f'rolling_high_{period}'] = df['high'].rolling(window=period).max()
            df[f'rolling_low_{period}'] = df['low'].rolling(window=period).min()

            # Distance from high/low (support/resistance proximity)
            df[f'dist_from_high_{period}'] = (
                (df[f'rolling_high_{period}'] - df['close']) / df['close'] * 100
            )
            df[f'dist_from_low_{period}'] = (
                (df['close'] - df[f'rolling_low_{period}']) / df['close'] * 100
            )

            # Position in range (0-100%)
            range_size = df[f'rolling_high_{period}'] - df[f'rolling_low_{period}']
            df[f'position_in_range_{period}'] = (
                (df['close'] - df[f'rolling_low_{period}']) / (range_size + 1e-10) * 100
            )

            # Breakout detection
            df[f'breaking_high_{period}'] = (
                df['high'] > df[f'rolling_high_{period}'].shift(1)
            ).astype(float)
            df[f'breaking_low_{period}'] = (
                df['low'] < df[f'rolling_low_{period}'].shift(1)
            ).astype(float)

        # Higher highs / Lower lows (trend structure)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(float)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(float)

        # Consecutive moves
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(float)
        df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(float)

        # Streaks (count consecutive up/down bars)
        df['up_streak'] = df.groupby(
            (df['close'] <= df['close'].shift(1)).cumsum()
        )['consecutive_up'].cumsum()

        df['down_streak'] = df.groupby(
            (df['close'] >= df['close'].shift(1)).cumsum()
        )['consecutive_down'].cumsum()

        return df

    def _calculate_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow features - buying/selling pressure."""

        # Volume changes
        df['volume_change'] = df['volume'].pct_change()

        # Volume ratios at different periods
        for period in [self.lookback_short, self.lookback_medium]:
            df[f'volume_ratio_{period}'] = (
                df['volume'] / df['volume'].rolling(window=period).mean()
            )

            # Volume spike detection
            df[f'volume_spike_{period}'] = (
                df[f'volume_ratio_{period}'] > 2.0
            ).astype(float)

        # Buying vs selling pressure (approximation from price action)
        # When close is in upper half of range with volume = buying pressure
        df['buying_pressure'] = df['close_position'] * df['volume']
        df['selling_pressure'] = (1 - df['close_position']) * df['volume']
        df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
        df['pressure_ratio'] = (
            df['buying_pressure'] / (df['selling_pressure'] + 1e-10)
        )

        # Rolling net pressure
        for period in [self.lookback_short, self.lookback_medium]:
            df[f'net_pressure_{period}'] = (
                df['net_pressure'].rolling(window=period).sum()
            )

        return df

    def _calculate_rejections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price rejection features - wicks showing failed attempts."""

        avg_range = df['candle_range'].rolling(window=self.lookback_medium).mean()

        # Strong rejections (large wicks)
        df['strong_upper_rejection'] = (
            (df['upper_shadow'] > avg_range * 0.5) &
            (df['upper_shadow_ratio'] > 0.5)
        ).astype(float)

        df['strong_lower_rejection'] = (
            (df['lower_shadow'] > avg_range * 0.5) &
            (df['lower_shadow_ratio'] > 0.5)
        ).astype(float)

        # Doji (indecision candle)
        df['is_doji'] = (df['body_to_range'] < 0.1).astype(float)

        # Pin bars (strong rejection candles)
        df['bullish_pin'] = (
            (df['lower_shadow_ratio'] > 0.6) &
            (df['body_to_range'] < 0.3)
        ).astype(float)

        df['bearish_pin'] = (
            (df['upper_shadow_ratio'] > 0.6) &
            (df['body_to_range'] < 0.3)
        ).astype(float)

        return df

    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pure price momentum (not indicator-based)."""

        periods = [3, 5, 10, 20]

        for period in periods:
            # Raw price change over period
            df[f'price_change_{period}'] = (
                (df['close'] - df['close'].shift(period)) /
                df['close'].shift(period) * 100
            )

            # Number of up bars in period
            def count_up_bars(series):
                if len(series) < 2:
                    return 0
                return (series.values[1:] > series.values[:-1]).sum()

            df[f'up_bars_{period}'] = (
                df['close'].rolling(window=period).apply(count_up_bars, raw=False)
            )

            # Strength of moves (volatility of returns)
            df[f'move_strength_{period}'] = (
                df['returns'].rolling(window=period).std() * 100
            )

        return df

    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gap features - pure price action."""

        # Gap up: current low > previous high
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(float)
        df['gap_up_size'] = np.where(
            df['gap_up'],
            (df['low'] - df['high'].shift(1)) / df['close'].shift(1) * 100,
            0
        )

        # Gap down: current high < previous low
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(float)
        df['gap_down_size'] = np.where(
            df['gap_down'],
            (df['low'].shift(1) - df['high']) / df['close'].shift(1) * 100,
            0
        )

        return df

    def _calculate_fractals(self, df: pd.DataFrame, period: int = 2) -> pd.DataFrame:
        """Calculate fractal highs and lows (Williams fractals)."""

        df['fractal_high'] = 0.0
        df['fractal_low'] = 0.0

        # Need enough data for fractals
        if len(df) < period * 2 + 1:
            return df

        for i in range(period, len(df) - period):
            window_highs = df['high'].iloc[i-period:i+period+1]
            window_lows = df['low'].iloc[i-period:i+period+1]

            if df['high'].iloc[i] == window_highs.max():
                df.loc[df.index[i], 'fractal_high'] = 1.0

            if df['low'].iloc[i] == window_lows.min():
                df.loc[df.index[i], 'fractal_low'] = 1.0

        return df

    def _calculate_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order block features - institutional footprints."""

        lookback = self.lookback_medium

        # Strong bullish move = price increase > 1% over lookback
        strong_bull = df['close'].pct_change(lookback) > 0.01

        # Strong bearish move = price decrease > 1% over lookback
        strong_bear = df['close'].pct_change(lookback) < -0.01

        # Bullish order block: last bearish candle before strong bull move
        df['bullish_order_block'] = (
            (df['close'] < df['open']) & strong_bull.shift(-1)
        ).astype(float)

        # Bearish order block: last bullish candle before strong bear move
        df['bearish_order_block'] = (
            (df['close'] > df['open']) & strong_bear.shift(-1)
        ).astype(float)

        return df

    def _calculate_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fair value gaps (price imbalances)."""

        # Bullish FVG: gap between current bar's low and 2 bars ago high
        df['bullish_fvg'] = (
            df['low'].shift(-1) > df['high'].shift(1)
        ).astype(float)

        df['bullish_fvg_size'] = np.where(
            df['bullish_fvg'],
            (df['low'].shift(-1) - df['high'].shift(1)) / df['close'] * 100,
            0
        )

        # Bearish FVG: gap between current bar's high and 2 bars ago low
        df['bearish_fvg'] = (
            df['high'].shift(-1) < df['low'].shift(1)
        ).astype(float)

        df['bearish_fvg_size'] = np.where(
            df['bearish_fvg'],
            (df['low'].shift(1) - df['high'].shift(-1)) / df['close'] * 100,
            0
        )

        return df

    def _calculate_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity zone features - where stops cluster."""

        lookback = self.lookback_long

        # Recent highs/lows (where stops likely sit)
        recent_high = df['high'].rolling(window=lookback).max()
        recent_low = df['low'].rolling(window=lookback).min()

        # Distance to liquidity zones
        df['dist_to_buy_stops'] = (
            (recent_low - df['close']) / df['close'] * 100
        )
        df['dist_to_sell_stops'] = (
            (recent_high - df['close']) / df['close'] * 100
        )

        # Price touching liquidity zones (stop hunt)
        df['touched_buy_stops'] = (
            df['low'] <= recent_low.shift(1)
        ).astype(float)
        df['touched_sell_stops'] = (
            df['high'] >= recent_high.shift(1)
        ).astype(float)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be created.

        Returns:
            List of feature column names
        """
        base_features = ['returns', 'log_returns']

        candle_features = [
            'body_size', 'body_pct', 'upper_shadow', 'lower_shadow',
            'candle_range', 'range_pct', 'is_bullish', 'body_to_range',
            'upper_shadow_ratio', 'lower_shadow_ratio', 'close_position'
        ]

        # Structure features (multiple periods)
        structure_features = []
        for period in [self.lookback_short, self.lookback_medium, self.lookback_long]:
            structure_features.extend([
                f'rolling_high_{period}', f'rolling_low_{period}',
                f'dist_from_high_{period}', f'dist_from_low_{period}',
                f'position_in_range_{period}', f'breaking_high_{period}',
                f'breaking_low_{period}'
            ])

        structure_features.extend([
            'higher_high', 'lower_low', 'higher_low', 'lower_high',
            'consecutive_up', 'consecutive_down', 'up_streak', 'down_streak'
        ])

        # Order flow features
        flow_features = ['volume_change', 'buying_pressure', 'selling_pressure',
                        'net_pressure', 'pressure_ratio']

        for period in [self.lookback_short, self.lookback_medium]:
            flow_features.extend([
                f'volume_ratio_{period}', f'volume_spike_{period}',
                f'net_pressure_{period}'
            ])

        # Rejection features
        rejection_features = [
            'strong_upper_rejection', 'strong_lower_rejection',
            'is_doji', 'bullish_pin', 'bearish_pin'
        ]

        # Momentum features
        momentum_features = []
        for period in [3, 5, 10, 20]:
            momentum_features.extend([
                f'price_change_{period}', f'up_bars_{period}',
                f'move_strength_{period}'
            ])

        # Other features
        other_features = [
            'gap_up', 'gap_up_size', 'gap_down', 'gap_down_size',
            'fractal_high', 'fractal_low',
            'bullish_order_block', 'bearish_order_block',
            'bullish_fvg', 'bullish_fvg_size', 'bearish_fvg', 'bearish_fvg_size',
            'dist_to_buy_stops', 'dist_to_sell_stops',
            'touched_buy_stops', 'touched_sell_stops'
        ]

        all_features = (
            base_features + candle_features + structure_features +
            flow_features + rejection_features + momentum_features +
            other_features
        )

        return all_features


def prepare_price_action_data(
    df: pd.DataFrame,
    lookback_short: int = 10,
    lookback_medium: int = 20,
    lookback_long: int = 50
) -> pd.DataFrame:
    """
    Convenience function to prepare data with all price action features.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        lookback_short: Short-term lookback period
        lookback_medium: Medium-term lookback period
        lookback_long: Long-term lookback period

    Returns:
        DataFrame with all price action features added
    """
    # Ensure column names are lowercase
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Calculate all price action features
    pa_features = PriceActionFeatures(
        lookback_short=lookback_short,
        lookback_medium=lookback_medium,
        lookback_long=lookback_long
    )

    df = pa_features.calculate_all_features(df)

    # Fill NaN values with 0 (for initial bars where lookback isn't available)
    df = df.fillna(0)

    # Replace inf values with 0
    df = df.replace([np.inf, -np.inf], 0)

    return df
