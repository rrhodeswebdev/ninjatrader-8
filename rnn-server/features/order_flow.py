"""
Order Flow & Volume Analysis Features - Complete Implementation

Implements all order flow concepts from the quant analysis:
- Volume Profile and Point of Control
- Delta analysis (buy/sell pressure)
- Volume imbalances and climaxes
- Large prints and iceberg detection
- Tape reading features
- Absorption detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


class OrderFlowFeatures:
    """
    Institutional order flow analysis
    """

    def __init__(self):
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.block_trade_std = 3.0  # 3 std devs for block trades

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict[str, np.ndarray]:
        """
        Volume at price levels (Volume Profile / Market Profile)

        Returns:
            poc_price: Point of Control (highest volume price)
            value_area_high: 70% volume upper bound
            value_area_low: 70% volume lower bound
            distance_from_poc: Current price vs POC
            volume_nodes: High volume price levels
            low_volume_nodes: Low volume price levels (air pockets)
        """
        n = len(df)

        poc_price = np.zeros(n)
        value_area_high = np.zeros(n)
        value_area_low = np.zeros(n)
        distance_from_poc = np.zeros(n)
        at_high_volume_node = np.zeros(n, dtype=bool)
        at_low_volume_node = np.zeros(n, dtype=bool)

        # Calculate rolling volume profile
        lookback = min(50, n)

        for i in range(lookback, n):
            window_prices = df['close'].iloc[i-lookback:i].values
            window_volumes = df['volume'].iloc[i-lookback:i].values

            if len(window_prices) < bins:
                continue

            try:
                # Create volume profile histogram
                hist, edges = np.histogram(window_prices, bins=bins, weights=window_volumes)

                # Point of Control (POC) - highest volume price level
                poc_idx = np.argmax(hist)
                poc_price[i] = (edges[poc_idx] + edges[poc_idx + 1]) / 2

                # Value Area (70% of volume)
                total_volume = hist.sum()
                target_volume = total_volume * 0.70

                # Find value area by expanding from POC
                value_volume = hist[poc_idx]
                lower_idx = poc_idx
                upper_idx = poc_idx

                while value_volume < target_volume and (lower_idx > 0 or upper_idx < len(hist) - 1):
                    # Add the side with more volume
                    lower_vol = hist[lower_idx - 1] if lower_idx > 0 else 0
                    upper_vol = hist[upper_idx + 1] if upper_idx < len(hist) - 1 else 0

                    if lower_vol > upper_vol:
                        lower_idx -= 1
                        value_volume += lower_vol
                    else:
                        upper_idx += 1
                        value_volume += upper_vol

                value_area_low[i] = edges[lower_idx]
                value_area_high[i] = edges[upper_idx + 1]

                # Distance from POC
                current_price = df['close'].iloc[i]
                distance_from_poc[i] = (current_price - poc_price[i]) / current_price * 100

                # High volume nodes (>1.5x average volume)
                avg_volume_per_bin = total_volume / bins
                high_volume_threshold = avg_volume_per_bin * 1.5
                low_volume_threshold = avg_volume_per_bin * 0.3

                # Check if current price is at high or low volume node
                current_bin = np.digitize(current_price, edges) - 1
                current_bin = min(max(current_bin, 0), len(hist) - 1)

                if hist[current_bin] > high_volume_threshold:
                    at_high_volume_node[i] = True
                elif hist[current_bin] < low_volume_threshold:
                    at_low_volume_node[i] = True

            except Exception:
                continue

        return {
            'poc_price': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'distance_from_poc': distance_from_poc,
            'at_high_volume_node': at_high_volume_node.astype(float),
            'at_low_volume_node': at_low_volume_node.astype(float),
        }

    def calculate_delta_analysis(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Buying vs Selling pressure (using bid/ask volume if available)

        Returns:
            cumulative_delta: Running buy/sell imbalance
            delta_divergence: Price up, delta down (bearish)
            delta_momentum: Rate of delta change
            aggressive_buyers: Market buy volume ratio
            aggressive_sellers: Market sell volume ratio
        """
        n = len(df)

        # Check if bid/ask volume available
        has_bid_ask = 'bid_volume' in df.columns and 'ask_volume' in df.columns

        cumulative_delta = np.zeros(n)
        delta_divergence = np.zeros(n)
        delta_momentum = np.zeros(n)
        aggressive_buyers = np.zeros(n)
        aggressive_sellers = np.zeros(n)

        if has_bid_ask:
            # Use actual bid/ask volume
            for i in range(n):
                bid_vol = df['bid_volume'].iloc[i]
                ask_vol = df['ask_volume'].iloc[i]

                # Delta: ask volume (buying pressure) - bid volume (selling pressure)
                delta = ask_vol - bid_vol

                # Cumulative delta
                if i > 0:
                    cumulative_delta[i] = cumulative_delta[i-1] + delta
                else:
                    cumulative_delta[i] = delta

                # Aggressive buyers/sellers as ratio of total volume
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    aggressive_buyers[i] = ask_vol / total_vol
                    aggressive_sellers[i] = bid_vol / total_vol

        else:
            # Approximate delta using price action and volume
            for i in range(n):
                # If close > open, assume more buying pressure
                close_position = (df['close'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-10)

                # Approximate delta based on where price closed in range
                approx_delta = (close_position - 0.5) * df['volume'].iloc[i]

                if i > 0:
                    cumulative_delta[i] = cumulative_delta[i-1] + approx_delta
                else:
                    cumulative_delta[i] = approx_delta

                # Approximate aggressive buyers/sellers
                aggressive_buyers[i] = close_position
                aggressive_sellers[i] = 1 - close_position

        # Delta divergence: price and delta moving in opposite directions
        for i in range(20, n):
            price_change = df['close'].iloc[i] - df['close'].iloc[i-20]
            delta_change = cumulative_delta[i] - cumulative_delta[i-20]

            # Divergence when signs are opposite
            if (price_change > 0 and delta_change < 0) or (price_change < 0 and delta_change > 0):
                delta_divergence[i] = abs(price_change) * abs(delta_change)

        # Delta momentum: rate of change in cumulative delta
        for i in range(10, n):
            delta_momentum[i] = cumulative_delta[i] - cumulative_delta[i-10]

        return {
            'cumulative_delta': cumulative_delta,
            'delta_divergence': delta_divergence,
            'delta_momentum': delta_momentum,
            'aggressive_buyers': aggressive_buyers,
            'aggressive_sellers': aggressive_sellers,
        }

    def calculate_volume_imbalances(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Detect absorption and exhaustion via volume

        Returns:
            buying_climax: High volume, small price gain
            selling_climax: High volume, small price drop
            volume_spike: Unusual volume (>2 std dev)
            volume_drought: Unusually low volume
            absorption_bullish: Price holds despite selling
            absorption_bearish: Price holds despite buying
        """
        n = len(df)

        buying_climax = np.zeros(n, dtype=bool)
        selling_climax = np.zeros(n, dtype=bool)
        volume_spike = np.zeros(n, dtype=bool)
        volume_drought = np.zeros(n, dtype=bool)
        absorption_bullish = np.zeros(n)
        absorption_bearish = np.zeros(n)

        for i in range(20, n):
            # Calculate volume statistics
            vol_mean = df['volume'].iloc[i-20:i].mean()
            vol_std = df['volume'].iloc[i-20:i].std()

            current_vol = df['volume'].iloc[i]

            # Volume spike: > 2 standard deviations
            if current_vol > vol_mean + 2 * vol_std:
                volume_spike[i] = True

            # Volume drought: < 0.5 standard deviations below mean
            if current_vol < vol_mean - 0.5 * vol_std:
                volume_drought[i] = True

            # Buying climax: high volume but price barely rises
            price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            if volume_spike[i] and 0 < price_change < 0.001:  # Less than 0.1% gain
                buying_climax[i] = True

            # Selling climax: high volume but price barely falls
            if volume_spike[i] and -0.001 < price_change < 0:  # Less than 0.1% loss
                selling_climax[i] = True

            # Absorption detection
            # Bullish absorption: price holds/rises despite high selling pressure
            if i >= 5:
                price_range = df['high'].iloc[i-5:i+1].max() - df['low'].iloc[i-5:i+1].min()
                avg_range = (df['high'].iloc[i-20:i] - df['low'].iloc[i-20:i]).mean()

                volume_ratio = current_vol / vol_mean

                # High volume, low movement = absorption
                if price_range < avg_range * 0.5 and volume_ratio > 1.5:
                    # Direction based on close position
                    close_position = (df['close'].iloc[i] - df['low'].iloc[i-5:i+1].min()) / (price_range + 1e-10)

                    if close_position > 0.6:  # Price holding near top
                        absorption_bullish[i] = volume_ratio
                    elif close_position < 0.4:  # Price holding near bottom
                        absorption_bearish[i] = volume_ratio

        return {
            'buying_climax': buying_climax.astype(float),
            'selling_climax': selling_climax.astype(float),
            'volume_spike': volume_spike.astype(float),
            'volume_drought': volume_drought.astype(float),
            'absorption_bullish': absorption_bullish,
            'absorption_bearish': absorption_bearish,
        }

    def calculate_tape_reading_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Micro-structure tape reading signals

        Returns:
            large_prints: Institutional size orders
            rapid_fire_orders: HFT activity
            iceberg_detected: Hidden liquidity
            sweep_detected: Stop run activity
        """
        n = len(df)

        large_prints = np.zeros(n, dtype=bool)
        rapid_fire_orders = np.zeros(n)
        iceberg_detected = np.zeros(n, dtype=bool)
        sweep_detected = np.zeros(n, dtype=bool)

        for i in range(50, n):
            # Large prints: volume > 3 std devs above mean
            vol_mean = df['volume'].iloc[i-50:i].mean()
            vol_std = df['volume'].iloc[i-50:i].std()

            if df['volume'].iloc[i] > vol_mean + self.block_trade_std * vol_std:
                large_prints[i] = True

            # Rapid fire: multiple bars with above average volume in short time
            if i >= 5:
                recent_vols = df['volume'].iloc[i-5:i+1]
                if (recent_vols > vol_mean).sum() >= 4:
                    rapid_fire_orders[i] = (recent_vols / vol_mean).mean()

            # Iceberg detection: repeated high volume at same price level
            if i >= 10:
                price_std = df['close'].iloc[i-10:i].std()
                price_range = df['high'].iloc[i-10:i].max() - df['low'].iloc[i-10:i].min()

                # Price barely moving but volume is high
                if price_range < price_std and (df['volume'].iloc[i-10:i] > vol_mean).sum() >= 7:
                    iceberg_detected[i] = True

            # Sweep detection: price quickly moves through level then reverses
            if i >= 3:
                # Quick move up then reversal
                if (df['high'].iloc[i-2] > df['high'].iloc[i-3] and
                    df['high'].iloc[i-1] > df['high'].iloc[i-2] and
                    df['close'].iloc[i] < df['low'].iloc[i-1]):
                    sweep_detected[i] = True

                # Quick move down then reversal
                elif (df['low'].iloc[i-2] < df['low'].iloc[i-3] and
                      df['low'].iloc[i-1] < df['low'].iloc[i-2] and
                      df['close'].iloc[i] > df['high'].iloc[i-1]):
                    sweep_detected[i] = True

        return {
            'large_prints': large_prints.astype(float),
            'rapid_fire_orders': rapid_fire_orders,
            'iceberg_detected': iceberg_detected.astype(float),
            'sweep_detected': sweep_detected.astype(float),
        }

    def calculate_vwap_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        VWAP (Volume Weighted Average Price) features

        Note: VWAP is NOT a lagging indicator - it's the actual average price
        institutions paid, making it a true market structure level.

        Returns:
            vwap: Current VWAP level
            distance_from_vwap: Price vs VWAP
            above_vwap: Price above VWAP (bullish)
            vwap_slope: VWAP trend direction
        """
        n = len(df)

        vwap = np.zeros(n)
        distance_from_vwap = np.zeros(n)
        above_vwap = np.zeros(n, dtype=bool)
        vwap_slope = np.zeros(n)

        # Calculate VWAP: cumulative(price * volume) / cumulative(volume)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tpv = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()

        for i in range(n):
            if cumulative_volume.iloc[i] > 0:
                vwap[i] = cumulative_tpv.iloc[i] / cumulative_volume.iloc[i]
                distance_from_vwap[i] = (df['close'].iloc[i] - vwap[i]) / df['close'].iloc[i] * 100
                above_vwap[i] = df['close'].iloc[i] > vwap[i]

            # VWAP slope
            if i >= 10 and vwap[i-10] > 0:
                vwap_slope[i] = (vwap[i] - vwap[i-10]) / vwap[i-10] * 100

        return {
            'vwap': vwap,
            'distance_from_vwap': distance_from_vwap,
            'above_vwap': above_vwap.astype(float),
            'vwap_slope': vwap_slope,
        }

    def calculate_all_orderflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all order flow features and add to dataframe

        Args:
            df: DataFrame with OHLCV data (and optionally bid_volume, ask_volume)

        Returns:
            DataFrame with all order flow features added
        """
        df = df.copy()

        # Calculate all order flow features
        features = {
            **self.calculate_volume_profile(df),
            **self.calculate_delta_analysis(df),
            **self.calculate_volume_imbalances(df),
            **self.calculate_tape_reading_features(df),
            **self.calculate_vwap_features(df),
        }

        # Add all features to dataframe
        for name, values in features.items():
            df[f'of_{name}'] = values

        return df
