"""
Enhanced Orderflow Features Module

Provides advanced orderflow and microstructure analysis for trading signals.
Based on institutional order flow, volume profile, and market microstructure theory.
"""

import numpy as np
from typing import Tuple, Optional


class OrderflowFeatures:
    """Advanced orderflow and microstructure feature extraction"""

    @staticmethod
    def cumulative_delta(bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
        """
        Cumulative volume delta - tracks institutional flow direction

        Positive values indicate aggressive buying (buyers hitting asks)
        Negative values indicate aggressive selling (sellers hitting bids)

        Args:
            bid_volume: Volume traded at bid (selling pressure)
            ask_volume: Volume traded at ask (buying pressure)

        Returns:
            Cumulative delta array
        """
        delta = ask_volume - bid_volume
        return np.cumsum(delta)

    @staticmethod
    def delta_divergence(price: np.ndarray, cum_delta: np.ndarray, window: int = 20) -> float:
        """
        Price-delta divergence - detects smart money vs dumb money

        When price rises but delta falls (or vice versa), it indicates
        potential reversal as institutional flow contradicts retail.

        Args:
            price: Price series
            cum_delta: Cumulative delta series
            window: Lookback period

        Returns:
            Divergence score (0-1), higher = stronger divergence
        """
        if len(price) < window or len(cum_delta) < window:
            return 0.0

        price_recent = price[-window:]
        delta_recent = cum_delta[-window:]

        # Calculate direction of movement
        price_direction = np.sign(np.diff(price_recent))
        delta_direction = np.sign(np.diff(delta_recent))

        # Count divergences
        divergences = np.sum(price_direction != delta_direction)
        divergence_ratio = divergences / (window - 1)

        return float(divergence_ratio)

    @staticmethod
    def bid_ask_imbalance(bid_volume: np.ndarray, ask_volume: np.ndarray,
                         window: int = 10) -> float:
        """
        Immediate supply/demand imbalance

        Measures recent buying vs selling pressure.
        Range: -1 (all asks/selling) to +1 (all bids/buying)

        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            window: Recent bars to analyze

        Returns:
            Imbalance score (-1 to +1)
        """
        if len(bid_volume) < window or len(ask_volume) < window:
            return 0.0

        recent_bid = np.sum(bid_volume[-window:])
        recent_ask = np.sum(ask_volume[-window:])

        total = recent_bid + recent_ask
        if total == 0:
            return 0.0

        imbalance = (recent_ask - recent_bid) / total
        return float(imbalance)

    @staticmethod
    def volume_at_price_profile(prices: np.ndarray, volumes: np.ndarray,
                                bins: int = 20) -> float:
        """
        Volume profile - identifies where institutional trading occurred

        Returns distance from Point of Control (POC) - the price level
        with highest volume, indicating institutional acceptance.

        Args:
            prices: Price series
            volumes: Volume series
            bins: Number of price bins

        Returns:
            POC distance as percentage of current price
        """
        if len(prices) < bins or len(volumes) < bins:
            return 0.0

        try:
            # Create volume profile histogram
            hist, edges = np.histogram(prices, bins=bins, weights=volumes)

            # Find Point of Control (highest volume price level)
            poc_idx = np.argmax(hist)
            poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2

            current_price = prices[-1]

            if current_price == 0:
                return 0.0

            # Distance from POC (positive = above POC, negative = below POC)
            poc_distance = (current_price - poc_price) / current_price

            return float(poc_distance)
        except Exception:
            return 0.0

    @staticmethod
    def absorption_detection(price: np.ndarray, volume: np.ndarray,
                           window: int = 5) -> float:
        """
        Detect absorption (large volume with minimal price movement)

        Indicates strong support/resistance where institutions are
        absorbing supply/demand without moving price.

        Args:
            price: Price series
            volume: Volume series
            window: Recent bars to analyze

        Returns:
            Absorption strength (0 = none, >2 = strong absorption)
        """
        if len(price) < max(window, 100) or len(volume) < max(window, 100):
            return 0.0

        # Recent price range
        price_range = np.max(price[-window:]) - np.min(price[-window:])

        # Average recent volume vs historical
        avg_volume = np.mean(volume[-window:])
        historical_avg = np.mean(volume[-100:])

        # Historical price volatility
        historical_std = np.std(price[-100:])

        if historical_std == 0 or historical_avg == 0:
            return 0.0

        # High volume but low price movement = absorption
        volume_ratio = avg_volume / historical_avg
        price_movement = price_range / historical_std

        # If price barely moved despite high volume
        if price_movement < 0.5:
            return float(volume_ratio)

        return 0.0

    @staticmethod
    def trade_intensity(volume: np.ndarray, time_delta: Optional[np.ndarray] = None,
                       window: int = 10) -> float:
        """
        Measure trade intensity (volume per time unit)

        High intensity indicates urgency from institutional traders.

        Args:
            volume: Volume series
            time_delta: Time between bars (optional, assumes uniform if None)
            window: Recent bars to analyze

        Returns:
            Trade intensity ratio vs historical average
        """
        if len(volume) < max(window, 100):
            return 1.0

        if time_delta is None:
            # Assume uniform time spacing
            recent_intensity = np.mean(volume[-window:])
        else:
            # Volume per unit time
            recent_intensity = np.sum(volume[-window:]) / np.sum(time_delta[-window:])

        historical_intensity = np.mean(volume[-100:-window])

        if historical_intensity == 0:
            return 1.0

        return float(recent_intensity / historical_intensity)

    @staticmethod
    def block_trade_detection(volume: np.ndarray, threshold_std: float = 3.0) -> float:
        """
        Detect unusually large trades (block trades) from institutions

        Args:
            volume: Volume series
            threshold_std: Standard deviations above mean to flag as block trade

        Returns:
            Block trade score (0 = normal, >1 = block trade detected)
        """
        if len(volume) < 50:
            return 0.0

        mean_vol = np.mean(volume[:-1])  # Exclude current bar
        std_vol = np.std(volume[:-1])

        if std_vol == 0:
            return 0.0

        current_vol = volume[-1]
        z_score = (current_vol - mean_vol) / std_vol

        if z_score > threshold_std:
            return float(z_score - threshold_std)

        return 0.0

    @staticmethod
    def iceberg_order_detection(bid_volume: np.ndarray, ask_volume: np.ndarray,
                                price: np.ndarray, window: int = 5) -> Tuple[float, str]:
        """
        Detect potential iceberg orders (hidden institutional orders)

        Identified by repeated absorption at same price level.

        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            price: Price series
            window: Bars to analyze

        Returns:
            Tuple of (iceberg_strength, direction)
        """
        if len(price) < window:
            return 0.0, "NONE"

        # Check if price is stuck at level with high volume
        price_recent = price[-window:]
        price_range = np.max(price_recent) - np.min(price_recent)
        price_std = np.std(price[-50:]) if len(price) >= 50 else 0.01

        # Price barely moving
        if price_range > price_std * 0.5:
            return 0.0, "NONE"

        # But high volume
        total_bid = np.sum(bid_volume[-window:])
        total_ask = np.sum(ask_volume[-window:])
        avg_volume = np.mean(bid_volume[-50:] + ask_volume[-50:])

        if avg_volume == 0:
            return 0.0, "NONE"

        volume_strength = (total_bid + total_ask) / (avg_volume * window)

        if volume_strength > 2.0:
            # Determine direction
            if total_bid > total_ask * 1.5:
                return float(volume_strength), "BID"
            elif total_ask > total_bid * 1.5:
                return float(volume_strength), "ASK"
            else:
                return float(volume_strength), "BOTH"

        return 0.0, "NONE"
