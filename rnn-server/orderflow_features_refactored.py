"""
Enhanced Orderflow Features Module (Refactored)

Provides advanced orderflow and microstructure analysis for trading signals.
Based on institutional order flow, volume profile, and market microstructure theory.

This refactored version improves:
- Readability: Clear constants, descriptive names, logical flow
- Maintainability: DRY principle, modular helpers, consistent patterns
- Robustness: Input validation, error handling, safe operations
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Default window sizes for analysis
DEFAULT_DIVERGENCE_WINDOW = 20
DEFAULT_IMBALANCE_WINDOW = 10
DEFAULT_ABSORPTION_WINDOW = 5
DEFAULT_ICEBERG_WINDOW = 5
DEFAULT_BLOCK_TRADE_WINDOW = 50
DEFAULT_VOLUME_PROFILE_BINS = 20

# Detection thresholds
BLOCK_TRADE_THRESHOLD_STD = 3.0  # Standard deviations above mean
ICEBERG_VOLUME_MULTIPLIER = 2.0   # Times normal volume
ICEBERG_IMBALANCE_THRESHOLD = 1.5 # Ratio for directional bias
ABSORPTION_PRICE_TOLERANCE = 0.5  # Fraction of volatility

# Valid return value ranges
IMBALANCE_RANGE = (-1.0, 1.0)  # -1 = all selling, +1 = all buying
DIVERGENCE_RANGE = (0.0, 1.0)  # 0 = no divergence, 1 = complete divergence


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class IcebergDirection(Enum):
    """Direction of detected iceberg order"""
    NONE = "NONE"
    BID = "BID"    # Large hidden buy order
    ASK = "ASK"    # Large hidden sell order
    BOTH = "BOTH"  # Activity on both sides


@dataclass(frozen=True)
class IcebergOrderResult:
    """
    Result from iceberg order detection

    Attributes:
        strength: Magnitude of iceberg activity (0.0 = none, >2.0 = strong)
        direction: Side where iceberg order detected
    """
    strength: float
    direction: IcebergDirection


# =============================================================================
# MAIN FEATURE EXTRACTION CLASS
# =============================================================================

class OrderflowFeatures:
    """
    Advanced orderflow and microstructure feature extraction

    This class provides static methods for analyzing order flow patterns
    that indicate institutional trading activity and market microstructure dynamics.

    All methods are stateless and can be called independently.

    Example:
        >>> orderflow = OrderflowFeatures()
        >>> bid_vol = np.array([100, 150, 200, 180])
        >>> ask_vol = np.array([120, 130, 250, 190])
        >>> cum_delta = orderflow.cumulative_delta(bid_vol, ask_vol)
        >>> imbalance = orderflow.bid_ask_imbalance(bid_vol, ask_vol, window=3)
    """

    # -------------------------------------------------------------------------
    # Core Orderflow Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def cumulative_delta(
        bid_volume: np.ndarray,
        ask_volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cumulative volume delta to track institutional flow direction

        Delta represents the difference between aggressive buying and selling.
        Positive cumulative delta indicates net buying pressure.
        Negative cumulative delta indicates net selling pressure.

        Args:
            bid_volume: Volume traded at bid (sellers hitting bid)
            ask_volume: Volume traded at ask (buyers hitting ask)

        Returns:
            Cumulative delta array where each element represents
            the net cumulative aggressive buying/selling pressure

        Example:
            >>> bid_vol = np.array([100, 150, 200])
            >>> ask_vol = np.array([120, 130, 250])
            >>> delta = OrderflowFeatures.cumulative_delta(bid_vol, ask_vol)
            >>> # delta = [20, 0, 50] cumulative
        """
        delta = ask_volume - bid_volume
        return np.cumsum(delta)

    @staticmethod
    def delta_divergence(
        price: np.ndarray,
        cum_delta: np.ndarray,
        window: int = DEFAULT_DIVERGENCE_WINDOW
    ) -> float:
        """
        Detect price-delta divergence (smart money vs retail flow)

        Divergence occurs when price and delta move in opposite directions.
        This often indicates institutional positioning against retail flow,
        potentially signaling an upcoming reversal.

        Args:
            price: Price series
            cum_delta: Cumulative delta series from cumulative_delta()
            window: Number of bars to analyze for divergence

        Returns:
            Divergence score:
            - 0.0: No divergence (price and delta aligned)
            - 0.5: Moderate divergence
            - 1.0: Complete divergence (opposite directions)
        """
        if not _validate_sufficient_length(price, window):
            return 0.0

        if not _validate_sufficient_length(cum_delta, window):
            return 0.0

        # Extract recent data
        price_recent = price[-window:]
        delta_recent = cum_delta[-window:]

        # Calculate directional movements
        price_direction = np.sign(np.diff(price_recent))
        delta_direction = np.sign(np.diff(delta_recent))

        # Count instances where directions don't match
        divergence_count = np.sum(price_direction != delta_direction)
        total_comparisons = window - 1

        # Calculate ratio and ensure in valid range
        divergence_ratio = divergence_count / total_comparisons

        return _clip_to_range(divergence_ratio, *DIVERGENCE_RANGE)

    @staticmethod
    def bid_ask_imbalance(
        bid_volume: np.ndarray,
        ask_volume: np.ndarray,
        window: int = DEFAULT_IMBALANCE_WINDOW
    ) -> float:
        """
        Calculate immediate supply/demand imbalance

        Measures the balance between recent buying and selling pressure.
        Useful for identifying short-term orderflow imbalances that
        may lead to price movement.

        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            window: Number of recent bars to analyze

        Returns:
            Imbalance score:
            - -1.0: Complete selling pressure (all volume on bid)
            - 0.0: Balanced (equal bid/ask volume)
            - +1.0: Complete buying pressure (all volume on ask)
        """
        if not (_validate_sufficient_length(bid_volume, window) and
                _validate_sufficient_length(ask_volume, window)):
            return 0.0

        # Calculate recent totals
        recent_bid_total = np.sum(bid_volume[-window:])
        recent_ask_total = np.sum(ask_volume[-window:])

        total_volume = recent_bid_total + recent_ask_total

        # Handle zero volume case
        if total_volume == 0:
            return 0.0

        # Calculate imbalance: positive = buying, negative = selling
        imbalance = (recent_ask_total - recent_bid_total) / total_volume

        return _clip_to_range(imbalance, *IMBALANCE_RANGE)

    # -------------------------------------------------------------------------
    # Advanced Microstructure Analysis
    # -------------------------------------------------------------------------

    @staticmethod
    def volume_at_price_profile(
        prices: np.ndarray,
        volumes: np.ndarray,
        bins: int = DEFAULT_VOLUME_PROFILE_BINS
    ) -> float:
        """
        Calculate distance from Point of Control (POC) in volume profile

        The POC is the price level with the highest traded volume,
        representing where institutions have shown acceptance of value.
        Distance from POC can indicate potential support/resistance.

        Args:
            prices: Price series
            volumes: Volume series
            bins: Number of price bins for volume histogram

        Returns:
            Percentage distance from POC:
            - Positive: Current price above POC
            - Negative: Current price below POC
            - 0.0: At POC or insufficient data
        """
        if not (_validate_sufficient_length(prices, bins) and
                _validate_sufficient_length(volumes, bins)):
            return 0.0

        try:
            poc_price = _calculate_point_of_control(prices, volumes, bins)
            current_price = prices[-1]

            return _safe_divide(
                current_price - poc_price,
                current_price,
                default=0.0
            )

        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def absorption_detection(
        price: np.ndarray,
        volume: np.ndarray,
        window: int = DEFAULT_ABSORPTION_WINDOW
    ) -> float:
        """
        Detect absorption (high volume with minimal price movement)

        Absorption occurs when large volume trades at a price level
        without significant price movement, indicating strong institutional
        support or resistance at that level.

        Args:
            price: Price series
            volume: Volume series
            window: Number of recent bars to analyze

        Returns:
            Absorption strength:
            - 0.0: No absorption detected
            - 1.0: Normal volume
            - >1.0: Volume higher than average with low price movement
            - >2.0: Strong absorption
        """
        min_history_required = max(window, 100)

        if not (_validate_sufficient_length(price, min_history_required) and
                _validate_sufficient_length(volume, min_history_required)):
            return 0.0

        # Calculate price movement
        recent_price_range = _calculate_price_range(price, window)
        historical_volatility = np.std(price[-100:])

        # Calculate volume metrics
        recent_avg_volume = np.mean(volume[-window:])
        historical_avg_volume = np.mean(volume[-100:])

        # Guard against division by zero
        if historical_volatility == 0 or historical_avg_volume == 0:
            return 0.0

        # Normalized price movement
        normalized_movement = recent_price_range / historical_volatility

        # Volume ratio vs historical
        volume_ratio = recent_avg_volume / historical_avg_volume

        # Absorption: high volume but low price movement
        if normalized_movement < ABSORPTION_PRICE_TOLERANCE:
            return float(volume_ratio)

        return 0.0

    @staticmethod
    def block_trade_detection(
        volume: np.ndarray,
        threshold_std: float = BLOCK_TRADE_THRESHOLD_STD
    ) -> float:
        """
        Detect unusually large trades (block trades) from institutions

        Block trades are orders significantly larger than typical volume,
        often indicating institutional activity that may impact price.

        Args:
            volume: Volume series
            threshold_std: Number of standard deviations above mean to flag

        Returns:
            Block trade score:
            - 0.0: Normal volume
            - >0.0: Number of std deviations above threshold
        """
        if not _validate_sufficient_length(volume, DEFAULT_BLOCK_TRADE_WINDOW):
            return 0.0

        historical_volume = volume[:-1]  # Exclude current bar from statistics

        mean_volume = np.mean(historical_volume)
        std_volume = np.std(historical_volume)

        if std_volume == 0:
            return 0.0

        current_volume = volume[-1]
        z_score = (current_volume - mean_volume) / std_volume

        # Return excess above threshold
        if z_score > threshold_std:
            return float(z_score - threshold_std)

        return 0.0

    @staticmethod
    def iceberg_order_detection(
        bid_volume: np.ndarray,
        ask_volume: np.ndarray,
        price: np.ndarray,
        window: int = DEFAULT_ICEBERG_WINDOW
    ) -> IcebergOrderResult:
        """
        Detect potential iceberg orders (hidden institutional orders)

        Iceberg orders are large hidden orders that only show a small
        portion. They're identified by repeated high volume at the same
        price level without significant price movement.

        Args:
            bid_volume: Bid volume series
            ask_volume: Ask volume series
            price: Price series
            window: Number of bars to analyze

        Returns:
            IcebergOrderResult containing:
            - strength: Magnitude of activity (>2.0 indicates strong iceberg)
            - direction: BID, ASK, BOTH, or NONE
        """
        if not _validate_sufficient_length(price, window):
            return IcebergOrderResult(0.0, IcebergDirection.NONE)

        # Check if price is consolidating (key iceberg characteristic)
        is_consolidating = _is_price_consolidating(price, window)

        if not is_consolidating:
            return IcebergOrderResult(0.0, IcebergDirection.NONE)

        # Calculate volume strength vs historical average
        volume_strength = _calculate_volume_strength(
            bid_volume, ask_volume, window
        )

        # Iceberg detected if volume is significantly elevated
        if volume_strength > ICEBERG_VOLUME_MULTIPLIER:
            direction = _determine_iceberg_direction(
                bid_volume, ask_volume, window
            )
            return IcebergOrderResult(volume_strength, direction)

        return IcebergOrderResult(0.0, IcebergDirection.NONE)

    @staticmethod
    def trade_intensity(
        volume: np.ndarray,
        time_delta: Optional[np.ndarray] = None,
        window: int = DEFAULT_IMBALANCE_WINDOW
    ) -> float:
        """
        Measure trade intensity (volume per time unit)

        High intensity indicates urgency from institutional traders
        trying to execute large orders quickly.

        Args:
            volume: Volume series
            time_delta: Time between bars (None assumes uniform spacing)
            window: Recent bars to analyze

        Returns:
            Intensity ratio vs historical average (>1.0 = elevated intensity)
        """
        min_required = max(window, 100)

        if not _validate_sufficient_length(volume, min_required):
            return 1.0  # Neutral value

        if time_delta is None:
            # Uniform time spacing
            recent_intensity = np.mean(volume[-window:])
            historical_intensity = np.mean(volume[-100:-window])
        else:
            # Calculate volume per time unit
            recent_total_volume = np.sum(volume[-window:])
            recent_total_time = np.sum(time_delta[-window:])

            historical_total_volume = np.sum(volume[-100:-window])
            historical_total_time = np.sum(time_delta[-100:-window])

            recent_intensity = _safe_divide(
                recent_total_volume, recent_total_time, default=1.0
            )
            historical_intensity = _safe_divide(
                historical_total_volume, historical_total_time, default=1.0
            )

        return _safe_divide(
            recent_intensity, historical_intensity, default=1.0
        )


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================

def _validate_sufficient_length(arr: np.ndarray, min_length: int) -> bool:
    """
    Check if array has sufficient length for analysis

    Args:
        arr: Array to validate
        min_length: Minimum required length

    Returns:
        True if array is long enough, False otherwise
    """
    return len(arr) >= min_length


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Perform division with zero-denominator protection

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero
    """
    if denominator == 0:
        return default

    return numerator / denominator


def _clip_to_range(value: float, min_val: float, max_val: float) -> float:
    """
    Clip value to specified range

    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clipped value as float
    """
    return float(np.clip(value, min_val, max_val))


def _calculate_price_range(price: np.ndarray, window: int) -> float:
    """
    Calculate price range (high - low) over window

    Args:
        price: Price series
        window: Number of bars

    Returns:
        Price range
    """
    price_recent = price[-window:]
    return np.max(price_recent) - np.min(price_recent)


def _calculate_point_of_control(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int
) -> float:
    """
    Calculate Point of Control (price with highest volume)

    Args:
        prices: Price series
        volumes: Volume series
        bins: Number of price bins

    Returns:
        Price level with highest volume
    """
    hist, edges = np.histogram(prices, bins=bins, weights=volumes)

    # Find bin with highest volume
    poc_idx = np.argmax(hist)

    # Return midpoint of POC bin
    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    return poc_price


def _is_price_consolidating(price: np.ndarray, window: int) -> bool:
    """
    Check if price is consolidating (low movement relative to volatility)

    Args:
        price: Price series
        window: Window to check

    Returns:
        True if consolidating, False otherwise
    """
    price_range = _calculate_price_range(price, window)

    # Calculate historical volatility for comparison
    historical_lookback = min(len(price), 50)
    historical_std = np.std(price[-historical_lookback:])

    if historical_std == 0:
        return False

    # Consolidating if range is small relative to volatility
    return price_range < (historical_std * ABSORPTION_PRICE_TOLERANCE)


def _calculate_volume_strength(
    bid_volume: np.ndarray,
    ask_volume: np.ndarray,
    window: int
) -> float:
    """
    Calculate volume strength vs historical average

    Args:
        bid_volume: Bid volume series
        ask_volume: Ask volume series
        window: Window to analyze

    Returns:
        Volume strength ratio (>1.0 = above average)
    """
    min_required = max(window, 50)

    if not (_validate_sufficient_length(bid_volume, min_required) and
            _validate_sufficient_length(ask_volume, min_required)):
        return 0.0

    # Recent volume
    recent_total_bid = np.sum(bid_volume[-window:])
    recent_total_ask = np.sum(ask_volume[-window:])
    recent_total = recent_total_bid + recent_total_ask

    # Historical average volume per bar
    historical_avg = np.mean(bid_volume[-50:] + ask_volume[-50:])

    # Expected total volume over window
    expected_total = historical_avg * window

    return _safe_divide(recent_total, expected_total, default=0.0)


def _determine_iceberg_direction(
    bid_volume: np.ndarray,
    ask_volume: np.ndarray,
    window: int
) -> IcebergDirection:
    """
    Determine which side has the iceberg order

    Args:
        bid_volume: Bid volume series
        ask_volume: Ask volume series
        window: Window to analyze

    Returns:
        IcebergDirection indicating where iceberg detected
    """
    total_bid = np.sum(bid_volume[-window:])
    total_ask = np.sum(ask_volume[-window:])

    # Check for strong directional bias
    if total_bid > total_ask * ICEBERG_IMBALANCE_THRESHOLD:
        return IcebergDirection.BID

    if total_ask > total_bid * ICEBERG_IMBALANCE_THRESHOLD:
        return IcebergDirection.ASK

    # Activity on both sides
    return IcebergDirection.BOTH
