"""
Adaptive Risk Management Module

Provides volatility-adjusted stops, dynamic targets, and position sizing
based on market conditions and ATR (Average True Range).
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class AdaptiveRiskManager:
    """
    Volatility-adjusted risk management using ATR

    Replaces fixed stop losses and profit targets with dynamic levels
    that adapt to current market volatility.
    """

    def __init__(self,
                 atr_period: int = 14,
                 stop_atr_multiple: float = 2.0,
                 target_atr_multiple: float = 3.0):
        """
        Args:
            atr_period: Period for ATR calculation
            stop_atr_multiple: Stop loss distance in ATR multiples
            target_atr_multiple: Profit target distance in ATR multiples
        """
        self.atr_period = atr_period
        self.stop_atr_multiple = stop_atr_multiple
        self.target_atr_multiple = target_atr_multiple

    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray) -> float:
        """
        Calculate Average True Range

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = average of true ranges over period

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            Current ATR value
        """
        if len(highs) < self.atr_period + 1:
            # Fallback to simple range
            if len(highs) > 0 and len(lows) > 0:
                return float(np.mean(highs - lows))
            return 1.0

        # Calculate true range components
        high_low = highs[1:] - lows[1:]
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])

        # True range is maximum of the three
        true_range = np.maximum(high_low, high_close)
        true_range = np.maximum(true_range, low_close)

        # ATR is simple moving average of true range
        if len(true_range) >= self.atr_period:
            atr = np.mean(true_range[-self.atr_period:])
        else:
            atr = np.mean(true_range)

        return float(atr)

    def get_stop_loss(self, entry_price: float, direction: int, atr: float) -> float:
        """
        Calculate adaptive stop loss based on ATR

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: Current ATR value

        Returns:
            Stop loss price
        """
        stop_distance = atr * self.stop_atr_multiple

        if direction == 1:  # Long position
            stop_price = entry_price - stop_distance
        else:  # Short position
            stop_price = entry_price + stop_distance

        return float(stop_price)

    def get_profit_target(self, entry_price: float, direction: int, atr: float) -> float:
        """
        Calculate adaptive profit target based on ATR

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: Current ATR value

        Returns:
            Profit target price
        """
        target_distance = atr * self.target_atr_multiple

        if direction == 1:  # Long position
            target_price = entry_price + target_distance
        else:  # Short position
            target_price = entry_price - target_distance

        return float(target_price)

    def get_multiple_targets(self, entry_price: float, direction: int,
                            atr: float, num_targets: int = 3) -> list:
        """
        Calculate multiple profit targets for scaling out

        Args:
            entry_price: Entry price
            direction: 1 for long, -1 for short
            atr: Current ATR value
            num_targets: Number of targets to generate

        Returns:
            List of target prices
        """
        targets = []
        base_multiple = self.target_atr_multiple

        for i in range(num_targets):
            # Each target is progressively further
            multiplier = base_multiple * (i + 1)
            distance = atr * multiplier

            if direction == 1:
                target = entry_price + distance
            else:
                target = entry_price - distance

            targets.append(float(target))

        return targets

    def adjust_for_regime(self, regime: str, regime_confidence: float) -> None:
        """
        Adjust stop/target multiples based on market regime

        Args:
            regime: Current market regime
            regime_confidence: Confidence in regime classification
        """
        # Store original values
        if not hasattr(self, '_original_stop_multiple'):
            self._original_stop_multiple = self.stop_atr_multiple
            self._original_target_multiple = self.target_atr_multiple

        # Adjust based on regime
        if regime == 'trending' and regime_confidence > 0.6:
            # Wider stops and targets in trends
            self.stop_atr_multiple = self._original_stop_multiple * 1.3
            self.target_atr_multiple = self._original_target_multiple * 1.5

        elif regime == 'mean_reverting' and regime_confidence > 0.6:
            # Tighter stops and targets in mean reversion
            self.stop_atr_multiple = self._original_stop_multiple * 0.8
            self.target_atr_multiple = self._original_target_multiple * 0.7

        elif regime == 'high_vol' and regime_confidence > 0.6:
            # Much wider stops in high volatility
            self.stop_atr_multiple = self._original_stop_multiple * 1.5
            self.target_atr_multiple = self._original_target_multiple * 1.8

        else:
            # Reset to original values
            self.stop_atr_multiple = self._original_stop_multiple
            self.target_atr_multiple = self._original_target_multiple

        logger.debug(f"Risk params adjusted for {regime}: "
                    f"stop={self.stop_atr_multiple:.2f}, "
                    f"target={self.target_atr_multiple:.2f}")

    def calculate_risk_reward_ratio(self, entry_price: float, stop_price: float,
                                   target_price: float) -> float:
        """
        Calculate risk/reward ratio for a trade

        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Profit target price

        Returns:
            Risk/reward ratio (reward/risk)
        """
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)

        if risk == 0:
            return 0.0

        rr_ratio = reward / risk

        return float(rr_ratio)


class TimeBasedExitManager:
    """
    Exit positions based on time in trade and price movement

    Prevents capital from being tied up in stagnant positions
    """

    def __init__(self,
                 max_bars_in_trade: int = 20,
                 min_movement_threshold: float = 0.002):
        """
        Args:
            max_bars_in_trade: Maximum bars before forcing exit
            min_movement_threshold: Minimum price movement required (as fraction)
        """
        self.max_bars_in_trade = max_bars_in_trade
        self.min_movement_threshold = min_movement_threshold

    def should_exit_time(self, entry_price: float, current_price: float,
                        bars_in_trade: int) -> Tuple[bool, Optional[str]]:
        """
        Determine if position should exit due to time/stagnation

        Args:
            entry_price: Entry price
            current_price: Current price
            bars_in_trade: Bars since entry

        Returns:
            (should_exit, exit_reason)
        """
        if bars_in_trade >= self.max_bars_in_trade:
            # Check if made any meaningful movement
            if entry_price == 0:
                return True, "TIME_EXIT_MAX_BARS"

            movement = abs(current_price - entry_price) / entry_price

            if movement < self.min_movement_threshold:
                return True, "TIME_EXIT_STAGNANT"
            else:
                return True, "TIME_EXIT_MAX_BARS"

        return False, None


class StructureBasedTrailingStop:
    """
    Trailing stop that respects market structure

    Trails based on swing points rather than fixed distances,
    allowing winners to run while protecting profits.
    """

    def __init__(self,
                 swing_lookback: int = 10,
                 min_trail_distance: float = 0.003):
        """
        Args:
            swing_lookback: Bars to look back for swing points
            min_trail_distance: Minimum trailing distance (as fraction)
        """
        self.swing_lookback = swing_lookback
        self.min_trail_distance = min_trail_distance

    def find_swing_point(self, highs: np.ndarray, lows: np.ndarray,
                        direction: int) -> float:
        """
        Find recent swing high/low for trailing stop placement

        Args:
            highs: High prices
            lows: Low prices
            direction: 1 for long (find swing low), -1 for short (find swing high)

        Returns:
            Swing point price
        """
        if len(highs) < self.swing_lookback or len(lows) < self.swing_lookback:
            if direction == 1:
                return float(np.min(lows)) if len(lows) > 0 else 0.0
            else:
                return float(np.max(highs)) if len(highs) > 0 else 0.0

        if direction == 1:
            # For longs, trail below recent swing lows
            swing_low = np.min(lows[-self.swing_lookback:])
            return float(swing_low)
        else:
            # For shorts, trail above recent swing highs
            swing_high = np.max(highs[-self.swing_lookback:])
            return float(swing_high)

    def update_trailing_stop(self, current_price: float, entry_price: float,
                            direction: int, highs: np.ndarray, lows: np.ndarray,
                            current_stop: float) -> Tuple[float, bool]:
        """
        Update trailing stop based on price action

        Args:
            current_price: Current market price
            entry_price: Original entry price
            direction: 1 for long, -1 for short
            highs: High price array
            lows: Low price array
            current_stop: Current stop loss price

        Returns:
            (new_stop_price, should_exit)
        """
        swing_point = self.find_swing_point(highs, lows, direction)

        if direction == 1:  # Long position
            # Never lower the stop (only trail upward)
            new_stop = max(current_stop, swing_point)

            # Ensure minimum profit protection once in profit
            if entry_price > 0:
                min_profit_stop = entry_price * (1 + self.min_trail_distance)

                if current_price > min_profit_stop:
                    new_stop = max(new_stop, min_profit_stop)

            # Check if stop hit
            should_exit = current_price <= new_stop

        else:  # Short position
            # Never raise the stop (only trail downward)
            new_stop = min(current_stop, swing_point)

            # Ensure minimum profit protection
            if entry_price > 0:
                min_profit_stop = entry_price * (1 - self.min_trail_distance)

                if current_price < min_profit_stop:
                    new_stop = min(new_stop, min_profit_stop)

            # Check if stop hit
            should_exit = current_price >= new_stop

        return float(new_stop), should_exit


class PartialExitManager:
    """
    Scale out of positions to lock profits while letting winners run
    """

    def __init__(self,
                 exit_levels: list = None,
                 exit_percentages: list = None):
        """
        Args:
            exit_levels: Multiples of initial target (e.g., [0.5, 1.0, 1.5])
            exit_percentages: Percentage to exit at each level (e.g., [0.33, 0.33, 0.34])
        """
        self.exit_levels = exit_levels or [0.5, 1.0, 1.5]
        self.exit_percentages = exit_percentages or [0.33, 0.33, 0.34]

        # Validate
        if len(self.exit_levels) != len(self.exit_percentages):
            raise ValueError("exit_levels and exit_percentages must have same length")

        if not np.isclose(sum(self.exit_percentages), 1.0):
            raise ValueError("exit_percentages must sum to 1.0")

    def calculate_partial_exit(self, entry_price: float, current_price: float,
                               initial_target: float, direction: int,
                               total_quantity: int,
                               current_exit_level: int) -> Tuple[int, int, bool]:
        """
        Calculate partial exit quantity at current price

        Args:
            entry_price: Entry price
            current_price: Current price
            initial_target: Initial profit target
            direction: 1 for long, -1 for short
            total_quantity: Total position size
            current_exit_level: Current exit level (0 = no exits yet)

        Returns:
            (exit_quantity, new_exit_level, should_exit)
        """
        if entry_price == 0 or initial_target == 0:
            return 0, current_exit_level, False

        # Calculate profit achieved vs target
        target_profit = abs(initial_target - entry_price)
        current_profit = abs(current_price - entry_price) if direction == 1 else abs(entry_price - current_price)

        if target_profit == 0:
            return 0, current_exit_level, False

        profit_ratio = current_profit / target_profit

        # Check if reached next exit level
        for i, level in enumerate(self.exit_levels):
            if i <= current_exit_level:
                continue  # Already exited this level

            if profit_ratio >= level:
                # Take partial profit
                exit_qty = int(total_quantity * self.exit_percentages[i])

                return exit_qty, i, True

        return 0, current_exit_level, False

    def get_remaining_quantity(self, total_quantity: int,
                              current_exit_level: int) -> int:
        """
        Calculate remaining position size after partial exits

        Args:
            total_quantity: Original position size
            current_exit_level: Current exit level

        Returns:
            Remaining quantity
        """
        exited_percentage = sum(self.exit_percentages[:current_exit_level + 1])
        remaining_percentage = 1.0 - exited_percentage

        remaining_qty = int(total_quantity * remaining_percentage)

        return remaining_qty
