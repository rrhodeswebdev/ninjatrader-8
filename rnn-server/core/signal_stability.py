"""Signal stability and persistence checks to prevent over-trading.

This module implements signal filtering to prevent rapid signal changes
that lead to over-trading. It enforces minimum holding periods and requires
strong conviction for signal reversals.
"""

from typing import Literal, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SignalConfig:
    """Configuration for signal stability rules."""
    min_bars_between_reversal: int = 2  # REDUCED: Was 5, now 2 bars (allow faster reversals)
    min_bars_in_hold: int = 1  # REDUCED: Was 3, now 1 bar (allow faster re-entry)
    required_confidence_increase: float = 0.05  # REDUCED: Was 0.15, now 0.05 (5% increase)
    enable_stability_check: bool = True  # Set to False to disable completely


class SignalState:
    """
    Track signal history to enforce stability and prevent over-trading.

    This class maintains state of previous signals to enforce rules like:
    - Minimum bars between buy↔sell reversals
    - Minimum hold period before re-entering
    - Required confidence increase for reversals
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialize signal state tracker.

        Args:
            config: Configuration for stability rules (uses defaults if None)
        """
        self.config = config or SignalConfig()

        self.last_signal: Literal["buy", "sell", "hold"] = "hold"
        self.last_signal_time: Optional[datetime] = None
        self.bars_since_last_change: int = 0
        self.last_confidence: float = 0.0
        self.signal_history: list[tuple[str, float, datetime]] = []

    def should_allow_signal(
        self,
        new_signal: Literal["buy", "sell", "hold"],
        new_confidence: float,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Determine if new signal should be allowed based on stability rules.

        Args:
            new_signal: The new signal to evaluate
            new_confidence: Confidence score for new signal (0-1)
            current_time: Current timestamp (uses now() if None)

        Returns:
            Tuple of (allow_signal: bool, reason: str)
            - allow_signal: True if signal should be executed
            - reason: Explanation for the decision

        Examples:
            >>> state = SignalState()
            >>> state.update_signal("buy", 0.7, datetime.now())
            >>> allowed, reason = state.should_allow_signal("sell", 0.8, datetime.now())
            >>> # Returns (False, "Too soon for reversal (0/5 bars)")
        """
        if not self.config.enable_stability_check:
            return True, "Stability check disabled"

        current_time = current_time or datetime.now()

        # Always allow same signal (no change)
        if new_signal == self.last_signal:
            return True, "Same signal - no change"

        # Rule 1: Prevent rapid reversals (buy -> sell or sell -> buy)
        if self._is_reversal(new_signal):
            if self.bars_since_last_change < self.config.min_bars_between_reversal:
                return False, (
                    f"Too soon for reversal "
                    f"({self.bars_since_last_change}/{self.config.min_bars_between_reversal} bars). "
                    f"Previous: {self.last_signal.upper()} @ {self.last_confidence:.1%}"
                )

            # Rule 3: Require significant confidence increase for reversals
            confidence_increase = new_confidence - self.last_confidence
            if confidence_increase < self.config.required_confidence_increase:
                return False, (
                    f"Insufficient confidence increase for reversal "
                    f"({confidence_increase:+.1%} < {self.config.required_confidence_increase:.1%}). "
                    f"Need {(self.last_confidence + self.config.required_confidence_increase):.1%} confidence, "
                    f"got {new_confidence:.1%}"
                )

        # Rule 2: Require minimum hold period before re-entering from hold
        if self.last_signal == "hold" and new_signal in ["buy", "sell"]:
            if self.bars_since_last_change < self.config.min_bars_in_hold:
                return False, (
                    f"Minimum hold period not met "
                    f"({self.bars_since_last_change}/{self.config.min_bars_in_hold} bars). "
                    f"Wait {self.config.min_bars_in_hold - self.bars_since_last_change} more bars"
                )

        return True, "Signal allowed"

    def update_signal(
        self,
        signal: Literal["buy", "sell", "hold"],
        confidence: float,
        current_time: Optional[datetime] = None
    ):
        """
        Update signal state after allowing a new signal.

        Args:
            signal: The new signal being executed
            confidence: Confidence score for the signal
            current_time: Current timestamp (uses now() if None)
        """
        current_time = current_time or datetime.now()

        # Track signal history
        self.signal_history.append((signal, confidence, current_time))

        # Keep only last 100 signals in history
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        # Update counters
        if signal != self.last_signal:
            self.bars_since_last_change = 0
        else:
            self.bars_since_last_change += 1

        # Update state
        self.last_signal = signal
        self.last_confidence = confidence
        self.last_signal_time = current_time

    def _is_reversal(self, new_signal: str) -> bool:
        """
        Check if new signal is opposite of last signal.

        Args:
            new_signal: Signal to check

        Returns:
            True if signals are opposite (buy↔sell)
        """
        reversals = {
            ("buy", "sell"),
            ("sell", "buy")
        }
        return (self.last_signal, new_signal) in reversals

    def get_statistics(self) -> dict:
        """
        Get statistics about signal stability.

        Returns:
            Dictionary with signal statistics:
            - total_signals: Total signals in history
            - reversals: Number of buy↔sell reversals
            - holds: Number of hold signals
            - avg_bars_between_changes: Average bars between signal changes
        """
        if not self.signal_history:
            return {
                "total_signals": 0,
                "reversals": 0,
                "holds": 0,
                "avg_bars_between_changes": 0
            }

        total = len(self.signal_history)
        holds = sum(1 for s, _, _ in self.signal_history if s == "hold")

        # Count reversals
        reversals = 0
        for i in range(1, len(self.signal_history)):
            prev_signal = self.signal_history[i-1][0]
            curr_signal = self.signal_history[i][0]
            if (prev_signal, curr_signal) in [("buy", "sell"), ("sell", "buy")]:
                reversals += 1

        # Calculate average bars between changes
        changes = 0
        bars_sum = 0
        for i in range(1, len(self.signal_history)):
            if self.signal_history[i][0] != self.signal_history[i-1][0]:
                changes += 1
                bars_sum += 1  # Each entry represents 1 bar

        avg_bars = bars_sum / changes if changes > 0 else 0

        return {
            "total_signals": total,
            "reversals": reversals,
            "holds": holds,
            "avg_bars_between_changes": avg_bars,
            "last_signal": self.last_signal,
            "bars_since_last_change": self.bars_since_last_change
        }

    def reset(self):
        """Reset signal state to initial conditions."""
        self.last_signal = "hold"
        self.last_signal_time = None
        self.bars_since_last_change = 0
        self.last_confidence = 0.0
        self.signal_history = []


# Global instance for use across requests
_global_signal_state: Optional[SignalState] = None


def get_signal_state(reset: bool = False) -> SignalState:
    """
    Get or create global signal state instance.

    Args:
        reset: If True, reset the state to initial conditions

    Returns:
        Global SignalState instance
    """
    global _global_signal_state

    if _global_signal_state is None or reset:
        _global_signal_state = SignalState()

    return _global_signal_state


def check_signal_stability(
    new_signal: Literal["buy", "sell", "hold"],
    confidence: float,
    state: Optional[SignalState] = None
) -> Tuple[bool, str]:
    """
    Convenience function to check signal stability using global or provided state.

    Args:
        new_signal: Signal to check
        confidence: Confidence score
        state: Optional SignalState instance (uses global if None)

    Returns:
        Tuple of (allowed, reason)
    """
    if state is None:
        state = get_signal_state()

    return state.should_allow_signal(new_signal, confidence)


def update_signal_state(
    signal: Literal["buy", "sell", "hold"],
    confidence: float,
    state: Optional[SignalState] = None
):
    """
    Convenience function to update signal state.

    Args:
        signal: New signal
        confidence: Confidence score
        state: Optional SignalState instance (uses global if None)
    """
    if state is None:
        state = get_signal_state()

    state.update_signal(signal, confidence)
