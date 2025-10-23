"""
Confidence-Based Position Sizing - Complete Implementation
"""

import numpy as np
from typing import Tuple


class ConfidenceBasedPositionSizing:
    """Dynamic position sizing based on confidence score"""

    def __init__(
        self,
        base_position_size: int = 1,
        max_position_size: int = 3,
        min_confidence: float = 60.0,
        max_confidence: float = 100.0
    ):
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def calculate_position_size(
        self,
        confidence: float,
        account_balance: float,
        risk_per_trade: float = 0.01
    ) -> Tuple[int, float]:
        """
        Calculate position size based on confidence and risk parameters

        Args:
            confidence: Composite confidence score (0-100)
            account_balance: Current account size
            risk_per_trade: Max % of account to risk (default 1%)

        Returns:
            Tuple of (position_size, risk_amount)
        """
        # Base risk amount
        max_risk_amount = account_balance * risk_per_trade

        # Scale position by confidence
        # Confidence 60 = 50% of max size
        # Confidence 100 = 100% of max size
        confidence_range = self.max_confidence - self.min_confidence
        confidence_scalar = (confidence - self.min_confidence) / confidence_range
        confidence_scalar = max(0, min(confidence_scalar, 1))

        # Calculate position size
        position_size = self.base_position_size + (
            (self.max_position_size - self.base_position_size) * confidence_scalar
        )

        # Calculate actual risk
        risk_amount = max_risk_amount * confidence_scalar

        return int(position_size), risk_amount
