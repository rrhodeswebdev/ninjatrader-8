"""
Safeguard System with Circuit Breakers - Complete Implementation
"""

from typing import Tuple, List, Dict


class SafeguardSystem:
    """Automated monitoring and circuit breakers"""

    def __init__(self):
        self.daily_loss_limit = -0.02  # 2% max daily loss
        self.drawdown_limit = -0.10    # 10% max drawdown
        self.min_confidence = 60        # Minimum confidence to trade

    def check_safeguards(self, current_state: Dict) -> Tuple[bool, List[str]]:
        """
        Check all safeguards before allowing trades

        Args:
            current_state: Dict with current trading state

        Returns:
            Tuple of (can_trade, alerts)
        """
        alerts = []

        # Daily loss check
        if current_state.get('daily_pnl', 0) < self.daily_loss_limit:
            alerts.append('CRITICAL: Daily loss limit reached')
            return False, alerts

        # Drawdown check
        if current_state.get('drawdown', 0) < self.drawdown_limit:
            alerts.append('CRITICAL: Maximum drawdown exceeded')
            return False, alerts

        # Model performance degradation
        if current_state.get('recent_win_rate', 0.5) < 0.40:
            alerts.append('WARNING: Win rate below 40%')

        # Confidence distribution shift
        if current_state.get('avg_confidence', 65) < 65:
            alerts.append('WARNING: Average confidence below threshold')

        return True, alerts
