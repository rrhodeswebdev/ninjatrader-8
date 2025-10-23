"""
Performance Monitoring Dashboard - Complete Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """Comprehensive performance tracking"""

    @staticmethod
    def calculate_metrics(trade_history: pd.DataFrame) -> Dict:
        """Calculate all performance metrics"""
        if len(trade_history) == 0:
            return {}

        # Return metrics
        total_return = trade_history['pnl'].sum()
        returns = trade_history['pnl'] / trade_history['equity'].shift(1)
        annual_return = returns.mean() * 252  # Annualized

        # Risk metrics
        max_drawdown = (trade_history['equity'] / trade_history['equity'].cummax() - 1).min()
        returns_std = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / returns_std if returns_std > 0 else 0

        # Trade metrics
        wins = trade_history[trade_history['pnl'] > 0]
        losses = trade_history[trade_history['pnl'] < 0]

        win_rate = len(wins) / len(trade_history) if len(trade_history) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'total_trades': len(trade_history)
        }
