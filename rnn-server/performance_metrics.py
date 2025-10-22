"""
Performance Metrics Tracker

Comprehensive tracking of trading performance metrics including
Sharpe ratio, win rate, profit factor, drawdown, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate and track comprehensive trading performance metrics
    """

    def __init__(self):
        """Initialize performance tracker"""
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [0.0]

    def add_trade(self, entry_time: datetime, exit_time: datetime,
                  entry_price: float, exit_price: float,
                  direction: int, quantity: int, pnl: float,
                  signal_confidence: Optional[float] = None,
                  regime: Optional[str] = None):
        """
        Record a completed trade

        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            direction: 1 for long, -1 for short
            quantity: Position size
            pnl: Profit/loss
            signal_confidence: Model confidence (optional)
            regime: Market regime (optional)
        """
        hold_time = (exit_time - entry_time).total_seconds() / 60  # Minutes

        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'quantity': quantity,
            'pnl': pnl,
            'hold_time': hold_time,
            'signal_confidence': signal_confidence,
            'regime': regime,
            'win': pnl > 0
        }

        self.trades.append(trade)
        self.equity_curve.append(self.equity_curve[-1] + pnl)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics

        Returns:
            Dictionary of performance metrics
        """
        if len(self.trades) == 0:
            return self._empty_metrics()

        df = pd.DataFrame(self.trades)

        metrics = {}

        # Basic metrics
        metrics['total_trades'] = len(df)
        metrics['winning_trades'] = (df['pnl'] > 0).sum()
        metrics['losing_trades'] = (df['pnl'] < 0).sum()
        metrics['breakeven_trades'] = (df['pnl'] == 0).sum()

        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        else:
            metrics['win_rate'] = 0.0

        # P&L metrics
        metrics['total_pnl'] = df['pnl'].sum()
        metrics['avg_pnl'] = df['pnl'].mean()

        winners = df[df['pnl'] > 0]['pnl']
        losers = df[df['pnl'] < 0]['pnl']

        metrics['avg_win'] = winners.mean() if len(winners) > 0 else 0.0
        metrics['avg_loss'] = losers.mean() if len(losers) > 0 else 0.0
        metrics['largest_win'] = winners.max() if len(winners) > 0 else 0.0
        metrics['largest_loss'] = losers.min() if len(losers) > 0 else 0.0

        # Profit factor
        total_wins = winners.sum() if len(winners) > 0 else 0.0
        total_losses = abs(losers.sum()) if len(losers) > 0 else 0.0

        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses
        else:
            metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0

        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown()
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(df['pnl'])
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(df['pnl'])
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(metrics['total_pnl'],
                                                                 metrics['max_drawdown'])

        # Trade duration
        metrics['avg_hold_time_winner'] = df[df['pnl'] > 0]['hold_time'].mean() \
            if len(df[df['pnl'] > 0]) > 0 else 0.0
        metrics['avg_hold_time_loser'] = df[df['pnl'] < 0]['hold_time'].mean() \
            if len(df[df['pnl'] < 0]) > 0 else 0.0
        metrics['avg_hold_time'] = df['hold_time'].mean()

        # Expectancy
        metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win']) + \
                               ((1 - metrics['win_rate']) * metrics['avg_loss'])

        # Kelly percentage (approximate)
        if metrics['avg_loss'] != 0:
            win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss'])
            metrics['kelly_percentage'] = (metrics['win_rate'] -
                                          (1 - metrics['win_rate']) / win_loss_ratio)
        else:
            metrics['kelly_percentage'] = 0.0

        # Consecutive wins/losses
        metrics['max_consecutive_wins'] = self._calculate_max_consecutive_wins()
        metrics['max_consecutive_losses'] = self._calculate_max_consecutive_losses()

        # By regime (if available)
        if 'regime' in df.columns and df['regime'].notna().any():
            regime_metrics = self._calculate_regime_metrics(df)
            metrics['regime_performance'] = regime_metrics

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'expectancy': 0.0
        }

    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown

        Returns:
            Maximum peak-to-trough decline
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = np.min(drawdown)

        return float(max_dd)

    def _calculate_sharpe_ratio(self, pnl_series: pd.Series,
                                risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (annualized)

        Args:
            pnl_series: Series of P&L values
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Annualized Sharpe ratio
        """
        if len(pnl_series) == 0:
            return 0.0

        # Convert to returns
        if len(self.equity_curve) > 1:
            equity = np.array(self.equity_curve[1:])  # Exclude starting 0
            returns = np.diff(equity) / (equity[:-1] + 1e-10)  # Avoid division by zero
        else:
            return 0.0

        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / (252 * 390)  # Daily risk-free rate

        if np.std(returns) == 0:
            return 0.0

        # Annualize (assuming ~252 trading days, ~390 minutes per day)
        sharpe = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(252 * 390)

        return float(sharpe)

    def _calculate_sortino_ratio(self, pnl_series: pd.Series,
                                 target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (focus on downside risk)

        Args:
            pnl_series: Series of P&L values
            target_return: Target return threshold

        Returns:
            Annualized Sortino ratio
        """
        if len(pnl_series) == 0:
            return 0.0

        # Convert to returns
        if len(self.equity_curve) > 1:
            equity = np.array(self.equity_curve[1:])
            returns = np.diff(equity) / (equity[:-1] + 1e-10)
        else:
            return 0.0

        if len(returns) == 0:
            return 0.0

        # Only consider downside deviation
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        # Annualize
        sortino = (np.mean(returns) - target_return) / np.std(downside_returns) * np.sqrt(252 * 390)

        return float(sortino)

    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)

        Args:
            total_return: Total return
            max_drawdown: Maximum drawdown

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        calmar = abs(total_return / max_drawdown)

        return float(calmar)

    def _calculate_max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        if len(self.trades) == 0:
            return 0

        wins = [t['win'] for t in self.trades]
        max_consecutive = 0
        current_consecutive = 0

        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if len(self.trades) == 0:
            return 0

        losses = [not t['win'] and t['pnl'] < 0 for t in self.trades]
        max_consecutive = 0
        current_consecutive = 0

        for loss in losses:
            if loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_regime_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate performance metrics by regime

        Args:
            df: DataFrame of trades

        Returns:
            Dictionary of metrics by regime
        """
        regime_metrics = {}

        for regime in df['regime'].unique():
            if pd.isna(regime):
                continue

            regime_df = df[df['regime'] == regime]

            regime_metrics[regime] = {
                'trades': len(regime_df),
                'win_rate': (regime_df['pnl'] > 0).mean(),
                'avg_pnl': regime_df['pnl'].mean(),
                'total_pnl': regime_df['pnl'].sum()
            }

        return regime_metrics

    def print_summary(self):
        """Print formatted performance summary"""
        metrics = self.calculate_metrics()

        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:       {metrics['total_trades']}")
        print(f"  Winners:            {metrics['winning_trades']} ({metrics['win_rate']:.1%})")
        print(f"  Losers:             {metrics['losing_trades']}")

        print(f"\nP&L Metrics:")
        print(f"  Total P&L:          ${metrics['total_pnl']:.2f}")
        print(f"  Average Win:        ${metrics['avg_win']:.2f}")
        print(f"  Average Loss:       ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
        print(f"  Expectancy:         ${metrics['expectancy']:.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:       ${metrics['max_drawdown']:.2f}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")

        print(f"\nTrade Duration:")
        print(f"  Avg Hold (Winners): {metrics['avg_hold_time_winner']:.1f} min")
        print(f"  Avg Hold (Losers):  {metrics['avg_hold_time_loser']:.1f} min")

        if 'regime_performance' in metrics:
            print(f"\nPerformance by Regime:")
            for regime, regime_metrics in metrics['regime_performance'].items():
                print(f"  {regime:15s}: {regime_metrics['trades']:3d} trades, "
                      f"{regime_metrics['win_rate']:.1%} win rate, "
                      f"${regime_metrics['total_pnl']:7.2f} P&L")

        print("="*60 + "\n")

    def get_equity_curve(self) -> List[float]:
        """
        Get equity curve

        Returns:
            List of cumulative P&L values
        """
        return self.equity_curve

    def export_trades_to_csv(self, filepath: str):
        """
        Export trades to CSV file

        Args:
            filepath: Path to save CSV
        """
        if len(self.trades) == 0:
            logger.warning("No trades to export")
            return

        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} trades to {filepath}")


class LivePerformanceTracker:
    """
    Track live performance metrics during trading session
    """

    def __init__(self):
        """Initialize live tracker"""
        self.session_start = datetime.now()
        self.metrics = PerformanceMetrics()
        self.open_positions: Dict[str, Dict] = {}

    def open_position(self, position_id: str, entry_time: datetime,
                     entry_price: float, direction: int, quantity: int,
                     signal_confidence: float, regime: str):
        """
        Record position opening

        Args:
            position_id: Unique position identifier
            entry_time: Entry timestamp
            entry_price: Entry price
            direction: 1 for long, -1 for short
            quantity: Position size
            signal_confidence: Model confidence
            regime: Market regime
        """
        self.open_positions[position_id] = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'direction': direction,
            'quantity': quantity,
            'signal_confidence': signal_confidence,
            'regime': regime
        }

        logger.info(f"Position opened: {position_id}")

    def close_position(self, position_id: str, exit_time: datetime,
                      exit_price: float, pnl: float):
        """
        Record position closing

        Args:
            position_id: Position identifier
            exit_time: Exit timestamp
            exit_price: Exit price
            pnl: Realized profit/loss
        """
        if position_id not in self.open_positions:
            logger.warning(f"Position {position_id} not found in open positions")
            return

        pos = self.open_positions.pop(position_id)

        self.metrics.add_trade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            direction=pos['direction'],
            quantity=pos['quantity'],
            pnl=pnl,
            signal_confidence=pos['signal_confidence'],
            regime=pos['regime']
        )

        logger.info(f"Position closed: {position_id}, P&L: ${pnl:.2f}")

    def get_session_metrics(self) -> Dict:
        """
        Get current session metrics

        Returns:
            Dictionary of session metrics
        """
        metrics = self.metrics.calculate_metrics()
        metrics['session_duration'] = (datetime.now() - self.session_start).total_seconds() / 3600
        metrics['open_positions'] = len(self.open_positions)

        return metrics
