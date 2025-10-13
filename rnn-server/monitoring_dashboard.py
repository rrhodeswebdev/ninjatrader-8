"""
Production Monitoring Dashboard
================================

Real-time tracking of critical trading metrics to detect issues early.

RED FLAGS (Stop Trading):
- 5 consecutive losses
- Sharpe < 0.5 over 100 trades
- Drawdown > 20%
- Win rate < 45% for high confidence trades (>0.7)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path


class TradingMonitor:
    """
    Real-time monitoring system for production trading
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_file: str = 'logs/trading_alerts.json'
    ):
        """
        Args:
            window_size: Number of recent trades to track
            alert_file: File to log alerts
        """
        self.window_size = window_size

        # Trade tracking
        self.trades = deque(maxlen=window_size)
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # High confidence trade tracking (>0.7)
        self.high_conf_trades = deque(maxlen=window_size)

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = None

        # Equity tracking
        self.equity_curve = []
        self.initial_equity = None

        # Alert system
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        self.alerts = []

        # Red flag thresholds
        self.thresholds = {
            'max_consecutive_losses': 5,
            'min_sharpe': 0.5,
            'max_drawdown': 0.20,  # 20%
            'min_high_conf_win_rate': 0.45  # 45%
        }

    def add_trade(
        self,
        timestamp: datetime,
        signal: str,
        confidence: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        contracts: int = 1
    ):
        """
        Record a completed trade

        Args:
            timestamp: Trade exit timestamp
            signal: 'long' or 'short'
            confidence: Model confidence (0-1)
            entry_price: Entry price
            exit_price: Exit price
            pnl: P&L in dollars
            contracts: Number of contracts
        """
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'contracts': contracts,
            'return': pnl / (entry_price * contracts * 50) if entry_price > 0 else 0  # ES: $50/point
        }

        self.trades.append(trade)

        # Track high confidence trades
        if confidence >= 0.7:
            self.high_conf_trades.append(trade)

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0

        # Update daily tracking
        if self.current_date is None or timestamp.date() != self.current_date:
            self.current_date = timestamp.date()
            self.daily_pnl = 0.0
            self.daily_trades = 0

        self.daily_pnl += pnl
        self.daily_trades += 1

        # Update equity curve
        if self.initial_equity is None:
            self.initial_equity = 25000.0  # Default initial equity

        current_equity = self.initial_equity + sum([t['pnl'] for t in self.trades])
        self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

        # Check for red flags
        self._check_alerts()

    def _check_alerts(self):
        """Check for red flag conditions"""
        if len(self.trades) < 10:
            return  # Need minimum trades

        alerts_triggered = []

        # 1. Consecutive losses
        if self.consecutive_losses >= self.thresholds['max_consecutive_losses']:
            alerts_triggered.append({
                'type': 'CRITICAL',
                'metric': 'consecutive_losses',
                'value': self.consecutive_losses,
                'threshold': self.thresholds['max_consecutive_losses'],
                'message': f"🔴 {self.consecutive_losses} consecutive losses - STOP TRADING",
                'timestamp': datetime.now().isoformat()
            })

        # 2. Sharpe ratio (if enough trades)
        if len(self.trades) >= 50:
            returns = np.array([t['return'] for t in self.trades])
            sharpe = self._calculate_sharpe(returns)

            if sharpe < self.thresholds['min_sharpe']:
                alerts_triggered.append({
                    'type': 'WARNING',
                    'metric': 'sharpe_ratio',
                    'value': sharpe,
                    'threshold': self.thresholds['min_sharpe'],
                    'message': f"⚠️  Sharpe ratio {sharpe:.2f} < {self.thresholds['min_sharpe']} - Poor risk-adjusted returns",
                    'timestamp': datetime.now().isoformat()
                })

        # 3. Drawdown
        if len(self.equity_curve) >= 10:
            drawdown = self._calculate_current_drawdown()

            if drawdown < -self.thresholds['max_drawdown']:
                alerts_triggered.append({
                    'type': 'CRITICAL',
                    'metric': 'drawdown',
                    'value': drawdown,
                    'threshold': -self.thresholds['max_drawdown'],
                    'message': f"🔴 Drawdown {drawdown*100:.1f}% > {self.thresholds['max_drawdown']*100:.0f}% - STOP TRADING",
                    'timestamp': datetime.now().isoformat()
                })

        # 4. High confidence win rate
        if len(self.high_conf_trades) >= 20:
            high_conf_wins = sum([1 for t in self.high_conf_trades if t['pnl'] > 0])
            high_conf_win_rate = high_conf_wins / len(self.high_conf_trades)

            if high_conf_win_rate < self.thresholds['min_high_conf_win_rate']:
                alerts_triggered.append({
                    'type': 'CRITICAL',
                    'metric': 'high_conf_win_rate',
                    'value': high_conf_win_rate,
                    'threshold': self.thresholds['min_high_conf_win_rate'],
                    'message': f"🔴 High confidence win rate {high_conf_win_rate:.1%} < {self.thresholds['min_high_conf_win_rate']:.0%} - Model broken",
                    'timestamp': datetime.now().isoformat()
                })

        # Log alerts
        if alerts_triggered:
            self.alerts.extend(alerts_triggered)
            self._save_alerts(alerts_triggered)

            # Print to console
            for alert in alerts_triggered:
                print(f"\n{alert['message']}\n")

    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252*390) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return sharpe

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if len(self.equity_curve) == 0:
            return 0.0

        equities = [e['equity'] for e in self.equity_curve]
        peak = max(equities)
        current = equities[-1]

        drawdown = (current - peak) / peak
        return drawdown

    def _save_alerts(self, alerts: List[Dict]):
        """Save alerts to file"""
        # Load existing alerts
        existing_alerts = []
        if self.alert_file.exists():
            with open(self.alert_file, 'r') as f:
                existing_alerts = json.load(f)

        # Append new alerts
        existing_alerts.extend(alerts)

        # Save
        with open(self.alert_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)

    def get_dashboard_metrics(self) -> Dict:
        """
        Get current dashboard metrics

        Returns:
            Dictionary with all key metrics
        """
        if len(self.trades) == 0:
            return {'error': 'No trades yet'}

        # Recent trades
        recent_trades = list(self.trades)
        returns = np.array([t['return'] for t in recent_trades])
        pnls = np.array([t['pnl'] for t in recent_trades])

        # Win rate
        wins = sum([1 for t in recent_trades if t['pnl'] > 0])
        win_rate = wins / len(recent_trades)

        # Sharpe ratio
        sharpe = self._calculate_sharpe(returns) if len(returns) >= 10 else 0.0

        # Profit factor
        gross_profit = sum([t['pnl'] for t in recent_trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in recent_trades if t['pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99

        # Expectancy
        expectancy = np.mean(pnls)

        # Current drawdown
        current_dd = self._calculate_current_drawdown()

        # Max drawdown
        if len(self.equity_curve) >= 2:
            equities = np.array([e['equity'] for e in self.equity_curve])
            running_max = np.maximum.accumulate(equities)
            drawdowns = (equities - running_max) / running_max
            max_dd = np.min(drawdowns)
        else:
            max_dd = 0.0

        # High confidence metrics
        if len(self.high_conf_trades) >= 5:
            high_conf_wins = sum([1 for t in self.high_conf_trades if t['pnl'] > 0])
            high_conf_win_rate = high_conf_wins / len(self.high_conf_trades)
            high_conf_pnl = sum([t['pnl'] for t in self.high_conf_trades])
        else:
            high_conf_win_rate = 0.0
            high_conf_pnl = 0.0

        # Daily metrics
        daily_return = self.daily_pnl / self.initial_equity if self.initial_equity else 0.0

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.trades),
            'total_pnl': sum([t['pnl'] for t in recent_trades]),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'high_conf_trades': len(self.high_conf_trades),
            'high_conf_win_rate': high_conf_win_rate,
            'high_conf_pnl': high_conf_pnl,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'daily_return_pct': daily_return * 100,
            'current_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_equity,
            'alerts_count': len(self.alerts)
        }

        # Add status flags
        metrics['status'] = self._get_status(metrics)

        return metrics

    def _get_status(self, metrics: Dict) -> str:
        """Determine overall system status"""
        # Critical checks
        if metrics['consecutive_losses'] >= self.thresholds['max_consecutive_losses']:
            return '🔴 CRITICAL - STOP TRADING'
        if metrics['current_drawdown'] < -self.thresholds['max_drawdown']:
            return '🔴 CRITICAL - STOP TRADING'
        if metrics['high_conf_trades'] >= 20 and metrics['high_conf_win_rate'] < self.thresholds['min_high_conf_win_rate']:
            return '🔴 CRITICAL - MODEL BROKEN'

        # Warning checks
        if metrics['total_trades'] >= 50 and metrics['sharpe_ratio'] < self.thresholds['min_sharpe']:
            return '⚠️  WARNING - Poor Performance'
        if metrics['consecutive_losses'] >= 3:
            return '⚠️  WARNING - Losing Streak'
        if metrics['current_drawdown'] < -0.10:
            return '⚠️  WARNING - Significant Drawdown'

        # All good
        return '✅ HEALTHY'

    def print_dashboard(self):
        """Print formatted dashboard"""
        metrics = self.get_dashboard_metrics()

        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return

        print("\n" + "="*70)
        print("TRADING MONITOR DASHBOARD")
        print("="*70)
        print(f"Status: {metrics['status']}")
        print(f"Last Update: {metrics['timestamp']}")

        print(f"\n📊 TRADE STATISTICS (Last {metrics['total_trades']} trades):")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"  Expectancy: ${metrics['expectancy']:.2f}")

        print(f"\n📈 RISK-ADJUSTED METRICS:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (target: > {self.thresholds['min_sharpe']:.1f})")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Current Drawdown: {metrics['current_drawdown']*100:.1f}% (limit: {self.thresholds['max_drawdown']*100:.0f}%)")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")

        print(f"\n🎯 CONFIDENCE METRICS:")
        print(f"  High Conf Trades (>70%): {metrics['high_conf_trades']}")
        print(f"  High Conf Win Rate: {metrics['high_conf_win_rate']:.1%} (target: > {self.thresholds['min_high_conf_win_rate']:.0%})")
        print(f"  High Conf P&L: ${metrics['high_conf_pnl']:,.2f}")

        print(f"\n⚠️  RISK INDICATORS:")
        print(f"  Consecutive Losses: {metrics['consecutive_losses']} (limit: {self.thresholds['max_consecutive_losses']})")
        print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")

        print(f"\n📅 TODAY:")
        print(f"  Trades: {metrics['daily_trades']}")
        print(f"  P&L: ${metrics['daily_pnl']:,.2f} ({metrics['daily_return_pct']:+.2f}%)")

        print(f"\n💰 EQUITY:")
        print(f"  Current: ${metrics['current_equity']:,.2f}")

        if metrics['alerts_count'] > 0:
            print(f"\n🚨 ALERTS: {metrics['alerts_count']} total")
            recent_alerts = self.alerts[-5:] if len(self.alerts) > 5 else self.alerts
            for alert in recent_alerts:
                print(f"  [{alert['type']}] {alert['message']}")

        print("\n" + "="*70 + "\n")

    def reset_daily_metrics(self):
        """Reset daily counters (call at start of each day)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_date = datetime.now().date()


# Example usage
if __name__ == '__main__':
    print("Production Monitoring Dashboard")
    print("="*50)
    print("\nUsage:")
    print("""
    from monitoring_dashboard import TradingMonitor

    # Initialize monitor
    monitor = TradingMonitor(window_size=100)

    # After each trade, log it
    monitor.add_trade(
        timestamp=datetime.now(),
        signal='long',
        confidence=0.75,
        entry_price=4500.00,
        exit_price=4505.00,
        pnl=250.00,  # $250 profit
        contracts=1
    )

    # Print dashboard
    monitor.print_dashboard()

    # Get metrics programmatically
    metrics = monitor.get_dashboard_metrics()
    if metrics['status'].startswith('🔴'):
        print("CRITICAL: Stop trading immediately!")
    """)
