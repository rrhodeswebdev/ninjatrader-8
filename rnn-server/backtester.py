"""
Event-Driven Backtesting Engine for Trading Strategies

Provides realistic backtesting with:
- Bar-by-bar execution
- Realistic fills (slippage, spread)
- Transaction costs
- Stop loss and take profit management
- Detailed trade-by-trade reporting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from risk_management import RiskManager
from trading_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_profit_factor,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_expectancy
)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    contracts: int
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop', 'target', 'signal', 'eod'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE

    def calculate_pnl(self):
        """Calculate P&L after exit"""
        if self.exit_price is None:
            return None

        if self.direction == 'long':
            gross_pnl = (self.exit_price - self.entry_price) * self.contracts * 50  # ES: $50/point
        else:  # short
            gross_pnl = (self.entry_price - self.exit_price) * self.contracts * 50

        # Subtract costs
        self.pnl = gross_pnl - self.commission - self.slippage
        self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price * 100 if self.direction == 'long' else (self.entry_price - self.exit_price) / self.entry_price * 100

        return self.pnl


@dataclass
class BacktestState:
    """Tracks current backtest state"""
    equity: float
    positions: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_goal: float = 500.0
    daily_max_loss: float = 250.0
    trades_today: int = 0
    max_trades_per_day: int = 10
    current_date: Optional[datetime] = None


class Backtester:
    """
    Event-driven backtesting engine
    """

    def __init__(
        self,
        initial_capital: float = 25000.0,
        commission_per_contract: float = 2.50,  # Round-trip commission
        slippage_ticks: int = 1,  # 1 tick slippage per side
        tick_value: float = 12.50,  # ES: $12.50 per tick
        daily_goal: float = 500.0,
        daily_max_loss: float = 250.0,
        max_trades_per_day: int = 10,
        risk_manager: Optional[RiskManager] = None
    ):
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.tick_value = tick_value
        self.daily_goal = daily_goal
        self.daily_max_loss = daily_max_loss
        self.max_trades_per_day = max_trades_per_day
        self.risk_manager = risk_manager or RiskManager()

        # State
        self.state = BacktestState(
            equity=initial_capital,
            daily_goal=daily_goal,
            daily_max_loss=daily_max_loss,
            max_trades_per_day=max_trades_per_day
        )

    def run(
        self,
        df: pd.DataFrame,
        model,
        df_secondary: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run backtest bar-by-bar

        Args:
            df: Primary timeframe DataFrame (1-min)
            model: Trained TradingModel instance
            df_secondary: Optional secondary timeframe (5-min)
            verbose: Print progress

        Returns:
            Dictionary with results and metrics
        """

        # Ensure datetime index
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])

        # Initialize equity curve
        self.state.equity_curve = [self.initial_capital]

        # Track predictions for analysis
        predictions_log = []

        print(f"\n{'='*60}")
        print(f"BACKTESTING: {len(df)} bars")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Commission: ${self.commission_per_contract}/contract round-trip")
        print(f"Slippage: {self.slippage_ticks} tick(s) per side")
        print(f"Daily Goal: ${self.daily_goal:.2f} | Max Loss: ${self.daily_max_loss:.2f}")
        print(f"{'='*60}\n")

        # Iterate through each bar
        for i in range(model.sequence_length, len(df)):
            current_bar = df.iloc[i]
            current_time = current_bar['time']

            # Check for new trading day
            if self.state.current_date is None or current_time.date() != self.state.current_date.date():
                self._reset_daily_limits(current_time)

            # Update existing positions (check stops/targets)
            self._update_positions(current_bar)

            # Check if we can trade (daily limits)
            if not self._can_trade():
                continue

            # Get prediction from model (only if no position)
            if len(self.state.positions) == 0:
                # Prepare data up to current bar
                historical_data = df.iloc[:i+1].copy()

                # Get prediction
                signal, confidence = model.predict(historical_data)

                # Get current ATR and regime for risk management
                atr = historical_data['atr'].iloc[-1] if 'atr' in historical_data.columns else 15.0

                # Detect regime
                from model import detect_market_regime
                regime = detect_market_regime(historical_data, lookback=min(100, len(historical_data)-1))

                # Log prediction
                predictions_log.append({
                    'time': current_time,
                    'signal': signal,
                    'confidence': confidence,
                    'regime': regime,
                    'close': current_bar['close']
                })

                # Calculate trade parameters
                if signal in ['long', 'short']:
                    # Entry price with slippage
                    entry_price = current_bar['close']
                    if signal == 'long':
                        entry_price += self.slippage_ticks * 0.25  # ES tick size: 0.25
                    else:
                        entry_price -= self.slippage_ticks * 0.25

                    # Get risk management parameters
                    trade_params = self.risk_manager.calculate_trade_parameters(
                        signal=signal,
                        confidence=confidence,
                        entry_price=entry_price,
                        atr=atr,
                        regime=regime,
                        account_balance=self.state.equity,
                        tick_value=self.tick_value
                    )

                    # Execute trade if approved
                    if trade_params['contracts'] > 0:
                        self._enter_position(
                            current_time,
                            trade_params['entry_price'],
                            trade_params['signal'],
                            trade_params['contracts'],
                            trade_params['stop_loss'],
                            trade_params['take_profit']
                        )

                        if verbose and len(self.state.closed_trades) % 10 == 0:
                            print(f"[{current_time}] {signal.upper()} @ ${entry_price:.2f} | "
                                  f"Contracts: {trade_params['contracts']} | "
                                  f"Stop: ${trade_params['stop_loss']:.2f} | "
                                  f"Target: ${trade_params['take_profit']:.2f} | "
                                  f"Conf: {confidence:.2%}")

            # Update equity curve
            current_equity = self._calculate_equity(current_bar['close'])
            self.state.equity_curve.append(current_equity)

        # Close any remaining positions at end
        if len(self.state.positions) > 0:
            final_bar = df.iloc[-1]
            for position in self.state.positions[:]:
                self._exit_position(
                    position,
                    final_bar['time'],
                    final_bar['close'],
                    'eod'
                )

        # Calculate final metrics
        results = self._calculate_metrics(predictions_log)

        if verbose:
            self._print_results(results)

        return results

    def _enter_position(
        self,
        entry_time: datetime,
        entry_price: float,
        direction: str,
        contracts: int,
        stop_loss: float,
        take_profit: float
    ):
        """Enter a new position"""

        # Calculate commission and slippage
        commission = contracts * self.commission_per_contract
        slippage = contracts * self.slippage_ticks * self.tick_value

        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            contracts=contracts,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
            slippage=slippage
        )

        self.state.positions.append(trade)
        self.state.trades_today += 1

    def _exit_position(
        self,
        position: Trade,
        exit_time: datetime,
        exit_price: float,
        reason: str
    ):
        """Exit an existing position"""

        # Add exit slippage
        if position.direction == 'long':
            exit_price -= self.slippage_ticks * 0.25
        else:
            exit_price += self.slippage_ticks * 0.25

        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = reason
        position.bars_held = (exit_time - position.entry_time).total_seconds() / 60  # Assuming 1-min bars

        # Calculate P&L
        position.calculate_pnl()

        # Update daily P&L
        self.state.daily_pnl += position.pnl

        # Update equity
        self.state.equity += position.pnl

        # Move to closed trades
        self.state.closed_trades.append(position)
        self.state.positions.remove(position)

    def _update_positions(self, current_bar: pd.Series):
        """Check and update all open positions"""

        for position in self.state.positions[:]:  # Copy list to allow removal
            # Update MFE and MAE
            if position.direction == 'long':
                favorable = current_bar['high'] - position.entry_price
                adverse = position.entry_price - current_bar['low']
            else:
                favorable = position.entry_price - current_bar['low']
                adverse = current_bar['high'] - position.entry_price

            position.max_favorable_excursion = max(position.max_favorable_excursion, favorable)
            position.max_adverse_excursion = max(position.max_adverse_excursion, adverse)

            # Check stop loss
            if position.direction == 'long':
                if current_bar['low'] <= position.stop_loss:
                    self._exit_position(position, current_bar['time'], position.stop_loss, 'stop')
                    continue
            else:  # short
                if current_bar['high'] >= position.stop_loss:
                    self._exit_position(position, current_bar['time'], position.stop_loss, 'stop')
                    continue

            # Check take profit
            if position.direction == 'long':
                if current_bar['high'] >= position.take_profit:
                    self._exit_position(position, current_bar['time'], position.take_profit, 'target')
                    continue
            else:  # short
                if current_bar['low'] <= position.take_profit:
                    self._exit_position(position, current_bar['time'], position.take_profit, 'target')
                    continue

    def _can_trade(self) -> bool:
        """Check if we can enter new trades based on daily limits"""

        # Check daily goal
        if self.state.daily_pnl >= self.state.daily_goal:
            return False

        # Check daily max loss
        if self.state.daily_pnl <= -self.state.daily_max_loss:
            return False

        # Check max trades per day
        if self.state.trades_today >= self.state.max_trades_per_day:
            return False

        return True

    def _reset_daily_limits(self, current_time: datetime):
        """Reset daily P&L limits for new trading day"""
        self.state.current_date = current_time
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions"""
        equity = self.state.equity

        for position in self.state.positions:
            # Mark to market
            if position.direction == 'long':
                unrealized_pnl = (current_price - position.entry_price) * position.contracts * 50
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.contracts * 50

            equity += unrealized_pnl

        return equity

    def _calculate_metrics(self, predictions_log: List[Dict]) -> Dict:
        """Calculate comprehensive backtest metrics"""

        if len(self.state.closed_trades) == 0:
            return {
                'total_trades': 0,
                'message': 'No trades executed'
            }

        # Convert trades to DataFrame
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'contracts': t.contracts,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'bars_held': t.bars_held,
                'mfe': t.max_favorable_excursion,
                'mae': t.max_adverse_excursion
            }
            for t in self.state.closed_trades
        ])

        # Calculate returns
        returns = trades_df['pnl'].values
        returns_pct = trades_df['pnl_pct'].values / 100

        # Equity curve
        equity_curve = np.array(self.state.equity_curve)

        # Calculate metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        metrics = {
            # Trade statistics
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'win_rate': calculate_win_rate(returns),

            # P&L metrics
            'total_pnl': returns.sum(),
            'total_return_pct': total_return * 100,
            'avg_trade_pnl': returns.mean(),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),

            # Risk-adjusted metrics
            'sharpe_ratio': calculate_sharpe_ratio(returns_pct, periods_per_year=252*390),  # 1-min bars
            'sortino_ratio': calculate_sortino_ratio(returns_pct, periods_per_year=252*390),
            'profit_factor': calculate_profit_factor(returns),
            'expectancy': calculate_expectancy(returns),
            'max_drawdown': calculate_max_drawdown(equity_curve - self.initial_capital) * 100,

            # Exit analysis
            'exits_at_stop': len(trades_df[trades_df['exit_reason'] == 'stop']),
            'exits_at_target': len(trades_df[trades_df['exit_reason'] == 'target']),
            'exits_other': len(trades_df[~trades_df['exit_reason'].isin(['stop', 'target'])]),

            # Holding time
            'avg_bars_held': trades_df['bars_held'].mean(),
            'avg_win_bars': trades_df[trades_df['pnl'] > 0]['bars_held'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'avg_loss_bars': trades_df[trades_df['pnl'] < 0]['bars_held'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,

            # Efficiency
            'avg_mfe': trades_df['mfe'].mean(),
            'avg_mae': trades_df['mae'].mean(),

            # Final equity
            'final_equity': equity_curve[-1],
            'peak_equity': equity_curve.max(),

            # Raw data
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'predictions_log': predictions_log
        }

        return metrics

    def _print_results(self, results: Dict):
        """Print formatted backtest results"""

        if results.get('total_trades', 0) == 0:
            print("No trades executed during backtest period.")
            return

        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")

        print(f"\n📊 TRADE STATISTICS")
        print(f"  Total Trades:        {results['total_trades']:>8d}")
        print(f"  Winning Trades:      {results['winning_trades']:>8d} ({results['win_rate']*100:>5.1f}%)")
        print(f"  Losing Trades:       {results['losing_trades']:>8d}")

        print(f"\n💰 P&L METRICS")
        print(f"  Total P&L:           ${results['total_pnl']:>8,.2f}")
        print(f"  Total Return:        {results['total_return_pct']:>8.2f}%")
        print(f"  Avg Trade P&L:       ${results['avg_trade_pnl']:>8,.2f}")
        print(f"  Avg Win:             ${results['avg_win']:>8,.2f}")
        print(f"  Avg Loss:            ${results['avg_loss']:>8,.2f}")
        print(f"  Largest Win:         ${results['largest_win']:>8,.2f}")
        print(f"  Largest Loss:        ${results['largest_loss']:>8,.2f}")

        print(f"\n📈 RISK-ADJUSTED METRICS")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:       {results['sortino_ratio']:>8.2f}")
        print(f"  Profit Factor:       {results['profit_factor']:>8.2f}")
        print(f"  Expectancy:          ${results['expectancy']:>8,.2f}")
        print(f"  Max Drawdown:        {results['max_drawdown']:>8.2f}%")

        print(f"\n🎯 EXIT ANALYSIS")
        print(f"  Exits at Stop:       {results['exits_at_stop']:>8d} ({results['exits_at_stop']/results['total_trades']*100:>5.1f}%)")
        print(f"  Exits at Target:     {results['exits_at_target']:>8d} ({results['exits_at_target']/results['total_trades']*100:>5.1f}%)")
        print(f"  Other Exits:         {results['exits_other']:>8d}")

        print(f"\n⏱️  HOLDING TIME (minutes)")
        print(f"  Avg Bars Held:       {results['avg_bars_held']:>8.1f}")
        print(f"  Avg Win Duration:    {results['avg_win_bars']:>8.1f}")
        print(f"  Avg Loss Duration:   {results['avg_loss_bars']:>8.1f}")

        print(f"\n🎲 EFFICIENCY")
        print(f"  Avg MFE (points):    {results['avg_mfe']:>8.2f}")
        print(f"  Avg MAE (points):    {results['avg_mae']:>8.2f}")

        print(f"\n💵 EQUITY")
        print(f"  Initial:             ${self.initial_capital:>8,.2f}")
        print(f"  Final:               ${results['final_equity']:>8,.2f}")
        print(f"  Peak:                ${results['peak_equity']:>8,.2f}")

        print(f"\n{'='*60}\n")


# Example usage
if __name__ == '__main__':
    print("Backtester module loaded. Use in conjunction with trained model.")
    print("\nExample usage:")
    print("""
    from backtester import Backtester
    from model import TradingModel
    import pandas as pd

    # Load data
    df = pd.read_csv('historical_data.csv')

    # Train model
    model = TradingModel(sequence_length=20)
    model.train(df.iloc[:2000])  # Train on first 2000 bars

    # Backtest on remaining data
    backtester = Backtester(initial_capital=25000)
    results = backtester.run(df.iloc[2000:], model)
    """)
