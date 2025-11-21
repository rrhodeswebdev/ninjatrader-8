"""
RNN Strategy Adapter for backintime Framework

This module bridges the RNN trading model with the backintime backtesting framework.
It allows you to backtest RNN strategies using backintime's sophisticated order execution,
margin management, and session handling.

Usage:
    from backintime_rnn_adapter import create_rnn_strategy, run_rnn_backtest
    from model import TradingModel

    # Train your model
    model = TradingModel(sequence_length=40)
    model.train(training_data)

    # Run backtest using backintime
    results = run_rnn_backtest(
        model=model,
        data_file='path/to/data.csv',
        since=datetime(...),
        until=datetime(...)
    )
"""

import os
import sys
import typing as t
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import pandas as pd
import numpy as np

# Import backintime components
try:
    from backintime import run_backtest
    from backintime.session import FuturesSession
    from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
    from backintime.timeframes import Timeframes as tf
    from backintime.utils import PREFETCH_SINCE
    from backintime.strategy import FuturesStrategy
    from backintime.broker import OrderSide
    BACKINTIME_AVAILABLE = True
except ImportError:
    BACKINTIME_AVAILABLE = False
    # Create dummy base class if backintime not available
    class FuturesStrategy:
        """Dummy base class when backintime is not available"""
        pass
    print("⚠️  backintime not installed - adapter functionality limited")

from model import TradingModel


class RNNFuturesStrategy(FuturesStrategy):
    """
    Adapter strategy that uses RNN model predictions within backintime framework.

    This strategy operates in an imperative style, calling the RNN model's predict()
    method on each tick to generate trading signals.
    """

    def __init__(self, model: TradingModel, atr_multiplier: float = 2.0):
        """
        Initialize RNN strategy.

        Args:
            model: Trained TradingModel instance
            atr_multiplier: ATR multiplier for stop loss/take profit (default: 2.0)
        """
        super().__init__()
        self.model = model
        self.atr_multiplier = atr_multiplier
        self.historical_data = []
        self.min_bars_required = model.sequence_length + 50  # Extra for indicators

        # Risk parameters (ES futures)
        self.max_position_size = 1  # Conservative: 1 contract
        self.tick_size = 0.25
        self.point_value = 50.0

    @property
    def indicators(self):
        """Define indicators needed for RNN model"""
        # ATR is needed for dynamic stops
        return [
            ('atr', tf.M1, {'period': 14})
        ]

    @property
    def candle_timeframes(self):
        """Define candle timeframes to track"""
        return [tf.M1]

    def tick(self, candle):
        """
        Called on each new candle close.

        This is where we:
        1. Accumulate historical data
        2. Get RNN prediction
        3. Execute trades based on prediction
        """
        # Build historical DataFrame
        current_bar = {
            'time': candle.open_time,
            'open': float(candle.open),
            'high': float(candle.high),
            'low': float(candle.low),
            'close': float(candle.close),
            'volume': float(candle.volume) if candle.volume else 0.0
        }
        self.historical_data.append(current_bar)

        # Wait for enough data
        if len(self.historical_data) < self.min_bars_required:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        df['time'] = pd.to_datetime(df['time'])

        # Get RNN prediction
        try:
            signal, confidence = self.model.predict(df)
        except Exception as e:
            print(f"Prediction error: {e}")
            return

        # Get current ATR for risk management
        atr = self.get_indicator('atr', tf.M1)
        if atr is None or atr <= 0:
            atr = 15.0  # Default fallback

        current_price = float(candle.close)

        # Check if we have an open position
        has_position = len(self.broker.get_active_orders()) > 0

        # Only trade on strong signals with high confidence
        if signal == 'long' and confidence > 0.55 and not has_position:
            # Calculate stops
            stop_loss = current_price - (atr * self.atr_multiplier)
            take_profit = current_price + (atr * self.atr_multiplier * 1.5)  # 1.5:1 RR

            # Enter long position
            self.broker.buy(
                quantity=self.max_position_size,
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit))
            )

        elif signal == 'short' and confidence > 0.55 and not has_position:
            # Calculate stops
            stop_loss = current_price + (atr * self.atr_multiplier)
            take_profit = current_price - (atr * self.atr_multiplier * 1.5)  # 1.5:1 RR

            # Enter short position
            self.broker.sell(
                quantity=self.max_position_size,
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit))
            )


def convert_rnn_data_to_backintime_format(
    input_file: str,
    output_file: str,
    time_col: str = 'time',
    date_format: str = None
) -> str:
    """
    Convert RNN server data format to backintime CSV format.

    Args:
        input_file: Path to RNN format CSV (time,open,high,low,close,volume)
        output_file: Path to output backintime format CSV
        time_col: Name of the time column
        date_format: Optional date format string

    Returns:
        Path to converted file
    """
    print(f"\n📊 Converting data format...")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    # Read RNN format
    df = pd.read_csv(input_file)

    # Ensure datetime
    if date_format:
        df['time'] = pd.to_datetime(df['time'], format=date_format)
    else:
        df['time'] = pd.to_datetime(df['time'])

    # Add close_time (1 minute after open_time for 1-min bars)
    df['close_time'] = df['time'] + pd.Timedelta(minutes=1)

    # Backintime expects: open_time, open, high, low, close, volume, close_time
    df_backintime = df[['time', 'open', 'high', 'low', 'close', 'volume', 'close_time']].copy()

    # Format timestamps for backintime
    df_backintime['time'] = df_backintime['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_backintime['close_time'] = df_backintime['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save
    df_backintime.to_csv(output_file, index=False, header=False)

    print(f"  ✓ Converted {len(df_backintime)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    return output_file


def run_rnn_backtest(
    model: TradingModel,
    data_file: str,
    since: datetime,
    until: datetime,
    initial_capital: float = 10000.0,
    atr_multiplier: float = 2.0,
    session_start: timedelta = timedelta(hours=9, minutes=30),
    session_end: timedelta = timedelta(hours=16, minutes=0),
    session_timezone: str = 'America/New_York',
    results_dir: str = None
) -> t.Any:
    """
    Run RNN strategy backtest using backintime framework.

    Args:
        model: Trained TradingModel instance
        data_file: Path to backintime format CSV file
        since: Start datetime for backtest
        until: End datetime for backtest
        initial_capital: Starting capital (default: $10,000)
        atr_multiplier: ATR multiplier for stops (default: 2.0)
        session_start: Trading session start time
        session_end: Trading session end time
        session_timezone: Timezone for session
        results_dir: Directory to save results (optional)

    Returns:
        Backtest results object from backintime

    Raises:
        ImportError: If backintime framework is not installed
        ValueError: If model is not trained
    """
    if not BACKINTIME_AVAILABLE:
        raise ImportError(
            "backintime framework is not installed. "
            "This feature requires the backintime package which is not included. "
            "Use the simple RNN backtester instead (see run_backtest.py)"
        )

    if not model.is_trained:
        raise ValueError("Model must be trained before backtesting")

    print("\n" + "="*70)
    print("RNN STRATEGY BACKTEST (using backintime framework)")
    print("="*70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Sequence Length: {model.sequence_length}")
    print(f"Data File: {data_file}")
    print(f"Period: {since} to {until}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("="*70 + "\n")

    # Setup data feed
    schema = CSVCandlesSchema(
        open_time=0, open=1, high=2,
        low=3, close=4, volume=5, close_time=6
    )

    feed = CSVCandlesFactory(
        data_file, 'ESNQ',  # ES or NQ futures
        tf.M1,
        delimiter=',',
        schema=schema,
        timezone=session_timezone
    )

    # Setup trading session
    session = FuturesSession(
        session_start=session_start,
        session_end=session_end,
        session_timezone=session_timezone,
        overnight_start=session_end,
        overnight_end=session_start,
        overnight_timezone=session_timezone,
        non_working_weekdays={5, 6}  # Saturday, Sunday
    )

    # Create strategy instance
    def create_strategy():
        return RNNFuturesStrategy(model=model, atr_multiplier=atr_multiplier)

    # Run backtest
    result = run_backtest(
        create_strategy,
        feed,
        initial_capital,
        since,
        until,
        None,  # until_candle
        session,
        prefetch_option=PREFETCH_SINCE,
        additional_collateral=Decimal('1500'),
        check_margin_call=True,
        per_contract_init_margin=Decimal('1699.64'),
        per_contract_maintenance_margin=Decimal('1500'),
        per_contract_overnight_margin=Decimal('2200')
    )

    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(result)
    print("\nSTATISTICS:")
    print(result.get_stats())
    print("="*70 + "\n")

    # Export results
    if results_dir:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        result.export_stats(str(results_path))
        result.export_orders(str(results_path))
        result.export_trades(str(results_path))

        print(f"✓ Results exported to {results_path}")

    return result


def create_rnn_strategy(model: TradingModel, **kwargs) -> RNNFuturesStrategy:
    """
    Factory function to create RNN strategy instance.

    Args:
        model: Trained TradingModel
        **kwargs: Additional strategy parameters

    Returns:
        RNNFuturesStrategy instance
    """
    return RNNFuturesStrategy(model=model, **kwargs)


if __name__ == '__main__':
    print(__doc__)
    print("\nExample usage:")
    print("""
    # 1. Train RNN model
    from model import TradingModel
    model = TradingModel(sequence_length=40)
    model.train(training_data)

    # 2. Convert data to backintime format (if needed)
    convert_rnn_data_to_backintime_format(
        'historical_data.csv',
        'backintime_data.csv'
    )

    # 3. Run backtest
    results = run_rnn_backtest(
        model=model,
        data_file='backintime_data.csv',
        since=datetime(2024, 1, 1, 9, 30),
        until=datetime(2024, 3, 1, 16, 0),
        initial_capital=25000.0
    )
    """)
