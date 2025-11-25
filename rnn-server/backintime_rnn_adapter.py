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
    from backintime import FuturesStrategy
    from backintime.broker import (
        OrderSide, LimitOrderOptions, MarketOrderOptions, TakeProfitOptions, StopLossOptions
    )
    from backintime.indicators import ATR
    BACKINTIME_AVAILABLE = True
except ImportError:
    BACKINTIME_AVAILABLE = False
    # Create dummy base class if backintime not available
    class FuturesStrategy:
        """Dummy base class when backintime is not available"""
        pass
    print("WARNING: backintime not installed - adapter functionality limited")

from model import TradingModel


class RNNFuturesStrategy(FuturesStrategy):
    """
    Adapter strategy that uses RNN model predictions within backintime framework.

    This strategy operates in an imperative style, calling the RNN model's predict()
    method on each tick to generate trading signals.
    """

    # Define as class attributes (not properties)
    candle_timeframes = {tf.M1}
    indicators = {
        ATR(tf.M1, period=14)  # ATR for dynamic stops
    }

    def __init__(self, model: TradingModel, atr_multiplier: float = 2.0,
                 broker_proxy=None, analyser=None, candles=None):
        """
        Initialize RNN strategy.

        Args:
            model: Trained TradingModel instance
            atr_multiplier: ATR multiplier for stop loss/take profit (default: 2.0)
            broker_proxy: backintime broker proxy (passed by framework)
            analyser: backintime analyser (passed by framework)
            candles: backintime candles buffer (passed by framework)
        """
        self.model = model
        self.atr_multiplier = atr_multiplier
        self.historical_data = []
        self.min_bars_required = model.sequence_length + 50  # Extra for indicators

        # Risk parameters (ES futures)
        self.max_position_size = 1  # Conservative: 1 contract
        self.tick_size = 0.25
        self.point_value = 50.0

        # Call parent init with backintime's expected arguments
        if broker_proxy is not None and analyser is not None and candles is not None:
            super().__init__(broker_proxy, analyser, candles)
        else:
            super().__init__()

    def tick(self):
        """
        Called on each new candle close.

        This is where we:
        1. Accumulate historical data
        2. Get RNN prediction
        3. Execute trades based on prediction
        """
        # Get current candle from backintime
        candle = self.candles.get(tf.M1)

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

        # Get current ATR for risk management using backintime's analyser
        atr_values = self.analyser.atr(tf.M1, period=14)
        if atr_values is None or len(atr_values) == 0:
            atr = 15.0  # Default fallback
        else:
            atr = float(atr_values[-1])

        current_price = float(candle.close)

        # Check if we have funds to trade
        if not self.broker.max_funds_for_futures:
            return

        # Conservative trading: Only trade when flat (no positions)
        # This avoids capital conflicts and position management complexities
        is_flat = not self.broker.in_long and not self.broker.in_short

        if signal == 'long' and confidence > 0.55 and is_flat:
            # Enter long position only when completely flat
            order = LimitOrderOptions(
                order_side=OrderSide.BUY,
                order_price=Decimal(str(current_price)),
                percentage_amount=Decimal('15')  # ~15% of capital = 1-2 contracts with margin
            )
            self.broker.submit_limit_order(order)

        elif signal == 'short' and confidence > 0.55 and is_flat:
            # Enter short position only when completely flat
            order = LimitOrderOptions(
                order_side=OrderSide.SELL,
                order_price=Decimal(str(current_price)),
                percentage_amount=Decimal('15')  # ~15% of capital = 1-2 contracts with margin
            )
            self.broker.submit_limit_order(order)


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
    print(f"\n Converting data format...")
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

    print(f"   Converted {len(df_backintime)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    return output_file


def _parse_date_local_tz(date_str: str, timezone: str = 'UTC') -> datetime:
    """
    Parse date string assuming it's already in the target timezone.

    This differs from backintime's default _parse_date which assumes dates are UTC.
    Our CSV dates are already in local time (e.g., America/New_York).

    Returns timezone-naive datetime to match backintime's internal expectations.
    """
    # Parse as timezone-naive and return as-is
    dt = pd.to_datetime(date_str)
    return dt.to_pydatetime()


def _parse_date_utc_to_local(date_str: str, timezone: str = 'UTC') -> datetime:
    """
    Parse date string assuming it's in UTC, then convert to target timezone.

    This is backintime's default behavior - assumes CSV timestamps are UTC
    and converts them to the specified local timezone.

    Args:
        date_str: Timestamp string to parse
        timezone: Target timezone to convert to (e.g., 'America/New_York')

    Returns:
        Timezone-naive datetime in the target timezone
    """
    import pytz

    # Parse as UTC
    dt = pd.to_datetime(date_str)

    # Localize as UTC if not already timezone-aware
    if dt.tzinfo is None:
        dt = dt.tz_localize('UTC')

    # Convert to target timezone
    target_tz = pytz.timezone(timezone)
    dt_local = dt.tz_convert(target_tz)

    # Return timezone-naive (backintime expects naive datetimes)
    return dt_local.tz_localize(None).to_pydatetime()


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
    results_dir: str = None,
    assume_utc: bool = False
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
        session_start: Trading session start time (default: 9:30 AM for stocks)
        session_end: Trading session end time (default: 4:00 PM for stocks)
        session_timezone: Timezone for session (default: America/New_York)
        results_dir: Directory to save results (optional)
        assume_utc: If True, CSV timestamps are treated as UTC and converted to session_timezone.
                    If False, timestamps are assumed to already be in session_timezone (default: False)

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

    # Select appropriate date parser based on timestamp format
    if assume_utc:
        # CSV timestamps are UTC - convert to session timezone
        date_parser = _parse_date_utc_to_local
        print(f"Timestamp mode: UTC -> {session_timezone} conversion")
    else:
        # CSV timestamps are already in session timezone
        date_parser = _parse_date_local_tz
        print(f"Timestamp mode: Local ({session_timezone}) - no conversion")

    feed = CSVCandlesFactory(
        data_file, 'ESNQ',  # ES or NQ futures
        tf.M1,
        delimiter=',',
        schema=schema,
        timezone=session_timezone,
        date_parser=date_parser
    )

    # Setup trading session (Stock market hours: Mon-Fri 9:30 AM - 4:00 PM ET)
    session = FuturesSession(
        session_start=session_start,
        session_end=session_end,
        session_timezone=session_timezone,
        overnight_start=session_end,  # 4:00 PM
        overnight_end=session_start,  # 9:30 AM
        overnight_timezone=session_timezone,
        non_working_weekdays={5, 6}  # Saturday, Sunday
    )

    # Create strategy class with parameters baked in
    class ParameterizedRNNStrategy(RNNFuturesStrategy):
        def __init__(self, broker_proxy, analyser, candles):
            super().__init__(model=model, atr_multiplier=atr_multiplier,
                           broker_proxy=broker_proxy, analyser=analyser, candles=candles)

    # Run backtest
    result = run_backtest(
        ParameterizedRNNStrategy,
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

        try:
            result.export_stats(str(results_path))
        except (AttributeError, TypeError) as e:
            # Handle case where there are no completed trades
            print(f"  WARNING: Could not export stats (possibly no completed trades): {e}")

        result.export_orders(str(results_path))
        result.export_trades(str(results_path))

        print(f" Results exported to {results_path}")

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

    # 3. Run backtest with local timestamps (default)
    results = run_rnn_backtest(
        model=model,
        data_file='backintime_data.csv',
        since=datetime(2024, 1, 1, 9, 30),
        until=datetime(2024, 3, 1, 16, 0),
        initial_capital=25000.0,
        assume_utc=False  # Timestamps are already in America/New_York
    )

    # Or with UTC timestamps (converts to session timezone)
    results = run_rnn_backtest(
        model=model,
        data_file='backintime_data_utc.csv',
        since=datetime(2024, 1, 1, 14, 30),  # 9:30 AM ET in UTC
        until=datetime(2024, 3, 1, 21, 0),   # 4:00 PM ET in UTC
        initial_capital=25000.0,
        session_timezone='America/New_York',
        assume_utc=True  # Timestamps are UTC, convert to ET
    )
    """)
