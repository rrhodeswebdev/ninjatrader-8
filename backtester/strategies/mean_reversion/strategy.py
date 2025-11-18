import os
import typing as t
from datetime import datetime, timedelta
from decimal import Decimal

from backintime import run_backtest
from backintime.session import FuturesSession
from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
from backintime.analyser.indicators.constants import OPEN, CandleProperties
from backintime.timeframes import Timeframes as tf
from backintime.utils import PREFETCH_SINCE
from backintime.declarative import (scalar, prices, ATR, KELTNER, 
                    AtrExpression, PriceExpression, BooleanExpression, 
                    DeclarativeStrategy, TP, SL)
from datetime import datetime, timedelta

import pytz

from backintime.session import FuturesSession


high:   PriceExpression = prices.high(tf.M1)
low:    PriceExpression = prices.low(tf.M1)
close:  PriceExpression = prices.close(tf.M1)

atr:    AtrExpression = ATR(tf.M5, period=5)[-1]


def buy_signal(source: CandleProperties = OPEN,
               ema_period: int = 5,
               atr_period: int = 5, 
               multiplier: t.Union[int, float] = 1.5) -> BooleanExpression:
    """
    Return an expression that is True if buy conditions are met, False otherwise.
    Buy/Long when the price touches or crosses the lower KC.
    """
    kc = KELTNER(tf.M5, source, period=ema_period,
                 atr_period=atr_period, multiplier=multiplier)
    kc_lower_band = kc[-1].lower_band   
    return (high >= kc_lower_band) & (low < kc_lower_band)


def sell_signal(source: CandleProperties = OPEN,
                ema_period: int = 5,
                atr_period: int = 5, 
                multiplier: t.Union[int, float] = 1.5) -> BooleanExpression:
    """
    Return an expression that is True if sell conditions are met, False otherwise.
    Sell/Short when the price touches or crosses the upper KC.
    """
    kc = KELTNER(tf.M5, source, period=ema_period,
                 atr_period=atr_period, multiplier=multiplier)
    kc_upper_band = kc[-1].upper_band   
    return (high >= kc_upper_band) & (low < kc_upper_band)


def long_exit(tp_ratio, sl_ratio) -> t.Tuple[t.Union[TP, SL]]:
    """
    Return a sequence of TP/SL to close Long:
        - TP at the current price + ATR*tp_ratio
        - SL at the current price - ATR*sl_ratio
    """
    atr: AtrExpression = ATR(tf.M5, period=5)[-1]
    return (
        TP(trigger_price=close + atr*scalar(tp_ratio)), 
        SL(trigger_price=close - atr*scalar(sl_ratio))
    )


def short_exit(tp_ratio, sl_ratio) -> t.Tuple[t.Union[TP, SL]]:
    """
    Return a sequence of TP/SL to close Short:
        - TP at the current price - ATR*tp_ratio
        - SL at the current price + ATR*sl_ratio
    """
    atr: AtrExpression = ATR(tf.M5, period=5)[-1]
    return (
        TP(trigger_price=close - atr*scalar(tp_ratio)), 
        SL(trigger_price=close + atr*scalar(sl_ratio))
    )

# Default implementation
class MeanReversion(DeclarativeStrategy):
    title = "Mean Reversion"
    buy_signal = buy_signal()
    sell_signal = sell_signal()

    long_entry = close      # Limit Price
    long_exit = (
            TP(trigger_price=close + atr*scalar(2)), 
            SL(trigger_price=close - atr*scalar(1.5))
    )

    short_entry = close     # Limit price
    short_exit = (
            TP(trigger_price=close - atr*scalar(2)), 
            SL(trigger_price=close + atr*scalar(1.5))
    )


def run_with_params(since: datetime, 
                    until: datetime,
                    ema_period: int = 5, 
                    source: CandleProperties = OPEN,
                    atr_period: int = 5, 
                    multiplier: t.Union[int, float] = 1.5,
                    tp_ratio=Decimal('0.333'), 
                    sl_ratio=Decimal('1'),
                    optimization=False,
                    lock = None):
    # Build strategy using expressions from `declarative`
    class MeanReversion(DeclarativeStrategy):
        title = "Mean Reversion"
        buy_signal = buy_signal(source, ema_period=ema_period,
                                atr_period=atr_period, 
                                multiplier=multiplier)

        sell_signal = sell_signal(source, ema_period=ema_period,
                                  atr_period=atr_period, 
                                  multiplier=multiplier)

        long_entry = close      # Limit Price
        long_exit = long_exit(tp_ratio, sl_ratio)

        short_entry = close     # Limit price
        short_exit = short_exit(tp_ratio, sl_ratio)
        # Used to distinguish different runs of the optimizer (will be logged to CSV)
        params = (f"EMA period={ema_period} "
                  f"source={source} "
                  f"ATR period={atr_period} "
                  f"multiplier={multiplier} "
                  f"TP ratio={tp_ratio} "
                  f"SL ratio={sl_ratio}")

    # Configure dirs
    dirname = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.dirname(os.path.dirname(dirname))

    data_dir = os.path.join(rootdir, 'data')
    results_dir = os.path.join(rootdir, 'results')
    # Configure feed
    datafile = os.path.join(data_dir, 'mnq_1m_20240310_fixed.csv')
    schema = CSVCandlesSchema(open_time=0, open=1, high=2,
                              low=3, close=4, volume=5, close_time=6)

    feed = CSVCandlesFactory(datafile, 'MNQUSD', tf.M1,
                             delimiter=',', schema=schema,
                             timezone='America/New_York')
    # Setup session
    '''Trading hours 6pm-5pm Sunday-Friday US/Central (CME)
    Overnight margin period 16:30-08:00 America/New_York
    '''
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=16, minutes=0),
                             session_timezone='America/New_York',
                             overnight_start=timedelta(hours=16, minutes=0),
                             overnight_end=timedelta(hours=9, minutes=30),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5, 6})  # Saturday, Sunday
    # Run bakctesting
    result = run_backtest(MeanReversion,
                          feed, 10_000, since, until, None, session,
                          prefetch_option=PREFETCH_SINCE,
                          additional_collateral=Decimal('1500'),
                          check_margin_call=True,
                          per_contract_init_margin = Decimal('1699.64'),
                          per_contract_maintenance_margin=Decimal('1500'),
                          per_contract_overnight_margin=Decimal('2200'))

    print(result)
    print(result.get_stats())
    result.export_stats(results_dir, optimization=optimization, lock=lock)
    result.export_orders(results_dir)
    result.export_trades(results_dir)

def test_session_is_open_with_us_central_time():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 15:00')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert not session.is_open(sample_time)

def test_session_is_open():
    session = FuturesSession(session_start=timedelta(hours=10, minutes=30),
                             session_end=timedelta(hours=15, minutes=50),
                             session_timezone='America/New_York',
                             overnight_start=timedelta(hours=16, minutes=0),
                             overnight_end=timedelta(hours=9, minutes=30),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5, 6})  # Saturday, Sunday

    sample_time = datetime.fromisoformat('2020-01-08 09:30')
    sample_time = pytz.timezone('America/New_York').localize(sample_time)
    assert session.is_open(sample_time)
        
if __name__ == '__main__':
    since = datetime.fromisoformat("2024-03-12 09:30:00-04:00")
    until = datetime.fromisoformat("2024-03-25 16:00:00-04:00")
    run_with_params(since=since, until=until)   # defaults for the rest
    # test_session_is_open()