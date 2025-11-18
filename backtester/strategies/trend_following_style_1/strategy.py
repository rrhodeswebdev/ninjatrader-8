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
from backintime.declarative import (scalar, prices, EMA, ATR, KELTNER,
                    AtrExpression, EmaExpression,
                    PriceExpression, BooleanExpression, 
                    DeclarativeStrategy, TP, SL)


close:  PriceExpression = prices.close(tf.M1)
atr:    AtrExpression = ATR(tf.M5, period=14)[-1]


def buy_signal(source: CandleProperties = OPEN,
               short_period: int = 5,
               long_period: int = 21) -> BooleanExpression:
    """
    Return True if buy conditions are met, False otherwise.

    Buy/Long when Short-Term EMA crosses above Long-
    Term EMA and the last 3 close prices are increasing.
    """
    short_ema = EMA(tf.M5, source, period=short_period)[-1]
    long_ema = EMA(tf.M5, source, period=long_period)[-1]
    m5_close: PriceExpression = prices.close(tf.M5)

    price_check = (m5_close[-1] > m5_close[-2]) & (m5_close[-2] > m5_close[-3])
    return (short_ema > long_ema) & price_check


def sell_signal(source: CandleProperties = OPEN,
                short_period: int = 5,
                long_period: int = 21) -> BooleanExpression:
    """
    Return True if sell conditions are met, False otherwise.

    Sell/Short when Short-Term EMA crosses below Long-
    Term EMA and the last 3 close prices are decreasing.
    """
    short_ema = EMA(tf.M5, source, period=short_period)[-1]
    long_ema = EMA(tf.M5, source, period=long_period)[-1]
    m5_close: PriceExpression = prices.close(tf.M5)

    price_check = (m5_close[-1] < m5_close[-2]) & (m5_close[-2] < m5_close[-3])
    return (short_ema < long_ema) & price_check


def long_exit(tp_ratio, sl_ratio) -> t.Tuple[t.Union[TP, SL]]:
    """
    Return a sequence of TP/SL to close Long:
        - TP at the current price + ATR*tp_ratio
        - SL at the current price - ATR*sl_ratio
    """
    atr: AtrExpression = ATR(tf.M5, period=14)[-1]
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
    atr: AtrExpression = ATR(tf.M5, period=14)[-1]
    return (
        TP(trigger_price=close - atr*scalar(tp_ratio)), 
        SL(trigger_price=close + atr*scalar(sl_ratio))
    )

# Default implementation
class TrendFollowingStyle1(DeclarativeStrategy):
    title = "Trend Following Style 1"
    buy_signal = buy_signal()
    sell_signal = sell_signal()

    long_entry = close      # Limit price
    long_exit = (
            TP(trigger_price=close + atr*scalar(1)), 
            SL(trigger_price=close - atr*scalar(0.5))
    )

    short_entry = close     # Limit price
    short_exit = (
            TP(trigger_price=close + atr*scalar(1)), 
            SL(trigger_price=close - atr*scalar(0.5))
    )


def run_with_params(since: datetime, 
                    until: datetime,
                    short_ema_period: int = 5, 
                    long_ema_period: int = 21, 
                    source: CandleProperties = OPEN,
                    tp_ratio = Decimal('1'), 
                    sl_ratio = Decimal('0.5')) -> None:
    # Build strategy using expressions from `declarative`
    class TrendFollowingStyle1(DeclarativeStrategy):
        title = "Trend Following Style 1"
        buy_signal = buy_signal(source, short_ema_period, long_ema_period)
        sell_signal = sell_signal(source, short_ema_period, long_ema_period)

        long_entry = close      # Limit price
        long_exit = long_exit(tp_ratio, sl_ratio)

        short_entry = close     # Limit price
        short_exit = short_exit(tp_ratio, sl_ratio)
        # Used to distinguish different runs of the optimizer (will be logged to CSV)
        params = (f"Short EMA period={short_ema_period} "
                  f"Long EMA period={long_ema_period} "
                  f"source={source} "
                  f"TP ratio={tp_ratio} "
                  f"SL ratio={sl_ratio}")

    # Setup dirs
    dirname = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.dirname(os.path.dirname(dirname))

    data_dir = os.path.join(rootdir, 'data')
    results_dir = os.path.join(rootdir, 'results')
    # Setup feed
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
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday
    # Run backtesting
    result = run_backtest(TrendFollowingStyle1, 
                          feed, 10_000, since, until, None, session,
                          prefetch_option=PREFETCH_SINCE,
                          additional_collateral=Decimal('1500'),
                          check_margin_call=True,
                          per_contract_init_margin = Decimal('1699.64'),
                          per_contract_maintenance_margin=Decimal('1500'),
                          per_contract_overnight_margin=Decimal('2200'))

    print(result)
    print(result.get_stats())
    result.export_stats(results_dir)
    result.export_orders(results_dir)
    result.export_trades(results_dir)


if __name__ == '__main__':
    since = datetime.fromisoformat("2024-03-10 18:00:00-04:00")
    until = datetime.fromisoformat("2024-03-25 17:00:00-04:00")
    run_with_params(since=since, until=until)   # defaults for the rest