import os
import typing as t
from datetime import datetime, timedelta
from decimal import Decimal

from backintime import run_backtest
from backintime.session import FuturesSession
from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
from backintime.analyser.indicators.constants import CLOSE, CandleProperties
from backintime.timeframes import Timeframes as tf
from backintime.utils import PREFETCH_SINCE
from backintime.declarative import (scalar, prices, VWEMA, VWMA,
                    VwemaExpression, VwmaExpression,
                    PriceExpression, BooleanExpression, 
                    DeclarativeStrategy, TP, SL)




def buy_signal(source: CandleProperties = CLOSE,
               short_period: int = 8,
               long_period: int = 13) -> BooleanExpression:
    """
    Return True if buy conditions are met, False otherwise.

    Buy/Long when Short-Term VWEMA crosses above Long-
    Term VWEMA.
    """
    short_vwema = VWEMA(tf.D1, source, period=short_period)
    long_vwema = VWEMA(tf.D1, source, period=long_period)
    vwma = VWMA(tf.D1, source, period=short_period)

    crossover_up = (short_vwema[-2] < long_vwema[-2]) & (short_vwema[-1] > long_vwema[-1]) 
    return crossover_up


def long_exit(source: CandleProperties = CLOSE,
               short_period: int = 8,
               long_period: int = 13) -> t.Tuple[t.Union[TP, SL]]:
    """
    Return a signal when the vwma crosses the long_vwema down
    """
    long_vwema = VWEMA(tf.D1, source, period=long_period)
    vwma = VWMA(tf.D1, source, period=short_period)
    crossover_down = (vwma[-2] > long_vwema[-2]) & (vwma[-1] < long_vwema[-1]) 
    
    # Trying exit with just signals and not hard TP/SL Need to figure something out with it
    return crossover_down


class VwemaLongStrategy(DeclarativeStrategy):
    title = "VWEMA Long Strategy"
    buy_signal = buy_signal()
    sell_signal = long_exit()

def run_with_params(since: datetime, 
                    until: datetime) -> None:

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
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday
    # Run backtesting
    result = run_backtest(VwemaLongStrategy, 
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
    since = datetime.fromisoformat("2019-05-05 18:12:00-04:00")
    until = datetime.fromisoformat("2024-04-26 17:00:00-04:00")
    run_with_params(since=since, until=until)   # defaults for the rest