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
from backintime.declarative import (scalar, prices, MACD, 
                    AtrExpression, PriceExpression, BooleanExpression, 
                    DeclarativeStrategy, TP, SL)


class MacdStrategy(DeclarativeStrategy):
    title = "MACD Strategy"
    buy_signal = (MACD(tf.M1)[-1].hist > scalar(0)) & (MACD(tf.M1)[-2].hist <= scalar(0))
    sell_signal = (MACD(tf.M1)[-1].hist <= scalar(0)) & (MACD(tf.M1)[-2].hist > scalar(0))


def run_with_params(since: datetime, until: datetime):
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
    # Run bakctesting
    result = run_backtest(MacdStrategy,
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