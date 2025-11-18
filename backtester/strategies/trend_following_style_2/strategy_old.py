import os
import typing as t
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from backintime import FuturesStrategy, run_backtest
from backintime.session import FuturesSession
from backintime.trading_strategy import AbstractStrategyFactory
from backintime.analyser.analyser import Analyser
from backintime.candles import Candles
from backintime.broker.futures.proxy import FuturesBrokerProxy
from backintime.analyser.indicators.constants import CLOSE, OPEN, CandleProperties
from backintime.analyser.indicators.keltner_channel import \
    KeltnerChannelResultSequence as KcResult
from backintime.broker import (LimitOrderOptions, OrderSide, StopLossOptions,
                               TakeProfitOptions)
from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
from backintime.indicators import EMA, KELTNER_CHANNEL
from backintime.timeframes import Timeframes as tf
from backintime.utils import PREFETCH_SINCE


class TrendFollowingStyle2(FuturesStrategy):
    title = "Trend Following Style 2"
    candle_timeframes = { tf.M1: 1, tf.M5: 3 }
    indicators = {
        EMA(tf.M5, OPEN, 5),        # Short-Term EMA
        EMA(tf.M5, OPEN, 21),       # Long-Term EMA
        KELTNER_CHANNEL(tf.M5, 5)   # KC with EMA period=5              
    }

    def __init__(self, 
                 broker: FuturesBrokerProxy,
                 analyser: Analyser,
                 candles: Candles, 
                 short_period: int, 
                 long_period: int, 
                 source: CandleProperties):
        self.source = source
        self.short_period = short_period
        self.long_period = long_period
        self._last_buy = None
        self._crossover_up = False
        self._crossover_down = False
        super().__init__(broker, analyser, candles)

    def _buy_signal(self, short_ema: np.ndarray, 
                    long_ema: np.ndarray, close_prices: list) -> bool:
        """
        Return True if buy conditions are met, False otherwise.

        Buy/Long when Short-Term EMA crosses above Long-
        Term EMA and the last 3 close prices are increasing.
        """
        price_check = close_prices[-1] > close_prices[-2] > close_prices[-3]
        return short_ema[-1] > long_ema[-1] and price_check

    def _sell_signal(self, short_ema: np.ndarray, 
                     long_ema: np.ndarray, close_prices: list) -> bool:
        """
        Return True if sell conditions are met, False otherwise.

        Sell/Short when Short-Term EMA crosses below Long-
        Term EMA and the last 3 close prices are decreasing.
        """
        price_check = close_prices[-1] < close_prices[-2] < close_prices[-3]
        return short_ema[-1] < long_ema[-1] and price_check

    def _make_take_profit(self, trigger_price: Decimal) -> TakeProfitOptions:
        """
        Make a Take Profit with trigger price at `trigger_price`.
        """
        return TakeProfitOptions(percentage_amount=Decimal('100.00'),
                                 trigger_price=trigger_price)

    def _make_stop_loss(self, trigger_price: Decimal) -> StopLossOptions:
        """
        Make a Stop Loss with trigger price at `trigger_price`.
        """
        return StopLossOptions(percentage_amount=Decimal('100.00'),
                               trigger_price=trigger_price)

    def _make_long(self, price: Decimal, 
                   keltner_channel: KcResult) -> LimitOrderOptions:
        """
        Make a Limit BUY for a long position at `price` with:
            - TP at Keltner Channel's upper band
            - SL at Keltner Channel's lower band
        """
        upper_band = Decimal(keltner_channel[-1].upper_band)
        lower_band = Decimal(keltner_channel[-1].lower_band)
        tp = self._make_take_profit(upper_band)
        sl = self._make_stop_loss(lower_band)
        return LimitOrderOptions(order_side=OrderSide.BUY,
                                 order_price=price, 
                                 percentage_amount=Decimal('100'),
                                 take_profit=tp,
                                 stop_loss=sl)

    def _make_short(self, price: Decimal, 
                    keltner_channel: KcResult) -> LimitOrderOptions:
        """
        Make a Limit SELL for a short position at `price` with:
            - TP at Keltner Channel's lower band
            - SL at Keltner Channel's upper band
        """
        upper_band = Decimal(keltner_channel[-1].upper_band)
        lower_band = Decimal(keltner_channel[-1].lower_band)
        tp = self._make_take_profit(lower_band)
        sl = self._make_stop_loss(upper_band)
        return LimitOrderOptions(order_side=OrderSide.SELL,
                                 order_price=price,
                                 percentage_amount=Decimal('100'),
                                 take_profit=tp,
                                 stop_loss=sl)

    def describe_params(self) -> str:
        return (f"source={self.source} "
                f"Short EMA period={self.short_period} "
                f"Long EMA period={self.long_period}")

    def tick(self):
        curr_close = self.candles.get(tf.M1).close

        source = self.source
        short_period = self.short_period
        long_period = self.long_period

        short_ema = self.analyser.ema(tf.M5, source, period=short_period)
        long_ema = self.analyser.ema(tf.M5, source, period=long_period)
        kc = self.analyser.keltner_channel(tf.M5, OPEN, period=5, 
                                           atr_period=5, multiplier=1)

        close_prices = [
            self.candles.get(tf.M5, -3).close,
            self.candles.get(tf.M5, -2).close,
            self.candles.get(tf.M5, -1).close
        ]

        if self.broker.max_funds_for_futures:
            if self._buy_signal(short_ema, long_ema, close_prices):     # Long
                long = self._make_long(curr_close, kc)
                self.broker.submit_limit_order(long)

            elif self._sell_signal(short_ema, long_ema, close_prices):  # Short
                short = self._make_short(curr_close, kc)
                self.broker.submit_limit_short(short)


class TrendFollowingStyle2Factory(AbstractStrategyFactory):
    def __init__(self, 
                 short_ema_period: int, 
                 long_ema_period: int, 
                 source: CandleProperties = OPEN):
        self.short_period = short_ema_period
        self.long_period = long_ema_period
        self.source = source

    def create(self, 
               broker: FuturesBrokerProxy,
               analyser: Analyser,
               candles: Candles) -> TrendFollowingStyle2:
        return TrendFollowingStyle2(broker, analyser, candles, 
                                    self.short_period,
                                    self.long_period, self.source)


def run_with_params(since: datetime, 
                    until: datetime,
                    short_ema_period: int = 5, 
                    long_ema_period: int = 21,
                    source: CandleProperties = OPEN) -> None:
    factory = TrendFollowingStyle2Factory(short_ema_period, 
                                          long_ema_period, source)
    # Alter `indicators` property using provided params
    TrendFollowingStyle2.indicators = {
            EMA(tf.M5, source, short_ema_period),      # Short-Term EMA
            EMA(tf.M5, source, long_ema_period),       # Long-Term EMA
            KELTNER_CHANNEL(tf.M5, 5)   # KC with EMA period=5              
    }

    dirname = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.dirname(os.path.dirname(dirname))

    data_dir = os.path.join(rootdir, 'data')
    results_dir = os.path.join(rootdir, 'results')

    datafile = os.path.join(data_dir, 'mnq_1m_20240310_fixed.csv')
    schema = CSVCandlesSchema(open_time=0, open=1, high=2,
                              low=3, close=4, volume=5, close_time=6)

    feed = CSVCandlesFactory(datafile, 'MNQUSD', tf.M1, 
                             delimiter=',', schema=schema,
                             timezone='America/New_York')

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

    '''Evaluate the trading strategy running backtesting.
    Since indicators require some amount of data to be already there
    in order to produce the first result, we're using a prefetching 
    feature here. 

    Since the `prefetch_option` is PREFETCH_SINCE,
    the data is collected SINCE the `since` date.
    Actual start time of the backtesting is then shifted.

    This option is useful when working with historical data 
    from CSV files. We just specify the first date in the file
    as the `since` date, and the engine will use the data
    from start to populate internal buffers so to have
    indicators already computed by the actual start of the backtesting.

    When working with data feed from APIs, PREFETCH_UNTIL should
    be used instead. Thus, the data UNTIL the `since` date
    is queried and used for the initial indicators computing.
    Then the actual start time of backtesting is not altered.
    
    Another thing to consider here is `additional_collateral`.
    This is the amount of funds (USD) used as a good-faith
    collateral in addition to the initial margin of an exchange.
    On the one hand, it significantly decreases the risk of
    a margin call (when you don't have enough funds to 
    cover the maintenance margin). On the other hand, it means
    that the possible amount of contracts you trade is decreased
    since for each contract held you're required to have more
    funds as a collateral.
    '''
    result = run_backtest(TrendFollowingStyle2,
                          feed, 10_000, since, until, factory, session,
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
