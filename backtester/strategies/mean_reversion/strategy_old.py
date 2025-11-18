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
from backintime.analyser.indicators.constants import OPEN, CandleProperties
from backintime.analyser.indicators.keltner_channel import \
    KeltnerChannelResultSequence as KcResult
from backintime.broker import (LimitOrderOptions, OrderSide, StopLossOptions,
                               TakeProfitOptions)
from backintime.data.csv import CSVCandlesFactory, CSVCandlesSchema
from backintime.indicators import ATR, KELTNER_CHANNEL
from backintime.timeframes import Timeframes as tf
from backintime.utils import PREFETCH_SINCE


class MeanReversion(FuturesStrategy):
    title = "Mean Reversion"
    candle_timeframes = { tf.M1 }
    indicators = {
        ATR(tf.M5, period=5),       # ATR  
        KELTNER_CHANNEL(tf.M5, period=5)   # KC with EMA period=5 
    }

    def __init__(self, 
                 broker: FuturesBrokerProxy,
                 analyser: Analyser,
                 candles: Candles, 
                 ema_period: int, 
                 source: CandleProperties, 
                 atr_period: int, 
                 multiplier: t.Union[int, float],
                 tp_ratio: Decimal, 
                 sl_ratio: Decimal):
        self.ema_period = ema_period
        self.source = source
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._last_buy = None
        self._tp_ratio = tp_ratio
        self._sl_ratio = sl_ratio
        super().__init__(broker, analyser, candles)

    def _buy_signal(self, high: Decimal, low: Decimal,
                    keltner_channel: KcResult) -> bool:
        """
        Return True if buy conditions are met, False otherwise.
        Buy/Long when the price touches or crosses the lower KC.
        """
        return high >= keltner_channel[-1].lower_band and \
                        low < keltner_channel[-1].lower_band

    def _sell_signal(self, high: Decimal, low: Decimal, 
                     keltner_channel: KcResult) -> bool:
        """
        Return True if sell conditions are met, False otherwise.
        Sell/Short when the price touches or crosses the upper KC.
        """
        return high >= keltner_channel[-1].upper_band and \
                        low < keltner_channel[-1].upper_band

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

    def _make_limit_long(self, price: Decimal, 
                         atr: np.ndarray) -> LimitOrderOptions:
        """
        Make a Limit BUY order for a long position at `price` with:
            - TP at the current price + ATR*tp_ratio
            - SL at the current price - ATR*sl_ratio
        """
        tp_trigger = price + Decimal(atr[-1])*self._tp_ratio
        sl_trigger = price - Decimal(atr[-1])*self._sl_ratio
        tp = self._make_take_profit(tp_trigger)
        sl = self._make_stop_loss(sl_trigger)
        return LimitOrderOptions(order_side=OrderSide.BUY,
                                 order_price=price, 
                                 percentage_amount=Decimal('100'),
                                 take_profit=tp,
                                 stop_loss=sl)

    def _make_limit_short(self, price: Decimal, 
                          atr: np.ndarray) -> LimitOrderOptions:
        """
        Make a Limit SELL order for a short position at `price` with:
            - TP at the current price - ATR*tp_ratio
            - SL at the current price + ATR*sl_ratio
        """
        tp_trigger = price - Decimal(atr[-1])*self._tp_ratio
        sl_trigger = price + Decimal(atr[-1])*self._sl_ratio
        tp = self._make_take_profit(tp_trigger)
        sl = self._make_stop_loss(sl_trigger)
        return LimitOrderOptions(order_side=OrderSide.SELL,
                                 order_price=price, 
                                 percentage_amount=Decimal('100'),
                                 take_profit=tp,
                                 stop_loss=sl)

    def describe_params(self) -> str:
        return (f"EMA period={self.ema_period} "
                f"source={self.source} "
                f"ATR period={self.atr_period} "
                f"multiplier={self.multiplier} "
                f"TP ratio={self._tp_ratio} "
                f"SL ratio={self._sl_ratio}")

    def tick(self):
        curr_m1 = self.candles.get(tf.M1)
        high = curr_m1.high
        low = curr_m1.low
        close = curr_m1.close

        source = self.source
        atr_period = self.atr_period
        ema_period = self.ema_period
        multiplier = self.multiplier

        atr = self.analyser.atr(tf.M5, period=14)
        kc = self.analyser.keltner_channel(tf.M5, source, period=ema_period,
                                           atr_period=atr_period, 
                                           multiplier=multiplier)

        if self.broker.max_funds_for_futures:   # Entry
            if not self.broker.in_short:
                if self._buy_signal(high, low, kc):  # Long
                    long = self._make_limit_long(close, atr)
                    self.broker.submit_limit_order(long)

            if not self.broker.in_long:
                if self._sell_signal(high, low, kc):  # Short
                    short = self._make_limit_short(close, atr)
                    self.broker.submit_limit_short(short)


class MeanReversionFactory(AbstractStrategyFactory):
    def __init__(self, 
                 ema_period: int, 
                 source: CandleProperties, 
                 atr_period: int, 
                 multiplier: t.Union[int, float],
                 tp_ratio: Decimal, 
                 sl_ratio: Decimal):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.source = source
        self.multiplier = multiplier
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio

    def create(self,
               broker: FuturesBrokerProxy,
               analyser: Analyser,
               candles: Candles) -> MeanReversion:
        return MeanReversion(broker, analyser, candles, 
                             self.ema_period, self.source,
                             self.atr_period, self.multiplier,
                             self.tp_ratio, self.sl_ratio)


def run_with_params(since: datetime, 
                    until: datetime,
                    ema_period: int = 5, 
                    source: CandleProperties = OPEN,
                    atr_period: int = 5, 
                    multiplier: t.Union[int, float] = 1.5,
                    tp_ratio=Decimal('2'), 
                    sl_ratio=Decimal('1.5')):
    factory = MeanReversionFactory(ema_period, source, atr_period, 
                                   multiplier, tp_ratio, sl_ratio)
    # Alter `indicators` property using provided params
    MeanReversion.indicators = {
            ATR(tf.M5, period=14),          # ATR
            KELTNER_CHANNEL(tf.M5, period=ema_period)    # KELTNER
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

    result = run_backtest(MeanReversion,
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