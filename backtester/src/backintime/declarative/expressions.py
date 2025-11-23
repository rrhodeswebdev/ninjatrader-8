import functools
import operator
import typing as t
from decimal import Decimal

from backintime import FuturesStrategy
from backintime.analyser.indicators.base import IndicatorParam
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.broker.base import (LimitOrderOptions, MarketOrderOptions,
                                    OrderSide, StopLossOptions,
                                    TakeProfitOptions)
from backintime.timeframes import Timeframes
from backintime.timeframes import Timeframes as tf

from .base import (BooleanExpression, Primitive, merge_indicators_meta,
                   merge_ohlcv_timeframes)
from .indicators import *


class ScalarExpression(Primitive):
    """Evaluates to provided numeric value."""
    def __init__(self, value):
        self.value=value
        super().__init__(concrete=self)

    def eval(self, broker, analyser, candles):
        return self.value


class PriceExpression(Primitive):
    """Evaluates to OHLC for a given timeframe."""
    def __init__(self, price, timeframe, index: int = -1):
        self.price=price
        self.timeframe=timeframe
        self.index=index
        super().__init__(concrete=self)

    def __getitem__(self, index: int = -1):
        return PriceExpression(self.price, self.timeframe, index)

    @property
    def ohlcv_timeframes(self) -> dict:
        return { self.timeframe: abs(self.index) }

    def eval(self, broker, analyser, candles):
        try:
            ohlcv = candles.get(self.timeframe, self.index)
        except IndexError as e:
            print(self.timeframe, self.index)
            raise e
        if self.price is OPEN:
            return ohlcv.open
        elif self.price is HIGH:
            return ohlcv.high
        elif self.price is LOW:
            return ohlcv.low
        elif self.price is CLOSE:
            return ohlcv.close


class MarketPriceStub:
    """
    Stub object that evaluates to nothing meaningful 
    and simply designates market price (i.e. OPEN price of the next bar).

    Used for Market Orders.
    """
    @property
    def ohlcv_timeframes(self):
        return dict()

    @property
    def indicators_meta(self): # list of indicator params
        return set()

    def eval(self, broker, analyser, candles):
        return self


class TakeProfitExpression:
    """Evaluates to TakeProfitOptions."""
    def __init__(self, trigger_price, order_price=None):
        self.trigger_price = trigger_price
        self.order_price = order_price
        self._ohlcv_timeframes = merge_ohlcv_timeframes(trigger_price, order_price)
        self._indicators_meta = merge_indicators_meta(trigger_price, order_price)

    @property
    def ohlcv_timeframes(self): # set
        return self._ohlcv_timeframes

    @property
    def indicators_meta(self): # list of indicator params
        return self._indicators_meta

    def eval(self, broker, analyser, candles) -> TakeProfitOptions:
        order_price = self.order_price
        if order_price:
            order_price = order_price.eval(broker, analyser, candles)
        trigger_price = self.trigger_price.eval(broker, analyser, candles)

        return TakeProfitOptions(trigger_price=trigger_price, 
                    order_price=order_price, percentage_amount=Decimal(100))


class StopLossExpression:
    """Evaluates to StopLossOptions."""
    def __init__(self, trigger_price, order_price=None):
        self.trigger_price = trigger_price
        self.order_price = order_price
        self._ohlcv_timeframes = merge_ohlcv_timeframes(trigger_price, order_price)
        self._indicators_meta = merge_indicators_meta(trigger_price, order_price)

    @property
    def ohlcv_timeframes(self): # set
        return self._ohlcv_timeframes

    @property
    def indicators_meta(self): # list of indicator params
        return self._indicators_meta

    def eval(self, broker, analyser, candles) -> StopLossOptions:
        order_price = self.order_price
        if order_price:
            order_price = order_price.eval(broker, analyser, candles)
        trigger_price = self.trigger_price.eval(broker, analyser, candles)

        return StopLossOptions(trigger_price=trigger_price, 
                    order_price=order_price, percentage_amount=Decimal(100))


def crossover_up(fst, snd) -> BooleanExpression:
    return (fst[-3] > snd[-3]) & (fst[-2] > snd[-2]) & (fst[-1] > snd[-1])


def crossover_down(fst, snd) -> BooleanExpression:
    return (fst[-3] < snd[-3]) & (fst[-2] < snd[-2]) & (fst[-1] < snd[-1])

