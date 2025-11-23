import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.sma import SmaOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class SmaExpression(Primitive):
    def __init__(self, timeframe: Timeframes, price, period: int, index: int):
        self.timeframe = timeframe
        self.price = price
        self.period = period
        self.index = index
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { SmaOptions(self.timeframe, self.price, self.period) }

    def eval(self, broker, analyser, candles):
        return Decimal(analyser.sma(self.price, self.timeframe)[self.index])


class SmaFactory:
    def __init__(self, timeframe: Timeframes, price, period: int):
        self.timeframe=timeframe
        self.price=price
        self.period=period

    def __getitem__(self, index):
        return SmaExpression(self.timeframe, self.price, self.period, index)
