import typing as t
from decimal import Decimal

from backintime.analyser.indicators.adx import AdxOptions
from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class AdxExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { AdxOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        return Decimal(analyser.adx(self.timeframe, self.period)[self.index])


class AdxFactory:
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index) -> AdxExpression:
        return AdxExpression(self.timeframe, self.period, index)


    