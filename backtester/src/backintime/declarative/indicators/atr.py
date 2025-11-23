import typing as t
from decimal import Decimal

from backintime.analyser.indicators.atr import AtrOptions
from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class AtrExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { AtrOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        return Decimal(analyser.atr(self.timeframe, self.period)[self.index])


class AtrFactory:
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index) -> AtrExpression:
        return AtrExpression(self.timeframe, self.period, index)
