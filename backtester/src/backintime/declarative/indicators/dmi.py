import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.dmi import DmiOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class DmiExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int, attr: str):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)

    @property
    def adx(self):
        self._attr = 'adx'
        return self

    @property
    def positive_di(self):
        self._attr = 'positive_di'
        return self

    @property
    def negative_di(self):
        self._attr = 'negative_di'
        return self

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]: 
        return { DmiOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        result = analyser.dmi(self.timeframe, self.period)
        return Decimal(getattr(result[self.index], self.attr))


class DmiFactoryWithIndex:
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index

    @property
    def adx(self) -> DmiExpression:
        attr = 'adx'
        return DmiExpression(self.timeframe, self.period, self.index, attr)

    @property
    def positive_di(self) -> DmiExpression:
        attr = 'positive_di'
        return DmiExpression(self.timeframe, self.period, self.index, attr)

    @property
    def negative_di(self) -> DmiExpression:
        attr = 'negative_di'
        return DmiExpression(self.timeframe, self.period, self.index, attr)


class DmiFactory:
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index):
        return DmiFactoryWithIndex(self.timeframe, self.period, index)