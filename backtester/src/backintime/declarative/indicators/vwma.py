import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN, VOLUME,
                                                      CandleProperties)
from backintime.analyser.indicators.vwma import VwmaOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class VwmaExpression(Primitive):
    def __init__(self, 
                 timeframe: Timeframes,
                 price: int,
                 period: int,
                 index: int,
                 attr: str):
        self.timeframe = timeframe
        self.price = price
        self.period = period
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)
        
    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return {
            VwmaOptions(self.timeframe, period = self.period)
        }

    def eval(self, broker, analyser, candles):
        result = analyser.vwma(self.price, self.timeframe, period = self.period)
        return Decimal(getattr(result[self.index], self.attr))


class VwmaFactoryWithIndex:
    def __init__(self, 
                 timeframe: Timeframes,
                 price: int,
                 period: int,
                 index: int):
        self.timeframe = timeframe
        self.price = price
        self.period = period
        self.index = index

    @property
    def vwma(self) -> VwmaExpression:
        attr = 'vwma'
        return VwmaExpression(
                    self.timeframe,
                    self.price, 
                    self.period, 
                    self.index, attr)
        
class VwmaFactory:
    def __init__(self, 
                 timeframe: Timeframes,
                 price: int,
                 period: int):
        self.timeframe = timeframe
        self.price = price
        self.period = period
    def __getitem__(self, index):
        return VwmaFactoryWithIndex(self.timeframe, self.price, self.period, index)
