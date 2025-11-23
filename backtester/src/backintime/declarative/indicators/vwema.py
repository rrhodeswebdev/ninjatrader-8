import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN, VOLUME,
                                                      CandleProperties)
from backintime.analyser.indicators.vwema import VwEmaOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class VwemaExpression(Primitive):
    def __init__(self, 
                 timeframe: Timeframes,
                 period: int,
                 price: int,
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
            VwEmaOptions(self.timeframe, period = self.period)
        }

    def eval(self, broker, analyser, candles):
        result = analyser.vwema(self.price, self.timeframe, period = self.period)
        return Decimal(getattr(result[self.index], self.attr))

class VwemaFactoryWithIndex:
    def __init__(self, 
                 timeframe: Timeframes,
                 period: int,
                 price: int,
                 index: int):
        self.timeframe = timeframe
        self.price = price
        self.period = period
        self.index = index

    @property
    def vwema(self) -> VwemaExpression:
        attr = 'vwema'
        return VwemaExpression(
                    self.timeframe, 
                    self.price,
                    self.period, 
                    self.index, attr)
        
class VwemaFactory:
    def __init__(self, 
                 timeframe: Timeframes,
                 price: int,
                 period: int):
        self.timeframe = timeframe
        self.price = price
        self.period = period
    def __getitem__(self, index):
        return VwemaFactoryWithIndex(self.timeframe, self.price, self.period, index)