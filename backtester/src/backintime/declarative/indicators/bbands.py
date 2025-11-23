import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.bbands import BbandsOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class BbandsExpression(Primitive):
    def __init__(self, 
                 timeframe: Timeframes, 
                 candle_property: CandleProperties,
                 period: int,
                 deviation_quotient: int,
                 index: int,
                 line: str):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.quotient = deviation_quotient
        self.index = index
        self.line = line
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return {
            BbandsOptions(self.timeframe, self.candle_property, 
                          self.period, self.quotient) 
        }

    def eval(self, broker, analyser, candles):
        result = analyser.bbands(self.timeframe, self.candle_property, 
                                 self.period, self.quotient)
        return Decimal(getattr(result[self.index], self.line))


class BbandsFactoryWithIndex:
    def __init__(self, 
                 timeframe: Timeframes, 
                 candle_property: CandleProperties,
                 period: int,
                 deviation_quotient: int,
                 index: int):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.quotient = deviation_quotient
        self.index = index

    @property
    def upper_band(self) -> BbandsExpression:
        return BbandsExpression(self.timeframe, self.candle_property, 
                self.period, self.quotient, self.index, 'upper_band')

    @property
    def middle_band(self) -> BbandsExpression:
        return BbandsExpression(self.timeframe, self.candle_property, 
                self.period, self.quotient, self.index, 'middle_band')

    @property
    def lower_band(self) -> BbandsExpression:
        return BbandsExpression(self.timeframe, self.candle_property, 
                self.period, self.quotient, self.index, 'lower_band')


class BbandsFactory:
    def __init__(self, 
                 timeframe: Timeframes, 
                 candle_property: CandleProperties = CLOSE,
                 period: int = 20,
                 deviation_quotient: int = 2):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.quotient = deviation_quotient

    def __getitem__(self, index) -> BbandsFactoryWithIndex:
        return BbandsFactoryWithIndex(self.timeframe, self.candle_property, 
                    self.period, self.quotient, index)