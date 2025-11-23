import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.ema import EmaOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class EmaExpression(Primitive):
    def __init__(self, timeframe: Timeframes,
                 candle_property: CandleProperties,
                 period: int,
                 index: int):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.index = index
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return {
            EmaOptions(self.timeframe, self.candle_property, self.period)
        }

    def eval(self, broker, analyser, candles):
        return Decimal(analyser.ema(self.timeframe, self.candle_property, self.period)[self.index])


class EmaFactory:
    def __init__(self, timeframe: Timeframes,
                 candle_property: CandleProperties = CLOSE,
                 period: int = 9):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    def __getitem__(self, index) -> EmaExpression:
        return EmaExpression(self.timeframe, self.candle_property, 
                    self.period, index)