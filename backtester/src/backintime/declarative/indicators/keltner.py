import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.keltner_channel import \
    KeltnerChannelOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class KeltnerChannelExpression(Primitive):
    def __init__(self, 
                 timeframe: Timeframes,
                 candle_property: CandleProperties,
                 period: int,
                 atr_period: int,
                 multiplier: int,
                 index: int,
                 line: str):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.index = index
        self.line = line
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return {
            KeltnerChannelOptions(self.timeframe, 
                        self.candle_property, self.period,
                        self.atr_period, self.multiplier)
        }

    def eval(self, broker, analyser, candles):
        result = analyser.keltner_channel(self.timeframe,
                        self.candle_property, self.period,
                        self.atr_period, self.multiplier)
        return Decimal(getattr(result[self.index], self.line))


class KeltnerChannelFactoryWithIndex:
    def __init__(self, 
                 timeframe: Timeframes,
                 candle_property: CandleProperties,
                 period: int,
                 atr_period: int,
                 multiplier: int,
                 index: int):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.index = index

    @property
    def upper_band(self) -> KeltnerChannelExpression:
        return KeltnerChannelExpression(
                    self.timeframe,
                    self.candle_property,
                    self.period,
                    self.atr_period,
                    self.multiplier,
                    self.index,
                    'upper_band')

    @property
    def middle_band(self) -> KeltnerChannelExpression:
        return KeltnerChannelExpression(
                    self.timeframe,
                    self.candle_property,
                    self.period,
                    self.atr_period,
                    self.multiplier,
                    self.index,
                    'middle_band')

    @property
    def lower_band(self) -> KeltnerChannelExpression:
        return KeltnerChannelExpression(
                    self.timeframe,
                    self.candle_property,
                    self.period,
                    self.atr_period,
                    self.multiplier,
                    self.index,
                    'lower_band')


class KeltnerChannelFactory:
    def __init__(self, 
                 timeframe: Timeframes,
                 candle_property: CandleProperties = CLOSE,
                 period: int = 20,
                 atr_period: int = 10,
                 multiplier: int = 2):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def __getitem__(self, index):
        return KeltnerChannelFactoryWithIndex(
                    self.timeframe, 
                    self.candle_property, 
                    self.period,
                    self.atr_period,
                    self.multiplier,
                    index)