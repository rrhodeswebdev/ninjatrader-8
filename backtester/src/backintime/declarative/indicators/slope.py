from decimal import Decimal
from backintime.timeframes import Timeframes
from backintime.analyser.indicators.base import IndicatorParam
from backintime.analyser.indicators.constants import OPEN, HIGH, LOW, CLOSE, CandleProperties
from backintime.declarative.base import Primitive


class SlopeExpression(Primitive):
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
    def input_timeseries(self):
        return [
            IndicatorParam(timeframe=self.timeframe, 
                           candle_property=self.candle_property, 
                           quantity=self.period**2),
        ]

    def eval(self, broker, analyser, candles):
        return Decimal(analyser.slope(self.timeframe, self.candle_property, self.period)[self.index])


class SlopeFactory:
    def __init__(self, timeframe: Timeframes,
                 candle_property: CandleProperties = OPEN,
                 period: int = 5):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    def __getitem__(self, index) -> SlopeExpression:
        return SlopeExpression(self.timeframe, self.candle_property, 
                    self.period, index)