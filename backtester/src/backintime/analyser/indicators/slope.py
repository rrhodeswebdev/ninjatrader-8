import typing as t

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import IndicatorOptions, InputTimeseries, MarketData
from .constants import OPEN, CandleProperties


def slope(market_data: MarketData, timeframe: Timeframes,
            candle_property: CandleProperties = OPEN,
            period: int = 5, price2bar_ratio: float = 8.725) -> numpy.ndarray:
    """Slope."""
    quantity = period**2
    values = market_data.get_values(timeframe, candle_property, quantity)
    values = pd.Series(values)
    
    ema = ta.trend.EMAIndicator(values, period).ema_indicator()
    slope = abs(numpy.degrees(numpy.arctan(ema.diff()/price2bar_ratio)))
    return slope.values


class SlopeOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, 
                 candle_property: CandleProperties = OPEN, 
                 period: int = 5):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'SLOPE'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, 
                    candle_property=self.candle_property, period=self.period)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, 
                            self.candle_property, self.period**2),
        )