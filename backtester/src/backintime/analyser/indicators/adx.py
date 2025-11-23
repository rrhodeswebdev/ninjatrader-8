import typing as t

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import IndicatorOptions, InputTimeseries, MarketData
from .constants import CLOSE, HIGH, LOW


def adx(market_data: MarketData, 
        timeframe: Timeframes,
        period: int = 14) -> numpy.ndarray:
    """
    Average Directional Movement Index (ADX).

    ADX does not indicate trend direction or momentum, 
    only trend strength. 
    Generally, ADX readings below 20 indicate trend weakness,
    and readings above 40 indicate trend strength. 
    An extremely strong trend is indicated by readings above 50.
    """
    quantity = period**2

    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = pd.Series(highs, dtype=numpy.float64)
        
    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = pd.Series(lows, dtype=numpy.float64)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = pd.Series(close, dtype=numpy.float64)

    adx = ta.trend.adx(highs, lows, close, period)
    return adx.values


class AdxOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'ADX'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, period=self.period)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, HIGH, self.period**2),
            InputTimeseries(self.timeframe, LOW, self.period**2),
            InputTimeseries(self.timeframe, CLOSE, self.period**2)
        )

    