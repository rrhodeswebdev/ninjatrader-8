import typing as t

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import IndicatorOptions, InputTimeseries, MarketData
from .constants import CLOSE, HIGH, LOW


def atr(market_data: MarketData, 
        timeframe: Timeframes, 
        period: int = 14) -> numpy.ndarray:
    """Average True Range (ATR)."""
    quantity = period**2

    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = pd.Series(highs)
        
    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = pd.Series(lows)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = pd.Series(close)

    atr = ta.volatility.AverageTrueRange(highs, lows, close, period)
    return atr.average_true_range().values


class AtrOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'ATR'

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
