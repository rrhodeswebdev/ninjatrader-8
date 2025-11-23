import typing as t

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import IndicatorOptions, InputTimeseries, MarketData
from .constants import CLOSE, CandleProperties


def sma(market_data: MarketData, 
        timeframe: Timeframes,
        candle_property: CandleProperties = CLOSE,
        period: int = 9) -> numpy.ndarray:
    """Simple moving average, also known as 'MA'."""
    values = market_data.get_values(timeframe, candle_property, period)
    values = pd.Series(values)
    sma = ta.trend.SMAIndicator(values, period).sma_indicator()
    return sma.values


class SmaOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, 
                 candle_property: CandleProperties = CLOSE, 
                 period: int = 9):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'SMA'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, 
                    candle_property=self.candle_property, period=self.period)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, 
                            self.candle_property, self.period),
        )
