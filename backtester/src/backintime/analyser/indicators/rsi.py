import typing as t
from dataclasses import dataclass

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import IndicatorOptions, InputTimeseries, MarketData
from .constants import CLOSE


def rsi(market_data: MarketData, timeframe: Timeframes,
            period: int = 14) -> numpy.ndarray:
    """
    Relative Strength Index (RSI).

    Momentum oscillator that measures the speed and change 
    of price movements. RSI oscillates between zero and 100. 
    Traditionally, and according to Wilder, RSI is considered 
    overbought when above 70 and oversold when below 30.
    """
    quantity = period**2
    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = pd.Series(close)
    rsi = ta.momentum.RSIIndicator(close, period).rsi()
    return rsi.values


class RsiOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'RSI'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, period=self.period)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, CLOSE, self.period**2),
        )
