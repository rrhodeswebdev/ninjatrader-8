import typing as t
from dataclasses import dataclass

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import (IndicatorOptions, IndicatorResultSequence, InputTimeseries,
                   MarketData)
from .constants import CLOSE, HIGH, LOW


@dataclass
class DMIResultItem:
    adx: numpy.float64
    positive_di: numpy.float64
    negative_di: numpy.float64


class DMIResultSequence(IndicatorResultSequence[DMIResultItem]):
    def __init__(self, 
                 adx: numpy.ndarray, 
                 positive_di: numpy.ndarray, 
                 negative_di: numpy.ndarray):
        self.adx = adx
        self.positive_di = positive_di
        self.negative_di = negative_di

    def crossover_up(self) -> bool:
        return self.positive_di[-1] > self.negative_di[-1] and \
                self.positive_di[-2] <= self.negative_di[-2]

    def crossover_down(self) -> bool:
        return self.negative_di[-1] > self.positive_di[-1] and \
                self.negative_di[-2] <= self.negative_di[-2]

    def adx_increases(self, period: int = 2) -> bool:
        values = self.adx[-period:]
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def adx_decreases(self, period: int = 2) -> bool:
        values = self.adx[-period:]
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def __iter__(self) -> t.Iterator[DMIResultItem]:
        zip_iter = zip(self.adx, self.positive_di, self.negative_di)
        return (
            DMIResultItem(adx, positive_di, negative_di) 
                for adx, positive_di, negative_di in zip_iter
        )

    def __reversed__(self) -> t.Iterator[DMIResultItem]:
        reversed_iter = zip(reversed(self.adx), 
                            reversed(self.positive_di), 
                            reversed(self.negative_di))
        return (
            DMIResultItem(adx, positive_di, negative_di) 
                for adx, positive_di, negative_di in reversed_iter
        )

    def __getitem__(self, index: int) -> DMIResultItem:
        return DMIResultItem(self.adx[index], 
                             self.positive_di[index], 
                             self.negative_di[index])

    def __len__(self) -> int:
        return min(len(self.adx),
                   len(self.positive_di),
                   len(self.negative_di))

    def __repr__(self) -> str:
        return (f"DMIResultSequence(adx={self.adx}, "
                f"positive_di={self.positive_di}, "
                f"negative_di={self.negative_di})")


def dmi(market_data: MarketData, timeframe: Timeframes,
            period: int = 14) -> DMIResultSequence:
    """Directional Movement Indicator (DMI)."""
    quantity = period**2
    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = pd.Series(highs, dtype=numpy.float64)

    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = pd.Series(lows, dtype=numpy.float64)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = pd.Series(close, dtype=numpy.float64)

    dmi = ta.trend.ADXIndicator(highs, lows, close, period)

    return DMIResultSequence(adx=dmi.adx().values, 
                             positive_di=dmi.adx_pos().values, 
                             negative_di=dmi.adx_neg().values)


class DmiOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, period: int = 14):
        self.timeframe = timeframe
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'DMI'

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
