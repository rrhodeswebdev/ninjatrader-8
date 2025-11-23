import typing as t
from dataclasses import dataclass
from enum import Enum

import numpy
import pandas as pd

from backintime.timeframes import Timeframes

from .base import (IndicatorOptions, IndicatorResultSequence, InputTimeseries,
                   MarketData)
from .constants import CLOSE, HIGH, LOW


@dataclass
class TraditionalPivotPointsItem:
    pivot:  numpy.float64
    s1:     numpy.float64
    s2:     numpy.float64
    s3:     numpy.float64
    s4:     numpy.float64
    s5:     numpy.float64
    r1:     numpy.float64
    r2:     numpy.float64
    r3:     numpy.float64
    r4:     numpy.float64
    r5:     numpy.float64


@dataclass
class ClassicPivotPointsItem:
    pivot:  numpy.float64
    s1:     numpy.float64
    s2:     numpy.float64
    s3:     numpy.float64
    s4:     numpy.float64
    r1:     numpy.float64
    r2:     numpy.float64
    r3:     numpy.float64
    r4:     numpy.float64


@dataclass
class FibonacciPivotPointsItem:
    pivot:  numpy.float64
    s1:     numpy.float64
    s2:     numpy.float64
    s3:     numpy.float64
    r1:     numpy.float64
    r2:     numpy.float64
    r3:     numpy.float64


class TraditionalPivotPoints(IndicatorResultSequence[TraditionalPivotPointsItem]):
    def __init__(self, 
                 pivot,
                 s1: numpy.ndarray,
                 s2: numpy.ndarray,
                 s3: numpy.ndarray,
                 s4: numpy.ndarray,
                 s5: numpy.ndarray,
                 r1: numpy.ndarray,
                 r2: numpy.ndarray,
                 r3: numpy.ndarray,
                 r4: numpy.ndarray,
                 r5: numpy.ndarray):
        self.pivot = pivot
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.r5 = r5

    def __iter__(self) -> t.Iterator[TraditionalPivotPointsItem]:
        zip_iter = zip(self.pivot, 
                       self.s1, self.s2, self.s3, self.s4, self.s5, 
                       self.r1, self.r2, self.r3, self.r4, self.r5)
        return (
            TraditionalPivotPointsItem(*values) 
                for values in zip_iter
        )

    def __reversed__(self) -> t.Iterator[TraditionalPivotPointsItem]:
        reversed_iter = zip(reversed(self.pivot), 
                            reversed(self.s1), 
                            reversed(self.s2), reversed(self.s3), 
                            reversed(self.s4), reversed(self.s5),
                            reversed(self.r1), reversed(self.r2), 
                            reversed(self.r3), reversed(self.r4), 
                            reversed(self.r5))
        return (
            TraditionalPivotPointsItem(*values) 
                for values in reversed_iter
        )

    def __getitem__(self, index: int) -> TraditionalPivotPointsItem:
        return TraditionalPivotPointsItem(self.pivot[index],
                                          self.s1[index], 
                                          self.s2[index], self.s3[index], 
                                          self.s4[index], self.s5[index],
                                          self.r1[index], self.r2[index], 
                                          self.r3[index], self.r4[index],
                                          self.r5[index])

    def __len__(self) -> int:
        return min(len(self.pivot), 
                   len(self.s1), len(self.s2), 
                   len(self.s3), len(self.s4), len(self.s5),
                   len(self.r1), len(self.r2), 
                   len(self.r3), len(self.r4), len(self.r5))

    def __repr__(self) -> str:
        return (f"TraditionalPivotPoints(pivot={self.pivot}, "
                f"s1={self.s1}, s2={self.s2}, "
                f"s3={self.s3}, s4={self.s4}, s5={self.s5}, "
                f"r1={self.r1}, r2={self.r2}, "
                f"r3={self.r3}, r4={self.r4}, r5={self.r5})")


class ClassicPivotPoints(IndicatorResultSequence[ClassicPivotPointsItem]):
    def __init__(self, 
                 pivot,
                 s1: numpy.ndarray,
                 s2: numpy.ndarray,
                 s3: numpy.ndarray,
                 s4: numpy.ndarray,
                 r1: numpy.ndarray,
                 r2: numpy.ndarray,
                 r3: numpy.ndarray,
                 r4: numpy.ndarray):
        self.pivot = pivot
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4

    def __iter__(self) -> t.Iterator[ClassicPivotPointsItem]:
        zip_iter = zip(self.pivot, 
                       self.s1, self.s2, self.s3, self.s4, 
                       self.r1, self.r2, self.r3, self.r4)
        return (
            ClassicPivotPointsItem(*values) 
                for values in zip_iter
        )

    def __reversed__(self) -> t.Iterator[ClassicPivotPointsItem]:
        reversed_iter = zip(reversed(self.pivot), 
                            reversed(self.s1), reversed(self.s2), 
                            reversed(self.s3), reversed(self.s4), 
                            reversed(self.r1), reversed(self.r2), 
                            reversed(self.r3), reversed(self.r4))
        return (
            ClassicPivotPointsItem(*values) 
                for values in reversed_iter
        )

    def __getitem__(self, index: int) -> ClassicPivotPointsItem:
        return ClassicPivotPointsItem(self.pivot[index],
                                      self.s1[index], self.s2[index], 
                                      self.s3[index], self.s4[index], 
                                      self.r1[index], self.r2[index], 
                                      self.r3[index], self.r4[index])

    def __len__(self) -> int:
        return min(len(self.pivot), 
                   len(self.s1), len(self.s2), len(self.s3), len(self.s4),
                   len(self.r1), len(self.r2), len(self.r3), len(self.r4))

    def __repr__(self) -> str:
        return (f"ClassicPivotPoints(pivot={self.pivot}, "
                f"s1={self.s1}, s2={self.s2}, "
                f"s3={self.s3}, s4={self.s4}, "
                f"r1={self.r1}, r2={self.r2}, "
                f"r3={self.r3}, r4={self.r4})")


class FibonacciPivotPoints(IndicatorResultSequence[FibonacciPivotPointsItem]):
    def __init__(self, 
                 pivot,
                 s1: numpy.ndarray,
                 s2: numpy.ndarray,
                 s3: numpy.ndarray,
                 r1: numpy.ndarray,
                 r2: numpy.ndarray,
                 r3: numpy.ndarray):
        self.pivot = pivot
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

    def __iter__(self) -> t.Iterator[FibonacciPivotPointsItem]:
        zip_iter = zip(self.pivot, 
                       self.s1, self.s2, self.s3, 
                       self.r1, self.r2, self.r3)
        return (
            FibonacciPivotPointsItem(*values) 
                for values in zip_iter
        )

    def __reversed__(self) -> t.Iterator[FibonacciPivotPointsItem]:
        reversed_iter = zip(reversed(self.pivot), 
                            reversed(self.s1), reversed(self.s2), 
                            reversed(self.s3), 
                            reversed(self.r1), reversed(self.r2), 
                            reversed(self.r3))
        return (
            FibonacciPivotPointsItem(*values) 
                for values in reversed_iter
        )

    def __getitem__(self, index: int) -> FibonacciPivotPointsItem:
        return FibonacciPivotPointsItem(self.pivot[index],
                                        self.s1[index], self.s2[index], 
                                        self.s3[index], 
                                        self.r1[index], self.r2[index], 
                                        self.r3[index])

    def __len__(self) -> int:
        return min(len(self.pivot), 
                   len(self.s1), len(self.s2), len(self.s3),
                   len(self.r1), len(self.r2), len(self.r3))

    def __repr__(self) -> str:
        return (f"FibonacciPivotPoints(pivot={self.pivot}, "
                f"s1={self.s1}, s2={self.s2}, s3={self.s3}, "
                f"r1={self.r1}, r2={self.r2}, r3={self.r3})")


def typical_price(highs: pd.Series, lows: pd.Series, 
                        close: pd.Series) -> pd.Series:
    return (highs + lows + close) / 3


def pivot(market_data: MarketData, timeframe: Timeframes,
            period: int = 15) -> TraditionalPivotPoints:
    """
    Tradtional Pivot Points.
    https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/

    Represents significant support and resistance levels 
    that can be used to determine potential trades.
    The pivot points come as a technical analysis indicator
    calculated using a security’s high, low, and close.
    """
    quantity = period + 1
    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = highs[:-1]   # or 1:?
    highs = pd.Series(highs, dtype=numpy.float64)

    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = lows[:-1]
    lows = pd.Series(lows, dtype=numpy.float64)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = close[:-1]
    close = pd.Series(close, dtype=numpy.float64)

    pivot = typical_price(highs, lows, close)  
    # TRADITIONAL
    s1 = (pivot * 2) - highs
    s2 = pivot - (highs - lows)
    s3 = lows - (2 * (highs - pivot))
    s4 = lows - (3 * (highs - pivot))
    s5 = lows - (4 * (highs - pivot))

    r1 = (pivot * 2) - lows
    r2 = pivot + (highs - lows)
    r3 = highs + (2 * (pivot - lows))
    r4 = highs + (3 * (pivot - lows))
    r5 = highs + (4 * (pivot - lows))

    return TraditionalPivotPoints(pivot.values, 
                                  s1.values, s2.values, s3.values,
                                  s4.values, s5.values, 
                                  r1.values, r2.values, r3.values, 
                                  r4.values, r5.values)


def pivot_fib(market_data: MarketData, timeframe: Timeframes, 
                period: int = 15) -> FibonacciPivotPoints:
    """
    Fibonacci Pivot Points.
    https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/

    Represents significant support and resistance levels 
    that can be used to determine potential trades.
    The pivot points come as a technical analysis indicator
    calculated using a security’s high, low, and close.
    """
    quantity = period + 1
    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = highs[:-1]   # or 1:?
    highs = pd.Series(highs, dtype=numpy.float64)

    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = lows[:-1]
    lows = pd.Series(lows, dtype=numpy.float64)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = close[:-1]
    close = pd.Series(close, dtype=numpy.float64)

    pivot = typical_price(highs, lows, close)
        # FIBONACCI
    s1 = pivot - 0.382 * (highs - lows)
    s2 = pivot - 0.618 * (highs - lows)
    s3 = pivot - (highs - lows)

    r1 = pivot + 0.382 * (highs - lows)
    r2 = pivot + 0.618 * (highs - lows)
    r3 = pivot + (highs - lows)

    return FibonacciPivotPoints(pivot.values, 
                                s1.values, s2.values, s3.values,
                                r1.values, r2.values, r3.values)


def pivot_classic(market_data: MarketData, timeframe: Timeframes, 
                    period: int = 15) -> ClassicPivotPoints:
    """
    Classic Pivot Points.
    https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/

    Represents significant support and resistance levels 
    that can be used to determine potential trades.
    The pivot points come as a technical analysis indicator
    calculated using a security’s high, low, and close.
    """
    quantity = period + 1
    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = highs[:-1]   # or 1:?
    highs = pd.Series(highs, dtype=numpy.float64)

    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = lows[:-1]
    lows = pd.Series(lows, dtype=numpy.float64)

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = close[:-1]
    close = pd.Series(close, dtype=numpy.float64)

    pivot = typical_price(highs, lows, close)        
    # CLASSIC
    s1 = (pivot * 2) - highs
    s2 = pivot - (highs - lows)
    s3 = pivot - 2 * (highs - lows)
    s4 = pivot - 3 * (highs - lows)

    r1 = (pivot * 2) - lows
    r2 = pivot + (highs - lows)
    r3 = pivot + 2 * (highs - lows)
    r4 = pivot + 3 * (highs - lows)

    return ClassicPivotPoints(pivot.values, 
                              s1.values, s2.values, 
                              s3.values, s4.values, 
                              r1.values, r2.values, 
                              r3.values, r4.values)


class PivotOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, period: int = 15):
        self.timeframe = timeframe
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'PIVOT'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, 
                    candle_property=self.candle_property, period=self.period)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, HIGH, self.period + 1),
            InputTimeseries(self.timeframe, LOW, self.period + 1),
            InputTimeseries(self.timeframe, CLOSE, self.period + 1),
        )


class FibonacciPivotOptions(PivotOptions):
    @property
    def indicator_name(self) -> str:
        return 'PIVOT_FIB'


class ClassicPivotOptions(PivotOptions):
    @property
    def indicator_name(self) -> str:
        return 'PIVOT_CLASSIC'
