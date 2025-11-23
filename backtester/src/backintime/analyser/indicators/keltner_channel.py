import typing as t
from dataclasses import dataclass

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .atr import atr
from .base import (IndicatorOptions, IndicatorResultSequence, InputTimeseries,
                   MarketData)
from .constants import CLOSE, HIGH, LOW, OPEN, CandleProperties


@dataclass
class KeltnerChannelResultItem:
    upper_band: numpy.float64
    middle_band: numpy.float64
    lower_band: numpy.float64


class KeltnerChannelResultSequence(IndicatorResultSequence[KeltnerChannelResultItem]):
    def __init__(self, 
                 upper_band: numpy.ndarray, 
                 middle_band: numpy.ndarray, 
                 lower_band: numpy.ndarray):
        self.upper_band = upper_band
        self.middle_band = middle_band
        self.lower_band = lower_band

    def __iter__(self) -> t.Iterator[KeltnerChannelResultItem]:
        zip_iter = zip(self.upper_band, self.middle_band, self.lower_band)
        return (
            KeltnerChannelResultItem(upper, middle, lower) 
                for upper, middle, lower in zip_iter
        )

    def __reversed__(self) -> t.Iterator[KeltnerChannelResultItem]:
        reversed_iter = zip(reversed(self.upper_band), 
                            reversed(self.middle_band), 
                            reversed(self.lower_band))
        return (
            KeltnerChannelResultItem(upper, middle, lower) 
                for upper, middle, lower in reversed_iter
        )

    def __getitem__(self, index: int) -> KeltnerChannelResultItem:
        return KeltnerChannelResultItem(self.upper_band[index], 
                                        self.middle_band[index], 
                                        self.lower_band[index])

    def __len__(self) -> int:
        return min(len(self.upper_band), 
                   len(self.middle_band), 
                   len(self.lower_band))

    def __repr__(self) -> str:
        return (f"KeltnerChannelResultSequence(upper_band={self.upper_band}, "
                f"middle_band={self.middle_band}, "
                f"lower_band={self.lower_band})")


def keltner_channel(market_data: MarketData, 
                    timeframe: Timeframes,
                    candle_property: CandleProperties = CLOSE,
                    period: int = 20,
                    atr_period: int = 10,
                    multiplier: int = 2) -> KeltnerChannelResultSequence:
    """
    """
    quantity = period**2
    
    highs = market_data.get_values(timeframe, HIGH, quantity)
    highs = pd.Series(map(lambda x: float(x), highs))

    lows = market_data.get_values(timeframe, LOW, quantity)
    lows = pd.Series(map(lambda x: float(x), lows))

    close = market_data.get_values(timeframe, CLOSE, quantity)
    close = pd.Series(map(lambda x: float(x), close))

    if candle_property is OPEN:
        open_ = market_data.get_values(timeframe, OPEN, quantity)
        open_ = pd.Series(map(lambda x: float(x), open_))
        middle_band = open_.ewm(span=period, 
                                min_periods=period, 
                                adjust=False).mean()
    elif candle_property is HIGH:
        middle_band = highs.ewm(span=period, min_periods=period, adjust=False).mean()
    elif candle_property is LOW:
        middle_band = lows.ewm(span=period, min_periods=period, adjust=False).mean()
    else:   # CLOSE 
        middle_band = close.ewm(span=period, min_periods=period, adjust=False).mean()

    atr_ = ta.volatility.AverageTrueRange(highs, lows, close, atr_period).average_true_range()
    upper_band = middle_band + (multiplier * atr_)
    lower_band = middle_band - (multiplier * atr_)
    return KeltnerChannelResultSequence(upper_band.values, middle_band.values, lower_band.values)


class KeltnerChannelOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes,
                 candle_property: CandleProperties = CLOSE,
                 period: int = 20,
                 atr_period: int = 10,
                 multiplier: int = 2):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier

    @property
    def indicator_name(self) -> str:
        return 'KELTNER_CHANNEL'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe,
                    candle_property=self.candle_property,
                    period=self.period,
                    atr_period=self.atr_period,
                    multiplier=self.multiplier)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, OPEN, self.period**2),
            InputTimeseries(self.timeframe, HIGH, self.period**2),
            InputTimeseries(self.timeframe, LOW, self.period**2),
            InputTimeseries(self.timeframe, CLOSE, self.period**2)
        )
