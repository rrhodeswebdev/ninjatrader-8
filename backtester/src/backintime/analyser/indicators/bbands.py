import typing as t
from dataclasses import dataclass

import numpy
import pandas as pd
import ta

from backintime.timeframes import Timeframes

from .base import (IndicatorOptions, IndicatorResultSequence, InputTimeseries,
                   MarketData)
from .constants import CLOSE, CandleProperties


@dataclass
class BbandsResultItem:
    upper_band: numpy.float64
    middle_band: numpy.float64
    lower_band: numpy.float64


class BbandsResultSequence(IndicatorResultSequence[BbandsResultItem]):
    def __init__(self, 
                 upper_band: numpy.ndarray, 
                 middle_band: numpy.ndarray, 
                 lower_band: numpy.ndarray):
        self.upper_band = upper_band
        self.middle_band = middle_band
        self.lower_band = lower_band

    def __iter__(self) -> t.Iterator[BbandsResultItem]:
        zip_iter = zip(self.upper_band, self.middle_band, self.lower_band)
        return (
            BbandsResultItem(upper, middle, lower) 
                for upper, middle, lower in zip_iter
        )

    def __reversed__(self) -> t.Iterator[BbandsResultItem]:
        reversed_iter = zip(reversed(self.upper_band), 
                            reversed(self.middle_band), 
                            reversed(self.lower_band))
        return (
            BbandsResultItem(upper, middle, lower) 
                for upper, middle, lower in reversed_iter
        )

    def __getitem__(self, index: int) -> BbandsResultItem:
        return BbandsResultItem(self.upper_band[index], 
                                self.middle_band[index], 
                                self.lower_band[index])

    def __len__(self) -> int:
        return min(len(self.upper_band), 
                   len(self.middle_band), 
                   len(self.lower_band))

    def __repr__(self) -> str:
        return (f"BbandsResultSequence(upper_band={self.upper_band}, "
                f"middle_band={self.middle_band}, "
                f"lower_band={self.lower_band})")


def bbands(market_data: MarketData, 
           timeframe: Timeframes,
           candle_property: CandleProperties = CLOSE,
           period: int = 20,
           deviation_quotient: int = 2) -> BbandsResultSequence:
    """
    Bollinger Bands (BBANDS).

    Bollinger Bands are volatility bands placed above 
    and below a moving average.
    Volatility is based on the standard deviation, 
    which changes as volatility increases and decreases.
    The bands automatically widen when volatility increases
    and narrow when volatility decreases.
    """
    quantity = period**2
    values = market_data.get_values(timeframe, candle_property, quantity)
    values = pd.Series(values)

    bbands = ta.volatility.BollingerBands(values, period, 
                                          deviation_quotient)
    upper_band = bbands.bollinger_hband().values
    middle_band = bbands.bollinger_mavg().values
    lower_band = bbands.bollinger_lband().values

    return BbandsResultSequence(upper_band, middle_band, lower_band)


class BbandsOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, 
                 candle_property: CandleProperties = CLOSE, 
                 period: int = 20,
                 deviation_quotient: int = 2):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period
        self.deviation_quotient = deviation_quotient

    @property
    def indicator_name(self) -> str:
        return 'BBANDS'

    @property
    def indicator_options(self) -> dict:
        return dict(timeframe=self.timeframe, 
                    candle_property=self.candle_property, period=self.period, 
                    deviation_quotient=self.deviation_quotient)

    @property
    def input_timeseries(self) -> t.Tuple[InputTimeseries]:
        return (
            InputTimeseries(self.timeframe, 
                            self.candle_property, self.period**2),
        )
