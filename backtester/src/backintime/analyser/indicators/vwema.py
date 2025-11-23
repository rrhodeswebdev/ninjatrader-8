import typing as t
from dataclasses import dataclass

import numpy
import pandas as pd
import ta
import ta.volume

from backintime.timeframes import Timeframes

from .base import (IndicatorOptions, IndicatorResultSequence, InputTimeseries,
                   MarketData)
from .constants import CLOSE, VOLUME, CandleProperties

@dataclass
class VwemaResultItem:
    vwema: numpy.float64

class VwemaResultSequence(IndicatorResultSequence[VwemaResultItem]):
    def __init__(self,
                 vwema: numpy.ndarray):
        self.vwema = vwema

    def __iter__(self) -> t.Iterator[VwemaResultItem]:
        zip_iter = zip(self.vwema)
        return (
            VwemaResultItem(vwema) 
                for vwema in zip_iter
        )

    def __reversed__(self) -> t.Iterator[VwemaResultItem]:
        reversed_iter = zip(reversed(self.vwema))
        return (
            VwemaResultItem(vwema) 
                for vwema in reversed_iter
        )

    def __getitem__(self, index: int) -> VwemaResultItem:
        return VwemaResultItem(self.vwema[index])

    def __len__(self) -> int:
        return min(len(self.vwema))

    def __repr__(self) -> str:
        return (f"VwemaResultSequence(vwema={self.vwema})")
    
def vwema(market_data: MarketData, 
        timeframe: Timeframes,
        candle_property: CandleProperties = CLOSE,
        period: int = 8) -> numpy.ndarray:
    """The vwema function returns ema of the volume-weighted moving average of CandleProperties for period bars back."""
    source_values = market_data.get_values(timeframe, CandleProperties, period)
    source_values = pd.Series(source_values)
    volume_values = market_data.get_values(timeframe, VOLUME, period)
    volume_values = pd.Series(volume_values)
    
    sma_source_volume = ta.trend.SMAIndicator(source_values * volume_values, period).sma_indicator()
    print(f'vwema: sma_source_volume: {sma_source_volume.values}')
    sma_volume = ta.trend.SMAIndicator(volume_values, period).sma_indicator()
    print(f'vwema: sma_volume: {sma_volume.values}')
    vwma = sma_source_volume.values/sma_volume.values
    print(f'vwema: vwma.values: {vwma.values}')
    vwema = ta.trend.EMAIndicator(vwma.values, period).ema_indicator()
    print(f'vwema: vwema.values: {vwema.values}')
    return vwema.values


class VwEmaOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, 
                 candle_property: CandleProperties = CLOSE, 
                 period: int = 8):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'VWEMA'

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
