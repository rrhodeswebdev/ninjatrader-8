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
class VwmaResultItem:
    vwma: numpy.float64

class VwmaResultSequence(IndicatorResultSequence[VwmaResultItem]):
    def __init__(self,
                 vwma: numpy.ndarray):
        self.vwma = vwma

    def __iter__(self) -> t.Iterator[VwmaResultItem]:
        zip_iter = zip(self.vwma)
        return (
            VwmaResultItem(vwma) 
                for vwma in zip_iter
        )

    def __reversed__(self) -> t.Iterator[VwmaResultItem]:
        reversed_iter = zip(reversed(self.vwma))
        return (
            VwmaResultItem(vwma) 
                for vwma in reversed_iter
        )

    def __getitem__(self, index: int) -> VwmaResultItem:
        return VwmaResultItem(self.vwma[index])

    def __len__(self) -> int:
        return min(len(self.vwma))

    def __repr__(self) -> str:
        return (f"VwmaResultSequence(vwma={self.vwma})")
            
def vwma(market_data: MarketData, 
        timeframe: Timeframes,
        candle_property: CandleProperties = VOLUME,
        period: int = 8) -> numpy.ndarray:
    """The vwma function returns the volume-weighted moving average of CandleProperties for period bars back."""
    source_values = market_data.get_values(timeframe, CandleProperties, period)
    source_values = pd.Series(source_values)
    volume_values = market_data.get_values(timeframe, CandleProperties, period)
    volume_values = pd.Series(volume_values)
    
    sma_source_volume = ta.trend.SMAIndicator(source_values * volume_values, period).sma_indicator()
    print(f'vwma: sma_source_volume: {sma_source_volume.values}')
    sma_volume = ta.trend.SMAIndicator(volume_values, period).sma_indicator()
    print(f'vwma: sma_volume: {sma_volume.values}')
    vwma = sma_source_volume.values/sma_volume.values
    print(f'vwma: vwma.values: {vwma.values}')
    return VwmaResultSequence(vwma.values)


class VwmaOptions(IndicatorOptions):
    def __init__(self, timeframe: Timeframes, 
                 candle_property: CandleProperties = VOLUME, 
                 period: int = 8):
        self.timeframe = timeframe
        self.candle_property = candle_property
        self.period = period

    @property
    def indicator_name(self) -> str:
        return 'VWMA'

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
