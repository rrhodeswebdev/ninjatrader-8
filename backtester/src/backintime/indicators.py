"""Aliases for declaring indciators."""
from .analyser.indicators.adx import AdxOptions
from .analyser.indicators.atr import AtrOptions
from .analyser.indicators.bbands import BbandsOptions
from .analyser.indicators.dmi import DmiOptions
from .analyser.indicators.ema import EmaOptions
from .analyser.indicators.keltner_channel import KeltnerChannelOptions
from .analyser.indicators.slope import SlopeOptions
from .analyser.indicators.macd import MacdOptions
from .analyser.indicators.pivot import (ClassicPivotOptions,
                                        FibonacciPivotOptions, PivotOptions)
from .analyser.indicators.rsi import RsiOptions
from .analyser.indicators.sma import SmaOptions
from .analyser.indicators.vwema import VwEmaOptions
from .analyser.indicators.vwma import VwmaOptions

ADX = AdxOptions
ATR = AtrOptions
BBANDS = BbandsOptions
DMI = DmiOptions
EMA = EmaOptions
MACD = MacdOptions
RSI = RsiOptions
SMA = SmaOptions
VWEMA = VwEmaOptions
VWMA = VwmaOptions
KELTNER_CHANNEL = KeltnerChannelOptions
SLOPE = SlopeOptions
PIVOT = PivotOptions
PIVOT_FIB = FibonacciPivotOptions
PIVOT_CLASSIC = ClassicPivotOptions