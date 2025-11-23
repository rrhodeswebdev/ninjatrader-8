import functools

from backintime.analyser.indicators.constants import CLOSE, HIGH, LOW, OPEN

from .base import ArithmeticExpression, BooleanExpression, Primitive
from .declarative import DeclarativeStrategy
from .expressions import *

# Shortcuts
scalar = ScalarExpression

ADX = AdxFactory
ATR = AtrFactory
BBANDS = BbandsFactory
DMI = DmiFactory
EMA = EmaFactory
SLOPE = SlopeFactory
KELTNER = KeltnerChannelFactory
MACD = MacdFactory
PIVOT = TraditionalPivotFactory
PIVOT_FIB = FibonacciPivotFactory
PIVOT_CLASSIC = ClassicPivotFactory
RSI = RsiFactory
SMA = SmaFactory
VWEMA = VwemaFactory
VWMA = VwmaFactory

TP = TakeProfitExpression
SL = StopLossExpression

class prices:
    open = functools.partial(PriceExpression, OPEN)
    high = functools.partial(PriceExpression, HIGH)
    low = functools.partial(PriceExpression, LOW)
    close = functools.partial(PriceExpression, CLOSE)
    market = MarketPriceStub()

