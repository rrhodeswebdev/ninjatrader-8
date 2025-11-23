import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.pivot import (ClassicPivotOptions,
                                                  FibonacciPivotOptions,
                                                  PivotOptions)
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


# Traditional Pivot
class TraditionalPivotPointsExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int, attr: str):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { PivotOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        result = analyser.pivot(self.timeframe, self.period)
        return Decimal(getattr(result[self.index], self.attr))


class TraditionalPivotFactoryWithIndex:
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index

    @property
    def pivot(self) -> TraditionalPivotPointsExpression:
        attr = 'pivot'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def s1(self) -> TraditionalPivotPointsExpression:
        attr = 's1'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def s2(self) -> TraditionalPivotPointsExpression:
        attr = 's2'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def s3(self) -> TraditionalPivotPointsExpression:
        attr = 's3'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def s4(self) -> TraditionalPivotPointsExpression:
        attr = 's4'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def s5(self) -> TraditionalPivotPointsExpression:
        attr = 's5'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def r1(self) -> TraditionalPivotPointsExpression:
        attr = 'r1'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def r2(self) -> TraditionalPivotPointsExpression:
        attr = 'r2'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def r3(self) -> TraditionalPivotPointsExpression:
        attr = 'r3'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def r4(self) -> TraditionalPivotPointsExpression:
        attr = 'r4'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)

    @property
    def r5(self) -> TraditionalPivotPointsExpression:
        attr = 'r5'
        return TraditionalPivotPointsExpression(
                    self.timeframe, 
                    self.period, 
                    self.index, attr)
    

class TraditionalPivotFactory:
    def __init__(self, timeframe: Timeframes, period: int = 15):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index):
        return TraditionalPivotFactoryWithIndex(
                    self.timeframe, self.period, index)
# End of Traditional Pivot

# Fibonacci Pivot Points
class FibonacciPivotPointsExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int, attr: str):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { FibonacciPivotOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        result = analyser.pivot_fib(self.timeframe, self.period)
        return Decimal(getattr(result[self.index], self.attr))


class FibonacciPivotFactoryWithIndex:
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index

    @property
    def pivot(self) -> FibonacciPivotPointsExpression:
        attr = 'pivot'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s1(self) -> FibonacciPivotPointsExpression:
        attr = 's1'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s2(self) -> FibonacciPivotPointsExpression:
        attr = 's2'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s3(self) -> FibonacciPivotPointsExpression:
        attr = 's3'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r1(self) -> FibonacciPivotPointsExpression:
        attr = 'r1'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r2(self) -> FibonacciPivotPointsExpression:
        attr = 'r2'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r3(self) -> FibonacciPivotPointsExpression:
        attr = 'r3'
        return FibonacciPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)


class FibonacciPivotFactory:
    def __init__(self, timeframe: Timeframes, period: int = 15):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index):
        return FibonacciPivotFactoryWithIndex(
                    self.timeframe,
                    self.period, index)
# End of Fibonacci Pivot

# Classic Pivot Points
class ClassicPivotPointsExpression(Primitive):
    def __init__(self, timeframe: Timeframes, period: int, index: int, attr: str):
        self.timeframe = timeframe
        self.period = period
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return { ClassicPivotOptions(self.timeframe, self.period) }

    def eval(self, broker, analyser, candles):
        result = analyser.pivot_classic(self.timeframe, self.period)
        return Decimal(getattr(result[self.index], self.attr))


class ClassicPivotFactoryWithIndex:
    def __init__(self, timeframe: Timeframes, period: int, index: int):
        self.timeframe = timeframe
        self.period = period
        self.index = index

    @property
    def pivot(self) -> ClassicPivotPointsExpression:
        attr = 'pivot'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s1(self) -> ClassicPivotPointsExpression:
        attr = 's1'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s2(self) -> ClassicPivotPointsExpression:
        attr = 's2'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s3(self) -> ClassicPivotPointsExpression:
        attr = 's3'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def s4(self) -> ClassicPivotPointsExpression:
        attr = 's4'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r1(self) -> ClassicPivotPointsExpression:
        attr = 'r1'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r2(self) -> ClassicPivotPointsExpression:
        attr = 'r2'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r3(self) -> ClassicPivotPointsExpression:
        attr = 'r3'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)

    @property
    def r4(self) -> ClassicPivotPointsExpression:
        attr = 'r4'
        return ClassicPivotPointsExpression(
                    self.timeframe,
                    self.period,
                    self.index, attr)


class ClassicPivotFactory:
    def __init__(self, timeframe: Timeframes, period: int = 15):
        self.timeframe = timeframe
        self.period = period

    def __getitem__(self, index):
        return ClassicPivotFactoryWithIndex(
                    self.timeframe,
                    self.period, index)
# End of Classic Pivot Points