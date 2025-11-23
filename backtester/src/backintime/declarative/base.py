import functools
import operator
import typing as t
from decimal import Decimal
from itertools import zip_longest

from backintime import FuturesStrategy
from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.broker.base import (LimitOrderOptions, MarketOrderOptions,
                                    OrderSide, StopLossOptions,
                                    TakeProfitOptions)
from backintime.timeframes import Timeframes
from backintime.timeframes import Timeframes as tf


def merge_indicators_meta(some, other=None) -> t.Set[IndicatorOptions]:
    other_indicators = other.indicators_meta if other else set()
    return some.indicators_meta | other_indicators


def merge_ohlcv_timeframes(fst, snd=None) -> dict:
    if not snd:
        return fst.ohlcv_timeframes

    fst = fst.ohlcv_timeframes
    snd = snd.ohlcv_timeframes if snd else dict()

    output = dict()
    iter_items = zip_longest(fst.items(), snd.items(), fillvalue=(None, None))

    for (fst_key, fst_value), (snd_key, snd_value) in iter_items:
        if fst_key:
            output[fst_key] = max(fst_value, snd.get(fst_key, -1))
        if snd_key:
            output[snd_key] = max(snd_value, fst.get(snd_key, -1))
    return output


class BooleanExpression:  # boolean ops and chaining
    """
    Boolean ops & chaining

    BooleanExpressions are built by applying overloaded logical ops 
    to Primitives (scalar, price, indicator), 
    ArithmeticExpressions, or other BooleanExpressions.
    """
    def __init__(self, op, left, right=None):
        self.op=op
        self.left=left
        self.right=right
        self._ohlcv_timeframes=merge_ohlcv_timeframes(left, right)
        self._indicators_meta=merge_indicators_meta(left, right)

    def __and__(self, other):
        return BooleanExpression(operator.__and__, self, other)

    def __or__(self, other):
        return BooleanExpression(operator.__or__, self, other)

    def __le__(self, other):
        return BooleanExpression(operator.le, self, other)

    def __lt__(self, other):
        return BooleanExpression(operator.lt, self, other)

    def __ge__(self, other):
        return BooleanExpression(operator.ge, self, other)

    def __gt__(self, other):
        return BooleanExpression(operator.gt, self, other)

    def __eq__(self, other):
        return BooleanExpression(operator.eq, self, other)

    def __invert__(self):
        return BooleanExpression(operator.__not__, self)
    
    @property
    def ohlcv_timeframes(self):
        return self._ohlcv_timeframes

    @property
    def indicators_meta(self):
        return self._indicators_meta

    def eval(self, broker, analyser, candles) -> bool:
        if self.right is None:
            return self.op(self.left.eval(broker, analyser, candles))
        else:
            try:
                return self.op(self.left.eval(broker, analyser, candles), self.right.eval(broker, analyser, candles))
            except Exception as e:
                print(f"OP: {self.op}")
                print(f"Left: {self.left}") 
                print(f"Right: {self.right}")
                raise e


class ArithmeticExpression: 
    """
    Arithmetic ops & chaining.

    ArithmeticExpressions are built by applying overloaded arithmetic ops
    to Primitives (scalar, price, indicator), or other ArithmeticExpressions.
    """
    def __init__(self, op, left, right=None):
        self.op=op
        self.left=left
        self.right=right
        self._ohlcv_timeframes=merge_ohlcv_timeframes(left, right)
        self._indicators_meta=merge_indicators_meta(left, right)

    def __and__(self, other):
        return BooleanExpression(operator.__and__, self, other)

    def __or__(self, other):
        return BooleanExpression(operator.__or__, self, other)

    def __le__(self, other):
        return BooleanExpression(operator.le, self, other)

    def __lt__(self, other):
        return BooleanExpression(operator.lt, self, other)

    def __ge__(self, other):
        return BooleanExpression(operator.ge, self, other)

    def __gt__(self, other):
        return BooleanExpression(operator.gt, self, other)

    def __eq__(self, other):
        return BooleanExpression(operator.eq, self, other)

    def __abs__(self):
        return ArithmeticExpression(operator.abs, self)

    def __neg__(self):
        return ArithmeticExpression(operator.neg, self)
    
    def __add__(self, other):
        return ArithmeticExpression(operator.add, self, other)

    def __sub__(self, other):
        return ArithmeticExpression(operator.sub, self, other)

    def __mul__(self, other):
        return ArithmeticExpression(operator.mul, self, other)

    def __truediv__(self, other):
        return ArithmeticExpression(operator.truediv, self, other)

    @property
    def ohlcv_timeframes(self):
        return self._ohlcv_timeframes

    @property
    def indicators_meta(self):
        return self._indicators_meta
    
    def eval(self, broker, analyser, candles) -> bool:
        if self.right is None:
            return self.op(self.left.eval(broker, analyser, candles))
        else:
            return self.op(self.left.eval(broker, analyser, candles), self.right.eval(broker, analyser, candles))


class Primitive:
    """
    Rich comparable and arithmetic, evaluates to scalar numeric value.

    Applying overloaded arithmetic ops creates new ArithmeticExpression
    Applying overloaded logical ops creates new BooleanExpression
    """
    def __init__(self, concrete):
        self.concrete=concrete

    def __abs__(self):
        return ArithmeticExpression(operator.abs, self)

    def __neg__(self):
        return ArithmeticExpression(operator.neg, self)
    
    def __add__(self, other):
        return ArithmeticExpression(operator.add, self, other)

    def __sub__(self, other):
        return ArithmeticExpression(operator.sub, self, other)

    def __mul__(self, other):
        return ArithmeticExpression(operator.mul, self, other)

    def __truediv__(self, other):
        return ArithmeticExpression(operator.truediv, self, other)

    def __le__(self, other) -> BooleanExpression:
        return BooleanExpression(operator.le, self, other)

    def __lt__(self, other):
        return BooleanExpression(operator.lt, self, other)

    def __ge__(self, other):
        return BooleanExpression(operator.ge, self, other)

    def __gt__(self, other):
        return BooleanExpression(operator.gt, self, other)

    def __eq__(self, other):
        return BooleanExpression(operator.eq, self, other)

    @property
    def ohlcv_timeframes(self) -> dict:
        return dict()

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return set()

    def eval(self, broker, analyser, candles):
        return self.concrete.eval(broker, analyser, candles)
