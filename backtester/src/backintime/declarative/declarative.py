import functools
import operator
import typing as t
from dataclasses import dataclass
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

from .base import ArithmeticExpression, BooleanExpression, Primitive
from .expressions import (MarketPriceStub, StopLossExpression,
                          TakeProfitExpression)


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    b = next(iterator, None)

    if not a:
        return
    if not b:
        yield a, b
        return

    yield a, b
    a = b
    
    for b in iterator:
        yield a, b
        a = b


def merge_ohlcv_timeframes(fst, snd=None):
    if not snd:
        snd = dict()

    output = dict()
    iter_items = zip_longest(fst.items(), snd.items(), fillvalue=(None, None))

    for (fst_key, fst_value), (snd_key, snd_value) in iter_items:
        if fst_key:
            output[fst_key] = max(fst_value, snd.get(fst_key, -1))
        if snd_key:
            output[snd_key] = max(snd_value, fst.get(snd_key, -1))
    return output


class DeclarativeStrategy:
    """
    DeclarativeStrategy is used to implement a strategy providing
    high level expressions.

    Expression objects can be evaluated in runtime and they produce different
    results depending on market conditions.

    BooleanExpressions are used to identify BUY/SELL signals.

    For Long Entry/Short Entry the expression must return order price.
    Thus, it accepts any expression that's evaluated to a numeric value
    plus MarketPriceStub, a special object that implies Market Order.

    For Long Exit/Short Exit the expression returns a sequence of TP/SL.

    All expressions carry hints about their dependencies:
    timeframes of OHLC prices and input timeseries specification
    of indicators used. These hints are used by the engine to make sure
    it will be able to evaluate the expressions at runtime.
    An expression chain (a nested expression) combines all the hints 
    of the inner expressions.
    """
    title: str

    buy_signal:     BooleanExpression
    sell_signal:    BooleanExpression
    long_entry:     t.Union[ArithmeticExpression, Primitive, MarketPriceStub]=MarketPriceStub()
    long_exit:      t.Optional[t.Tuple[t.Union[TakeProfitExpression, StopLossExpression]]]=None
    short_entry:    t.Union[ArithmeticExpression, Primitive, MarketPriceStub]=MarketPriceStub()
    short_exit:     t.Optional[t.Tuple[t.Union[TakeProfitExpression, StopLossExpression]]]=None
    # Used to distinguish different runs of the optimizer (will be logged to CSV)
    params: str=""

    @classmethod
    def get_candle_timeframes(cls) -> dict:
        if cls.long_exit and cls.short_exit:
            tfs = [
                cls.buy_signal.ohlcv_timeframes,
                cls.sell_signal.ohlcv_timeframes,
                cls.long_entry.ohlcv_timeframes,
                cls.long_exit[0].ohlcv_timeframes,
                cls.long_exit[1].ohlcv_timeframes,
                cls.short_entry.ohlcv_timeframes,
                cls.short_exit[0].ohlcv_timeframes,
                cls.short_exit[1].ohlcv_timeframes
            ]

            res = dict()
            for fst, snd in pairwise(tfs):
                # print(fst, snd)
                tmp = merge_ohlcv_timeframes(fst, snd)
                res = merge_ohlcv_timeframes(res, tmp)
                # print(f"merged: {res}")
            return res
        else:
            tfs = [
                cls.buy_signal.ohlcv_timeframes,
                cls.sell_signal.ohlcv_timeframes,
                cls.long_entry.ohlcv_timeframes,
                
                cls.short_entry.ohlcv_timeframes,
                
            ]

            res = dict()
            for fst, snd in pairwise(tfs):
                # print(fst, snd)
                tmp = merge_ohlcv_timeframes(fst, snd)
                res = merge_ohlcv_timeframes(res, tmp)
                # print(f"merged: {res}")
            return res

    @classmethod
    def get_indicators_meta(cls) -> t.Set[IndicatorOptions]:
        if cls.long_exit and cls.short_exit:
            return cls.buy_signal.indicators_meta | \
                    cls.sell_signal.indicators_meta | \
                    cls.long_entry.indicators_meta | \
                    cls.long_exit[0].indicators_meta | \
                    cls.long_exit[1].indicators_meta | \
                    cls.short_entry.indicators_meta | \
                    cls.short_exit[0].indicators_meta | \
                    cls.short_exit[1].indicators_meta
        else:
            print(cls.buy_signal)
            print(cls.buy_signal.indicators_meta)
            print(cls.sell_signal)
            print(cls.sell_signal.indicators_meta)
            print(cls.long_entry.indicators_meta)
            print(cls.short_entry.indicators_meta)

            return cls.buy_signal.indicators_meta | \
                    cls.sell_signal.indicators_meta | \
                    cls.long_entry.indicators_meta | \
                    cls.short_entry.indicators_meta

    @classmethod
    def repr_indicators(cls) -> str:
        return ', '.join([ repr(opts) for opts in cls.get_indicators_meta() ])


def enter_long(expr, broker, analyser, candles):
    order_price = expr.long_entry.eval(broker, analyser, candles)
    # Market price
    if isinstance(order_price, MarketPriceStub):
        market_buy = MarketOrderOptions(OrderSide.BUY, percentage_amount=Decimal('100'))
        broker.submit_market_order(market_buy)
    # Limit price
    else:       
        if not expr.long_exit:
            raise Exception('Expression for Long Exit must be provided')

        tp_opt, sl_opt = None, None
        for exp in expr.long_exit:
            if isinstance(exp, TakeProfitExpression):
                tp_opt = exp.eval(broker, analyser, candles)
            elif isinstance(exp, StopLossExpression):
                sl_opt = exp.eval(broker, analyser, candles)

        limit_buy = LimitOrderOptions(OrderSide.BUY, 
                        order_price, percentage_amount=Decimal('100'),
                        take_profit=tp_opt, stop_loss=sl_opt)
        broker.submit_limit_order(limit_buy)


def enter_short(expr, broker, analyser, candles):
    order_price = expr.short_entry.eval(broker, analyser, candles)
    # Market price
    if isinstance(order_price, MarketPriceStub):
        market_sell = MarketOrderOptions(OrderSide.SELL, percentage_amount=Decimal('100'))
        broker.submit_market_short(market_sell)
    # Limit price
    else:
        if not expr.short_exit:
            raise Exception('Expression for Short Exit must be provided')

        tp_opt, sl_opt = None, None
        for exp in expr.short_exit:
            if isinstance(exp, TakeProfitExpression):
                tp_opt = exp.eval(broker, analyser, candles)
            elif isinstance(exp, StopLossExpression):
                sl_opt = exp.eval(broker, analyser, candles)

        limit_sell = LimitOrderOptions(OrderSide.SELL, 
                        order_price, percentage_amount=Decimal('100'),
                        take_profit=tp_opt, stop_loss=sl_opt)
        broker.submit_limit_short(limit_sell)

'''
def generic_strategy(expr, broker, analyser, candles):
    """Generic strategy implementation for expressions."""
    if broker.max_funds_for_futures:  # Got funds for at least 1 contract?
        # Long/No position
        if not broker.in_short:
            if expr.buy_signal.eval(broker, analyser, candles):
                enter_long(expr, broker, analyser, candles)
        # Short/No position
        if not broker.in_long:
            if expr.sell_signal.eval(broker, analyser, candles):
                enter_short(expr, broker, analyser, candles)
'''

def generic_strategy(expr, broker, analyser, candles):
    """Generic strategy implementation for expressions."""
        # Long
    if broker.in_long:
        print("In Long")
        if broker.max_funds_for_futures:
            if expr.buy_signal.eval(broker, analyser, candles):
                enter_long(expr, broker, analyser, candles)
        # If No TP/SL for Long Exit, check SELL signal for exit
        if expr.long_exit is None:
            if broker._bm.available_contracts:
                if expr.sell_signal.eval(broker, analyser, candles):
                    market_sell = MarketOrderOptions(OrderSide.SELL, percentage_amount=Decimal('100'))
                    broker.submit_market_order(market_sell)
    # Short
    elif broker.in_short:
        print("In Short")
        if broker.max_funds_for_futures:
            if expr.sell_signal.eval(broker, analyser, candles):
                enter_short(expr, broker, analyser, candles)
        # If no TP/SL for Short Exit,check BUY signal for exit
        if expr.short_exit is None:
            if broker.max_funds_for_futures:
                if expr.buy_signal.eval(broker, analyser, candles):
                    market_buy = MarketOrderOptions(OrderSide.BUY, percentage_amount=Decimal('100'))
                    broker.submit_market_order(market_buy)
    # No position
    else:
        print("No position")
        if broker.max_funds_for_futures:
            if expr.buy_signal.eval(broker, analyser, candles): # Enter Long
                enter_long(expr, broker, analyser, candles)

            if expr.sell_signal.eval(broker, analyser, candles): # Enter Short
                enter_short(expr, broker, analyser, candles)
