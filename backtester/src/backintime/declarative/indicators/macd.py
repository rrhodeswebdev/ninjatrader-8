import typing as t
from decimal import Decimal

from backintime.analyser.indicators.base import IndicatorOptions
from backintime.analyser.indicators.constants import (CLOSE, HIGH, LOW, OPEN,
                                                      CandleProperties)
from backintime.analyser.indicators.macd import MacdOptions
from backintime.declarative.base import Primitive
from backintime.timeframes import Timeframes


class MacdExpression(Primitive):
    def __init__(self, 
                 timeframe: Timeframes,
                 fastperiod: int,
                 slowperiod: int,
                 signalperiod: int,
                 index: int,
                 attr: str):
        self.timeframe = timeframe
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.index = index
        self.attr = attr
        super().__init__(concrete=self)

    @property
    def indicators_meta(self) -> t.Set[IndicatorOptions]:
        return {
            MacdOptions(self.timeframe, self.fastperiod, 
                        self.slowperiod, self.signalperiod)
        }

    def eval(self, broker, analyser, candles):
        result = analyser.macd(self.timeframe, self.fastperiod, 
                    self.slowperiod, self.signalperiod)
        return Decimal(getattr(result[self.index], self.attr))


class MacdFactoryWithIndex:
    def __init__(self, 
                 timeframe: Timeframes,
                 fastperiod: int,
                 slowperiod: int,
                 signalperiod: int,
                 index: int):
        self.timeframe = timeframe
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.index = index

    @property
    def macd(self) -> MacdExpression:
        attr = 'macd'
        return MacdExpression(
                    self.timeframe, 
                    self.fastperiod, 
                    self.slowperiod, 
                    self.signalperiod, 
                    self.index, attr)

    @property
    def signal(self) -> MacdExpression:
        attr = 'signal'
        return MacdExpression(
                    self.timeframe, 
                    self.fastperiod, 
                    self.slowperiod, 
                    self.signalperiod, 
                    self.index, attr)

    @property
    def hist(self) -> MacdExpression:
        attr = 'hist'
        return MacdExpression(
                    self.timeframe, 
                    self.fastperiod, 
                    self.slowperiod, 
                    self.signalperiod, 
                    self.index, attr)


class MacdFactory:
    def __init__(self, 
                 timeframe: Timeframes,
                 fastperiod: int = 12,
                 slowperiod: int = 26,
                 signalperiod: int = 9):
        self.timeframe = timeframe
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod

    def __getitem__(self, index):
        return MacdFactoryWithIndex(
                    self.timeframe, self.fastperiod, 
                    self.slowperiod, self.signalperiod, index)