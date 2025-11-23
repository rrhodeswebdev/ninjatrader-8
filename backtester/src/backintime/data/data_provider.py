import typing as t
from abc import ABC, abstractmethod
from collections import abc
from datetime import datetime

from backintime.timeframes import Timeframes

from .candle import Candle


class DataProvider(abc.Iterable):
    """
    Provides candles in historical order.
    `DataProvider` is an iterable object that 
    can be created for specific date range (since, until);
    Yields OHLCV candle during iteration.
    """
    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @property
    @abstractmethod
    def timeframe(self) -> Timeframes:
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        pass

    @property
    @abstractmethod
    def since(self) -> datetime:
        pass

    @property
    @abstractmethod
    def until(self) -> datetime:
        pass

    @abstractmethod
    def __iter__(self) -> t.Iterator[Candle]:
        pass


class DataProviderFactory(ABC):
    @property
    @abstractmethod
    def timeframe(self) -> Timeframes:
        pass

    @abstractmethod
    def create(self, since: datetime, until: datetime):
        pass


class DataProviderError(Exception):
    """Base class for all data related errors."""
    pass


class ParsingError(DataProviderError):
    """Failed to parse candle from source."""
    pass
