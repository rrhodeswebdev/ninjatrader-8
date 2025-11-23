import typing as t
from abc import ABC, abstractmethod
from collections import abc
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from backintime.timeframes import Timeframes

from .constants import CandleProperties

# NOTE: in < 3.9 can't use collections.abc mixins with type subscriptions
ResultItem = t.TypeVar("ResultItem")

class IndicatorResultSequence(abc.Sequence, t.Generic[ResultItem]):
    @abstractmethod
    def __iter__(self) -> t.Iterator[ResultItem]:
        """Iterate over results in historical order: oldest first."""
        pass

    @abstractmethod
    def __reversed__(self) -> t.Iterator[ResultItem]:
        """Iterate over results in reversed order: most recent first."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> ResultItem:
        """Get result item by index."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get length."""
        pass


class MarketData(ABC):
    @abstractmethod
    def get_values(self, 
                   timeframe: Timeframes, 
                   candle_property: CandleProperties, 
                   limit: int) -> t.Sequence[Decimal]:
        pass


@dataclass(frozen=True)
class IndicatorParam:
    timeframe: Timeframes
    candle_property: CandleProperties
    quantity: int


InputTimeseries = IndicatorParam

class IndicatorOptions(ABC):
    @property
    @abstractmethod
    def indicator_name(self) -> str:
        pass

    @property
    @abstractmethod
    def indicator_options(self) -> dict:
        pass

    @property
    @abstractmethod
    def input_timeseries(self) -> t.Iterable[InputTimeseries]:
        pass

    def __eq__(self, other) -> bool:
        return self.indicator_name == other.indicator_name and \
                self.indicator_options == other.indicator_options

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        pairs = [
            f"{key}={value}" 
                for key, value in self.indicator_options.items()
        ]
        indicator_opts = ', '.join(pairs)
        return f"{self.indicator_name}({indicator_opts})"
