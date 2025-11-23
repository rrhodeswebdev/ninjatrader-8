import typing as t
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import zip_longest

from .data.candle import Candle as InputCandle
from .timeframes import Timeframes, estimate_close_time


@dataclass
class Candle:
    """
    Contains a snapshot of OHLCV data of a candle. 
    The `is_closed` attribute can be used to determine if 
    the data is in its final state (i.e., the candle is closed).
    """
    open_time:  datetime
    close_time: datetime
    open:       Decimal = Decimal('NaN')
    high:       Decimal = Decimal('NaN')
    low:        Decimal = Decimal('NaN')
    close:      Decimal = Decimal('NaN')
    volume:     Decimal = Decimal('NaN')
    is_closed:  bool = False


class CandleNotFound(Exception):
    def __init__(self, timeframe: Timeframes):
        message = f"Candle {timeframe} was not found in buffer."
        super().__init__(message)


def _create_placeholder_candle(open_time: datetime, 
                               timeframe: Timeframes) -> Candle:
    close_time = estimate_close_time(open_time, timeframe)
    return Candle(open_time=open_time, close_time=close_time)


def iter_config(config):
    if isinstance(config, dict):
        return config.items()

    elif isinstance(config, set):
        return zip_longest(config, [], fillvalue=1)


def make_candles(start_time, timeframe, length):
    placeholder = _create_placeholder_candle(start_time, timeframe)
    return deque([placeholder], maxlen=length)


class CandlesBuffer:
    def __init__(self, start_time: datetime, config):
        self._data = {
            timeframe: make_candles(start_time, timeframe, maxlen)
                for timeframe, maxlen in iter_config(config)
        }

    def get(self, timeframe: Timeframes, index: int = -1) -> Candle:
        try:
            return self._data[timeframe][index]
        except KeyError:  # index error?
            raise CandleNotFound(timeframe)

    def update(self, candle: InputCandle) -> None:
        """Update stored candles data in accordance with `candle`."""
        for timeframe in self._data:
            self._update_candle(timeframe, candle) 

    def _update_candle(self, 
                       timeframe: Timeframes, 
                       new_candle: InputCandle) -> None:
        history = self._data[timeframe]
        last_candle = history[-1]
        if new_candle.close_time > last_candle.close_time or \
                new_candle.open_time == last_candle.open_time:
            candle = Candle(open_time=new_candle.open_time,
                            close_time=estimate_close_time(new_candle.open_time, timeframe),
                            open=new_candle.open,
                            high=new_candle.high,
                            low=new_candle.low,
                            close=new_candle.close,
                            volume=new_candle.volume)
            history.append(candle)
        else:
            last_candle.high = max(last_candle.high, new_candle.high)
            last_candle.low = min(last_candle.low, new_candle.low)
            last_candle.close = new_candle.close
            last_candle.volume += new_candle.volume

        last_candle.is_closed = (new_candle.close_time == last_candle.close_time)


class Candles:
    """
    Provides the last candle representation for various timeframes.
    It is useful for checking properties of a candle 
    on one timeframe (H1, for example), while having data
    on another (for instance, M1).
    """
    def __init__(self, buffer: CandlesBuffer):
        self._buffer=buffer

    def get(self, timeframe: Timeframes, index: int = -1) -> Candle:
        """
        Get the last candle representation on `timeframe`.
        If the candle of `timeframe` is not found, 
        raises `CandleNotFound`.
        """
        return replace(self._buffer.get(timeframe, index))  # replace = copy for dataclasses
