import typing as t
from datetime import datetime, timedelta
from enum import Enum


class Timeframes(Enum):
    # seconds
    S1 = 1
    S5 = 5
    S15 = 15
    S30 = 30
    # minutes
    M1 = 60
    M3 = 180
    M5 = 300
    M15 = 900
    M30 = 1800
    M45 = 2700
    # hours
    H1 = 3600
    H2 = 7200
    H3 = 10800
    H4 = 14400
    # day
    D1 = 86400
    # week
    W1 = 604800

    def __str__(self) -> str:
        return self.name


def get_timeframes_ratio(first_timeframe: Timeframes, 
                         second_timeframe: Timeframes) -> t.Tuple[int, int]:
    """Get quotient and remainder of a division of two timeframes."""
    return divmod(first_timeframe.value, second_timeframe.value)


def get_seconds_duration(timeframe: Timeframes) -> int:
    """Get timeframe duration in seconds."""
    return timeframe.value - 1


def get_millis_duration(timeframe: Timeframes) -> int:
    """Get timeframe duration in milliseconds."""
    return get_seconds_duration(timeframe) * 1000 + 999


def estimate_open_time(time: datetime, 
                       timeframe: Timeframes, 
                       offset: int = 0) -> datetime:
    """
    Get open time of a candle on `timeframe` from `time` 
    and add `offset` closed candles.
    """
    seconds_after_open = time.timestamp() % timeframe.value
    delta = timedelta(seconds=seconds_after_open - offset * timeframe.value, 
                      milliseconds=time.microsecond/1000)
    return time - delta


def estimate_close_time(time: datetime, 
                        timeframe: Timeframes, 
                        offset: int = 0) -> datetime:
    """
    Get close time of a candle on `timeframe` from `time`
    and add `offset` closed candles.
    """
    open_time = estimate_open_time(time, timeframe, offset)
    return open_time + timedelta(milliseconds=get_millis_duration(timeframe))