import time
import typing as t
from collections import abc
from datetime import datetime, timezone
from decimal import Decimal

import requests as r

from backintime.timeframes import Timeframes, estimate_close_time

from .candle import Candle
from .data_provider import (DataProvider, DataProviderError,
                            DataProviderFactory, ParsingError)


def _to_ms(time: datetime) -> int:
    """Convert `datetime` to milliseconds timestamp."""
    return int(time.timestamp()*1000)


def _parse_time(millis_timestamp: int) -> datetime:
    """Convert milliseconds timestamp to `datetime`(UTC)."""
    timestamp = millis_timestamp/1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _parse_candle(candle: list) -> Candle:
    """Parse candle from a sequence"""
    try:
        return Candle(open_time=_parse_time(candle[0]),
                      open=Decimal(candle[1]),
                      high=Decimal(candle[2]),
                      low=Decimal(candle[3]),
                      close=Decimal(candle[4]),
                      volume=Decimal(candle[5]),
                      close_time=_parse_time(candle[6]))
    except Exception as e:
        raise ParsingError(str(e))


def _utcnow() -> datetime:
    """Return current timezone aware date (UTC)."""
    return datetime.now(timezone.utc)


class BinanceCandles(DataProvider):
    _url = 'https://api.binance.com/api/v3/klines'
    _intervals = {
        Timeframes.M1: '1m',
        Timeframes.M3: '3m',
        Timeframes.M5: '5m',
        Timeframes.M15: '15m',
        Timeframes.M30: '30m',
        Timeframes.H1: '1h',
        Timeframes.H2: '2h',
        Timeframes.H4: '4h',
        Timeframes.D1: '1d',
        Timeframes.W1: '1w'
    }

    def __init__(self, 
                 symbol: str, 
                 timeframe: Timeframes, 
                 since: datetime, 
                 until: t.Optional[datetime]=_utcnow()):
        self._symbol=symbol
        self._timeframe=timeframe
        self._since=since
        self._until=until
        self._interval = self._intervals[timeframe]

    @property
    def title(self) -> str:
        return "Binance klines API v3"

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> Timeframes:
        return self._timeframe
    
    @property
    def since(self) -> datetime:
        return self._since
    
    @property
    def until(self) -> datetime:
        return self._until

    def __iter__(self) -> t.Iterator[Candle]:
        """Return generator that will yield one candle at a time."""
        since = _to_ms(self._since)
        until = _to_ms(self._until)
        end_time = estimate_close_time(self._until, self._timeframe, -1)
        end_time = _to_ms(end_time)

        max_per_request = 1000
        tf_ms = self._timeframe.value * 1000
        max_time_step = max_per_request * tf_ms
        # requests counter
        counter = 0

        params = {
            "symbol": self._symbol,
            "interval": self._interval,
            "startTime": None,  # candle open >= startTime
            "endTime": None,    # candle open <= endTime
            "limit": max_per_request    # candles quantity <= limit
        }

        for start_time_ms in range(since, until, max_time_step):
            params["startTime"] = start_time_ms
            params["endTime"] = min(end_time, start_time_ms + max_time_step - tf_ms)
            
            try:
                res = r.get(self._url, params)
                res.raise_for_status()
            # Wrap exceptions so that caller will know it is something
            # related to a data provider without knowing about `requests`.
            except r.exceptions.ConnectionError as e:
                raise DataProviderError("Failed to connect")
            except r.exceptions.HTTPError as e:
                raise DataProviderError(str(e))
            else:
                counter += 1

            for item in res.json():
                yield _parse_candle(item)
            # sleep after every 20th call
            if counter % 20 == 0:
                time.sleep(1)


class BinanceCandlesFactory(DataProviderFactory):
    def __init__(self, ticker: str, timeframe: Timeframes):
        self.ticker = ticker
        self._timeframe = timeframe

    @property
    def timeframe(self) -> Timeframes:
        return self._timeframe

    def create(self, since: datetime, until: datetime):
        return BinanceCandles(self.ticker, self.timeframe, since, until)
