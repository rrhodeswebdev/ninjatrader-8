import typing as t
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class Candle:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    open_time: datetime
    close_time: datetime