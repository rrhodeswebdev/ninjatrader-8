from enum import Enum


class CandleProperties(Enum):
    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'
    VOLUME = 'VOLUME'

    def __str__(self) -> str:
        return self.name


OPEN = CandleProperties.OPEN
HIGH = CandleProperties.HIGH
LOW = CandleProperties.LOW
CLOSE = CandleProperties.CLOSE
VOLUME = CandleProperties.VOLUME