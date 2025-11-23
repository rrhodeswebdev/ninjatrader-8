"""Interfaces for broker."""
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import (  # https://docs.python.org/3/library/decimal.html
    ROUND_FLOOR, Decimal)
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

    def __str__(self) -> str:
        return self.value


class OrderStatus(Enum):
    CREATED = "CREATED"
    CANCELLED = "CANCELLED"
    EXECUTED = "EXECUTED"
    # Only for TP/SL orders
    SYS_CANCELLED = "SYS_CANCELLED"
    ACTIVATED = "ACTIVATED"

    def __str__(self) -> str:
        return self.value


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

    def __str__(self) -> str:
        return self.value


class OrderOptions:
    def __init__(self, 
                 order_type: OrderSide, 
                 amount: t.Optional[Decimal] = None,
                 percentage_amount: t.Optional[Decimal] = None,
                 order_price: t.Optional[Decimal] = None):
        self.order_type = order_type
        self.amount = amount
        self.percentage_amount = percentage_amount
        self.order_price = order_price


class OrderInfo(ABC):
    @property
    @abstractmethod
    def order_id(self) -> int:
        pass

    @property
    @abstractmethod
    def order_type(self) -> OrderType:
        pass

    @property
    @abstractmethod
    def order_side(self) -> OrderSide:
        pass

    @property
    @abstractmethod
    def amount(self) -> Decimal:
        pass

    @property
    @abstractmethod
    def date_created(self) -> datetime:
        pass

    @property
    @abstractmethod
    def order_price(self) -> t.Optional[Decimal]:
        pass

    @property
    @abstractmethod
    def status(self) -> OrderStatus:
        pass

    @property
    @abstractmethod
    def date_updated(self) -> datetime:
        pass

    @property
    @abstractmethod
    def fill_price(self) -> t.Optional[Decimal]:
        pass

    @property
    @abstractmethod
    def trading_fee(self) -> t.Optional[Decimal]:
        pass

    @property
    @abstractmethod
    def is_unfulfilled(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_canceled(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_executed(self) -> bool:
        pass


class MarketOrderOptions(OrderOptions):
    def __init__(self, 
                 order_side: OrderSide, 
                 amount: t.Optional[Decimal] = None,
                 percentage_amount: t.Optional[Decimal] = None):
        self.order_side = order_side
        super().__init__(OrderType.MARKET, amount, percentage_amount)


class MarketOrderInfo(OrderInfo): 
    pass


class StrategyOrderInfo(OrderInfo):
    @property
    @abstractmethod
    def trigger_price(self) -> Decimal:
        pass

    @property
    @abstractmethod
    def is_activated(self) -> bool:
        pass


class TakeProfitOptions(OrderOptions):
    def __init__(self, 
                 trigger_price: Decimal, 
                 amount: t.Optional[Decimal] = None,
                 percentage_amount: t.Optional[Decimal] = None,
                 order_price: t.Optional[Decimal] = None):
        self.trigger_price = trigger_price
        order_type = OrderType.TAKE_PROFIT_LIMIT if order_price \
                        else OrderType.TAKE_PROFIT
        super().__init__(order_type, amount, percentage_amount, order_price)


class TakeProfitInfo(StrategyOrderInfo):
    pass


class StopLossOptions(OrderOptions):
    def __init__(self, 
                 trigger_price: Decimal, 
                 amount: t.Optional[Decimal] = None,
                 percentage_amount: t.Optional[Decimal] = None,
                 order_price: t.Optional[Decimal] = None):
        self.trigger_price = trigger_price
        order_type = OrderType.STOP_LOSS_LIMIT if order_price \
                        else OrderType.STOP_LOSS
        super().__init__(order_type, amount, percentage_amount, order_price)


class StopLossInfo(StrategyOrderInfo):
    pass


class LimitOrderOptions(OrderOptions):
    def __init__(self, 
                 order_side: OrderSide, 
                 order_price: Decimal, 
                 amount: t.Optional[Decimal] = None,
                 percentage_amount: t.Optional[Decimal] = None,
                 take_profit: t.Optional[TakeProfitOptions] = None,
                 stop_loss: t.Optional[StopLossOptions] = None):
        self.order_side = order_side
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        super().__init__(OrderType.LIMIT, amount, percentage_amount, order_price)


class LimitOrderInfo(OrderInfo):
    @property
    @abstractmethod
    def take_profit(self) -> t.Optional[TakeProfitInfo]:
        pass

    @property
    @abstractmethod
    def stop_loss(self) -> t.Optional[StopLossInfo]:
        pass


class TradeInfo(ABC):
    @property
    @abstractmethod
    def trade_id(self) -> int:
        pass

    @property
    @abstractmethod
    def order(self) -> OrderInfo:
        pass

    @property
    @abstractmethod
    def result_balance(self) -> Decimal:
        # fiat balance at the moment of order execution
        pass


class BrokerException(Exception):
    """Base class for all broker-related exceptions."""
    pass


class OrderSubmissionError(BrokerException):
    """Generic exception for order submission error."""
    pass


class InvalidOrderData(OrderSubmissionError):
    """Order submission failed because order data is invalid."""
    pass


class InsufficientFunds(OrderSubmissionError):
    """Order submission failed due to insufficient funds."""
    pass


class OrderCancellationError(BrokerException):
    """Generic exception for order cancellation error."""
    pass


class AbstractBroker(ABC):
    """
    Broker provides orders management in a simulated
    market environment.
    """
    @property
    @abstractmethod
    def current_equity(self) -> Decimal:
        """Get current equity."""
        pass

    @abstractmethod
    def iter_orders(self) -> t.Iterator[OrderInfo]:
        """Get orders iterator."""
        pass

    @abstractmethod
    def iter_trades(self) -> t.Iterator[TradeInfo]:
        """Get trades iterator."""
        pass

    @abstractmethod
    def get_orders(self) -> t.Sequence[OrderInfo]:
        """Get orders sequence."""
        pass

    @abstractmethod
    def get_trades(self) -> t.Sequence[TradeInfo]:
        """Get trades sequence."""
        pass

    @abstractmethod
    def submit_market_order(self, 
                            options: MarketOrderOptions) -> MarketOrderInfo:
        """Submit market order."""
        pass

    @abstractmethod
    def submit_limit_order(self, 
                           options: LimitOrderOptions) -> LimitOrderInfo:
        """Submit limit order."""
        pass

    @abstractmethod
    def submit_take_profit_order(
                self, 
                order_side: OrderSide,
                options: TakeProfitOptions) -> TakeProfitInfo:
        """Submit Take Profit order."""
        pass

    @abstractmethod
    def submit_stop_loss_order(
                self, 
                order_side: OrderSide,
                options: StopLossOptions) -> StopLossInfo:
        """Submit Stop Loss order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> None:
        """Cancel order by id."""
        pass