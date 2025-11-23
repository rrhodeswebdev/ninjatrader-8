"""
Orders only hold required data, but the desired functions 
must be implemented somewhere else. 
All fields are public. It is up to a broker implementation to set 
`status` and provide TP/SL orders features:

    - Multiple TP/SL orders can be posted for the same position, 
      so the summarised value of opened TP/SL orders can be greater
      than the opened position. 
      In this case, only the first TP/SL order whose conditions
      are satisfied will be executed. Others must be cancelled.

    - When TP/SL order is activated (target_price meets market), 
      it must be treated as Market/Limit order,
      depending on whether `order_price` was provided.
"""
import typing as t
from dataclasses import dataclass
from datetime import datetime
from decimal import (  # https://docs.python.org/3/library/decimal.html
    ROUND_FLOOR, Decimal)

from backintime.broker import base
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    StopLossOptions, TakeProfitOptions)


# Orders implementation
class Order:
    """Base class for all orders."""
    def __init__(self, 
                 side: OrderSide, 
                 order_type: OrderType,
                 amount: Decimal,
                 is_short: bool,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime,
                 order_price: t.Optional[Decimal] = None):
        if side is OrderSide.BUY or is_short:
            self.amount = amount.quantize(min_usd, ROUND_FLOOR)
        else:
            self.amount = amount.quantize(min_contracts, ROUND_FLOOR)

        self.order_price = order_price.quantize(min_usd) \
                        if order_price \
                        else None

        self.side = side
        self.is_short = is_short
        self.order_type = order_type
        self.date_created = date_created
        self.date_updated = date_created
        self.status = OrderStatus.CREATED
        self._fill_price: t.Optional[Decimal] = None
        self._trading_fee: t.Optional[Decimal] = None
        # Used for rounding
        self.min_usd = min_usd
        self.min_contracts = min_contracts

    @property
    def fill_price(self) -> t.Optional[Decimal]:
        return self._fill_price

    @fill_price.setter
    def fill_price(self, fill_price: Decimal) -> None:
        fill_price = fill_price.quantize(self.min_usd)
        self._fill_price = fill_price

    @property
    def trading_fee(self) -> t.Optional[Decimal]:
        return self._trading_fee

    @trading_fee.setter
    def trading_fee(self, trading_fee: Decimal) -> None:
        trading_fee = trading_fee.quantize(self.min_usd)
        self._trading_fee = trading_fee


class MarketOrder(Order):
    def __init__(self, 
                 side: OrderSide, 
                 amount: Decimal,
                 is_short: bool,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime):
        super().__init__(side, OrderType.MARKET, amount, is_short,
                         min_usd, min_contracts, date_created)


# Strategy orders have trigger price
class StrategyOrder(Order):
    def __init__(self,
                 side: OrderSide,
                 order_type: OrderType,
                 amount: Decimal, 
                 trigger_price: Decimal,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime,
                 is_short: bool = False,
                 order_price: t.Optional[Decimal] = None):
        self.trigger_price = trigger_price
        self.date_activated: t.Optional[datetime] = None
        super().__init__(side, order_type, amount, is_short,
                         min_usd, min_contracts,
                         date_created, order_price)


class TakeProfitOrder(StrategyOrder):
    def __init__(self,
                 side: OrderSide,
                 amount: Decimal,
                 trigger_price: Decimal,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime,
                 order_price: t.Optional[Decimal] = None):
        order_type = OrderType.TAKE_PROFIT_LIMIT if order_price \
                        else OrderType.TAKE_PROFIT
        super().__init__(side, order_type, amount, 
                         trigger_price, min_usd, min_contracts,
                         date_created, order_price=order_price)


class StopLossOrder(StrategyOrder):
    def __init__(self,
                 side: OrderSide,
                 amount: Decimal,
                 trigger_price: Decimal,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime,
                 order_price: t.Optional[Decimal] = None):
        order_type = OrderType.STOP_LOSS_LIMIT if order_price \
                        else OrderType.STOP_LOSS
        super().__init__(side, order_type, amount, 
                         trigger_price, min_usd, min_contracts, 
                         date_created, order_price=order_price)


# Limit orders have optional TP/SL
class LimitOrder(Order):
    def __init__(self, 
                 side: OrderSide,
                 amount: Decimal,
                 order_price: Decimal,
                 is_short: bool,
                 min_usd: Decimal,
                 min_contracts: Decimal,
                 date_created: datetime,
                 take_profit: t.Optional[TakeProfitOptions] = None,
                 stop_loss: t.Optional[StopLossOptions] = None):
        self.take_profit_options = take_profit
        self.stop_loss_options = stop_loss
        self.take_profit: t.Optional[TakeProfitOrder] = None 
        self.stop_loss: t.Optional[StopLossOrder] = None
        super().__init__(side, OrderType.LIMIT, 
                         amount, is_short, min_usd, min_contracts, 
                         date_created, order_price)
# End of orders implementation


# Orders interface implementation
class OrderInfo(base.OrderInfo):
    """
    Wrapper around `Order` that provides a read-only view
    into the wrapped `Order` data.
    """
    def __init__(self, order_id: int, order: Order):
        self._order_id = order_id
        self._order = order

    @property
    def order_id(self) -> int:
        return self._order_id

    @property
    def order_type(self) -> OrderType:
        return self._order.order_type

    @property
    def order_side(self) -> OrderSide:
        return self._order.side

    @property 
    def amount(self) -> Decimal:
        return self._order.amount

    @property
    def date_created(self) -> datetime:
        return self._order.date_created

    @property 
    def order_price(self) -> t.Optional[Decimal]:
        return self._order.order_price

    @property
    def trigger_price(self) -> Decimal:
        if self._order.order_type is not OrderType.MARKET and \
                self._order.order_type is not OrderType.LIMIT:
            return self._order.trigger_price
        else:
            return Decimal('NaN')

    @property 
    def status(self) -> OrderStatus:
        return self._order.status

    @property
    def date_updated(self) -> datetime:
        return self._order.date_updated

    @property
    def fill_price(self) -> t.Optional[Decimal]:
        return self._order.fill_price

    @property
    def trading_fee(self) -> t.Optional[Decimal]:
        return self._order.trading_fee

    @property
    def is_short(self) -> bool:
        return self._order.is_short

    @property 
    def is_unfulfilled(self) -> bool:
        return not self._order.fill_price

    @property 
    def is_canceled(self) -> bool:
        return self._order.status is OrderStatus.CANCELLED or \
                self._order.status is OrderStatus.SYS_CANCELLED

    @property 
    def is_executed(self) -> bool:
        return self._order.status is OrderStatus.EXECUTED


class MarketOrderInfo(OrderInfo, base.MarketOrderInfo): 
    def __init__(self, order_id: int, order: MarketOrder):
        super().__init__(order_id, order)


class StrategyOrderInfo(OrderInfo, base.StrategyOrderInfo):
    def __init__(self, order_id: int, order: StrategyOrder):
        super().__init__(order_id, order)

    @property
    def trigger_price(self) -> Decimal:
        return self._order.trigger_price

    @property
    def is_activated(self) -> bool:
        return self._order.status is OrderStatus.ACTIVATED


class TakeProfitInfo(StrategyOrderInfo):
    def __init__(self, order_id: int, order: TakeProfitOrder):
        super().__init__(order_id, order)


class StopLossInfo(StrategyOrderInfo):
    def __init__(self, order_id: int, order: StopLossOrder):
        super().__init__(order_id, order)


@dataclass
class StrategyOrders:
    take_profit_id: t.Optional[int] = None
    stop_loss_id: t.Optional[int] = None


class LimitOrderInfo(OrderInfo, base.LimitOrderInfo):
    def __init__(self, 
                 order_id: int, 
                 order: LimitOrder, 
                 strategy_orders: StrategyOrders):
        self._strategy_orders = strategy_orders
        super().__init__(order_id, order)

    @property 
    def take_profit(self) -> t.Optional[TakeProfitInfo]:
        take_profit = self._order.take_profit
        if take_profit:
            return TakeProfitInfo(self._strategy_orders.take_profit_id,
                                  take_profit)

    @property
    def stop_loss(self) -> t.Optional[StopLossInfo]:
        stop_loss = self._order.stop_loss
        if stop_loss:
            return StopLossInfo(self._strategy_orders.stop_loss_id,
                                stop_loss)


class TradeInfo(base.TradeInfo):
    def __init__(self, 
                 trade_id: int, 
                 order_info: OrderInfo, 
                 result_balance: Decimal):
        self._trade_id = trade_id
        self._order_info = order_info
        self._result_balance = result_balance

    @property
    def trade_id(self) -> int:
        return self._trade_id

    @property
    def order(self) -> OrderInfo:
        return self._order_info

    @property
    def result_balance(self) -> Decimal:
        # fiat balance at the moment of order execution
        return self._result_balance
# End of orders' interface implementation