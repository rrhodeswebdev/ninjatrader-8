import typing as t
from collections import abc
from itertools import count

from .orders import (LimitOrder, MarketOrder, Order, StopLossOrder,
                     StrategyOrder, StrategyOrders, TakeProfitOrder)


class OrdersRepository(abc.Iterable):
    def __init__(self):
        self._market_orders: t.List[int] = []  # Market/Strategy ids
        self._limit_orders: t.List[int] = []    # Limit/Strategy ids
        self._strategy_orders: t.List[int] = []  # Strategy ids
        self._orders_counter = count()
        self._orders_map: t.Dict[int, Order] = {}
        self._linked_strategy_orders: t.Dict[int, StrategyOrders] = {}

    def get_next_order_id(self) -> int:
        return next(self._orders_counter)

    def get_order(self, order_id: int) -> t.Optional[Order]:
        return self._orders_map.get(order_id)

    def get_market_orders(self):
        for order_id in self._market_orders.copy():
            yield (order_id, self._orders_map[order_id])

    def get_limit_orders(self):
        for order_id in self._limit_orders.copy():
            yield (order_id, self._orders_map[order_id])

    def get_strategy_orders(self):
        for order_id in self._strategy_orders.copy():
            yield (order_id, self._orders_map[order_id])

    def get_linked_orders(self, order_id: int) -> StrategyOrders:
        return self._linked_strategy_orders[order_id]

    def add_market_order(self, order: MarketOrder) -> int:
        order_id = next(self._orders_counter)
        self._market_orders.append(order_id)
        self._orders_map[order_id] = order
        return order_id 

    def add_limit_order(self, order: LimitOrder) -> int:
        order_id = next(self._orders_counter)
        self._limit_orders.append(order_id)
        self._orders_map[order_id] = order
        # Create shared obj for linked TP/SL orders 
        strategy_orders = StrategyOrders()
        self._linked_strategy_orders[order_id] = strategy_orders 
        return order_id

    def add_take_profit_order(self, order_id, order: TakeProfitOrder):
        return self._add_strategy_order(order)

    def add_stop_loss_order(self, order_id, order: StopLossOrder):
        return self._add_strategy_order(order)

    def add_linked_take_profit_order(self,
                                     tp_id,
                                     order: TakeProfitOrder, 
                                     limit_order_id: int) -> int:
        self._add_strategy_order(tp_id, order)
        limit_order = self._orders_map[limit_order_id]
        limit_order.take_profit = order
        # NOTE: StrategyOrders is shared with LimitOrderInfo 
        linked = self._linked_strategy_orders[limit_order_id]
        linked.take_profit_id = tp_id

    def add_linked_stop_loss_order(self,
                                   sl_id,
                                   order: StopLossOrder, 
                                   limit_order_id: int) -> int:
        self._add_strategy_order(sl_id, order)
        limit_order = self._orders_map[limit_order_id]
        limit_order.stop_loss = order
        # NOTE: StrategyOrders is shared with LimitOrderInfo
        linked = self._linked_strategy_orders[limit_order_id]
        linked.stop_loss_id = sl_id

    def add_order_to_market_orders(self, order_id: int) -> None:
        self._market_orders.append(order_id)

    def add_order_to_limit_orders(self, order_id: int) -> None:
        self._limit_orders.append(order_id)

    def remove_market_orders(self) -> None:
        self._market_orders = []

    def remove_market_order(self, order_id: int) -> None:
        self._market_orders.remove(order_id)

    def remove_limit_order(self, order_id: int) -> None:
        self._limit_orders.remove(order_id)

    def remove_take_profit_order(self, order_id: int) -> None:
        self.remove_strategy_order(order_id)

    def remove_stop_loss_order(self, order_id: int) -> None:
        self.remove_strategy_order(order_id)

    def remove_strategy_order(self, order_id: int) -> None:
        if order_id in self._limit_orders:
            self._limit_orders.remove(order_id)
        if order_id in self._market_orders:
            self._market_orders.remove(order_id)
        if order_id in self._strategy_orders:
            self._strategy_orders.remove(order_id)

    def _add_strategy_order(self, order_id, order: StrategyOrder):
        self._limit_orders.append(order_id)
        self._strategy_orders.append(order_id)
        self._orders_map[order_id] = order

    def __iter__(self) -> t.Iterator[t.Tuple[int, Order]]:
        for order_id, order in self._orders_map.items():
            yield order_id, order
