import typing as t
from decimal import Decimal

from backintime.broker.base import OrderSide, OrderStatus
from .orders import (LimitOrder, LimitOrderInfo, MarketOrder, MarketOrderInfo,
                     Order, OrderInfo, StopLossInfo, StopLossOrder,
                     StrategyOrder, StrategyOrders, TakeProfitInfo,
                     TakeProfitOrder, TradeInfo)


def check_limit_with_open(order, open_price) -> t.Optional[Decimal]:
    """
    Check if limit price of the order matches the market open price.
    If it does, return suggested fill price. Return None otherwise.
    """
    if order.side is OrderSide.BUY:
        if order.order_price >= open_price:    # same or lower for buy
            return open_price

    elif order.side is OrderSide.SELL:
        if order.order_price <= open_price:    # same or higher for sell
            return open_price


def check_limit_with_high_low(order, high, low) -> t.Optional[Decimal]:
    """
    Check if limit price of the order matches the market high/low prices.
    If it does, return suggested fill price. Return None otherwise.
    """
    if order.side is OrderSide.BUY:
        if order.order_price >= low:
            fill_price = min(order.order_price, high)   # limit price or better
            return fill_price

    elif order.side is OrderSide.SELL:
        if order.order_price <= high:
            fill_price = max(order.order_price, low)    # limit price or better
            return fill_price


class MatchingEngine:
    @staticmethod
    def get_updates_for_open(orders, open_price):
        """
        Get status updates for orders with limited price (Limit, TP, SL) 
        using 'open' price.
        """
        updates = []
        for order_id, order in orders:
            # Review strategy orders with open price
            if isinstance(order, StrategyOrder):
                if order.status is OrderStatus.CREATED:
                    if order.trigger_price == open_price:
                        if not order.order_price:   # TP/SL market
                            updates.append((order, order_id, 'EXECUTE', open_price))
                        # TP/SL Limit
                        else:
                            if (fill := check_limit_with_open(order, open_price)):
                                updates.append((order, order_id, 'EXECUTE', fill))
                            else:
                                updates.append((order, order_id, 'ACTIVATE', None))

                elif order.status is OrderStatus.ACTIVATED:
                    if (fill := check_limit_with_open(order, open_price)):
                        updates.append((order, order_id, 'EXECUTE', fill))

            # Review limit orders with open price
            elif isinstance(order, LimitOrder):
                if (fill := check_limit_with_open(order, open_price)):
                    updates.append((order, order_id, 'EXECUTE', fill))

        return updates

    @staticmethod
    def get_updates_for_high_low(orders, high, low):
        """
        Get status updates for orders with limited price (Limit, TP, SL) 
        using 'high' and 'low' prices.
        """
        updates = []
        for order_id, order in orders:
            # Review strategy order with HIGH, LOW prices
            if isinstance(order, StrategyOrder):
                if order.status is OrderStatus.CREATED:
                    if order.trigger_price >= low and \
                            order.trigger_price <= high:
                        if not order.order_price:   # TP/SL market
                            updates.append((order, order_id, 'EXECUTE', order.trigger_price))
                        # TP/SL Limit
                        else:
                            if (fill := check_limit_with_high_low(order, high, low)):
                                updates.append((order, order_id, 'EXECUTE', fill))
                            else:
                                updates.append((order, order_id, 'ACTIVATE', None))

                elif order.status is OrderStatus.ACTIVATED:
                    if (fill := check_limit_with_high_low(order, high, low)):
                        updates.append((order, order_id, 'EXECUTE', fill))

            # Review limit order with HIGH, LOW prices
            elif isinstance(order, LimitOrder):
                if (fill := check_limit_with_high_low(order, high, low)):
                    updates.append((order, order_id, 'EXECUTE', fill))

        return updates