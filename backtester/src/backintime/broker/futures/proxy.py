import typing as t
from decimal import Decimal

from backintime.broker.base import (AbstractBroker, LimitOrderInfo,
                                    LimitOrderOptions, MarketOrderInfo,
                                    MarketOrderOptions, OrderInfo, OrderSide,
                                    StopLossInfo, StopLossOptions,
                                    TakeProfitInfo, TakeProfitOptions,
                                    TradeInfo)

from .balance import FuturesBalanceInfo
from .broker import FuturesBroker


class FuturesBrokerProxy(AbstractBroker):
    """
    Broker provides orders management in a simulated
    market environment.
    """
    def __init__(self, broker: FuturesBroker):
        self._broker = broker

    @property
    def in_long(self) -> bool:
        return self._broker.in_long

    @property
    def in_short(self) -> bool:
        return self._broker.in_short

    @property
    def balance(self) -> FuturesBalanceInfo:
        """Get balance info."""
        return self._broker.balance

    @property
    def current_equity(self) -> Decimal:
        """Get current equity."""
        return self._broker.current_equity

    @property
    def max_funds_for_futures(self):
        return self._broker.max_funds_for_futures

    @property
    def position(self):
        return self._broker.position

    def iter_orders(self) -> t.Iterator[OrderInfo]:
        """Get orders iterator."""
        return self._broker.iter_orders()

    def iter_trades(self) -> t.Iterator[TradeInfo]:
        """Get trades iterator."""
        return self._broker.iter_trades()

    def get_orders(self) -> t.Sequence[OrderInfo]:
        """Get orders sequence."""
        return self._broker.get_orders()

    def get_trades(self) -> t.Sequence[TradeInfo]:
        """Get trades sequence."""
        return self._broker.get_trades()

    def submit_market_order(self, 
                            options: MarketOrderOptions) -> MarketOrderInfo:
        """Submit market order."""
        return self._broker.submit_market_order(options)

    def submit_limit_order(self, 
                           options: LimitOrderOptions) -> LimitOrderInfo:
        """Submit limit order."""
        return self._broker.submit_limit_order(options)

    def submit_market_short(self, 
                            options: MarketOrderOptions) -> MarketOrderInfo:
        """Submit market Short."""
        return self._broker.submit_market_short(options)

    def submit_limit_short(self, 
                           options: LimitOrderOptions) -> LimitOrderInfo:
        """Submit limit Short."""
        return self._broker.submit_limit_short(options)

    def submit_take_profit_order(
                self, 
                order_side: OrderSide,
                options: TakeProfitOptions) -> TakeProfitInfo:
        """Submit Take Profit order."""
        return self._broker.submit_take_profit_order(order_side, options)

    def submit_stop_loss_order(
                self, 
                order_side: OrderSide,
                options: StopLossOptions) -> StopLossInfo:
        """Submit Stop Loss order."""
        return self._broker.submit_stop_loss_order(order_side, options)

    def cancel_order(self, order_id: int) -> None:
        """Cancel order by id."""
        return self._broker.cancel_order(order_id)
