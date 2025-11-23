import typing as t
from datetime import datetime
from decimal import (  # https://docs.python.org/3/library/decimal.html
    ROUND_FLOOR, ROUND_HALF_UP, Decimal)
from itertools import count

from backintime.broker.base import (AbstractBroker, BrokerException,
                                    LimitOrderOptions, MarketOrderOptions,
                                    OrderCancellationError, OrderSide,
                                    OrderStatus, OrderSubmissionError,
                                    OrderType, StopLossOptions,
                                    TakeProfitOptions)
from backintime.session import FuturesSession
from backintime.broker.default.repo import OrdersRepository
from backintime.broker.default.validators import (
    validate_limit_order_options, validate_market_order_options,
    validate_stop_loss_options, validate_take_profit_options)

from .contracts import ContractSettings
from .matching import MatchingEngine as matching
from .balance_manager import BalanceManager
from .balance import FuturesBalance, FuturesBalanceInfo
from .orders import (LimitOrder, LimitOrderInfo, MarketOrder, MarketOrderInfo,
                     Order, OrderInfo, StopLossInfo, StopLossOrder,
                     StrategyOrder, StrategyOrders, TakeProfitInfo,
                     TakeProfitOrder, TradeInfo)


class OrderNotFound(OrderCancellationError):
    def __init__(self, order_id: int):
        message = f"Order with order_id={order_id} was not found"
        super().__init__(message)


class FuturesBroker(AbstractBroker):
    def __init__(self, 
                 start_money: Decimal,
                 session: t.Optional[FuturesSession] = None,
                 min_usd: Decimal = Decimal('0.01'),
                 min_contracts: Decimal = Decimal('1'),
                 per_contract_init_margin = Decimal('1699.64'),
                 additional_collateral = Decimal('0'),
                 contract_quotient: Decimal = Decimal('2'),
                 per_contract_fee: Decimal = Decimal('0.62'),
                 check_margin_call: bool = False,
                 per_contract_maintenance_margin = Decimal('1500'),
                 per_contract_overnight_margin = Decimal('2200')):
        #
        self._session = session
        self._min_usd = min_usd
        self._min_contracts = min_contracts
        contract_settings = ContractSettings(
                per_contract_fee,
                contract_quotient,
                per_contract_init_margin,
                per_contract_maintenance_margin,
                per_contract_overnight_margin,
                additional_collateral
        )
        self._bm = BalanceManager(start_money, contract_settings)
        self._orders = OrdersRepository()
        # Let's just make it as a simple list as for now
        self._trades_counter = count()
        self._trades: t.List[TradeInfo] = []
        # Close time of the current candle
        self._current_time: t.Optional[datetime] = None
        # Close price of the current (last) candle
        self._current_price: t.Optional[Decimal] = None
        self._check_margin_call_flag = check_margin_call

    @property
    def in_long(self) -> bool:
        return self._bm.in_long

    @property
    def in_short(self) -> bool:
        return self._bm.in_short

    @property
    def balance(self) -> FuturesBalanceInfo:
        """Get balance info."""
        return self._bm.balance

    @property
    def current_equity(self) -> Decimal:
        """Get current equity."""
        return self._bm.get_equity(self._current_price)

    @property
    def max_funds_for_futures(self):
        return self._bm.max_funds_for_futures

    @property
    def position(self):
        return self._bm.position.available_amount

    def iter_orders(self) -> t.Iterator[OrderInfo]:
        """Get orders iterator."""
        for order_id, order in self._orders:
            yield OrderInfo(order_id, order) 

    def get_orders(self) -> t.List[OrderInfo]:
        """Get orders list."""
        return list(self.iter_orders())

    def iter_trades(self) -> t.Iterator[TradeInfo]:
        """Get trades iterator."""
        return iter(self._trades)

    def get_trades(self) -> t.List[TradeInfo]:
        """Get trades list."""
        return list(self._trades)

    def submit_market_order(
                self, 
                options: MarketOrderOptions) -> MarketOrderInfo:
        """Submit market order (Long)."""
        order = self._create_market_order(options)
        order_id = self._orders.add_market_order(order)
        return MarketOrderInfo(order_id, order)

    def submit_market_short(
                self, 
                options: MarketOrderOptions) -> MarketOrderInfo:
        """Submit market order (Short)."""
        order = self._create_market_order(options, short=True)
        order_id = self._orders.add_market_order(order)
        return MarketOrderInfo(order_id, order)

    def submit_limit_order(
                self, 
                options: LimitOrderOptions) -> LimitOrderInfo:
        """Submit limit order (Long)."""
        #print(f"Submit Limit {options.order_side} {options.amount} {self.balance.available_usd_balance}")
        order = self._create_limit_order(options)
        order_id = self._orders.add_limit_order(order)
        #print(f"Submit Limit order #{order_id} {order.amount}")
        strategy_orders = self._orders.get_linked_orders(order_id)
        return LimitOrderInfo(order_id, order, strategy_orders)

    def submit_limit_short(
                self, 
                options: LimitOrderOptions) -> LimitOrderInfo:
        """Submit limit order (Short)."""
        #print(f"Submit Limit Short {options.order_side} {options.amount} {self.balance.available_usd_balance}")
        order = self._create_limit_order(options, short=True)
        order_id = self._orders.add_limit_order(order)
        #print(f"Submit Limit Short #{order_id} {order.amount}")
        strategy_orders = self._orders.get_linked_orders(order_id)
        return LimitOrderInfo(order_id, order, strategy_orders)

    def submit_take_profit_order(
                self, 
                order_side: OrderSide, 
                options: TakeProfitOptions) -> TakeProfitInfo:
        """Submit Take Profit order."""
        order_id = self._orders.get_next_order_id()
        order = self._create_take_profit(order_id, order_side, options)
        self._orders.add_take_profit_order(order_id, order)
        return TakeProfitInfo(order_id, order)

    def submit_stop_loss_order(
                self, 
                order_side: OrderSide, 
                options: StopLossOptions) -> StopLossInfo:
        """Submit Stop Loss order."""
        order_id = self._orders.get_next_order_id()
        order = self._create_stop_loss(order_id, order_side, options)
        self._orders.add_stop_loss_order(order_id, order)
        return StopLossInfo(order_id, order)

    def cancel_order(self, order_id: int) -> None:
        """Cancel order by id."""
        order = self._orders.get_order(order_id)
        if not order:
            raise OrderNotFound(order_id)

        if not order.status is OrderStatus.CREATED and \
                not order.status is OrderStatus.ACTIVATED:
            raise OrderCancellationError(
                            f"Order can't be cancelled, because "
                            f"order status is {order.status}")

        if isinstance(order, MarketOrder):
            self._bm.release_funds(order.side, order.amount, order.is_short)
            self._orders.remove_market_order(order_id)
        elif isinstance(order, LimitOrder):
            self._bm.release_funds(order.side, order.amount, order.is_short)
            self._orders.remove_limit_order(order_id)
        elif isinstance(order, TakeProfitOrder):
            self._bm.release_tp(order_id)
            self._orders.remove_take_profit_order(order_id)
        elif isinstance(order, StopLossOrder):
            self._bm.release_sl(order_id)
            self._orders.remove_stop_loss_order(order_id)
        order.status = OrderStatus.CANCELLED

    def _add_trade(self, order_id: int, order: Order) -> None:
        """Add new trade."""
        trade_id = next(self._trades_counter)
        order_info = OrderInfo(order_id, order)
        balance = self.balance.usd_balance
        self._trades.append(TradeInfo(trade_id, order_info, balance))

    def _submit_linked_tpsl(self, order_id, order):
        # Invert order side
        #print(f"Submit linked TP/SL for #{order_id}")
        side = OrderSide.BUY if order.side is OrderSide.SELL \
                    else OrderSide.SELL

        if order.take_profit_options:
            tp_id = self._orders.get_next_order_id()
            tp = self._create_take_profit(tp_id, side, 
                            order.take_profit_options)
            #print(f"TP #{tp_id} {tp.amount}")
            self._orders.add_linked_take_profit_order(tp_id, tp, order_id)

        if order.stop_loss_options:
            sl_id = self._orders.get_next_order_id()
            sl = self._create_stop_loss(sl_id, side, 
                            order.stop_loss_options)
            #print(f"SL #{sl_id} {sl.amount}")
            self._orders.add_linked_stop_loss_order(sl_id, sl, order_id)

    def _create_take_profit(self, 
                            order_id,
                            order_side: OrderSide,
                            options: TakeProfitOptions) -> TakeProfitOrder:
        """
        Initialize Take Profit order and acquire amount for execution.

        Acquired amount can be shared with other TP/SL orders.
        Should new TP/SL be posted, it can then acquire funds
        from the shared position without modifying the balance.
        """
        validate_take_profit_options(options)
        amount = self._bm.hold_for_tp(order_side, order_id, options)
        # Store order and return info
        return TakeProfitOrder(order_side,
                               amount=amount,
                               trigger_price=options.trigger_price,
                               order_price=options.order_price,
                               min_usd=self._min_usd, 
                               min_contracts=self._min_contracts, 
                               date_created=self._current_time)

    def _create_stop_loss(self, 
                          order_id,
                          order_side: OrderSide,
                          options: StopLossOptions) -> StopLossOrder:
        """
        Initialize Stop Loss order and acquire amount for execution.

        Acquired amount can be shared with other TP/SL orders.
        Should new TP/SL be posted, it can then acquire funds
        from the shared position without modifying the balance.
        """
        validate_stop_loss_options(options)
        amount = self._bm.hold_for_sl(order_side, order_id, options)
        # Store order and return info
        return StopLossOrder(order_side,
                             amount=amount,
                             trigger_price=options.trigger_price,
                             order_price=options.order_price,
                             min_usd=self._min_usd, 
                             min_contracts=self._min_contracts, 
                             date_created=self._current_time)

    def _create_market_order(
                self, 
                options: MarketOrderOptions,
                short: bool=False) -> MarketOrder:
        """Initialize Market order and hold funds for execution."""
        validate_market_order_options(options)
        amount = self._bm.hold_funds(options, short=short)
        # Store order and return info
        return MarketOrder(options.order_side, 
                           amount=amount,
                           is_short=short,
                           min_usd=self._min_usd, 
                           min_contracts=self._min_contracts, 
                           date_created=self._current_time)

    def _create_limit_order(
                self, 
                options: LimitOrderOptions,
                short: bool=False) -> LimitOrder:
        """Initialize Limit order and hold funds for execution."""
        validate_limit_order_options(options)
        amount = self._bm.hold_funds(options, short=short)
        # Store order and return info
        return LimitOrder(options.order_side,
                          amount=amount,
                          order_price=options.order_price,
                          is_short=short,
                          take_profit=options.take_profit,
                          stop_loss=options.stop_loss,
                          min_usd=self._min_usd, 
                          min_contracts=self._min_contracts, 
                          date_created=self._current_time)

    def _execute_market_orders(self, market_price: Decimal) -> None:
        """Execute all market orders and remove them from the internal storage."""
        for order_id, order in self._orders.get_market_orders():
            if isinstance(order, MarketOrder):
                self._execute_market_order(order_id, order, market_price)
        self._orders.remove_market_orders()     # remove all

    def _execute_market_order(self, 
                              order_id: int,
                              order: MarketOrder,
                              fill_price: Decimal) -> None:
        """
        Execute Market order and modify balance accordingly.
        All TP/SL orders will be cancelled.
        """
        fee = self._bm.process_order(order, fill_price)
        order.status = OrderStatus.EXECUTED
        order.fill_price = fill_price
        order.trading_fee = fee
        order.date_updated = self._current_time
        self._add_trade(order_id, order)

    def _execute_limit_order(self,
                             order_id: int,
                             order: LimitOrder,
                             fill_price: Decimal) -> None:
        """
        Execute Limit order.
        Modify balance accordingly and remove the order from
        the internal storage. All TP/SL orders, except for those
        submitted for this Limit order, will be cancelled.
        """
        #print(f"Position: {self._bm.position.entries}")
        #print(f"Acq.: {list(map(lambda x: x.acquired, self._bm.position._entries.values()))}")
        #print(f"Execute Limit #{order_id}")
        fee = self._bm.process_order(order, fill_price)
        
        #print(f"Position: {self._bm.position.entries}")
        #print(f"Acq.: {list(map(lambda x: x.acquired, self._bm.position._entries.values()))}")
        
        order.status = OrderStatus.EXECUTED
        order.fill_price = fill_price
        order.trading_fee = fee
        order.date_updated = self._current_time
        self._orders.remove_limit_order(order_id)
        self._add_trade(order_id, order)
        self._submit_linked_tpsl(order_id, order)
        #print(f"Position: {self._bm.position.entries}")
        #print(f"Acq.: {list(map(lambda x: x.acquired, self._bm.position._entries.values()))}")

    def _activate_strategy_order(self, 
                                 order_id: int, 
                                 order: StrategyOrder) -> None:
        """
        Activate TP/SL order. The order will be then treated as
        usual Market/Limit order, depending on whether 
        the `order_price` is provided.
        """
        order.status = OrderStatus.ACTIVATED
        order.date_activated = self._current_time
        order.date_updated = self._current_time

    def _execute_strategy_order(self, 
                                order_id: int, 
                                order: StrategyOrder,
                                fill_price: Decimal) -> None:
        """
        Execute TP/SL order. 
        Modify balance accordingly and remove the order from
        the internal storage. Other TP/SL orders will be cancelled.
        """
        # Do not execute removed
        #print(f"[TP/SL] execute #{order_id}")
        if not (order_id, order) in self._orders.get_strategy_orders():
            return
        #print(f"It hasn't been cancelled...")

        #print(f"Position: {self._bm.position.entries}")
        #print(f"Acq.: {list(map(lambda x: x.acquired, self._bm.position._entries.values()))}")

        fee, to_cancel = self._bm.process_tpsl(order_id, order, fill_price)
        order.status = OrderStatus.EXECUTED
        order.fill_price = fill_price
        order.trading_fee = fee
        order.date_updated = self._current_time
        self._orders.remove_strategy_order(order_id)
        self._add_trade(order_id, order)

        #print(f"Fee: {fee}, to_cancel: {to_cancel}")

        for order_id in to_cancel:
            order = self._orders.get_order(order_id)
            if not order:
                raise OrderNotFound(order_id)

            order.status = OrderStatus.SYS_CANCELLED
            order.date_updated = self._current_time
            self._orders.remove_strategy_order(order_id)
            #print(f"SYS_CANCELLED #{order_id}")
            #print('')
        #print(f"Position: {self._bm.position.entries}")
        #print(f"Acq.: {list(map(lambda x: x.acquired, self._bm.position._entries.values()))}")


    def _check_margin_call_if_needed(self, candle):
        """
        Check margin call if `check_margin_call_flag` is set.
        If session is used, and current margin is overnight,
        overnight margin is used. Maintenance margin otherwise.
        """
        if self._check_margin_call_flag:
            if self._session:
                is_overnight = self._session.is_overnight(candle.close_time)
            else:
                is_overnight = False
            self._bm.check_margin_call(candle.high, candle.low, 
                        candle.close_time, is_overnight=is_overnight)

    def _review_orders(self, candle):
        # Execute all market orders
        self._execute_market_orders(candle.open)
        # Review orders with limited price
        high = candle.high
        low = candle.low
        open_price = candle.open

        # First review with OPEN price
        orders = list(self._orders.get_limit_orders())
        for upd in matching.get_updates_for_open(orders, open_price):
            order, order_id, status, fill_price = upd
            if isinstance(order, StrategyOrder):
                if status is 'ACTIVATE':
                    self._activate_strategy_order(order_id, order)
                elif status is 'EXECUTE':
                    self._execute_strategy_order(order_id, order, fill_price)

            elif isinstance(order, LimitOrder):
                self._execute_limit_order(order_id, order, fill_price)

        # Then with HIGH/LOW prices
        orders = list(self._orders.get_limit_orders())
        for upd in matching.get_updates_for_high_low(orders, high, low):
            order, order_id, status, fill_price = upd
            if isinstance(order, StrategyOrder):
                if status is 'ACTIVATE':
                    self._activate_strategy_order(order_id, order)
                elif status is 'EXECUTE':
                    self._execute_strategy_order(order_id, order, fill_price)

            elif isinstance(order, LimitOrder):
                self._execute_limit_order(order_id, order, fill_price)

    def update(self, candle) -> None:
        """Review whether orders can be executed."""
        # Check Margin Call
        self._check_margin_call_if_needed(candle)
        # Update some state vars
        self._current_time = candle.close_time
        self._current_price = candle.close
        # Check if session is open
        if not self._session or self._session.is_open(candle.close_time):
            self._review_orders(candle)