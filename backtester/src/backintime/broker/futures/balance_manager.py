from decimal import Decimal, ROUND_FLOOR
from dataclasses import dataclass
from datetime import datetime

from backintime.broker.futures.balance import FuturesBalance, FuturesBalanceInfo
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    MarketOrderOptions, LimitOrderOptions,
                                    StopLossOptions, TakeProfitOptions,
                                    BrokerException)
from .position import Position
from .contracts import ContractSettings, ContractUtils
from .orders import StopLossOrder, TakeProfitOrder


class MarginCall(BrokerException):
    def __init__(self, equity_low: Decimal, num_contracts: Decimal,
                 maintenance_margin: Decimal, curr_time: datetime):
        message = (f"The lowest equity for the candle closed at {curr_time} "
                   f"is {equity_low}. This is not enough to cover "
                   f"the maintenance margin {maintenance_margin} "
                   f"for {num_contracts} purchased contracts. "
                   f"Margin Call occured at {curr_time}.")
        super().__init__(message)


class BalanceManager:
    """BalanceManager modifies the balance in accordance with orders submission, cancellation and execution.
    It operates on a higher level than the underlying balance functions.

    BalanceManager assumes it may be either in Long, in Short, or in no position at all.
    If there's no position or it is Long, BUY order opens new position entry for Long,
    and SELL order closes position (fully or partially).
    If there's no position or it is Short, SELL order opens new position entry for Short,
    and BUY orders close position (fully or partially).

    Currently you have to be in either position to submit TP/SL orders.
    You can't go Long and Short simultaneously. From the BalanceManager point of view it
    is effectively the same as no position at all."""

    def __init__(self, start_money: Decimal, contract_settings = ContractSettings()):
        self.in_long = False
        self.in_short = False
        self._balance = FuturesBalance(start_money)
        self._balance_info = FuturesBalanceInfo(self._balance)
        self._position = Position()
        self._utils = ContractUtils(contract_settings)

    @property
    def balance(self):
        return self._balance_info

    @property
    def position(self):
        return self._position

    @property
    def traded_contracts(self):
        total_amount = self._position.total_amount
        init_margin = self._utils._settings.per_contract_init_margin
        additional_collateral = self._utils._settings.additional_collateral
        margin_req = init_margin + additional_collateral
        return (total_amount / margin_req).quantize(Decimal('1'), ROUND_FLOOR)

    @property
    def available_contracts(self):
        available = self._position.available_amount
        init_margin = self._utils._settings.per_contract_init_margin
        additional_collateral = self._utils._settings.additional_collateral
        margin_req = init_margin + additional_collateral
        return (available / margin_req).quantize(Decimal('1'), ROUND_FLOOR)

    @property
    def max_funds_for_futures(self):
        available_usd = self.balance.available_usd_balance
        contracts, collateral, fee = self._utils.estimate_init(available_usd)
        return collateral

    def get_equity(self, price):
        entries = self._position.entries
        fills = [ x.fill_price for x in entries ]
        contracts = [ self._utils.get_contracts(x.amount) for x in entries ]
        # collateral = self._utils.get_collateral(sum(contracts))
        net_gain, _ = self._utils.get_net_gain(price, fills, 
                                contracts, long=self.in_long)
        return self.balance.usd_balance + net_gain

    def check_margin_call(self, high, low, curr_time: datetime, is_overnight: bool = False) -> None:
        equity = 0
        contracts = self.traded_contracts
        # Long
        if self.in_long:
            equity = self.get_equity(low)
        # Short
        elif self.in_short:
            equity = self.get_equity(high)
        # Neither
        else:
            return
        # Compute margin requirements
        margin_req = self._utils.get_overnight_margin(contracts) if is_overnight \
                            else self._utils.get_maintenance_margin(contracts)
        # Raise Margin Call if current equity does not meet margin requirements
        if equity < margin_req:
            raise MarginCall(equity, contracts, margin_req, curr_time)

    def _get_usd_amount(self, amount=None, percentage_amount=None):
        if amount:
            num_contracts, amount, fee = self._utils.estimate_init_net(amount)
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')
            return amount, fee

        elif percentage_amount:
            amount = self._position.entries[-1].amount
            ratio = percentage_amount/100
            partial_amount = amount * ratio # quantize?
            num_contracts, amount, fee = self._utils.estimate_init_net(partial_amount)
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')
            return amount, fee
        else:
            raise Exception('Neither amount nor percentage_amount provided.')

    def hold_for_tp(self, order_side, order_id, options): # -> evaluated amount
        if not (self.in_long or self.in_short):
            raise Exception('No position found. TP/SL currently can not be posted')

        amount, fee = self._get_usd_amount(options.amount, options.percentage_amount)
        self._position.open_tp(order_id, amount)
        return amount

    def hold_for_sl(self, order_side, order_id, options): # -> evaluated amount
        if not (self.in_long or self.in_short):
            raise Exception('No position found. TP/SL currently can not be posted')

        amount, fee = self._get_usd_amount(options.amount, options.percentage_amount)
        self._position.open_sl(order_id, amount)
        return amount

    def release_tp(self, order_id):
        self._position.release_tp(order_id)

    def release_sl(self, order_id):
        self._position.release_sl(order_id)

    def _hold_position(self, options):  # TODO: handle 0 amount (exception?)
        if options.amount:
            result = self._utils.estimate_init_net(options.amount)
            num_contracts, margin, fee = result
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')
            self._position.acquire(margin)
            return margin

        else:
            ratio = options.percentage_amount / 100
            max_usd = self._position.available_amount
            partial_amount = max_usd * ratio
            result = self._utils.estimate_maintenance_net(partial_amount)
            num_contracts, margin, fee = result
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')
            self._position.acquire(margin)
            return margin

    def _hold_balance(self, options):   # TODO: handle 0 amount (exception?)
        if options.amount:
            result = self._utils.estimate_init_net(options.amount)
            num_contracts, init_req, fee = result
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')

            self._balance.hold_usd(init_req + fee)
            #return self._utils.get_collateral(num_contracts)
            return init_req

        else:
            ratio = options.percentage_amount / 100
            max_usd = self._balance.available_usd_balance
            partial_amount = max_usd * ratio
            result = self._utils.estimate_init(partial_amount)
            num_contracts, init_req, fee = result
            if num_contracts == 0:
                raise Exception('Not enough funds to trade at least 1 contract.')

            self._balance.hold_usd(init_req + fee)
            # return self._utils.get_collateral(num_contracts)
            return init_req

    def _release_position(self, amount):
        self._position.release(amount)

    def _release_balance(self, amount):
        _, init_req, fee = self._utils.estimate_init_net(amount)
        self._balance.release_usd(init_req + fee)

    def hold_funds(self, options, short=False):
        if self.in_long:  # Long
            if options.order_side == OrderSide.SELL and not short:  # Close Long
                return self._hold_position(options)
        elif self.in_short:  # Short
            if options.order_side == OrderSide.BUY and short:  # Close Short
                return self._hold_position(options)
        # Hold from the balance to enter position (Long/Short)
        return self._hold_balance(options)

    def release_funds(self, order_side, amount, is_short=False):
        # May it be the case that it was held for Long but it is Short now or no position at all?
        if self.in_long:  # Long
            if order_side == OrderSide.SELL and not is_short:  # it was close Long
                self._release_position(amount)
        elif self.in_short:  # Short
            if order_side == OrderSide.BUY and is_short:  # it was close Short
                self._release_position(amount)
        # Relase the balance
        else:
            self._release_balance(amount)

    def process_order(self, order, fill_price):
        if order.side is OrderSide.BUY:
            return self.buy(order.amount, fill_price)
        elif order.side is OrderSide.SELL:
            return self.sell(order.amount, fill_price)
        else:
            raise Exception('Invalid order side')

    def process_tpsl(self, order_id, order, fill_price):
        if order.side is OrderSide.BUY:
            if isinstance(order, TakeProfitOrder):
                return self.tp_buy(order_id, fill_price)
            elif isinstance(order, StopLossOrder):
                return self.sl_buy(order_id, fill_price)
        elif order.side is OrderSide.SELL:
            if isinstance(order, TakeProfitOrder):
                return self.tp_sell(order_id, fill_price)
            elif isinstance(order, StopLossOrder):
                return self.sl_sell(order_id, fill_price)
        else:
            raise Exception('Invalid order side')

    def buy(self, amount, fill_price):
        """Modify balance and positions in accordance with BUY order execution (Market/Limit).
        If in Long, or no position, add new Long position entry.
        If in Short, close short position."""
        if self.in_short:
            return self._close_short(amount, fill_price)
        else:   # Long or no position
            return self._open_long(amount, fill_price)

    def sell(self, amount, fill_price):
        """Modify balance and positions in accordance with SELL order execution (Market/Limit).
        If in Short, or no position, add new Short position entry.
        If in Long, close long position."""
        if self.in_long:
            return self._close_long(amount, fill_price)
        else:   # Short or no position
            return self._open_short(amount, fill_price)

    def tp_buy(self, order_id, fill_price):
        """Modify balance and positions in accordance with TP BUY execution."""
        if self.in_short:
            return self._close_short_tp(order_id, fill_price)
        else: # ?
            # return self._open_long(amount, fill_price)
            raise NotImplementedError('TP BUY is currently implemented for Short only.')

    def tp_sell(self, order_id, fill_price):
        """Modify balance and positions in accordance with TP SELL execution."""
        if self.in_long:
            return self._close_long_tp(order_id, fill_price)
        else: # ?
            # return self._open_short(fill_price)
            raise NotImplementedError('TP SELL is currently implemented for Long only.')

    def sl_buy(self, order_id, fill_price):
        """Modify balance and positions in accordance with SL BUY execution."""
        if self.in_short:
            return self._close_short_sl(order_id, fill_price)
        else: # ?
            # return self._open_long(amount, fill_price)
            raise NotImplementedError('SL BUY is currently implemented for Short only.')

    def sl_sell(self, order_id, fill_price):
        """Modify balance and positions in accordance with SL SELL execution."""
        if self.in_long:
            return self._close_long_sl(order_id, fill_price)
        else: # ?
            # return self._open_short(fill_price)
            raise NotImplementedError('SL SELL is currently implemented for Long only.')

    def _open_long(self, usd_amount, fill_price):
        num_contracts, init_req, init_fee = self._utils.estimate_init_net(usd_amount)
        if num_contracts == 0:
                raise Exception('Cannot open Long: order amount is too small to trade at least 1 contract.')

        self._balance.withdraw_usd(init_fee)
        self._position.open(usd_amount, fill_price)
        self.in_long = True
        return init_fee

    def _close_long(self, usd_amount, fill_price):
        # Remove long
        entries = self._position.close(usd_amount)
        # Estimate gain and collateral
        fills = [ x.fill_price for x in entries ]
        contracts = [ self._utils.get_contracts(x.amount) for x in entries ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        self.in_long = len(self._position) > 0
        return fee

    def _close_long_tp(self, order_id, fill_price):
        entry, to_release, cancelled = self._position.close_tp(order_id)
        # Estimate gain, get collateral
        fills = [ entry.fill_price ]
        contracts = [ self._utils.get_contracts(entry.amount) ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        self.in_long = len(self._position) > 0
        return fee, cancelled

    def _close_long_sl(self, order_id, fill_price):
        entry, to_release, cancelled = self._position.close_sl(order_id)
        # Estimate gain, get collateral
        fills = [ entry.fill_price ]
        contracts = [ self._utils.get_contracts(entry.amount) ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        self.in_long = len(self._position) > 0
        return fee, cancelled

    def _open_short(self, usd_amount, fill_price):
        num_contracts, init_req, init_fee = self._utils.estimate_init_net(usd_amount)  # TODO: handle 0 contracts case
        if num_contracts == 0:
                raise Exception('Cannot open Short: order amount is too small to trade at least 1 contract.')

        self._balance.withdraw_usd(init_fee)
        self._position.open(usd_amount, fill_price)
        self.in_short = True
        return init_fee

    def _close_short(self, usd_amount, fill_price):
        # Remove Short
        entries = self._position.close(usd_amount)
        # Estimate gain, get collateral
        fills = [ x.fill_price for x in entries ]
        contracts = [ self._utils.get_contracts(x.amount) for x in entries ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts, long=False)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        self.in_short = len(self._position) > 0
        return fee

    def _close_short_tp(self, order_id, fill_price):
        # Remove Short
        entry, to_release, cancelled = self._position.close_tp(order_id)
        # Estimate gain, get collateral
        fills = [ entry.fill_price ]
        contracts = [ self._utils.get_contracts(entry.amount) ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts, long=False)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        to_release and self._balance.release_usd(to_release)
        self.in_short = len(self._position) > 0
        return fee, cancelled

    def _close_short_sl(self, order_id, fill_price):
        # Remove Short
        entry, to_release, cancelled = self._position.close_sl(order_id)
        # Estimate gain, get collateral
        fills = [ entry.fill_price ]
        contracts = [ self._utils.get_contracts(entry.amount) ]
        collateral = self._utils.get_collateral(sum(contracts))
        gain, fee = self._utils.get_net_gain(fill_price, fills, contracts, long=False)
        # Modify balance
        self._balance.release_with_gain(collateral, gain)
        to_release and self._balance.release_usd(to_release)
        self.in_short = len(self._position) > 0
        return fee, cancelled