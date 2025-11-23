import typing as t
from datetime import datetime
from decimal import (  # https://docs.python.org/3/library/decimal.html
    ROUND_HALF_UP, ROUND_FLOOR, Decimal)

from backintime.broker.base import InsufficientFunds as BaseInsufficientFunds
from backintime.broker.base import BrokerException, OrderSubmissionError

from .orders import (LimitOrder, LimitOrderInfo, MarketOrder, MarketOrderInfo,
                     Order, OrderInfo, StopLossInfo, StopLossOrder,
                     StrategyOrder, StrategyOrders, TakeProfitInfo,
                     TakeProfitOrder, TradeInfo, OrderSide)


class InsufficientFunds(BaseInsufficientFunds):
    def __init__(self, required: Decimal, available: Decimal):
        message = f"Need {required:.8f} but only have {available:.8f}"
        super().__init__(message)


from decimal import Decimal, ROUND_HALF_UP


class FuturesBalance:
    """Contracts & USD Balance implementation for futures broker."""
    def __init__(self, 
                 usd_balance: Decimal, 
                 contracts: Decimal = Decimal(0),
                 min_usd: Decimal = Decimal('0.01'),
                 min_contracts: Decimal = Decimal('1')):
        self._usd_balance = usd_balance
        self._available_usd_balance = usd_balance
        self._contracts = contracts
        self._available_contracts = contracts
        # Used for rounding
        self._min_usd = min_usd
        self._min_contracts = min_contracts

    @property
    def available_usd_balance(self) -> Decimal:
        """Get usd available for trading."""
        return self._available_usd_balance

    @property
    def available_contracts(self) -> Decimal:
        """Get contracts available for trading."""
        return self._available_contracts
    
    @property
    def usd_balance(self) -> Decimal:
        """Get usd balance."""
        return self._usd_balance

    @property
    def contracts(self) -> Decimal:
        """Get contracts balance."""
        return self._contracts

    def hold_usd(self, amount: Decimal) -> None:
        """
        Ensure there are enough usd available for trading and
        and decrease it.
        """
        amount = amount.quantize(self._min_usd, ROUND_HALF_UP)
        if amount > self._available_usd_balance:
            raise InsufficientFunds(amount, self._available_usd_balance)
        self._available_usd_balance -= amount

    def hold_contracts(self, amount: Decimal) -> None:
        """
        Ensure there are enough contracts available for trading and
        and decrease it.
        """
        amount = amount.quantize(self._min_contracts, ROUND_HALF_UP)
        if amount > self._available_contracts:
            raise InsufficientFunds(amount, self._available_contracts)
        self._available_contracts -= amount

    def release_usd(self, amount: Decimal) -> None:
        """Increase usd available for trading."""
        amount = amount.quantize(self._min_usd, ROUND_HALF_UP)
        self._available_usd_balance += amount

    def release_contracts(self, amount: Decimal) ->  None:
        """Increase contracts available for trading."""
        amount = amount.quantize(self._min_contracts, ROUND_HALF_UP)
        self._available_contracts += amount

    def withdraw_usd(self, amount: Decimal) -> None:
        """Decrease usd balance."""
        amount = amount.quantize(self._min_usd, ROUND_HALF_UP)
        if amount > self._usd_balance:
            raise InsufficientFunds(amount, self._usd_balance)
        self._usd_balance -= amount

    def withdraw_contracts(self, amount: Decimal) -> None:
        """Decrease contracts balance."""
        amount = amount.quantize(self._min_contracts, ROUND_HALF_UP)
        if amount > self._contracts:
            raise InsufficientFunds(amount, self._contracts)
        self._contracts -= amount

    def deposit_usd(self, amount: Decimal) -> None:
        """Increase usd balance and the amount available for trading."""
        amount = amount.quantize(self._min_usd, ROUND_HALF_UP)
        self._usd_balance += amount
        self._available_usd_balance += amount

    def deposit_contracts(self, amount: Decimal) -> None:
        """Increase contracts balance and the amount available for trading."""
        amount = amount.quantize(self._min_contracts, ROUND_HALF_UP)
        self._contracts += amount
        self._available_contracts += amount

    def release_with_gain(self, collateral: Decimal, gain: Decimal) -> None:
        self._usd_balance += gain.quantize(self._min_usd, ROUND_HALF_UP)
        self._available_usd_balance += (collateral + gain).quantize(self._min_usd, ROUND_HALF_UP)

    def __repr__(self) -> str:
        usd_balance = self._usd_balance
        available_usd = self._available_usd_balance
        contracts = self._contracts
        available_contracts = self._available_contracts

        return (f"Balance(usd_balance={usd_balance:.2f}, "
                f"available_usd_balance={available_usd:.2f}, "
                f"contracts={contracts}, "
                f"available_contracts={available_contracts})")


class FuturesBalanceInfo:
    """
    Wrapper around `FuturesBalance` that provides a read-only view
    into the wrapped `FuturesBalance` data.
    """
    def __init__(self, data: FuturesBalance):
        self._data = data

    @property
    def available_usd_balance(self) -> Decimal:
        """Get usd available for trading."""
        return self._data.available_usd_balance

    @property
    def available_contracts(self) -> Decimal:
        """Get contracts available for trading."""
        return self._data.available_contracts

    @property
    def usd_balance(self) -> Decimal:
        """Get usd balance."""
        return self._data.usd_balance

    @property
    def contracts(self) -> Decimal:
        """Get contracts balance."""
        return self._data.contracts

    def __repr__(self) -> str:
        usd_balance = self.usd_balance
        available_usd = self.available_usd_balance
        contracts = self.contracts
        available_contracts = self.available_contracts

        return (f"BalanceInfo(usd_balance={usd_balance:.2f}, "
                f"available_usd_balance={available_usd:.2f}, "
                f"contracts={contracts}, "
                f"available_contracts={available_contracts})")
