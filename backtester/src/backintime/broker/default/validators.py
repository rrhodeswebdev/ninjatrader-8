"""Base validation for orders. Only check prices and amounts."""
from decimal import Decimal

from backintime.broker.base import (InvalidOrderData, LimitOrderOptions,
                                    MarketOrderOptions, StopLossOptions,
                                    TakeProfitOptions)


def validate_market_order_options(options: MarketOrderOptions) -> None:
    if options.amount is not None:
        if options.amount <= 0:
            message = "`amount` must be > 0 or None. "
            raise InvalidOrderData(f"[MarketOrderOptions]: {message}")
    elif options.percentage_amount is not None:
        if options.percentage_amount > Decimal('100') or \
                options.percentage_amount <= Decimal('0'):
            message = "`percentage_amount` must be in (0, 100] or None. "
            raise InvalidOrderData(f"[MarketOrderOptions]: {message}")
    else:
        message = "provide `amount` or `percentage_amount`. "
        raise InvalidOrderData(f"[MarketOrderOptions]: {message}")


def validate_take_profit_options(options: TakeProfitOptions) -> None:
    message = ''
    if options.amount is not None:
        if options.amount <= 0:
            message += "`amount` must be > 0 or None. "
    elif options.percentage_amount is not None:
        if options.percentage_amount > Decimal('100') or \
                options.percentage_amount <= Decimal('0'):
            message += "`percentage_amount` must be in (0, 100] or None. "
    else:
        message += "provide `amount` or `percentage_amount`. "

    if options.trigger_price <= 0:
        message += "`trigger_price` must be > 0. "
    if options.order_price and options.order_price <= 0:
        message += "`order_price` must be > 0. "
    if message:
        raise InvalidOrderData(f"[TakeProfitOptions]: {message}")


def validate_stop_loss_options(options: StopLossOptions) -> None:
    message = ''
    if options.amount is not None:
        if options.amount <= 0:
            message += "`amount` must be > 0 or None. "
    elif options.percentage_amount is not None:
        if options.percentage_amount > Decimal('100') or \
                options.percentage_amount <= Decimal('0'):
            message += "`percentage_amount` must be in (0, 100] or None. "
    else:
        message += "provide `amount` or `percentage_amount`. "

    if options.trigger_price <= 0:
        message += "`trigger_price` must be > 0. "
    if options.order_price and options.order_price <= 0:
        message += "`order_price` must be > 0. "
    if message:
        raise InvalidOrderData(f"[StopLossOptions]: {message}")


def validate_limit_order_options(options: LimitOrderOptions) -> None:
    message = ''
    
    if options.amount is not None:
        if options.amount <= 0:
            message += "`amount` must be > 0 or None. "
    elif options.percentage_amount is not None:
        if options.percentage_amount > Decimal('100') or \
                options.percentage_amount <= Decimal('0'):
            message += "`percentage_amount` must be in (0, 100] or None. "
    else:
        message += "provide `amount` or `percentage_amount`. "

    if options.order_price <= 0:
        message += "`order_price` must be > 0. "

    if options.take_profit:
        try:
            validate_take_profit_options(options.take_profit)
        except InvalidOrderData as e:
            message += str(e)

    if options.stop_loss:
        try:
            validate_stop_loss_options(options.stop_loss)
        except InvalidOrderData as e:
            message += str(e)

    if message:
        raise InvalidOrderData(f"[LimitOrderOptions]: {message}")