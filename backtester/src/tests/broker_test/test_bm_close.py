from decimal import Decimal

import pytest

from backintime.broker.futures.balance_manager import BalanceManager
from backintime.broker.futures.contracts import ContractSettings
from backintime.broker.futures.balance import FuturesBalance, FuturesBalanceInfo
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    MarketOrderOptions, LimitOrderOptions,
                                    StopLossOptions, TakeProfitOptions)


@pytest.fixture
def bm_1601() -> BalanceManager:
    """
    Setup BalanceManager with initial balance of 1601,
    initial margin=1600, mantenance margin=1500,
    overnight_margin=2200, fee=0.62 and quotient=2.
    """
    settings = ContractSettings(
        per_contract_fee = Decimal('0.62'),
        per_contract_quotient = Decimal('2'),
        per_contract_init_margin = Decimal('1600'),
        per_contract_maintenance_margin = Decimal('1500'),
        per_contract_overnight_margin = Decimal('2200'),
        additional_collateral = Decimal('0'))

    return BalanceManager(Decimal('1601'), settings)


@pytest.fixture
def bm_3202() -> BalanceManager:
    """
    Setup BalanceManager with initial balance of 3202,
    initial margin=1600, mantenance margin=1500,
    overnight_margin=2200, fee=0.62 and quotient=2.
    """
    settings = ContractSettings(
        per_contract_fee = Decimal('0.62'),
        per_contract_quotient = Decimal('2'),
        per_contract_init_margin = Decimal('1600'),
        per_contract_maintenance_margin = Decimal('1500'),
        per_contract_overnight_margin = Decimal('2200'),
        additional_collateral = Decimal('0'))

    return BalanceManager(Decimal('3202'), settings)


def test_close_long_with_single_order(bm_3202):
    # Do some initial BUY first to enter Long
    bm = bm_3202
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # 3202 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same

    bm.buy(Decimal('1600'), fill_price=Decimal('100'))
    assert bm.in_long == True
    assert bm.balance.available_usd_balance == Decimal('1601.38') # 3202 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3201.38')  # 3202 - fee(0.62)
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')  # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for SELL
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    amount = bm.hold_funds(sell_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38')  # same
    assert bm.balance.usd_balance == Decimal('3201.38')  # same
    assert bm.position.total_amount == Decimal('1600')  # same
    assert bm.position.available_amount == Decimal('0')  # 1600 - amount (1600)
    assert bm.traded_contracts == Decimal('1')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    bm.sell(amount, Decimal('110'))
    assert bm.balance.available_usd_balance == Decimal('3220.76')  # 3201.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3220.76')  # 3201.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_close_short_with_single_order(bm_3202):
    # Do some initial SELL first to enter Short
    bm = bm_3202
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    amount = bm.hold_funds(sell_opt, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # 3202 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same

    bm.sell(Decimal(1600), fill_price=Decimal(100))
    assert bm.in_short == True
    assert bm.balance.available_usd_balance == Decimal('1601.38')  # 3202 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3201.38')  # minus 0.62 fee
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')  # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for BUY
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal(1600))
    amount = bm.hold_funds(buy_opt, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # same
    assert bm.balance.usd_balance == Decimal('3201.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1500)
    assert bm.traded_contracts == Decimal('1')     # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    bm.buy(amount, Decimal('90'))
    assert bm.balance.available_usd_balance == Decimal('3220.76')  # 3201.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3220.76')  # 3201.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_close_long_with_tp(bm_1601):
    # Do some initial BUY first to enter Long
    bm = bm_1601
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same

    bm.buy(Decimal('1600'), Decimal('100'))
    assert bm.in_long == True
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')   # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for TP SELL
    tp_id = 0
    tp_opt = TakeProfitOptions(amount=Decimal('1600'), trigger_price=Decimal('110'))
    amount = bm.hold_for_tp(OrderSide.SELL, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1500)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    # Hold for SL SELL
    sl_id = 1
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.SELL, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: minus amount(1500)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract

    # Close Long with TP SELL
    fee, to_cancel = bm.tp_sell(tp_id, fill_price=Decimal('110'))
    assert fee == Decimal('0.62')
    assert to_cancel == [sl_id]
    assert bm.balance.available_usd_balance == Decimal('1619.76')  # 1600.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1619.76')  # 1600.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_close_long_with_sl(bm_1601):
    # Do some initial BUY first to enter Long
    bm = bm_1601
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same

    bm.buy(Decimal('1600'), Decimal('100'))
    assert bm.in_long == True
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')   # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for TP SELL
    tp_id = 0
    tp_opt = TakeProfitOptions(amount=Decimal('1600'), trigger_price=Decimal('110'))
    amount = bm.hold_for_tp(OrderSide.SELL, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    # Hold for SL SELL
    sl_id = 1
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.SELL, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract

    # Close Long with SL SELL
    fee, to_cancel = bm.sl_sell(sl_id, fill_price=Decimal('90'))
    assert fee == Decimal('0.62')
    assert to_cancel == [tp_id]
    assert bm.balance.available_usd_balance == Decimal('1579.76')  # 1600.38 - price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1579.76')  # 1600.38 - price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_close_short_with_tp(bm_1601):
    # Do some initial SELL first to enter Short
    bm = bm_1601
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    amount = bm.hold_funds(sell_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same

    bm.sell(Decimal('1600'), Decimal('100'))
    assert bm.in_short == True
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')  # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for TP BUY
    tp_id = 0
    tp_opt = TakeProfitOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_tp(OrderSide.BUY, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 - amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    # Hold for SL BUY
    sl_id = 1
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.BUY, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract

    # Close Short with TP BUY
    fee, to_cancel = bm.tp_buy(tp_id, fill_price=Decimal('90'))
    assert fee == Decimal('0.62')
    assert to_cancel == [sl_id]
    assert bm.balance.available_usd_balance == Decimal('1619.76')  # 1600.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1619.76')  # 1600.38 + price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_close_short_with_sl(bm_1601):
    # Do some initial SELL first to enter Short
    bm = bm_1601
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    amount = bm.hold_funds(sell_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same

    bm.sell(Decimal('1600'), Decimal('100'))
    assert bm.in_short == True
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)
    assert bm.position.total_amount == Decimal('1600')
    assert bm.position.available_amount == Decimal('1600')
    assert bm.traded_contracts == Decimal('1')  # 1600 == 1 contract
    assert bm.available_contracts == Decimal('1')  # 1600 == 1 contract

    # Hold for TP BUY
    tp_id = 0
    tp_opt = TakeProfitOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_tp(OrderSide.BUY, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 - amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    # Hold for SL BUY
    sl_id = 1
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.BUY, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: 1600 - amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract

    # Close Short with SL BUY
    fee, to_cancel = bm.sl_buy(sl_id, fill_price=Decimal('110'))
    assert fee == Decimal('0.62')
    assert to_cancel == [tp_id]
    assert bm.balance.available_usd_balance == Decimal('1579.76')  # 1600.38 - price_diff(10)*quotient(2) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1579.76')  # 1600.38 - price_diff(10)*quotient(2) - fee(0.62)
    assert bm.position.total_amount == Decimal('0')
    assert bm.position.available_amount == Decimal('0')
    assert bm.traded_contracts == Decimal('0')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract
