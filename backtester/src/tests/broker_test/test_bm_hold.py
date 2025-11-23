from decimal import Decimal

import pytest

from backintime.broker.futures.balance_manager import BalanceManager
from backintime.broker.futures.balance import FuturesBalance, FuturesBalanceInfo
from backintime.broker.futures.contracts import ContractSettings
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    MarketOrderOptions, LimitOrderOptions,
                                    StopLossOptions, TakeProfitOptions)
# order amounts is in USD, in maintenance margin
# when posted, the funds are held for initial margin
# when executed the diff between the init and maintenance is returned
# so the margin held is the maintenance
# when cancelled, initial is returned
# when posted, funds are held for initial margin + fee, but what's the order amount?
# maybe order amount is in USD meaning maintenance + additional_collateral you can afford for this order?
'''
When order is posted, its amount is in USD and means either:
    - an amount of funds (maintenance margin + add. collateral) you can afford for this order
        (when you are entering Long/Short position)
    - or an amount from position to buy/sell.
        (when you are closgin Long/Short position)

    When you entered position the maintenance margin is held while you maintain it
    So your available balance will be decreased so to prohibit entering other positions

    bm.balance.available_usd_balance
    bm.balance.usd_balance

    bm.balance.contracts?

    bm.position.total_amount
    bm.position.available_amount
'''
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


def test_hold_funds_for_market_long_buy_with_abs_amount(bm_1601):
    bm = bm_1601
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same


def test_hold_funds_for_market_long_buy_with_percentage_amount(bm_3202):
    bm = bm_3202
    buy_opt = MarketOrderOptions(OrderSide.BUY, percentage_amount=Decimal('50'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38')  # 3202 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same


def test_hold_funds_for_market_long_sell_with_abs_amount(bm_3202):
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


def test_hold_funds_for_market_short_sell_with_abs_amount(bm_1601):
    bm = bm_1601
    buy_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    amount = bm.hold_funds(buy_opt, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38') # 1601 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same


def test_hold_funds_for_market_short_sell_with_percentage_amount(bm_3202):
    bm = bm_3202
    buy_opt = MarketOrderOptions(OrderSide.SELL, percentage_amount=Decimal('50'))
    amount = bm.hold_funds(buy_opt, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # 3202 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same 


def test_hold_funds_for_market_short_buy_with_abs_amount(bm_3202):
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
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')     # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_funds_for_limit_long_buy_with_abs_amount(bm_1601):
    bm = bm_1601
    buy_opt = LimitOrderOptions(OrderSide.BUY, Decimal('100'), Decimal('1600'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same


def test_hold_funds_for_limit_long_buy_with_percentage_amount(bm_3202):
    bm = bm_3202
    buy_opt = LimitOrderOptions(OrderSide.BUY, Decimal('100'), percentage_amount=Decimal('50'))
    amount = bm.hold_funds(buy_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38')  # 3202 - (1600 + 0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same


def test_hold_funds_for_limit_long_sell_with_abs_amount(bm_3202):
    # Do some initial BUY first to enter Long
    bm = bm_3202
    buy_opt = LimitOrderOptions(OrderSide.BUY, Decimal('100'), Decimal('1600'))
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
    sell_opt = LimitOrderOptions(OrderSide.SELL, Decimal('110'), Decimal('1600'))
    amount = bm.hold_funds(sell_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38')  # same
    assert bm.balance.usd_balance == Decimal('3201.38')  # same
    assert bm.position.total_amount == Decimal('1600')  # same
    assert bm.position.available_amount == Decimal('0')  # 1600 - amount (1600)
    assert bm.traded_contracts == Decimal('1')
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_funds_for_limit_short_sell_with_abs_amount(bm_1601):
    bm = bm_1601
    opts = LimitOrderOptions(OrderSide.SELL, Decimal('100'), Decimal('1600'))
    amount = bm.hold_funds(opts, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # 1600 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('1601')  # same


def test_hold_funds_for_limit_short_sell_with_percentage_amount(bm_3202):
    bm = bm_3202
    opts = LimitOrderOptions(OrderSide.SELL, Decimal('100'), percentage_amount=Decimal('50'))
    amount = bm.hold_funds(opts, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # 3202 - init(1600) - fee(0.62)
    assert bm.balance.usd_balance == Decimal('3202')  # same 


def test_hold_funds_for_limit_short_buy_with_abs_amount(bm_3202):
    # Do some initial SELL first to enter Short
    bm = bm_3202
    sell_opt = LimitOrderOptions(OrderSide.SELL, Decimal('100'), Decimal('1600'))
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
    buy_opt = LimitOrderOptions(OrderSide.BUY, Decimal('90'), Decimal(1600))
    amount = bm.hold_funds(buy_opt, short=True)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('1601.38') # same
    assert bm.balance.usd_balance == Decimal('3201.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')     # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_long_tp_with_abs_amount(bm_1601):
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


def test_hold_for_long_tp_with_percentage_amount(bm_1601):
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
    tp_opt = TakeProfitOptions(percentage_amount=Decimal('100'), 
                               trigger_price=Decimal('110'))
    amount = bm.hold_for_tp(OrderSide.SELL, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_long_sl_with_abs_amount(bm_1601):
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

    # Hold for SL SELL
    sl_id = 0
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.SELL, sl_id, sl_opt)

    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_long_sl_with_percentage_amount(bm_1601):
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

    # Hold for SL SELL
    sl_id = 0
    sl_opt = StopLossOptions(percentage_amount=Decimal('100'), 
                             trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.SELL, sl_id, sl_opt)

    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_long_tpsl_with_abs_amount(bm_1601):
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


def test_hold_for_long_tpsl_with_percentage_amount(bm_1601):
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
    tp_opt = TakeProfitOptions(percentage_amount=Decimal('100'), 
                               trigger_price=Decimal('110'))
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
    sl_opt = StopLossOptions(percentage_amount=Decimal('100'), 
                             trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.SELL, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: minus amount(1600)
    assert bm.traded_contracts == Decimal('1')   # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract


def test_hold_for_short_tp_with_abs_amount(bm_1601):
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
    assert bm.position.available_amount == Decimal('0')  # 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_short_tp_with_percentage_amount(bm_1601):
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
    tp_opt = TakeProfitOptions(percentage_amount=Decimal('100'), 
                               trigger_price=Decimal('90'))
    amount = bm.hold_for_tp(OrderSide.BUY, tp_id, tp_opt)

    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_short_sl_with_abs_amount(bm_1601):
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

    # Hold for SL BUY
    sl_id = 0
    sl_opt = StopLossOptions(amount=Decimal('1600'), trigger_price=Decimal('110'))
    amount = bm.hold_for_sl(OrderSide.BUY, sl_id, sl_opt)

    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_short_sl_with_percentage_amount(bm_1601):
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

    # Hold for SL BUY
    sl_id = 0
    sl_opt = StopLossOptions(percentage_amount=Decimal('100'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.BUY, sl_id, sl_opt)

    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract


def test_hold_for_short_tpsl_with_percentage_amount(bm_1601):
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
    tp_opt = TakeProfitOptions(percentage_amount=Decimal('100'), 
                               trigger_price=Decimal('90'))
    amount = bm.hold_for_tp(OrderSide.BUY, tp_id, tp_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # minus 1 contract

    # Hold for SL BUY
    sl_id = 1
    sl_opt = StopLossOptions(percentage_amount=Decimal('100'), trigger_price=Decimal('90'))
    amount = bm.hold_for_sl(OrderSide.BUY, sl_id, sl_opt)
    assert amount == Decimal('1600')
    assert bm.balance.available_usd_balance == Decimal('0.38')  # same
    assert bm.balance.usd_balance == Decimal('1600.38')  # same
    assert bm.position.total_amount == Decimal('1600')   # same
    assert bm.position.available_amount == Decimal('0')  # same: 1600 -amount(1600)
    assert bm.traded_contracts == Decimal('1')  # same
    assert bm.available_contracts == Decimal('0')  # same: minus 1 contract


def test_hold_for_short_tpsl_with_abs_amount(bm_1601):
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
