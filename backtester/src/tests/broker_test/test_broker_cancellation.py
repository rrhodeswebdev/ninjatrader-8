from backintime.broker.futures.broker import FuturesBroker
from decimal import Decimal
from datetime import datetime

import pytest

from backintime.data.candle import Candle
from backintime.broker.futures.balance import FuturesBalance, FuturesBalanceInfo
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    MarketOrderOptions, LimitOrderOptions,
                                    StopLossOptions, TakeProfitOptions)


@pytest.fixture
def broker_1601() -> FuturesBroker:
    """
    Setup FuturesBroker with initial balance of 1601,
    initial margin=1600, mantenance margin=1500,
    overnight_margin=2200, fee=0.62 and quotient=2.
    """
    return FuturesBroker(
                start_money = Decimal('1601'),
                contract_quotient = Decimal('2'),
                per_contract_fee = Decimal('0.62'),
                per_contract_init_margin = Decimal('1600'),
                per_contract_maintenance_margin = Decimal('1500'),
                per_contract_overnight_margin = Decimal('2200'),
                additional_collateral = Decimal('0'))


def test_cancel_long_market_buy(broker_1601):
    broker = broker_1601

    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    market_buy = broker.submit_market_order(buy_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert broker.balance.usd_balance == Decimal('1601')  # same

    broker.cancel_order(market_buy.order_id)
    assert market_buy.status is OrderStatus.CANCELLED
    assert market_buy.trading_fee == None
    assert market_buy.fill_price == None
    assert broker.in_long == False
    assert broker.in_short == False
    assert broker.balance.available_usd_balance == Decimal('1601') # restored
    assert broker.balance.usd_balance == Decimal('1601')  # restored


def test_cancel_long_market_sell(broker_1601):
    broker = broker_1601
    sample_candles = [
        Candle(open_time=datetime.now(),
               open=Decimal(1000),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        
        Candle(open_time=datetime.now(),
               open=Decimal(1010),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
    ]

    # BUY
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    market_buy = broker.submit_market_order(buy_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert broker.balance.usd_balance == Decimal('1601')  # same

    broker.update(sample_candles[0])
    assert market_buy.status is OrderStatus.EXECUTED
    assert market_buy.trading_fee == Decimal('0.62')
    assert market_buy.fill_price == Decimal('1000')
    assert broker.in_long == True
    assert broker.balance.available_usd_balance == Decimal('0.38') # 1601 - maintenance(1600) - fee(0.62)
    assert broker.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)

    # SELL
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    market_sell = broker.submit_market_order(sell_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38')  # same
    assert broker.balance.usd_balance == Decimal('1600.38')  # same

    broker.cancel_order(market_sell.order_id)
    assert market_sell.status is OrderStatus.CANCELLED
    assert market_sell.trading_fee == None
    assert market_sell.fill_price == None
    assert broker.in_long == True    # still in Long
    assert broker.in_short == False
    assert broker.balance.available_usd_balance == Decimal('0.38')  # same
    assert broker.balance.usd_balance == Decimal('1600.38')  # same


def test_cancel_short_market_sell(broker_1601):
    broker = broker_1601
    test_candle = Candle(open_time=datetime.now(),
                         open=Decimal(1000),
                         high=Decimal(1100),
                         low=Decimal(900),
                         close=Decimal(1050),
                         close_time=datetime.now(),
                         volume=Decimal(10_000))

    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    market_short = broker.submit_market_short(sell_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert broker.balance.usd_balance == Decimal('1601')  # same

    broker.cancel_order(market_short.order_id)
    assert market_short.status is OrderStatus.CANCELLED
    assert market_short.trading_fee == None
    assert market_short.fill_price == None
    assert broker.in_long == False
    assert broker.in_short == False
    assert broker.balance.available_usd_balance == Decimal('1601') # restored
    assert broker.balance.usd_balance == Decimal('1601')  # restored


def test_cancel_short_market_buy(broker_1601):
    broker = broker_1601
    sample_candles = [
        Candle(open_time=datetime.now(),
               open=Decimal(1000),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        
        Candle(open_time=datetime.now(),
               open=Decimal(1010),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
    ]

    # SELL
    sell_opt = MarketOrderOptions(OrderSide.SELL, Decimal('1600'))
    market_short = broker.submit_market_short(sell_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38')  # 1601 - (1600 + 0.62)
    assert broker.balance.usd_balance == Decimal('1601')  # same

    broker.update(sample_candles[0])
    assert market_short.status is OrderStatus.EXECUTED
    assert market_short.trading_fee == Decimal('0.62')
    assert market_short.fill_price == Decimal('1000')
    assert broker.in_short == True
    assert broker.balance.available_usd_balance == Decimal('0.38') # 1601 - maintenance(1600) - fee(0.62)
    assert broker.balance.usd_balance == Decimal('1600.38')  # 1601 - fee(0.62)

    # BUY
    buy_opt = MarketOrderOptions(OrderSide.BUY, Decimal('1600'))
    short_buy = broker.submit_market_short(buy_opt)
    assert broker.balance.available_usd_balance == Decimal('0.38') # same
    assert broker.balance.usd_balance == Decimal('1600.38')  # same

    broker.cancel_order(short_buy.order_id)
    assert short_buy.status is OrderStatus.CANCELLED
    assert short_buy.trading_fee == None
    assert short_buy.fill_price == None
    assert broker.in_long == False
    assert broker.in_short == True   # still in Short
    assert broker.balance.available_usd_balance == Decimal('0.38')  # same
    assert broker.balance.usd_balance == Decimal('1600.38')  # same


def test_cancel_long_tpsl(broker_1601):
    broker = broker_1601
    sample_candles = [
        Candle(open_time=datetime.now(),
               open=Decimal(1000),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        
        Candle(open_time=datetime.now(),
               open=Decimal(1010),
               high=Decimal(1200),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
    ]

    # Submit Limit BUY (Long) & TP/SL
    tp_opt = TakeProfitOptions(trigger_price=Decimal('1200'), 
                    percentage_amount=Decimal('100'))

    sl_opt = StopLossOptions(trigger_price=Decimal('800'),
                    percentage_amount=Decimal('100'))

    buy_opt = LimitOrderOptions(OrderSide.BUY, 
                Decimal('1000'), Decimal('1600'),
                take_profit=tp_opt, stop_loss=sl_opt)

    long_buy = broker.submit_limit_order(buy_opt)
    broker.update(sample_candles[0])

    assert long_buy.status is OrderStatus.EXECUTED
    assert long_buy.trading_fee == Decimal('0.62')
    assert long_buy.fill_price == Decimal('1000')

    broker.cancel_order(long_buy.take_profit.order_id)
    broker.cancel_order(long_buy.stop_loss.order_id)

    assert long_buy.take_profit.status is OrderStatus.CANCELLED
    assert long_buy.take_profit.trading_fee == None
    assert long_buy.take_profit.fill_price == None

    assert long_buy.stop_loss.status is OrderStatus.CANCELLED
    assert long_buy.stop_loss.trading_fee is None
    assert long_buy.stop_loss.fill_price is None

    assert broker.in_long == True   # still in Long
    assert broker.in_short == False
    assert broker.balance.available_usd_balance == Decimal('0.38')
    assert broker.balance.usd_balance == Decimal('1600.38')


def test_cancel_short_tpsl(broker_1601):
    broker = broker_1601
    sample_candles = [
        Candle(open_time=datetime.now(),
               open=Decimal(1000),
               high=Decimal(1100),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        
        Candle(open_time=datetime.now(),
               open=Decimal(800),
               high=Decimal(1200),
               low=Decimal(900),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
    ]

    # Submit Limit SELL (Short) & TP/SL
    tp_opt = TakeProfitOptions(trigger_price=Decimal('800'), 
                    percentage_amount=Decimal('100'))

    sl_opt = StopLossOptions(trigger_price=Decimal('1200'),
                    percentage_amount=Decimal('100'))

    sell_opt = LimitOrderOptions(OrderSide.SELL, 
                Decimal('1000'), Decimal('1600'),
                take_profit=tp_opt, stop_loss=sl_opt)

    short_sell = broker.submit_limit_short(sell_opt)
    broker.update(sample_candles[0])

    assert short_sell.status is OrderStatus.EXECUTED
    assert short_sell.trading_fee == Decimal('0.62')
    assert short_sell.fill_price == Decimal('1000')

    broker.cancel_order(short_sell.take_profit.order_id)
    broker.cancel_order(short_sell.stop_loss.order_id)

    assert short_sell.take_profit.status is OrderStatus.CANCELLED
    assert short_sell.take_profit.trading_fee == None
    assert short_sell.take_profit.fill_price == None

    assert short_sell.stop_loss.status is OrderStatus.CANCELLED
    assert short_sell.stop_loss.trading_fee is None
    assert short_sell.stop_loss.fill_price is None

    assert broker.in_long == False
    assert broker.in_short == True   # still in Short
    assert broker.balance.available_usd_balance == Decimal('0.38')
    assert broker.balance.usd_balance == Decimal('1600.38')

