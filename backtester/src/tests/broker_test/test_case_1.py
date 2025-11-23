from backintime.broker.futures.broker import FuturesBroker
from decimal import Decimal
from datetime import datetime

import pytest

from backintime.data.candle import Candle
from backintime.broker.futures.balance import FuturesBalance, FuturesBalanceInfo
from backintime.broker.base import (OrderSide, OrderStatus, OrderType,
                                    MarketOrderOptions, LimitOrderOptions,
                                    StopLossOptions, TakeProfitOptions)


def test_case_1():
    broker = FuturesBroker(
                start_money = Decimal('9326.82'),
                contract_quotient = Decimal('2'),
                per_contract_fee = Decimal('0.62'),
                per_contract_init_margin = Decimal('1600'),
                per_contract_maintenance_margin = Decimal('1500'),
                per_contract_overnight_margin = Decimal('2200'),
                additional_collateral = Decimal('0'))

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
    tp_opt = TakeProfitOptions(trigger_price=Decimal('80'), 
                    percentage_amount=Decimal('100'))

    sl_opt = StopLossOptions(trigger_price=Decimal('12000'),
                    percentage_amount=Decimal('100'))

    sell_opt = LimitOrderOptions(OrderSide.SELL, 
                Decimal('1000'), Decimal('6000'),
                take_profit=tp_opt, stop_loss=sl_opt)

    short_sell = broker.submit_limit_short(sell_opt)
    for candle in sample_candles:
        broker.update(candle)

    print(broker.in_long)
    print(broker.in_short)
    print(broker.balance.available_usd_balance)
    print(broker.balance.usd_balance)
    print(broker._bm.max_funds_for_futures)

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
                    amount=Decimal('3000'))

    sl_opt = StopLossOptions(trigger_price=Decimal('1200'),
                    amount=Decimal('3000'))

    sell_opt = LimitOrderOptions(OrderSide.SELL, 
                Decimal('1000'), Decimal('3000'),
                take_profit=tp_opt, stop_loss=sl_opt)

    short_sell = broker.submit_limit_short(sell_opt)
    for candle in sample_candles:
        broker.update(candle)

    print(broker.in_long)
    print(broker.in_short)
    print(broker.balance.available_usd_balance)
    print(broker.balance.usd_balance)
    print(broker._bm.max_funds_for_futures)

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
                Decimal('1000'), Decimal('3000'),
                take_profit=tp_opt, stop_loss=sl_opt)

    long_buy = broker.submit_limit_order(buy_opt)
    for candle in sample_candles:
        broker.update(candle)


#test_case_1()