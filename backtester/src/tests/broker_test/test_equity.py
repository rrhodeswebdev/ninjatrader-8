from datetime import datetime, timezone
from decimal import Decimal

from backintime.broker.base import (BrokerException, InsufficientFunds,
                                    InvalidOrderData, LimitOrderOptions,
                                    MarketOrderOptions, OrderCancellationError,
                                    OrderSide, OrderStatus,
                                    OrderSubmissionError, OrderType,
                                    StopLossOptions, TakeProfitOptions)
from backintime.broker.futures.broker import FuturesBroker, OrderNotFound
from backintime.data.candle import Candle


def test_limit_buy_execution_no_tpsl():
    """Test equity after the execution of a valid Limit order."""
    broker = FuturesBroker(Decimal(3500))
    test_side = OrderSide.BUY
    test_order_price=Decimal('1050')
    test_amount = broker.max_funds_for_futures   # 3399.28 = 2 contracts
    # 2 contracts at 500 then price is 600 = 100*2*2 = +400 gain 398.76 net
    # available(99.48) + collateral(3399.28) + net_gain(398.76) = 3897.52
    test_candle = Candle(open_time=datetime.now(),
                         open=Decimal(500),
                         high=Decimal(1100),
                         low=Decimal(400),
                         close=Decimal(600),
                         close_time=datetime.now(),
                         volume=Decimal(10_000))

    expected_equity = Decimal('3897.52')

    options = LimitOrderOptions(test_side, 
                                amount=test_amount, 
                                order_price=test_order_price)
    limit_order = broker.submit_limit_order(options)
    broker.update(test_candle)

    print(broker.balance)

    assert broker.current_equity == expected_equity


def test_limit_buy_execution_and_take_profit_limit_activation():
    broker = FuturesBroker(Decimal(3500))
    test_side = OrderSide.BUY
    test_order_price = Decimal('500')
    test_amount = broker.max_funds_for_futures   # 3399.28 = 2 contracts
    # 2 contracts at 500 then price is 1300 = 800*2*2 = +3200 gain +3198.76 net 
    # 99.48 + 3399.28 + 3198.76  = 6697.52
    test_candles = [
        # One for limit execution
        Candle(open_time=datetime.now(),
               open=Decimal(500),
               high=Decimal(1100),
               low=Decimal(400),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        # Another for TP activation
        Candle(open_time=datetime.now(),
               open=Decimal(1000),
               high=Decimal(1500),
               low=Decimal(900),
               close=Decimal(1300),
               close_time=datetime.now(),
               volume=Decimal(10_000))
    ]

    expected_equity = Decimal('6697.52')

    tp_options = TakeProfitOptions(percentage_amount=Decimal('100.00'),
                                   trigger_price=Decimal(1200),
                                   order_price=Decimal(1600))

    options = LimitOrderOptions(test_side,                      # BUY
                                amount=test_amount,             # 3399.28
                                order_price=test_order_price,   # 500
                                take_profit=tp_options)

    limit_order = broker.submit_limit_order(options)

    for candle in test_candles:
        broker.update(candle)

    print(broker.balance)

    assert broker.current_equity == expected_equity
