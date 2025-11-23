from datetime import datetime, timedelta, timezone
from decimal import Decimal

from backintime.session import FuturesSession
from backintime.broker.base import (BrokerException, InsufficientFunds,
                                    InvalidOrderData, LimitOrderOptions,
                                    MarketOrderOptions, OrderCancellationError,
                                    OrderSide, OrderStatus,
                                    OrderSubmissionError, OrderType,
                                    StopLossOptions, TakeProfitOptions)
from backintime.broker.futures.broker import FuturesBroker, OrderNotFound
from backintime.broker.futures.balance_manager import MarginCall
from backintime.data.candle import Candle


def test_limit_buy_margin_call():
    """Test execution of a valid Limit order."""
    broker = FuturesBroker(Decimal(3500), check_margin_call=True, 
                           per_contract_maintenance_margin=Decimal('1500'))
    test_side = OrderSide.BUY
    test_order_price=Decimal('500')
    test_amount = broker.max_funds_for_futures

    test_candles = [
        # One for Limit SELL execution
        Candle(open_time=datetime.now(),
               open=Decimal(500),
               high=Decimal(1100),
               low=Decimal(400),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown
        Candle(open_time=datetime.now(),
               open=Decimal(500),
               high=Decimal(1100),
               low=Decimal(300),
               close=Decimal(1050),
               close_time=datetime.now(),
               volume=Decimal(10_000))
    ]

    options = LimitOrderOptions(test_side, 
                                amount=test_amount, 
                                order_price=test_order_price)
    limit_order = broker.submit_limit_order(options)
    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
    except MarginCall as e:
        print(e)
        margin_call_raised = True
    
    assert margin_call_raised


def test_limit_buy_overnight_margin_call():
    """Test execution of a valid Limit order."""
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    broker = FuturesBroker(Decimal(3500), session, check_margin_call=True, 
                           per_contract_maintenance_margin=Decimal('1500'),
                           per_contract_overnight_margin=Decimal('2200'))

    test_side = OrderSide.BUY
    test_order_price=Decimal('500')
    test_amount = broker.max_funds_for_futures
    print(f"Test amount: {test_amount}")
    # Times for regular open session
    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')
    # Times for overnight
    open_time_2 = datetime.fromisoformat('2020-01-08 19:00-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 19:00:59-06:00')

    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(500),
               high=Decimal(1100),
               low=Decimal(400),
               close=Decimal(1050),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown
        Candle(open_time=open_time_2,
               open=Decimal(500),
               high=Decimal(1100),
               low=Decimal(300),
               close=Decimal(1050),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    options = LimitOrderOptions(test_side, 
                                amount=test_amount, 
                                order_price=test_order_price)
    limit_order = broker.submit_limit_order(options)
    print(broker.balance)
    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
            print(broker.current_equity)
    except MarginCall as e:
        print(e)
        margin_call_raised = True
    
    assert margin_call_raised


def test_limit_buy_no_overnight_margin_call():
    """Test execution of a valid Limit order."""
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    broker = FuturesBroker(Decimal(2300), session, check_margin_call=True, 
                           per_contract_maintenance_margin=Decimal('1500'),
                           per_contract_overnight_margin=Decimal('2200'))

    test_side = OrderSide.BUY
    test_order_price=Decimal('1000')
    test_amount = broker.max_funds_for_futures   # 3399.404
    # Times for regular open session
    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')
    # Times for overnight
    open_time_2 = datetime.fromisoformat('2020-01-08 19:00-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 19:00:59-06:00')

    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(1000),
               high=Decimal(1010),
               low=Decimal(1000),
               close=Decimal(1010),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown
        Candle(open_time=open_time_2,
               open=Decimal(975),
               high=Decimal(1000),
               low=Decimal(975),
               close=Decimal(1000),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    options = LimitOrderOptions(test_side, 
                                amount=test_amount, 
                                order_price=test_order_price)
    limit_order = broker.submit_limit_order(options)
    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
    except MarginCall as e:
        margin_call_raised = True

    assert not margin_call_raised


def test_limit_short_margin_call():
    broker = FuturesBroker(Decimal('3001.24'), 
                           per_contract_init_margin=Decimal('1700'),
                           per_contract_maintenance_margin=Decimal('1500'),
                           per_contract_fee=Decimal('0.62'),
                           check_margin_call=True)
    # Enter Short position
    broker.submit_market_short(MarketOrderOptions(order_side=OrderSide.SELL, amount=Decimal('1700')))
    print(broker.balance)

    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')

    open_time_2 = datetime.fromisoformat('2020-01-08 12:01-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 12:01:59-06:00')
    '''
    purchased 1 contract at 1100
    then, in the next candle the highest price is 1900
    price_diff = 800
    short gain = 800(1100 - 1900) * contract_quotient(2) = -1600, minus fee
    '''
    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(1100),
               high=Decimal(1100),
               low=Decimal(1000),
               close=Decimal(1010),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown of the short position
        Candle(open_time=open_time_2,
               open=Decimal(1800),
               high=Decimal(1900),
               low=Decimal(1775),
               close=Decimal(1775),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
            print(broker.current_equity)
    except MarginCall as e:
        margin_call_raised = True

    assert margin_call_raised


def test_limit_short_no_margin_call():
    broker = FuturesBroker(Decimal('3001.24'), 
                           per_contract_init_margin=Decimal('1500'),
                           per_contract_maintenance_margin=Decimal('1700'),
                           per_contract_fee=Decimal('0.62'),
                           check_margin_call=True)
    # Enter Short position
    broker.submit_market_short(MarketOrderOptions(order_side=OrderSide.SELL, amount=Decimal('1500')))

    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')

    open_time_2 = datetime.fromisoformat('2020-01-08 12:01-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 12:01:59-06:00')

    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(1100),
               high=Decimal(1100),
               low=Decimal(1000),
               close=Decimal(1010),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate non-critical drawdown of the short position
        Candle(open_time=open_time_2,
               open=Decimal(1700),
               high=Decimal(1750),
               low=Decimal(1675),
               close=Decimal(1675),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
    except MarginCall as e:
        margin_call_raised = True

    assert not margin_call_raised


def test_limit_short_overnight_margin_call():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    broker = FuturesBroker(Decimal('3001.24'), session, check_margin_call=True,
                           per_contract_init_margin=Decimal('1700'),
                           per_contract_maintenance_margin=Decimal('1500'),
                           per_contract_overnight_margin=Decimal('2200'),
                           per_contract_fee=Decimal('0.62'))
    # Enter Short position
    broker.submit_market_short(MarketOrderOptions(order_side=OrderSide.SELL, amount=Decimal('1700')))
    # Times for regular open session
    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')
    # Times for overnight
    open_time_2 = datetime.fromisoformat('2020-01-08 19:00-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 19:00:59-06:00')

    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(1100),
               high=Decimal(1100),
               low=Decimal(1000),
               close=Decimal(1010),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown of the short position
        Candle(open_time=open_time_2,
               open=Decimal(1400),
               high=Decimal(1550),
               low=Decimal(1375),
               close=Decimal(1375),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
    except MarginCall as e:
        margin_call_raised = True

    assert margin_call_raised


def test_limit_short_no_overnight_margin_call():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    broker = FuturesBroker(Decimal('3001.24'), session, check_margin_call=True,
                           per_contract_init_margin=Decimal('1700'),
                           per_contract_maintenance_margin=Decimal('1500'),
                           per_contract_overnight_margin=Decimal('2200'),
                           per_contract_fee=Decimal('0.62'))
    # Enter Short position
    broker.submit_market_short(MarketOrderOptions(order_side=OrderSide.SELL, amount=Decimal('1700')))
    # Times for regular open session
    open_time_1 = datetime.fromisoformat('2020-01-08 12:00-06:00')
    close_time_1 = datetime.fromisoformat('2020-01-08 12:00:59-06:00')
    # Times for overnight
    open_time_2 = datetime.fromisoformat('2020-01-08 19:00-06:00')
    close_time_2 = datetime.fromisoformat('2020-01-08 19:00:59-06:00')

    test_candles = [
        # One for Limit BUY execution
        Candle(open_time=open_time_1,
               open=Decimal(1100),
               high=Decimal(1100),
               low=Decimal(1000),
               close=Decimal(1010),
               close_time=close_time_1,
               volume=Decimal(10_000)),
        # Another one to simulate critical drawdown of the short position
        Candle(open_time=open_time_2,
               open=Decimal(1400),
               high=Decimal(1500),
               low=Decimal(1375),
               close=Decimal(1375),
               close_time=close_time_2,
               volume=Decimal(10_000))
    ]

    margin_call_raised = False

    try:
        for candle in test_candles:
            broker.update(candle)
    except MarginCall as e:
        margin_call_raised = True

    assert not margin_call_raised