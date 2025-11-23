import typing as t
from datetime import datetime
from decimal import Decimal
from pytest import fixture
from backintime.timeframes import Timeframes as tf
from backintime.data.candle import Candle
from backintime.declarative import prices
from backintime.candles import (
    Candles, 
    CandlesBuffer, 
    CandleNotFound
)


@fixture
def sample_h1_candles() -> t.List[Candle]:
    """Four valid H1 candles collected manually."""
    return [
        Candle(open_time=datetime.fromisoformat('2022-12-01 00:00+00:00'),
               open=Decimal('17165.53'),
               high=Decimal('17236.29'),
               low=Decimal('17122.65'),
               close=Decimal('17161.55'),
               close_time=datetime.fromisoformat('2022-12-01 00:59:59.999000+00:00'),
               volume=Decimal('14453')),

        Candle(open_time=datetime.fromisoformat('2022-12-01 01:00+00:00'),
               open=Decimal('17161.55'),
               high=Decimal('17170.32'),
               low=Decimal('17105.37'),
               close=Decimal('17117.13'),
               close_time=datetime.fromisoformat('2022-12-01 01:59:59.999000+00:00'),
               volume=Decimal('8650')),

        Candle(open_time=datetime.fromisoformat('2022-12-01 02:00+00:00'),
               open=Decimal('17117.13'),
               high=Decimal('17142.99'),
               low=Decimal('17088.01'),
               close=Decimal('17123.98'),
               close_time=datetime.fromisoformat('2022-12-01 02:59:59.999000+00:00'),
               volume=Decimal('7981')),

        Candle(open_time=datetime.fromisoformat('2022-12-01 03:00+00:00'),
               open=Decimal('17124.53'),
               high=Decimal('17169.73'),
               low=Decimal('17122.18'),
               close=Decimal('17150.98'),
               close_time=datetime.fromisoformat('2022-12-01 03:59:59.999000+00:00'),
               volume=Decimal('7343'))
    ]


def _candles_equal(first_candle: Candle, second_candle: Candle) -> bool:
    return (first_candle.open_time == second_candle.open_time and \
            first_candle.open == second_candle.open and \
            first_candle.high == second_candle.high and \
            first_candle.low == second_candle.low and \
            first_candle.close == second_candle.close and \
            first_candle.close_time == second_candle.close_time and \
            first_candle.volume == second_candle.volume)


def test_prices(sample_h1_candles):
    open_expr = prices.open(tf.H1)
    high_expr = prices.high(tf.H1)
    low_expr = prices.low(tf.H1)
    close_expr = prices.close(tf.H1)

    expected_open_time = datetime.fromisoformat('2022-12-01 03:00+00:00')
    expected_close_time = '2022-12-01 03:59:59.999000+00:00'
    expected_close_time = datetime.fromisoformat(expected_close_time)
    expected_h1_candle = Candle(open_time=expected_open_time,
                                open=Decimal('17124.53'),
                                high=Decimal('17169.73'),
                                low=Decimal('17122.18'),
                                close=Decimal('17150.98'),
                                close_time=expected_close_time,
                                volume=Decimal('7343'))

    since = sample_h1_candles[0].open_time
    timeframe = tf.H1
    candles_buffer = CandlesBuffer(since, {timeframe})
    candles = Candles(candles_buffer)

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_open = open_expr.eval(None, None, candles)
    result_high = high_expr.eval(None, None, candles)
    result_low = low_expr.eval(None, None, candles)
    result_close = close_expr.eval(None, None, candles)

    assert result_open == expected_h1_candle.open
    assert result_high == expected_h1_candle.high
    assert result_low == expected_h1_candle.low
    assert result_close == expected_h1_candle.close


def test_prices_upsampling(sample_h1_candles):
    open_expr = prices.open(tf.H4)
    high_expr = prices.high(tf.H4)
    low_expr = prices.low(tf.H4)
    close_expr = prices.close(tf.H4)

    expected_open_time = datetime.fromisoformat('2022-12-01 00:00+00:00')
    expected_close_time = '2022-12-01 03:59:59.999000+00:00'
    expected_close_time = datetime.fromisoformat(expected_close_time)
    expected_h4_candle = Candle(open_time=expected_open_time,
                                open=Decimal('17165.53'),
                                high=Decimal('17236.29'),
                                low=Decimal('17088.01'),
                                close=Decimal('17150.98'),
                                close_time=expected_close_time,
                                volume=Decimal('38427'))

    since = sample_h1_candles[0].open_time
    timeframe = tf.H4
    candles_buffer = CandlesBuffer(since, {timeframe})
    candles = Candles(candles_buffer)

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_open = open_expr.eval(None, None, candles)
    result_high = high_expr.eval(None, None, candles)
    result_low = low_expr.eval(None, None, candles)
    result_close = close_expr.eval(None, None, candles)

    assert result_open == expected_h4_candle.open
    assert result_high == expected_h4_candle.high
    assert result_low == expected_h4_candle.low
    assert result_close == expected_h4_candle.close


def test_ohlcv_history(sample_h1_candles):
    since = sample_h1_candles[0].open_time
    candles_buffer = CandlesBuffer(since, {tf.H1: 4})
    candles = Candles(candles_buffer)

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h1 = [
        [
            prices.open(tf.H1)[idx].eval(None, None, candles),
            prices.high(tf.H1)[idx].eval(None, None, candles),
            prices.low(tf.H1)[idx].eval(None, None, candles),
            prices.close(tf.H1)[idx].eval(None, None, candles),
        ] for idx in range(-len(sample_h1_candles), 0, 1)
    ]

    for result, expected in zip(result_h1, sample_h1_candles):
        assert result[0] == expected.open
        assert result[1] == expected.high
        assert result[2] == expected.low
        assert result[3] == expected.close


def test_ohlcv_history_with_upsampling(sample_h1_candles):
    expected_h2_candles = [
        Candle(open_time=sample_h1_candles[0].open_time,
               open=Decimal('17165.53'),
               high=Decimal('17236.29'),
               low=Decimal('17105.37'),
               close=Decimal('17117.13'),
               close_time=sample_h1_candles[1].close_time,
               volume=sample_h1_candles[0].volume + sample_h1_candles[1].volume),

        Candle(open_time=sample_h1_candles[2].open_time,
               open=Decimal('17117.13'),
               high=Decimal('17169.73'),
               low=Decimal('17088.01'),
               close=Decimal('17150.98'),
               close_time=sample_h1_candles[3].close_time,
               volume=sample_h1_candles[2].volume + sample_h1_candles[3].volume)
    ]

    since = sample_h1_candles[0].open_time
    candles_buffer = CandlesBuffer(since, {tf.H1: 4, tf.H2: 2})
    candles = Candles(candles_buffer)

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h2 = [
        [
            prices.open(tf.H2)[idx].eval(None, None, candles),
            prices.high(tf.H2)[idx].eval(None, None, candles),
            prices.low(tf.H2)[idx].eval(None, None, candles),
            prices.close(tf.H2)[idx].eval(None, None, candles),
        ] for idx in range(-len(expected_h2_candles), 0, 1)
    ]

    for result, expected in zip(result_h2, expected_h2_candles):
        assert result[0] == expected.open
        assert result[1] == expected.high
        assert result[2] == expected.low
        assert result[3] == expected.close

