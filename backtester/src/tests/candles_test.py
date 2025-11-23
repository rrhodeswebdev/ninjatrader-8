import typing as t
from datetime import datetime
from decimal import Decimal
from pytest import fixture
from backintime.timeframes import Timeframes as tf
from backintime.data.candle import Candle
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


def test_candles_buffer_candle_compression(sample_h1_candles):
    """
    Test candle compression feature of `CandleBuffer`. 
    Candle compression = build one candle of larger timeframe from 
    multiple candles of shorter one.
    """
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

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h4_candle = candles_buffer.get(timeframe)
    assert _candles_equal(result_h4_candle, expected_h4_candle)


def test_candle_compression(sample_h1_candles):
    """
    Test candle compression feature of `Candles`. 
    Candle compression = build one candle of larger timeframe from 
    multiple candles of shorter one.
    """
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

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h4_candle = Candles(candles_buffer).get(timeframe)
    assert _candles_equal(result_h4_candle, expected_h4_candle)


def test_ohlcv_history(sample_h1_candles):
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
    candles_buffer = CandlesBuffer(since, {tf.H1: 4, tf.H4: 1})

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h1 = [
        candles_buffer.get(tf.H1, -4),
        candles_buffer.get(tf.H1, -3),
        candles_buffer.get(tf.H1, -2),
        candles_buffer.get(tf.H1, -1),
    ]

    for result, expected in zip(result_h1, sample_h1_candles):
        assert _candles_equal(result, expected)


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

    for candle in sample_h1_candles:
        candles_buffer.update(candle)

    result_h2 = [
        candles_buffer.get(tf.H2, -2),
        candles_buffer.get(tf.H2, -1),
    ]

    for result in result_h2:
        print(f"S: {result.open_time}")
        print(f"O: {result.open}")
        print(f"H: {result.high}")
        print(f"L: {result.low}")
        print(f"C: {result.close}")
        print(f"E: {result.close_time}")
        print('')

    for result, expected in zip(result_h2, expected_h2_candles):
        assert _candles_equal(result, expected)


def test_candles_buffer_unexpected_timeframe_will_raise():
    """
    Ensure that accessing timeframe not passed to ctor before
    will raise `CandleNotFound` (not `KeyError` or whatever).
    """
    since = datetime.fromisoformat("2020-12-12 00:00+00:00")
    timeframes = { tf.M1, tf.H1 }
    unexpected_timeframe = tf.D1
    candle_not_found_raised = False
    candles_buffer = CandlesBuffer(since, timeframes)

    try:
        candles_buffer.get(unexpected_timeframe)
    except CandleNotFound as e:
        candle_not_found_raised = True
    assert candle_not_found_raised


def test_unexpected_timeframe_will_raise():
    """
    Ensure that accessing timeframe not passed to ctor before
    will raise `CandleNotFound` (not `KeyError` or whatever).
    """
    since = datetime.fromisoformat("2020-12-12 00:00+00:00")
    timeframes = { tf.M1, tf.H1 }
    unexpected_timeframe = tf.D1
    candle_not_found_raised = False
    candles_buffer = CandlesBuffer(since, timeframes)

    try:
        Candles(candles_buffer).get(unexpected_timeframe)
    except CandleNotFound as e:
        candle_not_found_raised = True
    assert candle_not_found_raised

