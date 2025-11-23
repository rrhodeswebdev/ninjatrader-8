from datetime import datetime
from backintime.timeframes import Timeframes as tf
from backintime.timeframes import (
	estimate_open_time, 
	estimate_close_time,
	get_seconds_duration,
	get_millis_duration,
	get_timeframes_ratio
)


def test_timeframes_ratio():
    quotient, remainder = get_timeframes_ratio(tf.H4, tf.H1)
    assert quotient == 4 and remainder == 0


def test_seconds_duration():
    assert get_seconds_duration(tf.H4) == 14_399


def test_millis_duration():
    assert get_millis_duration(tf.H1) == 3_599_999


def test_estimate_open_time():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    expected_date = datetime.fromisoformat("2020-12-02 00:00+00:00")
    assert estimate_open_time(some_date, tf.H1) == expected_date


def test_estimate_open_time_negative_offset():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    offset = -2
    expected_date = datetime.fromisoformat("2020-12-01 22:00+00:00")
    assert estimate_open_time(some_date, tf.H1, offset) == expected_date


def test_estimate_open_time_positive_offset():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    offset = 3
    expected_date = datetime.fromisoformat("2020-12-02 03:00+00:00")
    assert estimate_open_time(some_date, tf.H1, offset) == expected_date


def test_estimate_close_time():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    expected_date = datetime.fromisoformat("2020-12-02 00:59:59.999000+00:00")
    assert estimate_close_time(some_date, tf.H1) == expected_date


def test_estimate_close_time_negative_offset():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    offset = -2
    expected_date = datetime.fromisoformat("2020-12-01 22:59:59.999000+00:00")
    assert estimate_close_time(some_date, tf.H1, offset) == expected_date


def test_estimate_close_time_positive_offset():
    some_date = datetime.fromisoformat("2020-12-02 00:30+00:00")
    offset = 3
    expected_date = datetime.fromisoformat("2020-12-02 03:59:59.999000+00:00")
    assert estimate_close_time(some_date, tf.H1, offset) == expected_date
