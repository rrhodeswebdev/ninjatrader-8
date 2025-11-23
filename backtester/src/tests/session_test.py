from datetime import datetime, timedelta
from backintime.session import FuturesSession

import pytz


def test_session_is_open_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 16:00+00:00')
    assert session.is_open(sample_time)


def test_session_is_closed_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 23:00+00:00')
    assert not session.is_open(sample_time)


def test_session_is_open_with_us_central_time():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 15:00')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert session.is_open(sample_time)


def test_session_is_closed_with_us_central_time():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 17:00')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert not session.is_open(sample_time)


def test_session_is_open_with_utc_time_skip_first_hour():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 16:30+00:00')
    assert session.is_open(sample_time, first_hours_to_skip=1)


def test_session_is_closed_with_utc_time_skip_first_hour():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 16:00+00:00')
    assert not session.is_open(sample_time, first_hours_to_skip=1)


def test_session_is_open_with_us_central_time_skip_first_hour():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 10:30')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert session.is_open(sample_time, first_hours_to_skip=1)


def test_session_is_closed_with_us_central_time_skip_first_hour():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 09:30')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert not session.is_open(sample_time, first_hours_to_skip=1)


def test_session_is_closed_on_non_working_weekday():
    session = FuturesSession(session_start=timedelta(hours=9, minutes=30),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2024-04-13 09:30')    # Saturday
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert not session.is_open(sample_time)


def test_overnight_session_is_open_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 15:00+00:00')
    assert session.is_open(sample_time)


def test_overnight_session_is_closed_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 23:00+00:00')
    assert not session.is_open(sample_time)


def test_overnight_session_is_open_with_us_central_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 15:00')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert session.is_open(sample_time)


def test_overnight_session_is_closed_with_us_central_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 17:00')
    sample_time = pytz.timezone('US/Central').localize(sample_time)
    assert not session.is_open(sample_time)


def test_overnight_is_applied_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 23:00+00:00')
    assert session.is_overnight(sample_time)


def test_overnight_is_not_applied_with_utc_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 17:00+00:00')
    assert not session.is_overnight(sample_time)


def test_overnight_is_applied_with_ny_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 17:00')
    sample_time = pytz.timezone('America/New_York').localize(sample_time)
    assert session.is_overnight(sample_time)


def test_overnight_is_not_applied_with_ny_time():
    session = FuturesSession(session_start=timedelta(hours=18, minutes=0),
                             session_end=timedelta(hours=17, minutes=0),
                             session_timezone='US/Central',
                             overnight_start=timedelta(hours=16, minutes=30),
                             overnight_end=timedelta(hours=8, minutes=0),
                             overnight_timezone='America/New_York',
                             non_working_weekdays={5})  # Saturday

    sample_time = datetime.fromisoformat('2020-01-08 08:00')
    sample_time = pytz.timezone('America/New_York').localize(sample_time)
    assert not session.is_overnight(sample_time)

