import typing as t
from datetime import datetime, timedelta

import pytz


class TradingSession:
    def __init__(self,
                 session_start: timedelta,
                 session_end: timedelta,
                 session_timezone: str,
                 session_break_start: t.Optional[timedelta] = None,
                 session_break_end: t.Optional[timedelta] = None,
                 non_working_weekdays: t.Set[int] = set()):
        """
        params:
            - session_start: 
                Start time of the session. Can be hours, minutes, seconds,
                and any other measurements a timedelta object can store.
            - session_end:
                End time of the session. Can be hours, minutes, seconds,
                and any other measurements a timedelta object can store.
            - session_timezone:
                Timezone for session_start and session_end, e.g. 'US/Central'
                or 'UTC'. Basically any valid string for pytz.timezone().
            - session_break_start:
                (Optional) start time of the session break (if there is one).
            - session_break_end:
                (Optional) end time of the session break (if there is one).
            - non_working_weekdays:
                Set of non working weekdays represented as ints beginning
                from Monday being 0. Default to empty set.
        """
        self._overnight_session = session_start > session_end
        if session_break_start and session_break_end:
            assert (session_break_start < session_break_end, 
                    f"Break start must be less than break end.")

        self._session_start = session_start
        self._session_end = session_end
        self._session_break_start = session_break_start
        self._session_break_end = session_break_end
        self._session_timezone = pytz.timezone(session_timezone)
        self._non_working_weekdays = non_working_weekdays

    def _check_session(self, time: timedelta, first_hours_to_skip: int = 0) -> bool:
        """
        Check if `time` falls within session. 
        Note that breaks, if specified for session, are not taken into account
        and should be additionally checked with `_check_break` method.
        """
        session_start = self._session_start + timedelta(hours=first_hours_to_skip)
        if self._overnight_session:
            return time >= session_start or time < self._session_end
        else:
            return time >= session_start and time < self._session_end

    def _check_break(self, time: timedelta) -> bool:
        """
        Check if `time` falls within session break.
        If session break is not specified return False.
        """
        if not self._session_break_start or not self._session_break_end:
            return False
        return time >= self._session_break_start and time < self._session_break_end

    def is_open(self, time: datetime, first_hours_to_skip: int = 0) -> bool:
        """Check if market is open at `time`."""
        time = time.astimezone(self._session_timezone)  # cast timezone to session's
        # Check if the weekday is non working
        if time.weekday() in self._non_working_weekdays:
            return False

        time = timedelta(hours=time.hour, minutes=time.minute)
        in_session = self._check_session(time, first_hours_to_skip)
        in_break = self._check_break(time)
        return in_session and not in_break


class FuturesSession(TradingSession):
    def __init__(self,
                 session_start: timedelta,
                 session_end: timedelta,
                 session_timezone: str,
                 non_working_weekdays: t.Set[int],
                 overnight_start: timedelta,
                 overnight_end: timedelta,
                 overnight_timezone: str,
                 session_break_start: t.Optional[timedelta] = None,
                 session_break_end: t.Optional[timedelta] = None):
        """
        params:
            - session_start: 
                Start time of the session. Can be hours, minutes, seconds,
                and any other measurements a timedelta object can store.
            - session_end:
                End time of the session. Can be hours, minutes, seconds,
                and any other measurements a timedelta object can store.
            - session_timezone:
                Timezone for session_start and session_end, e.g. 'US/Central'
                or 'UTC'. Basically any valid string for pytz.timezone().
            - session_break_start:
                (Optional) start time of the session break (if there is one).
            - session_break_end:
                (Optional) end time of the session break (if there is one).
            - non_working_weekdays:
                Set of non working weekdays represented as ints beginning
                from Monday being 0. Default to empty set.
            - overnight_start:
                Start time of overnight margin period. Can be hours, minutes,
                seconds, and any other measurements a timedelta object can store.
            - overnight_end:
                End time of overnight margin period. Can be hours, minutes,
                seconds, and any other measurements a timedelta object can store.
            - overnight_timezone:
                Timezone for overnight_start and overnight_end, e.g. 'US/Central'
                or 'UTC'. Basically any valid string for pytz.timezone().
        """
        assert (overnight_start > overnight_end, 
                f"Overnight start must be greater than overnight end.")

        self._overnight_start = overnight_start
        self._overnight_end = overnight_end
        self._overnight_timezone = pytz.timezone(overnight_timezone)
        super().__init__(session_start, session_end, 
                         session_timezone, 
                         session_break_start, session_break_end, 
                         non_working_weekdays)

    def is_overnight(self, time: datetime) -> bool:
        """Check if overnight margin is applied at `time`."""
        time = time.astimezone(self._overnight_timezone)  # cast timezone to overnight's
        time = timedelta(hours=time.hour, minutes=time.minute)
        return time >= self._overnight_start or time < self._overnight_end
