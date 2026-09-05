"""Deterministic US cash-market session calendar and UTC audit helpers.

The calendar models regular NYSE-style sessions used for daily S&P 500 data.
It intentionally uses only the Python standard library so historical session
semantics remain reproducible without an external calendar dependency.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    day = date(year, month, 1)
    shift = (weekday - day.weekday()) % 7
    return day + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        day = date(year, month + 1, 1) - timedelta(days=1)
    return day - timedelta(days=(day.weekday() - weekday) % 7)


def _observed_fixed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    # Gregorian computus (Meeus/Jones/Butcher), sufficient for 2000 onward.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def market_holidays(year: int) -> set[date]:
    """Return regular full-day US cash-market holidays for a year."""
    holidays = {
        _observed_fixed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),       # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),       # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),          # Memorial Day
        _nth_weekday(year, 9, 0, 1),         # Labor Day
        _nth_weekday(year, 11, 3, 4),        # Thanksgiving
        _observed_fixed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_fixed(date(year, 6, 19)))
    holidays.add(_observed_fixed(date(year, 7, 4)))
    return holidays


def early_close_dates(year: int) -> set[date]:
    """Return the principal 13:00 ET early-close dates used by the audit."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    day_after = thanksgiving + timedelta(days=1)
    christmas_eve = date(year, 12, 24)
    july3 = date(year, 7, 3)
    result = {day_after}
    if christmas_eve.weekday() < 5 and christmas_eve not in market_holidays(year):
        result.add(christmas_eve)
    if july3.weekday() < 5 and july3 not in market_holidays(year):
        result.add(july3)
    return result


def is_market_session(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def session_close(day: date) -> datetime:
    """Return the session close as an aware UTC timestamp."""
    if not is_market_session(day):
        raise ValueError(f"{day.isoformat()} is not a market session")
    close_hour = 13 if day in early_close_dates(day.year) else 16
    return datetime(day.year, day.month, day.day, close_hour, 0, tzinfo=NEW_YORK).astimezone(UTC)


def session_table(start: str | date, end: str | date) -> pd.DataFrame:
    """Build a dated session table with local and UTC close information."""
    start_day = pd.Timestamp(start).date()
    end_day = pd.Timestamp(end).date()
    if end_day < start_day:
        raise ValueError("end must be on or after start")
    rows = []
    day = start_day
    while day <= end_day:
        if is_market_session(day):
            close_utc = session_close(day)
            rows.append({
                "session_date": day,
                "session_timezone": "America/New_York",
                "close_local": close_utc.astimezone(NEW_YORK),
                "close_utc": close_utc,
                "early_close": day in early_close_dates(day.year),
            })
        day += timedelta(days=1)
    return pd.DataFrame(rows).set_index("session_date")


def audit_market_timestamps(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Audit UTC timestamps against their expected NY session date/close."""
    if index.tz is None:
        raise ValueError("market timestamps must be timezone-aware")
    rows = []
    for timestamp in pd.DatetimeIndex(index).tz_convert("UTC"):
        local = timestamp.tz_convert(NEW_YORK)
        session = local.date()
        valid_day = is_market_session(session)
        expected = session_close(session) if valid_day else pd.NaT
        rows.append({
            "timestamp_utc": timestamp,
            "session_date": session,
            "valid_market_day": valid_day,
            "expected_close_utc": expected,
            "at_or_before_close": bool(valid_day and timestamp <= expected),
            "after_close": bool(valid_day and timestamp > expected),
            "weekend": session.weekday() >= 5,
            "holiday": bool(session in market_holidays(session.year)),
        })
    return pd.DataFrame(rows).set_index("timestamp_utc")
