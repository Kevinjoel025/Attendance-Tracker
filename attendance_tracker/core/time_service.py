"""
core/time_service.py
────────────────────
Provides runtime date/time in IST (Asia/Kolkata).

IMPORTANT:
- Never hardcode "today" anywhere else in the codebase.
- Always call these functions at runtime.
- Never bake date/time into SLM weights or training data.
- All datetime objects returned are timezone-aware.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from attendance_tracker.core.config import config

# ── Timezone ──────────────────────────────────────────────────────────────────

_TZ = ZoneInfo(config.timezone)  # "Asia/Kolkata"


# ── Core functions ────────────────────────────────────────────────────────────

def now() -> datetime:
    """Return the current datetime in IST, timezone-aware."""
    return datetime.now(_TZ)


def today() -> date:
    """Return today's date in IST."""
    return now().date()


def current_time() -> time:
    """Return the current time in IST."""
    return now().time().replace(tzinfo=None)  # naive time for comparisons


def today_str() -> str:
    """Return today's date as ISO string: '2026-09-19'."""
    return today().isoformat()


def now_str() -> str:
    """Return current datetime as ISO string: '2026-09-19T00:03:00+05:30'."""
    return now().isoformat()


def day_of_week() -> int:
    """Return today's day of week: 0=Monday, 6=Sunday."""
    return today().weekday()


def day_name() -> str:
    """Return today's day name: 'Monday', 'Tuesday', etc."""
    return today().strftime("%A")


# ── Observation window ────────────────────────────────────────────────────────

def is_within_observation_window() -> bool:
    """
    Return True if current time is before the end of the same-day observation window.

    The observation window extends until OBSERVATION_WINDOW_END (default 23:00 IST).
    Within this window, a missing attendance update may still arrive from the portal.
    After this window, the system infers POTENTIALLY_UNREGISTERED.
    """
    window_end_str = config.portal.observation_window_end  # e.g. "23:00"
    h, m = map(int, window_end_str.split(":"))
    window_end = time(h, m)
    return current_time() < window_end


def parse_date(date_str: str) -> date:
    """
    Parse a date string in ISO format ('2026-09-19') to a date object.
    Raises ValueError if format is invalid.
    """
    return date.fromisoformat(date_str)


def parse_time(time_str: str) -> time:
    """
    Parse a time string in HH:MM format to a time object.
    Raises ValueError if format is invalid.
    """
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected 'HH:MM'.")
    return time(int(parts[0]), int(parts[1]))


def period_has_ended(end_time_str: str, reference_date: date | None = None) -> bool:
    """
    Return True if the given period end time (e.g. '11:00') has already passed today.

    Parameters
    ----------
    end_time_str : str
        Period end time in 'HH:MM' 24-hour format.
    reference_date : date | None
        The date to check against. Defaults to today().
    """
    ref_date = reference_date or today()
    if ref_date < today():
        return True  # Past dates — period definitely ended
    if ref_date > today():
        return False  # Future date — period hasn't started

    # Same day
    period_end = parse_time(end_time_str)
    return current_time() >= period_end
