"""
timetable/service.py
────────────────────
Timetable service for loading, querying, and reasoning about scheduled classes.

RULES:
- Timezone-aware (Asia/Kolkata).
- Loads timetable definition from JSON data.
- Distinguishes attendance-bearing subjects from non-attendance activities.
- Handles effective dates, lunch gap, period time boundaries, and day-of-week queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.time_service import parse_date, parse_time

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PeriodInfo:
    """Information about a single class period time slot."""
    period_num: int
    start_time: str  # "HH:MM" (24h)
    end_time: str    # "HH:MM" (24h)

    @property
    def start_time_obj(self) -> time:
        return parse_time(self.start_time)

    @property
    def end_time_obj(self) -> time:
        return parse_time(self.end_time)


@dataclass(frozen=True)
class ScheduledClass:
    """A scheduled class entry in the timetable."""
    period_num: int
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM"
    subject: str     # Short name, e.g. "DE", "OE-1", "LIB"
    subject_code: str  # Canonical code or activity code, e.g. "22PC1AM202"
    is_attendance_bearing: bool

    @property
    def start_time_obj(self) -> time:
        return parse_time(self.start_time)

    @property
    def end_time_obj(self) -> time:
        return parse_time(self.end_time)


# ── Service Class ─────────────────────────────────────────────────────────────

class TimetableService:
    """
    Encapsulates timetable loading, caching, and querying logic.
    """

    def __init__(self, data_path: Path | str | None = None) -> None:
        if data_path is None:
            data_path = Path(config.paths.timetable)
        self.data_path = Path(data_path)
        self._effective_from: Optional[date] = None
        self._periods: dict[int, PeriodInfo] = {}
        self._lunch_start: Optional[time] = None
        self._lunch_end: Optional[time] = None
        self._non_attendance_activities: set[str] = set()
        self._schedule: dict[str, list[ScheduledClass]] = {}
        self._load()

    def _load(self) -> None:
        """Load and validate timetable JSON."""
        if not self.data_path.exists():
            logger.warning(f"Timetable file not found at {self.data_path}")
            return

        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # 1. Effective date
        effective_from_str = raw.get("effective_from", config.timetable.effective_from)
        self._effective_from = parse_date(effective_from_str)

        # 2. Periods
        periods_data = raw.get("periods", [])
        for p in periods_data:
            p_num = int(p["period_num"])
            self._periods[p_num] = PeriodInfo(
                period_num=p_num,
                start_time=p["start_time"],
                end_time=p["end_time"],
            )

        # 3. Lunch break
        lunch_data = raw.get("lunch", {})
        if lunch_data:
            self._lunch_start = parse_time(lunch_data["start_time"])
            self._lunch_end = parse_time(lunch_data["end_time"])

        # 4. Non-attendance activities
        non_att = raw.get("non_attendance_activities", config.timetable.non_attendance_activities)
        self._non_attendance_activities = {act.upper() for act in non_att}

        # 5. Schedule
        schedule_data = raw.get("schedule", {})
        for day_name, entries in schedule_data.items():
            classes = []
            for entry in entries:
                p_num = int(entry["period_num"])
                p_info = self._periods.get(p_num)
                if p_info is None:
                    raise ValueError(f"Unknown period {p_num} in timetable for {day_name}")

                subj = entry["subject"]
                subj_code = entry.get("subject_code", subj)
                is_att = entry.get("is_attendance_bearing")
                if is_att is None:
                    is_att = subj.upper() not in self._non_attendance_activities

                classes.append(
                    ScheduledClass(
                        period_num=p_num,
                        start_time=p_info.start_time,
                        end_time=p_info.end_time,
                        subject=subj,
                        subject_code=subj_code,
                        is_attendance_bearing=bool(is_att),
                    )
                )
            classes.sort(key=lambda c: c.period_num)
            self._schedule[day_name.capitalize()] = classes

    def get_effective_date(self) -> date:
        """Return the effective start date of this timetable."""
        return self._effective_from or parse_date(config.timetable.effective_from)

    def is_effective(self, target_date: date) -> bool:
        """Return True if the timetable is in effect on target_date."""
        return target_date >= self.get_effective_date()

    def get_period_info(self, period_num: int) -> Optional[PeriodInfo]:
        """Return period time info for the given period number."""
        return self._periods.get(period_num)

    def get_classes_for_date(self, target_date: date) -> list[ScheduledClass]:
        """
        Return all scheduled classes for a specific calendar date.
        Returns an empty list if target_date is before the timetable effective date
        or if it is a Sunday.
        """
        if not self.is_effective(target_date):
            return []

        # 0=Monday, 5=Saturday, 6=Sunday
        if target_date.weekday() == 6:
            return []

        day_name = target_date.strftime("%A")
        return list(self._schedule.get(day_name, []))

    def get_schedule_for_day(self, day_name: str) -> list[ScheduledClass]:
        """Return scheduled classes for a named day of the week (e.g. 'Monday')."""
        return list(self._schedule.get(day_name.strip().capitalize(), []))

    def get_attendance_bearing_classes(self, target_date: date) -> list[ScheduledClass]:
        """
        Return only attendance-bearing classes scheduled on target_date.
        Excludes non-attendance activities like LIB, SPORTS, ECA/CCA, SL, MTP.
        """
        classes = self.get_classes_for_date(target_date)
        return [c for c in classes if c.is_attendance_bearing]

    def get_classes_for_subject(
        self,
        target_date: date,
        subject: str,
    ) -> list[ScheduledClass]:
        """
        Return all classes scheduled for a specific subject (by code or short name)
        on target_date.
        """
        query = subject.strip().upper()
        classes = self.get_classes_for_date(target_date)
        return [
            c for c in classes
            if c.subject.upper() == query or c.subject_code.upper() == query
        ]

    def has_class_today(
        self,
        subject: str,
        target_date: date | None = None,
    ) -> bool:
        """
        Return True if at least one attendance-bearing class is scheduled for
        this subject on target_date (defaults to today in IST).
        """
        if target_date is None:
            target_date = datetime.now(_TZ).date()

        classes = self.get_classes_for_subject(target_date, subject)
        return any(c.is_attendance_bearing for c in classes)

    def is_lunch_break(self, target_time: time | datetime) -> bool:
        """Return True if the given time falls within the scheduled lunch gap (13:00–13:40)."""
        if isinstance(target_time, datetime):
            if target_time.tzinfo is not None:
                target_time = target_time.astimezone(_TZ).time()
            else:
                target_time = target_time.time()

        if self._lunch_start is None or self._lunch_end is None:
            return time(13, 0) <= target_time < time(13, 40)

        return self._lunch_start <= target_time < self._lunch_end

    def get_class_at_datetime(self, dt: datetime) -> Optional[ScheduledClass]:
        """
        Return the scheduled class occurring at the specified datetime in IST.
        Returns None if before timetable start, on Sunday, outside class hours,
        or during lunch.
        """
        if dt.tzinfo is not None:
            dt_ist = dt.astimezone(_TZ)
        else:
            dt_ist = dt.replace(tzinfo=_TZ)

        target_date = dt_ist.date()
        target_time = dt_ist.time()

        if not self.is_effective(target_date) or target_date.weekday() == 6:
            return None

        if self.is_lunch_break(target_time):
            return None

        classes = self.get_classes_for_date(target_date)
        for c in classes:
            if c.start_time_obj <= target_time < c.end_time_obj:
                return c

        return None

    def is_attendance_bearing(self, subject: str) -> bool:
        """
        Return True if a subject/activity name is attendance-bearing.
        """
        return subject.strip().upper() not in self._non_attendance_activities


# ── Global Singleton and Convenience Functions ────────────────────────────────

_default_service: Optional[TimetableService] = None


def get_timetable_service() -> TimetableService:
    """Return the singleton TimetableService instance."""
    global _default_service
    if _default_service is None:
        _default_service = TimetableService()
    return _default_service


def get_classes_for_date(target_date: date) -> list[ScheduledClass]:
    return get_timetable_service().get_classes_for_date(target_date)


def get_attendance_bearing_classes(target_date: date) -> list[ScheduledClass]:
    return get_timetable_service().get_attendance_bearing_classes(target_date)


def get_classes_for_subject(target_date: date, subject: str) -> list[ScheduledClass]:
    return get_timetable_service().get_classes_for_subject(target_date, subject)


def has_class_today(subject: str, target_date: date | None = None) -> bool:
    return get_timetable_service().has_class_today(subject, target_date)


def get_class_at_datetime(dt: datetime) -> Optional[ScheduledClass]:
    return get_timetable_service().get_class_at_datetime(dt)


def is_lunch_break(target_time: time | datetime) -> bool:
    return get_timetable_service().is_lunch_break(target_time)


def is_attendance_bearing(subject: str) -> bool:
    return get_timetable_service().is_attendance_bearing(subject)
