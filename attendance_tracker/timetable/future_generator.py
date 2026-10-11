"""
timetable/future_generator.py
──────────────────────────────
Deterministic future class generator from current date until the last working day.

Combines:
- Start date (default: today in IST)
- End date / Semester Last Working Day (from repository / calendar config)
- Regular weekly timetable (Monday–Saturday schedule)
- Academic calendar & named holidays
- Sunday rule & 2nd/4th Saturday rules
- Mid-sem examination holidays
- User-defined working-day overrides & holiday overrides
- Non-attendance period filtering (LIB, SPORTS, ECA/CCA, SL, MTP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.calendar.service import CalendarService, get_calendar_service
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver
from attendance_tracker.timetable.service import ScheduledClass, TimetableService, get_timetable_service

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FutureClass:
    """A deterministic scheduled future class instance."""
    date: date
    period_num: int
    subject_code: str
    short_name: str
    full_name: str
    start_time: str
    end_time: str
    is_attendance_bearing: bool = True


@dataclass(frozen=True)
class FutureClassSchedule:
    """Aggregated future classes container."""
    start_date: date
    end_date: date
    total_classes: int
    classes: list[FutureClass]
    by_subject: dict[str, list[FutureClass]]
    by_date: dict[date, list[FutureClass]]
    working_days_count: int
    holidays_count: int


# ── Generator Class ───────────────────────────────────────────────────────────

class FutureClassGenerator:
    """
    Authoritative single-source generator for future attendance-bearing classes.
    """

    def __init__(
        self,
        timetable_service: Optional[TimetableService] = None,
        calendar_service: Optional[CalendarService] = None,
        subject_resolver: Optional[SubjectResolver] = None,
    ) -> None:
        self.timetable = timetable_service or get_timetable_service()
        self.calendar = calendar_service or get_calendar_service()
        self.resolver = subject_resolver or get_subject_resolver()

    def generate_classes(
        self,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> FutureClassSchedule:
        """
        Generate all future attendance-bearing classes occurring between
        start_date and end_date (inclusive).

        Parameters
        ----------
        start_date : date | str | None
            Start date of projection (defaults to today in IST).
        end_date : date | str | None
            End date / Last working day (defaults to configured last working day).
        """
        # 1. Normalize start_date
        if start_date is None:
            curr_start = datetime.now(_TZ).date()
        elif isinstance(start_date, (date, datetime)):
            curr_start = start_date if isinstance(start_date, date) else start_date.date()
        else:
            curr_start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d").date()

        # 2. Normalize end_date
        if end_date is None:
            curr_end = self.calendar.get_last_working_day()
        elif isinstance(end_date, (date, datetime)):
            curr_end = end_date if isinstance(end_date, date) else end_date.date()
        else:
            curr_end = datetime.strptime(str(end_date).strip(), "%Y-%m-%d").date()

        if curr_start > curr_end:
            return FutureClassSchedule(
                start_date=curr_start,
                end_date=curr_end,
                total_classes=0,
                classes=[],
                by_subject={},
                by_date={},
                working_days_count=0,
                holidays_count=0,
            )

        all_classes: list[FutureClass] = []
        by_subject: dict[str, list[FutureClass]] = {}
        by_date: dict[date, list[FutureClass]] = {}
        working_days = 0
        holidays = 0

        curr = curr_start
        try:
            mt1_start, mt1_end = self.calendar.get_mid_term_1()
        except Exception:
            mt1_start, mt1_end = date(2026, 9, 7), date(2026, 9, 12)

        while curr <= curr_end:
            cal_status = self.calendar.get_calendar_status(curr)
            is_midterm = (mt1_start <= curr <= mt1_end) and not (cal_status.is_override and cal_status.is_working_day)

            if cal_status.cancels_classes or cal_status.is_holiday or is_midterm:
                holidays += 1
                by_date[curr] = []
            else:
                working_days += 1
                scheduled_list = self.timetable.get_attendance_bearing_classes(curr)
                day_classes: list[FutureClass] = []

                for sc in scheduled_list:
                    # Resolve full name & short name from canonical resolver
                    subj = self.resolver.resolve(sc.subject_code or sc.subject)
                    code = subj.code if subj else sc.subject_code
                    short_name = subj.short_name if subj else sc.subject
                    full_name = subj.full_name if subj else sc.subject

                    fc = FutureClass(
                        date=curr,
                        period_num=sc.period_num,
                        subject_code=code,
                        short_name=short_name,
                        full_name=full_name,
                        start_time=sc.start_time,
                        end_time=sc.end_time,
                        is_attendance_bearing=True,
                    )
                    day_classes.append(fc)
                    all_classes.append(fc)

                    # Group by canonical code and short name
                    if code not in by_subject:
                        by_subject[code] = []
                    by_subject[code].append(fc)

                    if short_name.upper() != code.upper():
                        if short_name not in by_subject:
                            by_subject[short_name] = []
                        by_subject[short_name].append(fc)

                by_date[curr] = day_classes

            curr += timedelta(days=1)

        return FutureClassSchedule(
            start_date=curr_start,
            end_date=curr_end,
            total_classes=len(all_classes),
            classes=all_classes,
            by_subject=by_subject,
            by_date=by_date,
            working_days_count=working_days,
            holidays_count=holidays,
        )

    # Alias for generate_classes
    generate_schedule_between = generate_classes

    def get_classes_for_subject(
        self,
        subject_query: str,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> list[FutureClass]:
        """Return all future classes for a specific subject."""
        schedule = self.generate_classes(start_date=start_date, end_date=end_date)
        subj = self.resolver.resolve(subject_query)
        if subj:
            return schedule.by_subject.get(subj.code, [])
        query_norm = subject_query.strip().upper()
        return schedule.by_subject.get(query_norm, [])

    def get_classes_for_date(self, target_date: date | str) -> list[FutureClass]:
        """Return all scheduled attendance-bearing classes on a specific date."""
        if isinstance(target_date, (date, datetime)):
            d = target_date if isinstance(target_date, date) else target_date.date()
        else:
            d = datetime.strptime(str(target_date).strip(), "%Y-%m-%d").date()

        schedule = self.generate_classes(start_date=d, end_date=d)
        return schedule.by_date.get(d, [])

    def count_remaining_classes(
        self,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> int:
        """Count total remaining future attendance-bearing classes."""
        schedule = self.generate_classes(start_date=start_date, end_date=end_date)
        return schedule.total_classes

    def count_remaining_classes_by_subject(
        self,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> dict[str, int]:
        """Count remaining future classes per canonical subject code."""
        schedule = self.generate_classes(start_date=start_date, end_date=end_date)
        counts: dict[str, int] = {}
        for code, class_list in schedule.by_subject.items():
            # Only count canonical codes to avoid duplicate short names
            if code.startswith("22"):
                counts[code] = len(class_list)
        return counts


# ── Global Singleton and Accessor ─────────────────────────────────────────────

_default_generator: Optional[FutureClassGenerator] = None


def get_future_class_generator(
    timetable_service: Optional[TimetableService] = None,
    calendar_service: Optional[CalendarService] = None,
    subject_resolver: Optional[SubjectResolver] = None,
) -> FutureClassGenerator:
    """Return the singleton FutureClassGenerator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = FutureClassGenerator(
            timetable_service=timetable_service,
            calendar_service=calendar_service,
            subject_resolver=subject_resolver,
        )
    return _default_generator
