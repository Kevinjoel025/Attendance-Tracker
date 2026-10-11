"""
attendance/forecast.py
───────────────────────
Authoritative deterministic semester forecast & planning engine.

CRITICAL PRODUCT RULES:
- Final decisions are ALWAYS based on OVERALL ATTENDANCE.
- Combines live SQLite snapshot with future classes generated until the Last Working Day.
- Answers:
  1. Maximum achievable overall attendance (attend 100% remaining).
  2. Minimum achievable overall attendance (miss 100% remaining).
  3. Feasibility of 75% college threshold & 70% personal target.
  4. Number of future classes required to reach targets.
  5. Maximum safe future classes to skip before semester ends.
  6. Tomorrow / specific date skip impact on overall attendance.
  7. Subject-specific skip impact on overall attendance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.attendance.repository import AttendanceRepository, get_repository
from attendance_tracker.calendar.service import CalendarService, get_calendar_service
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver
from attendance_tracker.timetable.future_generator import (
    FutureClass,
    FutureClassGenerator,
    FutureClassSchedule,
    get_future_class_generator,
)

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Result Dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SemesterForecastResult:
    """Complete forecast from current attendance state to semester end."""
    current_attended: int
    current_conducted: int
    current_percentage: float
    future_classes_count: int
    final_conducted_total: int
    max_possible_attended: int
    max_possible_percentage: float
    min_possible_percentage: float
    last_working_day: date
    days_remaining: int
    working_days_remaining: int
    target_75_feasible: bool
    classes_needed_for_75: int
    max_future_misses_for_75: int
    target_70_feasible: bool
    classes_needed_for_70: int
    max_future_misses_for_70: int
    remaining_by_subject: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DaySkipForecastResult:
    """Forecast of skipping classes on a specific date (e.g. tomorrow)."""
    target_date: date
    date_str: str
    day_name: str
    is_holiday: bool
    holiday_name: Optional[str]
    is_working_day: bool
    classes_count: int
    scheduled_classes: list[FutureClass]
    current_attended: int
    current_conducted: int
    current_percentage: float
    if_attend_all_attended: int
    if_attend_all_conducted: int
    if_attend_all_percentage: float
    if_skip_all_attended: int
    if_skip_all_conducted: int
    if_skip_all_percentage: float
    delta_if_skipped: float
    can_safely_skip_75: bool
    can_safely_skip_70: bool
    verdict: str


@dataclass(frozen=True)
class SubjectSkipImpactResult:
    """Result of skipping specific classes of a subject on overall attendance."""
    subject_code: str
    subject_short: str
    subject_full: str
    miss_count: int
    current_overall_attended: int
    current_overall_conducted: int
    current_overall_percentage: float
    projected_overall_attended: int
    projected_overall_conducted: int
    projected_overall_percentage: float
    overall_delta: float
    current_subject_attended: int
    current_subject_conducted: int
    current_subject_percentage: float
    projected_subject_attended: int
    projected_subject_conducted: int
    projected_subject_percentage: float
    subject_delta: float
    meets_75_overall: bool
    meets_70_overall: bool
    verdict: str


# ── Forecast Engine Class ─────────────────────────────────────────────────────

class SemesterForecastEngine:
    """
    Core deterministic semester attendance forecasting and planning engine.
    """

    def __init__(
        self,
        repository: Optional[AttendanceRepository] = None,
        generator: Optional[FutureClassGenerator] = None,
        calendar_service: Optional[CalendarService] = None,
        subject_resolver: Optional[SubjectResolver] = None,
    ) -> None:
        self.repo = repository or get_repository()
        self.generator = generator or get_future_class_generator()
        self.calendar = calendar_service or get_calendar_service()
        self.resolver = subject_resolver or get_subject_resolver()

    def get_current_totals(self) -> tuple[int, int, float]:
        """Fetch latest authoritative overall attended, conducted, percentage."""
        overall = self.repo.get_all_current_attendance()
        if overall is None:
            return 0, 0, 0.0
        return overall.total_attended, overall.total_conducted, overall.overall_percentage

    def forecast_semester(
        self,
        start_date: Optional[date | str] = None,
        end_date: Optional[date | str] = None,
    ) -> SemesterForecastResult:
        """
        Forecast overall attendance boundaries from start_date until semester end.
        """
        curr_att, curr_cond, curr_pct = self.get_current_totals()
        schedule: FutureClassSchedule = self.generator.generate_classes(
            start_date=start_date,
            end_date=end_date,
        )

        future_count = schedule.total_classes
        final_cond = curr_cond + future_count
        max_att = curr_att + future_count

        max_pct = current_percentage(max_att, final_cond)
        min_pct = current_percentage(curr_att, final_cond)

        college_t = config.attendance.college_threshold
        personal_t = config.attendance.personal_target

        # 75% calculations
        feasible_75 = max_pct >= college_t
        if feasible_75:
            # Number of future classes that MUST be attended to end at >= 75%
            # (curr_att + x) / (curr_cond + future_count) >= 0.75
            # curr_att + x >= 0.75 * final_cond
            # x >= 0.75 * final_cond - curr_att
            min_needed_75 = max(0, math.ceil((college_t / 100.0) * final_cond - curr_att))
            max_misses_75 = max(0, future_count - min_needed_75)
        else:
            min_needed_75 = future_count + 1  # impossible
            max_misses_75 = 0

        # 70% calculations
        feasible_70 = max_pct >= personal_t
        if feasible_70:
            min_needed_70 = max(0, math.ceil((personal_t / 100.0) * final_cond - curr_att))
            max_misses_70 = max(0, future_count - min_needed_70)
        else:
            min_needed_70 = future_count + 1
            max_misses_70 = 0

        # Count remaining per canonical subject code
        remaining_by_sub: dict[str, int] = {}
        for code, clist in schedule.by_subject.items():
            if code.startswith("22"):
                remaining_by_sub[code] = len(clist)

        now_date = schedule.start_date
        days_rem = max(0, (schedule.end_date - now_date).days + 1)

        return SemesterForecastResult(
            current_attended=curr_att,
            current_conducted=curr_cond,
            current_percentage=curr_pct,
            future_classes_count=future_count,
            final_conducted_total=final_cond,
            max_possible_attended=max_att,
            max_possible_percentage=max_pct,
            min_possible_percentage=min_pct,
            last_working_day=schedule.end_date,
            days_remaining=days_rem,
            working_days_remaining=schedule.working_days_count,
            target_75_feasible=feasible_75,
            classes_needed_for_75=min_needed_75 if feasible_75 else 0,
            max_future_misses_for_75=max_misses_75,
            target_70_feasible=feasible_70,
            classes_needed_for_70=min_needed_70 if feasible_70 else 0,
            max_future_misses_for_70=max_misses_70,
            remaining_by_subject=remaining_by_sub,
        )

    def forecast_day_skip(
        self,
        target_date: Optional[date | str] = None,
    ) -> DaySkipForecastResult:
        """
        Evaluate the impact of skipping classes on a specific date (default: tomorrow).
        """
        if target_date is None:
            eval_date = datetime.now(_TZ).date() + timedelta(days=1)
        elif isinstance(target_date, (date, datetime)):
            eval_date = target_date if isinstance(target_date, date) else target_date.date()
        else:
            eval_date = datetime.strptime(str(target_date).strip(), "%Y-%m-%d").date()

        date_str = eval_date.strftime("%Y-%m-%d")
        day_name = eval_date.strftime("%A")

        curr_att, curr_cond, curr_pct = self.get_current_totals()
        cal_status = self.calendar.get_calendar_status(eval_date)

        if cal_status.cancels_classes or cal_status.is_holiday:
            return DaySkipForecastResult(
                target_date=eval_date,
                date_str=date_str,
                day_name=day_name,
                is_holiday=True,
                holiday_name=cal_status.event_name or "Holiday",
                is_working_day=False,
                classes_count=0,
                scheduled_classes=[],
                current_attended=curr_att,
                current_conducted=curr_cond,
                current_percentage=curr_pct,
                if_attend_all_attended=curr_att,
                if_attend_all_conducted=curr_cond,
                if_attend_all_percentage=curr_pct,
                if_skip_all_attended=curr_att,
                if_skip_all_conducted=curr_cond,
                if_skip_all_percentage=curr_pct,
                delta_if_skipped=0.0,
                can_safely_skip_75=True,
                can_safely_skip_70=True,
                verdict=f"{day_name} ({date_str}) is a holiday ({cal_status.event_name or 'No classes'}). No attendance-bearing classes are scheduled.",
            )

        classes_today = self.generator.get_classes_for_date(eval_date)
        k = len(classes_today)

        if k == 0:
            return DaySkipForecastResult(
                target_date=eval_date,
                date_str=date_str,
                day_name=day_name,
                is_holiday=False,
                holiday_name=None,
                is_working_day=True,
                classes_count=0,
                scheduled_classes=[],
                current_attended=curr_att,
                current_conducted=curr_cond,
                current_percentage=curr_pct,
                if_attend_all_attended=curr_att,
                if_attend_all_conducted=curr_cond,
                if_attend_all_percentage=curr_pct,
                if_skip_all_attended=curr_att,
                if_skip_all_conducted=curr_cond,
                if_skip_all_percentage=curr_pct,
                delta_if_skipped=0.0,
                can_safely_skip_75=True,
                can_safely_skip_70=True,
                verdict=f"No attendance-bearing classes are scheduled on {day_name} ({date_str}).",
            )

        attend_all_att = curr_att + k
        attend_all_cond = curr_cond + k
        attend_all_pct = current_percentage(attend_all_att, attend_all_cond)

        skip_all_att = curr_att
        skip_all_cond = curr_cond + k
        skip_all_pct = current_percentage(skip_all_att, skip_all_cond)

        delta = round(skip_all_pct - curr_pct, 2)
        college_t = config.attendance.college_threshold
        personal_t = config.attendance.personal_target

        can_skip_75 = skip_all_pct >= college_t
        can_skip_70 = skip_all_pct >= personal_t

        if can_skip_75:
            verdict = (
                f"You have {k} attendance-bearing classes on {day_name}. "
                f"Skipping them drops overall attendance from {curr_pct:.2f}% to {skip_all_pct:.2f}%, "
                f"which remains safely above the 75.0% college threshold."
            )
        elif can_skip_70:
            verdict = (
                f"You have {k} attendance-bearing classes on {day_name}. "
                f"Skipping them drops overall attendance from {curr_pct:.2f}% to {skip_all_pct:.2f}%. "
                f"This is below the 75.0% college requirement, but above your 70.0% personal target."
            )
        else:
            verdict = (
                f"You have {k} attendance-bearing classes on {day_name}. "
                f"Skipping them drops overall attendance from {curr_pct:.2f}% to {skip_all_pct:.2f}%. "
                f"At your current overall attendance ({curr_pct:.2f}%), you have no margin to skip."
            )

        return DaySkipForecastResult(
            target_date=eval_date,
            date_str=date_str,
            day_name=day_name,
            is_holiday=False,
            holiday_name=None,
            is_working_day=True,
            classes_count=k,
            scheduled_classes=classes_today,
            current_attended=curr_att,
            current_conducted=curr_cond,
            current_percentage=curr_pct,
            if_attend_all_attended=attend_all_att,
            if_attend_all_conducted=attend_all_cond,
            if_attend_all_percentage=attend_all_pct,
            if_skip_all_attended=skip_all_att,
            if_skip_all_conducted=skip_all_cond,
            if_skip_all_percentage=skip_all_pct,
            delta_if_skipped=delta,
            can_safely_skip_75=can_skip_75,
            can_safely_skip_70=can_skip_70,
            verdict=verdict,
        )

    def forecast_subject_skip(
        self,
        subject_query: str,
        miss_count: int = 1,
    ) -> SubjectSkipImpactResult:
        """
        Evaluate the impact of missing `miss_count` classes of a specific subject
        with OVERALL ATTENDANCE as the primary decision metric.
        """
        subj = self.resolver.resolve(subject_query)
        if subj is None:
            raise ValueError(f"Subject '{subject_query}' could not be resolved.")

        curr_overall_att, curr_overall_cond, curr_overall_pct = self.get_current_totals()
        subj_rec = self.repo.get_attendance_for_subject(subj.code)

        s_att = subj_rec.attended if subj_rec else 0
        s_cond = subj_rec.conducted if subj_rec else 0
        s_pct = subj_rec.percentage if subj_rec else 0.0

        # Primary calculation: OVERALL impact
        proj_overall_att = curr_overall_att
        proj_overall_cond = curr_overall_cond + miss_count
        proj_overall_pct = current_percentage(proj_overall_att, proj_overall_cond)
        overall_delta = round(proj_overall_pct - curr_overall_pct, 2)

        # Secondary calculation: Subject-level impact
        proj_subj_att = s_att
        proj_subj_cond = s_cond + miss_count
        proj_subj_pct = current_percentage(proj_subj_att, proj_subj_cond)
        subj_delta = round(proj_subj_pct - s_pct, 2)

        college_t = config.attendance.college_threshold
        personal_t = config.attendance.personal_target

        meets_75 = proj_overall_pct >= college_t
        meets_70 = proj_overall_pct >= personal_t

        if meets_75:
            verdict = (
                f"Missing {miss_count} {subj.short_name} classes changes your overall attendance "
                f"from {curr_overall_att}/{curr_overall_cond} ({curr_overall_pct:.2f}%) to "
                f"{proj_overall_att}/{proj_overall_cond} ({proj_overall_pct:.2f}%), "
                f"which remains above the 75.0% college threshold."
            )
        elif meets_70:
            verdict = (
                f"Missing {miss_count} {subj.short_name} classes changes your overall attendance "
                f"from {curr_overall_att}/{curr_overall_cond} ({curr_overall_pct:.2f}%) to "
                f"{proj_overall_att}/{proj_overall_cond} ({proj_overall_pct:.2f}%). "
                f"This drops overall attendance below 75.0%, but stays above your 70.0% target."
            )
        else:
            verdict = (
                f"Missing {miss_count} {subj.short_name} classes changes your overall attendance "
                f"from {curr_overall_att}/{curr_overall_cond} ({curr_overall_pct:.2f}%) to "
                f"{proj_overall_att}/{proj_overall_cond} ({proj_overall_pct:.2f}%). "
                f"You are below the 75.0% college requirement and cannot safely skip these classes."
            )

        return SubjectSkipImpactResult(
            subject_code=subj.code,
            subject_short=subj.short_name,
            subject_full=subj.full_name,
            miss_count=miss_count,
            current_overall_attended=curr_overall_att,
            current_overall_conducted=curr_overall_cond,
            current_overall_percentage=curr_overall_pct,
            projected_overall_attended=proj_overall_att,
            projected_overall_conducted=proj_overall_cond,
            projected_overall_percentage=proj_overall_pct,
            overall_delta=overall_delta,
            current_subject_attended=s_att,
            current_subject_conducted=s_cond,
            current_subject_percentage=s_pct,
            projected_subject_attended=proj_subj_att,
            projected_subject_conducted=proj_subj_cond,
            projected_subject_percentage=proj_subj_pct,
            subject_delta=subj_delta,
            meets_75_overall=meets_75,
            meets_70_overall=meets_70,
            verdict=verdict,
        )


# ── Global Singleton and Accessor ─────────────────────────────────────────────

_default_forecast_engine: Optional[SemesterForecastEngine] = None


def get_forecast_engine(
    repository: Optional[AttendanceRepository] = None,
    generator: Optional[FutureClassGenerator] = None,
    calendar_service: Optional[CalendarService] = None,
    subject_resolver: Optional[SubjectResolver] = None,
) -> SemesterForecastEngine:
    """Return the singleton SemesterForecastEngine instance."""
    global _default_forecast_engine
    if _default_forecast_engine is None:
        _default_forecast_engine = SemesterForecastEngine(
            repository=repository,
            generator=generator,
            calendar_service=calendar_service,
            subject_resolver=subject_resolver,
        )
    return _default_forecast_engine
