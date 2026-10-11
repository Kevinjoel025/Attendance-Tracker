"""
attendance/change_detector.py
──────────────────────────────
Deterministic attendance change detector.

Compares two attendance snapshots (or current vs previous observations)
and determines the real-world attendance event using timetable, calendar,
and variable portal registration timing rules.

RULES:
- Deterministic logic only (no SLM).
- Uses actual snapshot timestamps, not hardcoded "yesterday".
- Observation window ends at 23:00 IST (OBSERVATION_WINDOW_END = "23:00").
- Does not fabricate individual class attribution when multiple classes change.
- Distinguishes all 10+ status codes defined by Version 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from attendance_tracker.calendar.service import (
    CalendarService,
    get_calendar_service,
)
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.time_service import parse_time
from attendance_tracker.timetable.service import (
    TimetableService,
    get_timetable_service,
)

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Status Enum ───────────────────────────────────────────────────────────────

class AttendanceStatus(str, Enum):
    """Authoritative attendance change detection status codes."""
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    HOLIDAY = "HOLIDAY"
    NO_CLASS_SCHEDULED = "NO_CLASS_SCHEDULED"
    CLASS_NOT_YET_OCCURRED = "CLASS_NOT_YET_OCCURRED"
    CLASS_MAY_NOT_BE_REGISTERED_YET = "CLASS_MAY_NOT_BE_REGISTERED_YET"
    DATA_MAY_BE_STALE = "DATA_MAY_BE_STALE"
    POTENTIALLY_UNREGISTERED = "POTENTIALLY_UNREGISTERED"
    MULTIPLE_CLASSES_REGISTERED = "MULTIPLE_CLASSES_REGISTERED"
    DATA_ANOMALY = "DATA_ANOMALY"
    NO_PREVIOUS_SNAPSHOT = "NO_PREVIOUS_SNAPSHOT"
    SCRAPE_FAILED = "SCRAPE_FAILED"


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttendanceChangeResult:
    """Structured result of attendance change detection."""
    subject_code: str
    status: AttendanceStatus
    explanation: str
    delta_attended: int
    delta_conducted: int
    prev_attended: Optional[int]
    prev_conducted: Optional[int]
    curr_attended: int
    curr_conducted: int
    prev_snapshot_time: Optional[datetime]
    curr_snapshot_time: datetime
    reference_date: date
    is_anomalous: bool = False


# ── Change Detector Class ─────────────────────────────────────────────────────

class AttendanceChangeDetector:
    """
    Evaluates deltas between snapshots against timetable and calendar context.
    """

    def __init__(
        self,
        timetable_service: Optional[TimetableService] = None,
        calendar_service: Optional[CalendarService] = None,
    ) -> None:
        self.timetable = timetable_service or get_timetable_service()
        self.calendar = calendar_service or get_calendar_service()

    def _get_observation_window_end(self) -> time:
        """Parse the observation window end time (default 23:00)."""
        window_end_str = config.portal.observation_window_end
        return parse_time(window_end_str)

    def detect_change(
        self,
        subject_code: str,
        curr_attended: int,
        curr_conducted: int,
        prev_attended: Optional[int] = None,
        prev_conducted: Optional[int] = None,
        prev_snapshot_time: Optional[datetime] = None,
        curr_snapshot_time: Optional[datetime] = None,
        reference_date: Optional[date] = None,
        current_time_override: Optional[datetime] = None,
        scrape_status: str = "SUCCESS",
    ) -> AttendanceChangeResult:
        """
        Detect and classify the change in attendance between two observations.

        Parameters
        ----------
        subject_code : str
            Canonical subject code or short name.
        curr_attended : int
            Current attended count.
        curr_conducted : int
            Current conducted count.
        prev_attended : Optional[int]
            Previous attended count (None if this is the first snapshot).
        prev_conducted : Optional[int]
            Previous conducted count (None if this is the first snapshot).
        prev_snapshot_time : Optional[datetime]
            Timestamp of the previous successful snapshot.
        curr_snapshot_time : Optional[datetime]
            Timestamp of current snapshot (defaults to now in IST).
        reference_date : Optional[date]
            Date to evaluate timetable/calendar against (defaults to date of curr_snapshot_time).
        current_time_override : Optional[datetime]
            Allows overriding the evaluation time (useful for unit tests).
        scrape_status : str
            "SUCCESS", "FAILED", or "PARTIAL".
        """
        # Determine evaluation times in IST
        now_dt = current_time_override or datetime.now(_TZ)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=_TZ)
        else:
            now_dt = now_dt.astimezone(_TZ)

        if curr_snapshot_time is None:
            curr_snapshot_dt = now_dt
        else:
            if curr_snapshot_time.tzinfo is None:
                curr_snapshot_dt = curr_snapshot_time.replace(tzinfo=_TZ)
            else:
                curr_snapshot_dt = curr_snapshot_time.astimezone(_TZ)

        if reference_date is None:
            ref_date = curr_snapshot_dt.date()
        else:
            ref_date = reference_date

        # ── Step 1: Check scrape failure ──────────────────────────────
        if scrape_status.upper() == "FAILED":
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.SCRAPE_FAILED,
                explanation="Portal scrape failed. Cannot determine attendance changes.",
                delta_attended=0,
                delta_conducted=0,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # ── Step 2: Check first snapshot (no previous baseline) ───────
        if prev_attended is None or prev_conducted is None:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.NO_PREVIOUS_SNAPSHOT,
                explanation=(
                    f"Initial baseline snapshot established: {curr_attended}/{curr_conducted}."
                ),
                delta_attended=0,
                delta_conducted=0,
                prev_attended=None,
                prev_conducted=None,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # ── Step 3: Compute deltas ────────────────────────────────────
        delta_attended = curr_attended - prev_attended
        delta_conducted = curr_conducted - prev_conducted

        # ── Step 4: Data anomalies ────────────────────────────────────
        is_anomaly = (
            curr_attended > curr_conducted
            or delta_conducted < 0
            or delta_attended < 0
            or delta_attended > delta_conducted
        )
        if is_anomaly:
            explanation = (
                f"Data anomaly detected: "
                f"attended={curr_attended}/{curr_conducted} (delta={delta_attended}/{delta_conducted}). "
                f"Counts cannot decrease and attended cannot exceed conducted."
            )
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.DATA_ANOMALY,
                explanation=explanation,
                delta_attended=delta_attended,
                delta_conducted=delta_conducted,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
                is_anomalous=True,
            )

        # ── Step 5: Direct registration cases ─────────────────────────
        # Exactly 1 class registered and attended
        if delta_conducted == 1 and delta_attended == 1:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.PRESENT,
                explanation="1 class registered on portal and attended (+1/+1). Marked Present.",
                delta_attended=1,
                delta_conducted=1,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # Exactly 1 class registered and absent
        if delta_conducted == 1 and delta_attended == 0:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.ABSENT,
                explanation="1 class registered on portal and missed (+0/+1). Marked Absent.",
                delta_attended=0,
                delta_conducted=1,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # Multiple classes registered simultaneously
        if delta_conducted > 1:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.MULTIPLE_CLASSES_REGISTERED,
                explanation=(
                    f"Multiple classes registered between snapshots: "
                    f"+{delta_attended} attended out of +{delta_conducted} conducted."
                ),
                delta_attended=delta_attended,
                delta_conducted=delta_conducted,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # ── Step 6: Delta = 0, Delta = 0 (Reasoning Required) ─────────
        # delta_conducted == 0 and delta_attended == 0

        # 6a. Check if reference_date is a holiday
        if self.calendar.is_holiday(ref_date):
            holiday_name = self.calendar.get_holiday_name(ref_date) or "Holiday"
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.HOLIDAY,
                explanation=f"No class was expected because {ref_date} is a holiday ({holiday_name}).",
                delta_attended=0,
                delta_conducted=0,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # 6b. Check timetable for attendance-bearing classes
        scheduled_classes = self.timetable.get_classes_for_subject(ref_date, subject_code)
        att_bearing_classes = [c for c in scheduled_classes if c.is_attendance_bearing]

        if not att_bearing_classes:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.NO_CLASS_SCHEDULED,
                explanation=(
                    f"No attendance-bearing class was scheduled for {subject_code} on "
                    f"{ref_date.strftime('%A')} ({ref_date})."
                ),
                delta_attended=0,
                delta_conducted=0,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # 6c. Attendance-bearing class(es) WERE scheduled
        # Find earliest start and latest end among scheduled periods
        latest_end_class = max(att_bearing_classes, key=lambda c: c.end_time_obj)
        earliest_start_class = min(att_bearing_classes, key=lambda c: c.start_time_obj)
        class_end_time = latest_end_class.end_time_obj
        class_start_time = earliest_start_class.start_time_obj

        today_dt_date = now_dt.date()
        obs_window_end = self._get_observation_window_end()

        # Case 1: Future date
        if ref_date > today_dt_date:
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.CLASS_NOT_YET_OCCURRED,
                explanation=f"Class on {ref_date} is in the future and has not yet occurred.",
                delta_attended=0,
                delta_conducted=0,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        # Case 2: Today
        if ref_date == today_dt_date:
            current_eval_time = now_dt.time()

            # Before class has ended
            if current_eval_time < class_end_time:
                return AttendanceChangeResult(
                    subject_code=subject_code,
                    status=AttendanceStatus.CLASS_NOT_YET_OCCURRED,
                    explanation=(
                        f"Today's {subject_code} class ({earliest_start_class.start_time}–{latest_end_class.end_time}) "
                        f"has not yet concluded."
                    ),
                    delta_attended=0,
                    delta_conducted=0,
                    prev_attended=prev_attended,
                    prev_conducted=prev_conducted,
                    curr_attended=curr_attended,
                    curr_conducted=curr_conducted,
                    prev_snapshot_time=prev_snapshot_time,
                    curr_snapshot_time=curr_snapshot_dt,
                    reference_date=ref_date,
                )

            # Class has ended today
            # Check if the snapshot data was taken before the class ended (stale snapshot)
            if (
                curr_snapshot_dt.date() == ref_date
                and curr_snapshot_dt.time() < class_end_time
            ):
                return AttendanceChangeResult(
                    subject_code=subject_code,
                    status=AttendanceStatus.DATA_MAY_BE_STALE,
                    explanation=(
                        f"Latest portal snapshot was recorded at {curr_snapshot_dt.strftime('%H:%M')}, "
                        f"which was before {subject_code} class concluded at {latest_end_class.end_time}."
                    ),
                    delta_attended=0,
                    delta_conducted=0,
                    prev_attended=prev_attended,
                    prev_conducted=prev_conducted,
                    curr_attended=curr_attended,
                    curr_conducted=curr_conducted,
                    prev_snapshot_time=prev_snapshot_time,
                    curr_snapshot_time=curr_snapshot_dt,
                    reference_date=ref_date,
                )

            # Fresh snapshot taken after class end
            if current_eval_time < obs_window_end:
                return AttendanceChangeResult(
                    subject_code=subject_code,
                    status=AttendanceStatus.CLASS_MAY_NOT_BE_REGISTERED_YET,
                    explanation=(
                        f"Class concluded at {latest_end_class.end_time}. "
                        f"Attendance may not have been registered on the portal yet "
                        f"(same-day observation window open until {config.portal.observation_window_end})."
                    ),
                    delta_attended=0,
                    delta_conducted=0,
                    prev_attended=prev_attended,
                    prev_conducted=prev_conducted,
                    curr_attended=curr_attended,
                    curr_conducted=curr_conducted,
                    prev_snapshot_time=prev_snapshot_time,
                    curr_snapshot_time=curr_snapshot_dt,
                    reference_date=ref_date,
                )
            else:
                # After 23:00 IST
                return AttendanceChangeResult(
                    subject_code=subject_code,
                    status=AttendanceStatus.POTENTIALLY_UNREGISTERED,
                    explanation=(
                        f"Class concluded at {latest_end_class.end_time} and observation window ended at "
                        f"{config.portal.observation_window_end}, but portal attendance remains unchanged. "
                        f"Attendance may not have been registered."
                    ),
                    delta_attended=0,
                    delta_conducted=0,
                    prev_attended=prev_attended,
                    prev_conducted=prev_conducted,
                    curr_attended=curr_attended,
                    curr_conducted=curr_conducted,
                    prev_snapshot_time=prev_snapshot_time,
                    curr_snapshot_time=curr_snapshot_dt,
                    reference_date=ref_date,
                )

        # Case 3: Past date (ref_date < today_dt_date)
        # Check if current snapshot was recorded after class ended on that date
        if (
            curr_snapshot_dt.date() < ref_date
            or (curr_snapshot_dt.date() == ref_date and curr_snapshot_dt.time() < class_end_time)
        ):
            return AttendanceChangeResult(
                subject_code=subject_code,
                status=AttendanceStatus.DATA_MAY_BE_STALE,
                explanation=(
                    f"Snapshot was taken before {subject_code} class ended on {ref_date}."
                ),
                delta_attended=0,
                delta_conducted=0,
                prev_attended=prev_attended,
                prev_conducted=prev_conducted,
                curr_attended=curr_attended,
                curr_conducted=curr_conducted,
                prev_snapshot_time=prev_snapshot_time,
                curr_snapshot_time=curr_snapshot_dt,
                reference_date=ref_date,
            )

        return AttendanceChangeResult(
            subject_code=subject_code,
            status=AttendanceStatus.POTENTIALLY_UNREGISTERED,
            explanation=(
                f"Class on past date {ref_date} concluded without any registered update on the portal."
            ),
            delta_attended=0,
            delta_conducted=0,
            prev_attended=prev_attended,
            prev_conducted=prev_conducted,
            curr_attended=curr_attended,
            curr_conducted=curr_conducted,
            prev_snapshot_time=prev_snapshot_time,
            curr_snapshot_time=curr_snapshot_dt,
            reference_date=ref_date,
        )


# ── Global Singleton and Convenience Function ─────────────────────────────────

_default_detector: Optional[AttendanceChangeDetector] = None


def get_change_detector() -> AttendanceChangeDetector:
    """Return the singleton AttendanceChangeDetector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AttendanceChangeDetector()
    return _default_detector


def detect_change(
    subject_code: str,
    curr_attended: int,
    curr_conducted: int,
    prev_attended: Optional[int] = None,
    prev_conducted: Optional[int] = None,
    prev_snapshot_time: Optional[datetime] = None,
    curr_snapshot_time: Optional[datetime] = None,
    reference_date: Optional[date] = None,
    current_time_override: Optional[datetime] = None,
    scrape_status: str = "SUCCESS",
) -> AttendanceChangeResult:
    return get_change_detector().detect_change(
        subject_code=subject_code,
        curr_attended=curr_attended,
        curr_conducted=curr_conducted,
        prev_attended=prev_attended,
        prev_conducted=prev_conducted,
        prev_snapshot_time=prev_snapshot_time,
        curr_snapshot_time=curr_snapshot_time,
        reference_date=reference_date,
        current_time_override=current_time_override,
        scrape_status=scrape_status,
    )
