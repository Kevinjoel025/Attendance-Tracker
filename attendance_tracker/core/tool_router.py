"""
core/tool_router.py
───────────────────
Deterministic Tool Router & Response Pipeline.

Coordinates:
1. Natural language understanding via NLUInferenceEngine.
2. Subject resolution via SubjectResolver.
3. Execution of deterministic calculations in attendance.engine & forecast.py.
4. Database queries via AttendanceRepository.
5. Response text formatting via response.templates.
6. Overall attendance as authoritative primary decision metric.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.attendance.change_detector import (
    AttendanceChangeDetector,
    AttendanceChangeResult,
    AttendanceStatus,
)
from attendance_tracker.attendance.comparator import (
    SnapshotComparisonEngine,
    SnapshotComparisonResult,
    get_comparison_engine,
)
from attendance_tracker.attendance.engine import (
    ClassesRequiredResult,
    MaxCanMissResult,
    ProjectionResult,
    classes_required_for_target,
    compute_projection,
    max_classes_can_miss,
)
from attendance_tracker.attendance.forecast import (
    DaySkipForecastResult,
    SemesterForecastEngine,
    SemesterForecastResult,
    SubjectSkipImpactResult,
    get_forecast_engine,
)
from attendance_tracker.attendance.repository import (
    AttendanceChangeRecord,
    AttendanceRepository,
    OverallAttendance,
    Subject,
    SubjectAttendance,
    get_repository,
)
from attendance_tracker.calendar.service import CalendarService, get_calendar_service
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import (
    SubjectResolver,
    get_subject_resolver,
)
from attendance_tracker.core.thresholds import (
    COLLEGE_THRESHOLD,
    PERSONAL_TARGET,
    resolve_threshold,
)
from attendance_tracker.response.templates import (
    format_change_updates,
    format_classes_required,
    format_date_schedule,
    format_day_skip_forecast,
    format_error,
    format_help,
    format_max_can_miss,
    format_overall_attendance,
    format_projection,
    format_semester_forecast,
    format_snapshot_comparison,
    format_subject_attendance,
    format_subject_skip_impact,
    format_unknown,
)
from attendance_tracker.slm.inference.nlu import (
    NLUInferenceEngine,
    NLUResult,
    get_nlu_engine,
)
from attendance_tracker.timetable.future_generator import (
    FutureClassGenerator,
    get_future_class_generator,
)
from attendance_tracker.timetable.service import TimetableService, get_timetable_service

logger = get_logger(__name__)
_TZ = ZoneInfo(config.timezone)


@dataclass(frozen=True)
class ToolResult:
    """Structured result returned by tool execution."""
    success: bool
    intent: str
    tool_name: str
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    subject: Optional[Subject] = None


@dataclass(frozen=True)
class PipelineResponse:
    """End-to-end response containing NLU output, tool execution, and rendered text."""
    query: str
    nlu: NLUResult
    tool: ToolResult
    response_text: str
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "nlu": self.nlu.to_dict(),
            "tool": {
                "success": self.tool.success,
                "intent": self.tool.intent,
                "tool_name": self.tool.tool_name,
                "error_message": self.tool.error_message,
                "error_type": self.tool.error_type,
                "subject": self.tool.subject.short_name if self.tool.subject else None,
            },
            "response_text": self.response_text,
            "total_time_ms": round(self.total_time_ms, 2),
        }


class ToolRouter:
    """
    Routes parsed NLU intents and planning questions to deterministic tools.
    """

    def __init__(
        self,
        repository: Optional[AttendanceRepository] = None,
        subject_resolver: Optional[SubjectResolver] = None,
        nlu_engine: Optional[NLUInferenceEngine] = None,
        calendar_service: Optional[CalendarService] = None,
        timetable_service: Optional[TimetableService] = None,
    ) -> None:
        self.repo = repository or get_repository()
        self.resolver = subject_resolver or get_subject_resolver(repository=self.repo)
        self._nlu_engine = nlu_engine
        self.calendar = calendar_service or get_calendar_service(repository=self.repo)
        self.timetable = timetable_service or get_timetable_service()
        self.generator = get_future_class_generator(
            timetable_service=self.timetable,
            calendar_service=self.calendar,
            subject_resolver=self.resolver,
        )
        self.forecast_engine = get_forecast_engine(
            repository=self.repo,
            generator=self.generator,
            calendar_service=self.calendar,
            subject_resolver=self.resolver,
        )
        self.comparator = get_comparison_engine(
            repository=self.repo,
            subject_resolver=self.resolver,
        )
        self.change_detector = AttendanceChangeDetector(
            timetable_service=self.timetable,
            calendar_service=self.calendar,
        )

    @property
    def nlu_engine(self) -> NLUInferenceEngine:
        if self._nlu_engine is None:
            self._nlu_engine = get_nlu_engine()
        return self._nlu_engine

    # ── Tool Implementations ──────────────────────────────────────────────────

    def route_get_attendance(self) -> ToolResult:
        """Fetch overall attendance across all subjects."""
        overall = self.repo.get_all_current_attendance()
        if overall is None or not overall.subject_records:
            return ToolResult(
                success=False,
                intent="GET_ATTENDANCE",
                tool_name="get_all_current_attendance",
                error_message="No attendance records found in database.",
                error_type="DATABASE_EMPTY",
            )
        return ToolResult(
            success=True,
            intent="GET_ATTENDANCE",
            tool_name="get_all_current_attendance",
            data=overall,
        )

    def route_get_subject_attendance(self, subject_str: Optional[str]) -> ToolResult:
        """Fetch attendance for a specific resolved subject."""
        if not subject_str:
            return ToolResult(
                success=False,
                intent="GET_SUBJECT_ATTENDANCE",
                tool_name="get_attendance_for_subject",
                error_message="Please specify a subject (e.g. OS, DE, IoT, ACD).",
                error_type="MISSING_SUBJECT",
            )

        subj = self.resolver.resolve(subject_str)
        if subj is None:
            return ToolResult(
                success=False,
                intent="GET_SUBJECT_ATTENDANCE",
                tool_name="get_attendance_for_subject",
                error_message=f"Could not recognize subject '{subject_str}'.",
                error_type="UNKNOWN_SUBJECT",
            )

        record = self.repo.get_attendance_for_subject(subj.code)
        if record is None:
            return ToolResult(
                success=False,
                intent="GET_SUBJECT_ATTENDANCE",
                tool_name="get_attendance_for_subject",
                error_message=f"No attendance records found for {subj.full_name} ({subj.short_name}).",
                error_type="NO_RECORDS",
                subject=subj,
            )

        return ToolResult(
            success=True,
            intent="GET_SUBJECT_ATTENDANCE",
            tool_name="get_attendance_for_subject",
            data=record,
            subject=subj,
        )

    def route_projection(
        self,
        subject_str: Optional[str],
        count: Optional[int],
        raw_query: str = "",
    ) -> ToolResult:
        """Calculate attendance projection for attending or missing classes."""
        if not subject_str:
            return ToolResult(
                success=False,
                intent="ATTENDANCE_PROJECTION",
                tool_name="calculate_projection",
                error_message="Please specify a subject for the projection.",
                error_type="MISSING_SUBJECT",
            )

        subj = self.resolver.resolve(subject_str)
        if subj is None:
            return ToolResult(
                success=False,
                intent="ATTENDANCE_PROJECTION",
                tool_name="calculate_projection",
                error_message=f"Could not recognize subject '{subject_str}'.",
                error_type="UNKNOWN_SUBJECT",
            )

        record = self.repo.get_attendance_for_subject(subj.code)
        if record is None:
            return ToolResult(
                success=False,
                intent="ATTENDANCE_PROJECTION",
                tool_name="calculate_projection",
                error_message=f"No attendance records found for {subj.full_name}.",
                error_type="NO_RECORDS",
                subject=subj,
            )

        n_classes = count if (count is not None and count > 0) else 1

        q_lower = raw_query.lower()
        if any(w in q_lower for w in ["miss", "skip", "bunk", "absent"]):
            action = "MISS"
        else:
            action = "ATTEND"

        result = compute_projection(
            subject_code=subj.code,
            attended=record.attended,
            conducted=record.conducted,
            action=action,
            count=n_classes,
            college_threshold=subj.threshold,
        )

        return ToolResult(
            success=True,
            intent="ATTENDANCE_PROJECTION",
            tool_name="calculate_projection",
            data=result,
            subject=subj,
        )

    def route_classes_required(
        self,
        subject_str: Optional[str],
        target: Optional[float] = None,
        raw_query: str = "",
    ) -> ToolResult:
        """Calculate classes required to reach a target percentage."""
        if not subject_str:
            return ToolResult(
                success=False,
                intent="CLASSES_REQUIRED_FOR_TARGET",
                tool_name="classes_required_for_target",
                error_message="Please specify a subject to calculate required classes.",
                error_type="MISSING_SUBJECT",
            )

        subj = self.resolver.resolve(subject_str)
        if subj is None:
            return ToolResult(
                success=False,
                intent="CLASSES_REQUIRED_FOR_TARGET",
                tool_name="classes_required_for_target",
                error_message=f"Could not recognize subject '{subject_str}'.",
                error_type="UNKNOWN_SUBJECT",
            )

        record = self.repo.get_attendance_for_subject(subj.code)
        if record is None:
            return ToolResult(
                success=False,
                intent="CLASSES_REQUIRED_FOR_TARGET",
                tool_name="classes_required_for_target",
                error_message=f"No attendance records found for {subj.full_name}.",
                error_type="NO_RECORDS",
                subject=subj,
            )

        target_pct = resolve_threshold(
            query=raw_query,
            extracted_target=target,
            intent="CLASSES_REQUIRED_FOR_TARGET",
            default_threshold=subj.threshold,
        )

        raw_result = classes_required_for_target(
            record.attended,
            record.conducted,
            target_pct,
        )

        result = ClassesRequiredResult(
            subject_code=subj.code,
            current_attended=raw_result.current_attended,
            current_conducted=raw_result.current_conducted,
            current_percentage=raw_result.current_percentage,
            target_percentage=target_pct,
            classes_required=raw_result.classes_required,
            already_above_target=raw_result.already_above_target,
            impossible=raw_result.impossible,
        )

        return ToolResult(
            success=True,
            intent="CLASSES_REQUIRED_FOR_TARGET",
            tool_name="classes_required_for_target",
            data=result,
            subject=subj,
        )

    def route_max_can_miss(
        self,
        subject_str: Optional[str],
        threshold: Optional[float] = None,
        raw_query: str = "",
    ) -> ToolResult:
        """Calculate maximum classes that can be missed while staying above threshold."""
        if not subject_str:
            return ToolResult(
                success=False,
                intent="MAX_CLASSES_CAN_MISS",
                tool_name="max_classes_can_miss",
                error_message="Please specify a subject to calculate bunk margin.",
                error_type="MISSING_SUBJECT",
            )

        subj = self.resolver.resolve(subject_str)
        if subj is None:
            return ToolResult(
                success=False,
                intent="MAX_CLASSES_CAN_MISS",
                tool_name="max_classes_can_miss",
                error_message=f"Could not recognize subject '{subject_str}'.",
                error_type="UNKNOWN_SUBJECT",
            )

        record = self.repo.get_attendance_for_subject(subj.code)
        if record is None:
            return ToolResult(
                success=False,
                intent="MAX_CLASSES_CAN_MISS",
                tool_name="max_classes_can_miss",
                error_message=f"No attendance records found for {subj.full_name}.",
                error_type="NO_RECORDS",
                subject=subj,
            )

        threshold_pct = resolve_threshold(
            query=raw_query,
            extracted_target=threshold,
            intent="MAX_CLASSES_CAN_MISS",
            default_threshold=subj.threshold,
        )

        raw_result = max_classes_can_miss(
            record.attended,
            record.conducted,
            threshold_pct,
        )

        result = MaxCanMissResult(
            subject_code=subj.code,
            current_attended=raw_result.current_attended,
            current_conducted=raw_result.current_conducted,
            current_percentage=raw_result.current_percentage,
            threshold_percentage=threshold_pct,
            max_can_miss=raw_result.max_can_miss,
            already_below_threshold=raw_result.already_below_threshold,
        )

        return ToolResult(
            success=True,
            intent="MAX_CLASSES_CAN_MISS",
            tool_name="max_classes_can_miss",
            data=result,
            subject=subj,
        )

    def route_check_attendance_update(
        self,
        subject_str: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> ToolResult:
        """Check for recent attendance changes or snapshot updates."""
        try:
            res = self.comparator.compare_latest_with_previous()
            return ToolResult(
                success=True,
                intent="CHECK_ATTENDANCE_UPDATE",
                tool_name="compare_latest_with_previous",
                data=res,
            )
        except Exception as e:
            logger.error(f"Error checking attendance updates: {e}")
            return ToolResult(
                success=False,
                intent="CHECK_ATTENDANCE_UPDATE",
                tool_name="compare_latest_with_previous",
                error_message=str(e),
                error_type="COMPARISON_ERROR",
            )

    # ── Phase 13 Planning & Forecast Routes ───────────────────────────────────

    def route_can_skip_tomorrow(self, target_date: Optional[date] = None) -> ToolResult:
        """Evaluate if student can skip classes tomorrow with overall attendance impact."""
        try:
            res: DaySkipForecastResult = self.forecast_engine.forecast_day_skip(target_date=target_date)
            return ToolResult(
                success=True,
                intent="CAN_SKIP_DAY",
                tool_name="forecast_day_skip",
                data=res,
            )
        except Exception as e:
            logger.error(f"Error forecasting day skip: {e}")
            return ToolResult(
                success=False,
                intent="CAN_SKIP_DAY",
                tool_name="forecast_day_skip",
                error_message=str(e),
            )

    def route_semester_forecast(self) -> ToolResult:
        """Forecast maximum possible attendance, total remaining classes, and bunk margins."""
        try:
            res: SemesterForecastResult = self.forecast_engine.forecast_semester()
            return ToolResult(
                success=True,
                intent="SEMESTER_FORECAST",
                tool_name="forecast_semester",
                data=res,
            )
        except Exception as e:
            logger.error(f"Error forecasting semester: {e}")
            return ToolResult(
                success=False,
                intent="SEMESTER_FORECAST",
                tool_name="forecast_semester",
                error_message=str(e),
            )

    def route_subject_skip_impact(self, subject_str: str, miss_count: int) -> ToolResult:
        """Calculate overall attendance impact when missing N classes of a subject."""
        try:
            subj = self.resolver.resolve(subject_str)
            if subj is None:
                return ToolResult(
                    success=False,
                    intent="SUBJECT_SKIP_IMPACT",
                    tool_name="forecast_subject_skip",
                    error_message=f"Could not recognize subject '{subject_str}'.",
                    error_type="UNKNOWN_SUBJECT",
                )

            res: SubjectSkipImpactResult = self.forecast_engine.forecast_subject_skip(
                subject_query=subj.code,
                miss_count=miss_count,
            )
            return ToolResult(
                success=True,
                intent="SUBJECT_SKIP_IMPACT",
                tool_name="forecast_subject_skip",
                data=res,
                subject=subj,
            )
        except Exception as e:
            logger.error(f"Error forecasting subject skip: {e}")
            return ToolResult(
                success=False,
                intent="SUBJECT_SKIP_IMPACT",
                tool_name="forecast_subject_skip",
                error_message=str(e),
            )

    def route_schedule_for_date(self, target_date: date) -> ToolResult:
        """Fetch schedule and holiday status for a specific date."""
        try:
            date_str = target_date.strftime("%Y-%m-%d")
            day_name = target_date.strftime("%A")
            status = self.calendar.get_calendar_status(target_date)
            classes = self.generator.get_classes_for_date(target_date)

            return ToolResult(
                success=True,
                intent="GET_SCHEDULE",
                tool_name="get_schedule_for_date",
                data={
                    "date_str": date_str,
                    "day_name": day_name,
                    "is_holiday": status.is_holiday or status.cancels_classes,
                    "holiday_name": status.event_name,
                    "classes": classes,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                intent="GET_SCHEDULE",
                tool_name="get_schedule_for_date",
                error_message=str(e),
            )

    def route_help(self) -> ToolResult:
        """Return system capabilities and canonical subjects."""
        return ToolResult(
            success=True,
            intent="HELP",
            tool_name="help",
            data=self.resolver.get_canonical_subjects(),
        )

    def route_unknown(self, query: str, reason: Optional[str] = None) -> ToolResult:
        """Handle unclassified or malformed intents."""
        return ToolResult(
            success=False,
            intent="UNKNOWN",
            tool_name="unknown_intent_handler",
            error_message=reason or f"Could not determine intent for query: '{query}'",
            error_type="UNKNOWN_INTENT",
        )

    # ── Dispatcher & End-to-End Pipeline ──────────────────────────────────────

    def execute(self, nlu_result: NLUResult, raw_query: str = "") -> ToolResult:
        """
        Execute the appropriate tool given an NLUResult and raw query.
        """
        intent = nlu_result.intent

        if intent == "GET_ATTENDANCE":
            return self.route_get_attendance()

        if intent == "GET_SUBJECT_ATTENDANCE":
            return self.route_get_subject_attendance(nlu_result.subject)

        if intent == "ATTENDANCE_PROJECTION":
            return self.route_projection(nlu_result.subject, nlu_result.count, raw_query=raw_query)

        if intent == "CLASSES_REQUIRED_FOR_TARGET":
            return self.route_classes_required(nlu_result.subject, nlu_result.target, raw_query=raw_query)

        if intent == "MAX_CLASSES_CAN_MISS":
            return self.route_max_can_miss(nlu_result.subject, nlu_result.target, raw_query=raw_query)

        if intent == "CHECK_ATTENDANCE_UPDATE":
            return self.route_check_attendance_update(nlu_result.subject, nlu_result.date)

        if intent == "CAN_SKIP_DAY":
            return self.route_can_skip_tomorrow()

        if intent == "SEMESTER_FORECAST":
            return self.route_semester_forecast()

        if intent == "SUBJECT_SKIP_IMPACT":
            return self.route_subject_skip_impact(
                subject_str=nlu_result.subject or "",
                miss_count=nlu_result.count or 1,
            )

        if intent == "GET_SCHEDULE":
            tomorrow_date = datetime.now(_TZ).date() + timedelta(days=1)
            return self.route_schedule_for_date(tomorrow_date)

        if intent == "HELP":
            return self.route_help()

        return self.route_unknown(raw_query or nlu_result.raw_output, reason=nlu_result.malformed_reason)

    def format_response(self, tool_result: ToolResult, raw_query: str = "") -> str:
        """
        Render natural language text response for a tool execution result.
        """
        if not tool_result.success:
            if tool_result.intent == "UNKNOWN":
                return format_unknown(raw_query, reason=tool_result.error_message)
            return format_error(tool_result.error_message or "An error occurred during tool execution.")

        intent = tool_result.intent

        if intent == "GET_ATTENDANCE" and isinstance(tool_result.data, OverallAttendance):
            return format_overall_attendance(tool_result.data)

        if intent == "GET_SUBJECT_ATTENDANCE" and isinstance(tool_result.data, SubjectAttendance):
            return format_subject_attendance(tool_result.data)

        if intent == "ATTENDANCE_PROJECTION" and isinstance(tool_result.data, ProjectionResult):
            subj_name = tool_result.subject.full_name if tool_result.subject else None
            return format_projection(tool_result.data, subject_name=subj_name)

        if intent == "CLASSES_REQUIRED_FOR_TARGET" and isinstance(tool_result.data, ClassesRequiredResult):
            subj_name = tool_result.subject.full_name if tool_result.subject else None
            return format_classes_required(tool_result.data, subject_name=subj_name)

        if intent == "MAX_CLASSES_CAN_MISS" and isinstance(tool_result.data, MaxCanMissResult):
            subj_name = tool_result.subject.full_name if tool_result.subject else None
            return format_max_can_miss(tool_result.data, subject_name=subj_name)

        if intent == "CHECK_ATTENDANCE_UPDATE":
            if isinstance(tool_result.data, SnapshotComparisonResult):
                return format_snapshot_comparison(tool_result.data)
            latest = self.repo.get_latest_snapshot()
            time_str = latest.scraped_at if latest else None
            subj_str = tool_result.subject.short_name if tool_result.subject else None
            return format_change_updates(tool_result.data or [], snapshot_time=time_str, subject_filter=subj_str)

        if intent == "CAN_SKIP_DAY" and isinstance(tool_result.data, DaySkipForecastResult):
            return format_day_skip_forecast(tool_result.data)

        if intent == "SEMESTER_FORECAST" and isinstance(tool_result.data, SemesterForecastResult):
            return format_semester_forecast(tool_result.data)

        if intent == "SUBJECT_SKIP_IMPACT" and isinstance(tool_result.data, SubjectSkipImpactResult):
            return format_subject_skip_impact(tool_result.data)

        if intent == "GET_SCHEDULE" and isinstance(tool_result.data, dict):
            d = tool_result.data
            return format_date_schedule(
                date_str=d["date_str"],
                day_name=d["day_name"],
                is_holiday=d["is_holiday"],
                holiday_name=d["holiday_name"],
                classes=d["classes"],
            )

        if intent == "HELP":
            return format_help(tool_result.data)

        return format_unknown(raw_query)

    def process(self, query: str) -> PipelineResponse:
        """
        Execute full end-to-end processing: NLU / Semantic Routing -> Tool -> Formatting.
        """
        start_time = time.perf_counter()
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # ── Pre-routing for Phase 13 Assistant Planning & What-If Queries ─────

        # 1. Skip Tomorrow / Day Planning
        if any(p in q_lower for p in [
            "can i skip tomorrow",
            "can i bunk tomorrow",
            "should i go tomorrow",
            "can i miss tomorrow",
            "skip tomorrow",
            "bunk tomorrow",
            "if i skip tomorrow",
            "if i miss tomorrow",
        ]):
            nlu_res = NLUResult(
                raw_output="CAN_SKIP_DAY|NONE|NONE|NONE|tomorrow",
                intent="CAN_SKIP_DAY",
                subject=None,
                target=None,
                count=None,
                date="tomorrow",
                is_valid=True,
                confidence=1.0,
            )
            tool_res = self.route_can_skip_tomorrow()
            resp_text = self.format_response(tool_res, raw_query=q_clean)
            return PipelineResponse(
                query=q_clean,
                nlu=nlu_res,
                tool=tool_res,
                response_text=resp_text,
                total_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 2. Semester Forecast / Maximum Achievable Attendance / Remaining Classes
        if any(p in q_lower for p in [
            "maximum attendance",
            "highest attendance",
            "max attendance",
            "maximum overall attendance",
            "classes left this semester",
            "classes left",
            "classes remaining",
            "how many classes are left",
            "can i still reach 75",
            "can i reach 75",
            "how many classes can i skip before the semester ends",
            "how many classes can i skip until the end of the semester",
            "how many classes can i skip this semester",
            "how many classes can i miss before the semester ends",
            "when is my last working day",
            "semester end",
        ]):
            nlu_res = NLUResult(
                raw_output="SEMESTER_FORECAST|NONE|75.0|NONE|NONE",
                intent="SEMESTER_FORECAST",
                subject=None,
                target=75.0,
                count=None,
                date=None,
                is_valid=True,
                confidence=1.0,
            )
            tool_res = self.route_semester_forecast()
            resp_text = self.format_response(tool_res, raw_query=q_clean)
            return PipelineResponse(
                query=q_clean,
                nlu=nlu_res,
                tool=tool_res,
                response_text=resp_text,
                total_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 3. Daily Changes / Attendance History / Updates
        if any(p in q_lower for p in [
            "what changed today",
            "what changed yesterday",
            "did my attendance update today",
            "what classes were registered today",
            "which subjects changed today",
            "show my attendance history",
            "attendance history",
            "how much did my attendance increase today",
        ]):
            nlu_res = NLUResult(
                raw_output="CHECK_ATTENDANCE_UPDATE|NONE|NONE|NONE|today",
                intent="CHECK_ATTENDANCE_UPDATE",
                subject=None,
                target=None,
                count=None,
                date="today",
                is_valid=True,
                confidence=1.0,
            )
            tool_res = self.route_check_attendance_update()
            resp_text = self.format_response(tool_res, raw_query=q_clean)
            return PipelineResponse(
                query=q_clean,
                nlu=nlu_res,
                tool=tool_res,
                response_text=resp_text,
                total_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 4. Schedule & Holiday Lookups
        if any(p in q_lower for p in [
            "what classes do i have tomorrow",
            "classes tomorrow",
            "schedule tomorrow",
            "timetable tomorrow",
            "is tomorrow a holiday",
            "is today a holiday",
            "is saturday a working day",
        ]):
            target_d = datetime.now(_TZ).date() + timedelta(days=1)
            nlu_res = NLUResult(
                raw_output="GET_SCHEDULE|NONE|NONE|NONE|tomorrow",
                intent="GET_SCHEDULE",
                subject=None,
                target=None,
                count=None,
                date="tomorrow",
                is_valid=True,
                confidence=1.0,
            )
            tool_res = self.route_schedule_for_date(target_d)
            resp_text = self.format_response(tool_res, raw_query=q_clean)
            return PipelineResponse(
                query=q_clean,
                nlu=nlu_res,
                tool=tool_res,
                response_text=resp_text,
                total_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 5. Subject-Specific Miss / Skip Count with Overall Attendance Decision (e.g. "If I miss 5 ACD classes", "What happens if I miss 5 classes in OS?")
        # Exclude projection/target queries (e.g. "what is my projected attendance")
        if not any(w in q_lower for w in ["project", "reach", "target", "%", "percent", "need"]):
            skip_subject_match = re.search(r"(?:what\s+happens\s+if\s+i|what\s+if\s+i|if\s+i)\s+(?:miss|skip|bunk)\s+(\d+)\s+classes?(?:\s+of|\s+in)?\s+([a-zA-Z0-9\s\-]+)", q_lower)
            if not skip_subject_match:
                skip_subject_match = re.search(r"(?:what\s+happens\s+if\s+i|what\s+if\s+i|if\s+i)\s+(?:miss|skip|bunk)\s+(\d+)\s+([a-zA-Z0-9\-]+)(?:\s+classes)?", q_lower)

            if skip_subject_match:
                count_val = int(skip_subject_match.group(1))
                raw_sub = skip_subject_match.group(2).strip()
                resolved = self.resolver.resolve(raw_sub)
                if resolved is not None:
                    nlu_res = NLUResult(
                        raw_output=f"SUBJECT_SKIP_IMPACT|{resolved.short_name}|NONE|{count_val}|NONE",
                        intent="SUBJECT_SKIP_IMPACT",
                        subject=resolved.short_name,
                        target=None,
                        count=count_val,
                        date=None,
                        is_valid=True,
                        confidence=1.0,
                    )
                    tool_res = self.route_subject_skip_impact(resolved.code, count_val)
                    resp_text = self.format_response(tool_res, raw_query=q_clean)
                    return PipelineResponse(
                        query=q_clean,
                        nlu=nlu_res,
                        tool=tool_res,
                        response_text=resp_text,
                        total_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    )

        # ── Standard NLU Flow via Frozen SLM ──────────────────────────────────
        nlu_result = self.nlu_engine.predict(q_clean)
        tool_result = self.execute(nlu_result, raw_query=q_clean)
        response_text = self.format_response(tool_result, raw_query=q_clean)
        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        return PipelineResponse(
            query=q_clean,
            nlu=nlu_result,
            tool=tool_result,
            response_text=response_text,
            total_time_ms=total_elapsed,
        )


# ── Singleton Accessor ────────────────────────────────────────────────────────

_default_router: Optional[ToolRouter] = None


def get_tool_router(
    repository: Optional[AttendanceRepository] = None,
    subject_resolver: Optional[SubjectResolver] = None,
    nlu_engine: Optional[NLUInferenceEngine] = None,
) -> ToolRouter:
    """Return or initialize the singleton ToolRouter."""
    global _default_router
    if _default_router is None or repository is not None:
        _default_router = ToolRouter(
            repository=repository,
            subject_resolver=subject_resolver,
            nlu_engine=nlu_engine,
        )
    return _default_router
