"""
web/state.py
────────────
Centralized runtime state and singleton service management for FastAPI application.
Ensures SLM checkpoint, NLU engine, and backend services are loaded ONCE at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from attendance_tracker.attendance.change_detector import (
    AttendanceChangeDetector,
    get_change_detector,
)
from attendance_tracker.attendance.repository import (
    AttendanceRepository,
    get_repository,
)
from attendance_tracker.calendar.service import (
    CalendarService,
    get_calendar_service,
)
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import (
    SubjectResolver,
    get_subject_resolver,
)
from attendance_tracker.core.tool_router import (
    ToolRouter,
    get_tool_router,
)
from attendance_tracker.portal.service import (
    PortalAttendanceService,
    get_portal_service,
)
from attendance_tracker.slm.inference.nlu import (
    NLUInferenceEngine,
    get_nlu_engine,
)
from attendance_tracker.timetable.service import (
    TimetableService,
    get_timetable_service,
)

logger = get_logger(__name__)


@dataclass
class AppServices:
    """Container for pre-initialized singleton backend services."""

    repository: AttendanceRepository
    subject_resolver: SubjectResolver
    nlu_engine: NLUInferenceEngine
    tool_router: ToolRouter
    portal_service: PortalAttendanceService
    timetable_service: TimetableService
    calendar_service: CalendarService
    change_detector: AttendanceChangeDetector
    is_ready: bool = False


_services: Optional[AppServices] = None


def init_app_services() -> AppServices:
    """
    Initialize all persistent backend components once during application startup.
    Loads the SLM model and tokenizer into memory.
    """
    global _services
    if _services is not None and _services.is_ready:
        return _services

    logger.info("Initializing Attendance Tracker Web backend services...")

    repo = get_repository()
    resolver = get_subject_resolver(repository=repo)
    timetable = get_timetable_service()
    calendar = get_calendar_service(repository=repo)
    if calendar.repository is None:
        calendar.repository = repo
    change_detector = get_change_detector()
    portal = get_portal_service()

    # Pre-warm frozen SLM inference engine once
    nlu = get_nlu_engine()
    router = get_tool_router(repository=repo, subject_resolver=resolver, nlu_engine=nlu)

    _services = AppServices(
        repository=repo,
        subject_resolver=resolver,
        nlu_engine=nlu,
        tool_router=router,
        portal_service=portal,
        timetable_service=timetable,
        calendar_service=calendar,
        change_detector=change_detector,
        is_ready=True,
    )

    logger.info("All Attendance Tracker Web services successfully initialized.")
    return _services


def get_app_services() -> AppServices:
    """Retrieve initialized application services container."""
    global _services
    if _services is None or not _services.is_ready:
        return init_app_services()
    return _services
