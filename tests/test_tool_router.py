"""
tests/test_tool_router.py
─────────────────────────
Unit tests for ToolRouter (core/tool_router.py).
"""

from pathlib import Path
import pytest

from attendance_tracker.attendance.engine import (
    ClassesRequiredResult,
    MaxCanMissResult,
    ProjectionResult,
)
from attendance_tracker.attendance.repository import (
    AttendanceRepository,
    OverallAttendance,
    SubjectAttendance,
)
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.core.tool_router import ToolResult, ToolRouter
from attendance_tracker.slm.inference.nlu import NLUResult


@pytest.fixture
def repo(tmp_path: Path) -> AttendanceRepository:
    db_file = tmp_path / "test_attendance.db"
    r = AttendanceRepository(db_path=db_file)
    r.seed_all()
    return r


@pytest.fixture
def router(repo: AttendanceRepository) -> ToolRouter:
    resolver = SubjectResolver(repository=repo)
    return ToolRouter(repository=repo, subject_resolver=resolver)


def test_route_get_attendance(router: ToolRouter) -> None:
    res = router.route_get_attendance()
    assert res.success is True
    assert res.intent == "GET_ATTENDANCE"
    assert isinstance(res.data, OverallAttendance)
    assert res.data.total_attended == 119
    assert res.data.total_conducted == 236


def test_route_get_subject_attendance(router: ToolRouter) -> None:
    res = router.route_get_subject_attendance("OS")
    assert res.success is True
    assert res.intent == "GET_SUBJECT_ATTENDANCE"
    assert isinstance(res.data, SubjectAttendance)
    assert res.data.subject.short_name == "OS"
    assert res.data.attended == 17
    assert res.data.conducted == 28


def test_route_get_subject_attendance_missing_subject(router: ToolRouter) -> None:
    res = router.route_get_subject_attendance(None)
    assert res.success is False
    assert res.error_type == "MISSING_SUBJECT"


def test_route_get_subject_attendance_unknown_subject(router: ToolRouter) -> None:
    res = router.route_get_subject_attendance("NonExistent123")
    assert res.success is False
    assert res.error_type == "UNKNOWN_SUBJECT"


def test_route_projection_attend(router: ToolRouter) -> None:
    res = router.route_projection("IoTAP", count=3, raw_query="attend next 3 classes of IoT")
    assert res.success is True
    assert res.intent == "ATTENDANCE_PROJECTION"
    assert isinstance(res.data, ProjectionResult)
    assert res.data.action == "ATTEND"
    assert res.data.count == 3
    # IoT initial seed: 15/27 -> attend 3 -> 18/30 (60.0%)
    assert res.data.projected_attended == 18
    assert res.data.projected_conducted == 30
    assert res.data.projected_percentage == 60.0


def test_route_projection_miss(router: ToolRouter) -> None:
    res = router.route_projection("OS", count=2, raw_query="miss next 2 classes of OS")
    assert res.success is True
    assert res.intent == "ATTENDANCE_PROJECTION"
    assert isinstance(res.data, ProjectionResult)
    assert res.data.action == "MISS"
    assert res.data.count == 2
    # OS initial seed: 17/28 -> miss 2 -> 17/30 (56.67%)
    assert res.data.projected_attended == 17
    assert res.data.projected_conducted == 30


def test_route_classes_required(router: ToolRouter) -> None:
    res = router.route_classes_required("DE", target=75.0)
    assert res.success is True
    assert res.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert isinstance(res.data, ClassesRequiredResult)
    # DE: 11/28 -> target 75% -> ceil((75*28 - 100*11)/(100-75)) = ceil((2100-1100)/25) = ceil(1000/25) = 40
    assert res.data.classes_required == 40


def test_route_max_can_miss(router: ToolRouter) -> None:
    res = router.route_max_can_miss("ACD", threshold=75.0)
    assert res.success is True
    assert res.intent == "MAX_CLASSES_CAN_MISS"
    assert isinstance(res.data, MaxCanMissResult)
    # ACD: 21/33 (63.64%) -> below 75% -> max_can_miss = 0, already_below_threshold = True
    assert res.data.max_can_miss == 0
    assert res.data.already_below_threshold is True


def test_route_check_attendance_update(router: ToolRouter) -> None:
    res = router.route_check_attendance_update()
    assert res.success is True
    assert res.intent == "CHECK_ATTENDANCE_UPDATE"


def test_route_help(router: ToolRouter) -> None:
    res = router.route_help()
    assert res.success is True
    assert res.intent == "HELP"
    assert len(res.data) == 9


def test_route_unknown(router: ToolRouter) -> None:
    res = router.route_unknown("What is the meaning of life?")
    assert res.success is False
    assert res.intent == "UNKNOWN"
    assert res.error_type == "UNKNOWN_INTENT"


def test_execute_dispatcher(router: ToolRouter) -> None:
    nlu = NLUResult(
        raw_output="GET_ATTENDANCE|NONE|NONE|NONE|NONE",
        intent="GET_ATTENDANCE",
        subject=None,
        target=None,
        count=None,
        date=None,
        is_valid=True,
    )
    res = router.execute(nlu)
    assert res.success is True
    assert res.intent == "GET_ATTENDANCE"
