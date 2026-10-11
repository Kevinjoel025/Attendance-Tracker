"""
tests/test_thresholds.py
────────────────────────
Unit tests for centralized threshold resolution (core/thresholds.py)
and threshold-aware tool routing.
"""

from pathlib import Path
import pytest

from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.core.thresholds import (
    COLLEGE_THRESHOLD,
    PERSONAL_TARGET,
    extract_explicit_percentage,
    resolve_threshold,
)
from attendance_tracker.core.tool_router import ToolRouter


def test_constants() -> None:
    assert COLLEGE_THRESHOLD == 75.0
    assert PERSONAL_TARGET == 70.0


def test_extract_explicit_percentage() -> None:
    assert extract_explicit_percentage("How many ACD classes can I miss at 80%?") == 80.0
    assert extract_explicit_percentage("How many classes to reach 85% in DE?") == 85.0
    assert extract_explicit_percentage("How many classes to reach 70.5% in OS?") == 70.5
    assert extract_explicit_percentage("What if I stay at 80 in OS?") == 80.0
    assert extract_explicit_percentage("How many classes can I miss in ACD?") is None
    assert extract_explicit_percentage("How many classes to reach my target?") is None


def test_resolve_threshold_explicit_percentage() -> None:
    # Explicit percentage in query takes highest precedence
    assert resolve_threshold(query="How many ACD classes can I miss at 80%?") == 80.0
    assert resolve_threshold(query="How many ACD classes can I miss at 70%?") == 70.0
    assert resolve_threshold(query="How many classes to reach 85% in DE?") == 85.0
    assert resolve_threshold(query="How many classes to reach 75% in DE?") == 75.0


def test_resolve_threshold_personal_target_language() -> None:
    # Personal target phrases resolve to 70.0%
    assert resolve_threshold(query="How many ACD classes can I miss at my target?") == 70.0
    assert resolve_threshold(query="How many ACD classes can I miss while staying above my target?") == 70.0
    assert resolve_threshold(query="How many ACD classes can I miss while staying above my personal target?") == 70.0
    assert resolve_threshold(query="How many ACD classes do I need to attend to reach my target?") == 70.0
    assert resolve_threshold(query="How many ACD classes do I need to attend to reach my goal?") == 70.0


def test_resolve_threshold_college_requirement_language() -> None:
    # College requirement phrases resolve to 75.0%
    assert resolve_threshold(query="How many ACD classes can I miss while meeting the college requirement?") == 75.0
    assert resolve_threshold(query="How many ACD classes can I miss for the college requirement?") == 75.0
    assert resolve_threshold(query="How many ACD classes do I need to attend to meet the college requirement?") == 75.0
    assert resolve_threshold(query="How many ACD classes do I need to attend for the college requirement?") == 75.0
    assert resolve_threshold(query="How many classes to reach the minimum attendance in DE?") == 75.0


def test_resolve_threshold_no_target_specified_default() -> None:
    # No target specified defaults to 75.0%
    assert resolve_threshold(query="How many classes can I miss in ACD?", intent="MAX_CLASSES_CAN_MISS") == 75.0
    assert resolve_threshold(query="How many ACD classes can I miss?", intent="MAX_CLASSES_CAN_MISS") == 75.0
    assert resolve_threshold(query="How many classes do I need to attend in DE?", intent="CLASSES_REQUIRED_FOR_TARGET") == 75.0
    assert resolve_threshold(query="", intent="MAX_CLASSES_CAN_MISS") == 75.0


def test_resolve_threshold_slm_extracted_target_filtering() -> None:
    # If SLM extracted 70.0 but user query did not contain "70" or target language, ignore SLM default
    query = "How many classes can I miss in ACD?"
    res = resolve_threshold(query=query, extracted_target=70.0, intent="MAX_CLASSES_CAN_MISS")
    assert res == 75.0

    # If SLM extracted 80.0 and query actually contained "80", accept it
    query_with_80 = "How many classes can I miss at 80% in ACD?"
    res_80 = resolve_threshold(query=query_with_80, extracted_target=80.0, intent="MAX_CLASSES_CAN_MISS")
    assert res_80 == 80.0


@pytest.fixture
def test_router(tmp_path: Path) -> ToolRouter:
    db_file = tmp_path / "test_attendance.db"
    repo = AttendanceRepository(db_path=db_file)
    repo.seed_all()
    resolver = SubjectResolver(repository=repo)
    return ToolRouter(repository=repo, subject_resolver=resolver)


def test_router_max_can_miss_thresholds(test_router: ToolRouter) -> None:
    # 1. Default (no target) -> 75%
    res_default = test_router.route_max_can_miss("ACD", threshold=None, raw_query="How many classes can I miss in ACD?")
    assert res_default.success is True
    assert res_default.data.threshold_percentage == 75.0

    # 2. Explicit 80% -> 80%
    res_80 = test_router.route_max_can_miss("ACD", threshold=80.0, raw_query="How many ACD classes can I miss at 80%?")
    assert res_80.success is True
    assert res_80.data.threshold_percentage == 80.0

    # 3. Explicit 70% -> 70%
    res_70 = test_router.route_max_can_miss("ACD", threshold=70.0, raw_query="How many ACD classes can I miss at 70%?")
    assert res_70.success is True
    assert res_70.data.threshold_percentage == 70.0

    # 4. Personal target language -> 70%
    res_target = test_router.route_max_can_miss("ACD", threshold=None, raw_query="How many ACD classes can I miss at my target?")
    assert res_target.success is True
    assert res_target.data.threshold_percentage == 70.0

    # 5. College requirement language -> 75%
    res_college = test_router.route_max_can_miss("ACD", threshold=None, raw_query="How many ACD classes can I miss for the college requirement?")
    assert res_college.success is True
    assert res_college.data.threshold_percentage == 75.0


def test_router_classes_required_thresholds(test_router: ToolRouter) -> None:
    # 1. Explicit 85% -> 85%
    res_85 = test_router.route_classes_required("ACD", target=85.0, raw_query="How many ACD classes do I need to attend to reach 85%?")
    assert res_85.success is True
    assert res_85.data.target_percentage == 85.0

    # 2. Explicit 75% -> 75%
    res_75 = test_router.route_classes_required("ACD", target=75.0, raw_query="How many ACD classes do I need to attend to reach 75%?")
    assert res_75.success is True
    assert res_75.data.target_percentage == 75.0

    # 3. Personal target language -> 70%
    res_target = test_router.route_classes_required("ACD", target=None, raw_query="How many ACD classes do I need to attend to reach my target?")
    assert res_target.success is True
    assert res_target.data.target_percentage == 70.0

    # 4. College requirement language -> 75%
    res_college = test_router.route_classes_required("ACD", target=None, raw_query="How many ACD classes do I need to attend for the college requirement?")
    assert res_college.success is True
    assert res_college.data.target_percentage == 75.0
