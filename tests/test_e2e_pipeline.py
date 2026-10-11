"""
tests/test_e2e_pipeline.py
──────────────────────────
End-to-end integration tests for the full College Attendance Tracker pipeline:
SLM inference -> NLU parser -> Subject resolver -> Tool router -> Response templates.
"""

from pathlib import Path
import pytest

from attendance_tracker.attendance.engine import classes_required_for_target
from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.core.tool_router import PipelineResponse, ToolRouter
from attendance_tracker.slm.inference.nlu import get_nlu_engine


@pytest.fixture(scope="module")
def shared_router() -> ToolRouter:
    repo = AttendanceRepository()
    repo.seed_all()
    resolver = SubjectResolver(repository=repo)
    nlu_engine = get_nlu_engine()
    return ToolRouter(repository=repo, subject_resolver=resolver, nlu_engine=nlu_engine)


def test_e2e_overall_attendance(shared_router: ToolRouter) -> None:
    query = "What is my overall attendance percentage?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "GET_ATTENDANCE"
    assert resp.tool.success is True

    overall = shared_router.repo.get_all_current_attendance()
    assert overall is not None
    assert f"Overall Attendance: {overall.overall_percentage:.2f}%" in resp.response_text
    assert f"{overall.total_attended} / {overall.total_conducted} classes" in resp.response_text
    assert resp.total_time_ms > 0.0


def test_e2e_subject_attendance(shared_router: ToolRouter) -> None:
    query = "How many classes have I attended in OS?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "GET_SUBJECT_ATTENDANCE"
    assert resp.tool.success is True

    os_att = shared_router.repo.get_attendance_for_subject("OS")
    assert os_att is not None
    assert "Operating Systems" in resp.response_text
    assert f"{os_att.percentage:.2f}%" in resp.response_text
    assert f"{os_att.attended} attended / {os_att.conducted} conducted" in resp.response_text


def test_e2e_attendance_projection(shared_router: ToolRouter) -> None:
    query = "What happens if I attend next 3 classes of IoT?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "ATTENDANCE_PROJECTION"
    assert resp.tool.success is True
    assert "Attendance Projection for IoT Architecture and Protocols" in resp.response_text

    iot_att = shared_router.repo.get_attendance_for_subject("IoT")
    assert iot_att is not None
    proj_att = iot_att.attended + 3
    proj_cond = iot_att.conducted + 3
    proj_pct = (proj_att / proj_cond) * 100.0
    assert f"{proj_att}/{proj_cond}" in resp.response_text
    assert f"{proj_pct:.2f}%" in resp.response_text


def test_e2e_classes_required(shared_router: ToolRouter) -> None:
    query = "How many classes do I need to attend to reach 75% in DE?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert resp.tool.success is True
    assert "Target Goal for Data Engineering" in resp.response_text

    de_att = shared_router.repo.get_attendance_for_subject("DE")
    assert de_att is not None
    calc_res = classes_required_for_target(de_att.attended, de_att.conducted, 75.0)
    assert f"{calc_res.classes_required} consecutive class(es)" in resp.response_text
    assert "75.0%" in resp.response_text


def test_e2e_max_can_miss(shared_router: ToolRouter) -> None:
    query = "How many classes can I miss in ACD?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "MAX_CLASSES_CAN_MISS"
    assert resp.tool.success is True
    assert resp.tool.data.threshold_percentage == 75.0
    assert "Bunk Margin for Automata and Compiler Design" in resp.response_text
    assert "cannot miss any classes" in resp.response_text
    assert "75.0%" in resp.response_text


def test_e2e_max_can_miss_at_80(shared_router: ToolRouter) -> None:
    query = "How many ACD classes can I miss at 80%?"
    resp = shared_router.process(query)

    assert resp.nlu.intent == "MAX_CLASSES_CAN_MISS"
    assert resp.tool.success is True
    assert resp.tool.data.threshold_percentage == 80.0
    assert "80.0%" in resp.response_text


def test_e2e_max_can_miss_at_my_target(shared_router: ToolRouter) -> None:
    query = "How many ACD classes can I miss at my target?"
    resp = shared_router.process(query)

    assert resp.nlu.intent == "MAX_CLASSES_CAN_MISS"
    assert resp.tool.success is True
    assert resp.tool.data.threshold_percentage == 70.0
    assert "70.0%" in resp.response_text


def test_e2e_max_can_miss_college_requirement(shared_router: ToolRouter) -> None:
    query = "How many ACD classes can I miss while meeting the college requirement?"
    resp = shared_router.process(query)

    assert resp.nlu.intent == "MAX_CLASSES_CAN_MISS"
    assert resp.tool.success is True
    assert resp.tool.data.threshold_percentage == 75.0
    assert "75.0%" in resp.response_text


def test_e2e_classes_required_at_85(shared_router: ToolRouter) -> None:
    query = "How many ACD classes do I need to attend to reach 85%?"
    resp = shared_router.process(query)

    assert resp.nlu.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert resp.tool.success is True
    assert resp.tool.data.target_percentage == 85.0
    assert "85.0%" in resp.response_text


def test_e2e_classes_required_my_target(shared_router: ToolRouter) -> None:
    query = "How many ACD classes do I need to attend to reach my target?"
    resp = shared_router.process(query)

    assert resp.nlu.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert resp.tool.success is True
    assert resp.tool.data.target_percentage == 70.0
    assert "70.0%" in resp.response_text


def test_e2e_check_update(shared_router: ToolRouter) -> None:
    query = "Has my attendance been updated today?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "CHECK_ATTENDANCE_UPDATE"
    assert resp.tool.success is True
    assert (
        "Attendance Changes Detected" in resp.response_text
        or "Attendance Update Status" in resp.response_text
    )


def test_e2e_help(shared_router: ToolRouter) -> None:
    query = "What can you do?"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "HELP"
    assert resp.tool.success is True
    assert "Help & Capabilities" in resp.response_text
    assert "Tracked Subjects:" in resp.response_text


def test_e2e_unknown(shared_router: ToolRouter) -> None:
    query = "Tell me a joke"
    resp = shared_router.process(query)

    assert isinstance(resp, PipelineResponse)
    assert resp.nlu.intent == "UNKNOWN"
    assert resp.tool.success is False
    assert "couldn't understand your request" in resp.response_text


def test_pipeline_response_serialization(shared_router: ToolRouter) -> None:
    resp = shared_router.process("What is my attendance in OS?")
    d = resp.to_dict()
    assert isinstance(d, dict)
    assert d["query"] == "What is my attendance in OS?"
    assert d["nlu"]["intent"] == "GET_SUBJECT_ATTENDANCE"
    assert d["tool"]["success"] is True
    assert "response_text" in d
    assert "total_time_ms" in d
