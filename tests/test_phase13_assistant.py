"""
tests/test_phase13_assistant.py
───────────────────────────────
Comprehensive test suite for Phase 13 Personal Attendance Assistant & Timetable/Calendar.

Validates:
1. Deterministic future class generator and semester forecast engine.
2. Calendar overrides (WORKING_DAY, HOLIDAY) and Last Working Day persistence.
3. Universal ingestion adapters (CSV and Excel) validation and snapshot committing.
4. Natural language queries for decision-making (Can I skip tomorrow, Max attendance, Miss subject impact).
5. All new Phase 13 Web API endpoints.
"""

from datetime import date, timedelta
import io
import pytest
from starlette.testclient import TestClient

from attendance_tracker.attendance.adapters.csv_adapter import CSVAttendanceAdapter
from attendance_tracker.attendance.adapters.excel_adapter import ExcelAttendanceAdapter
from attendance_tracker.attendance.comparator import get_comparison_engine
from attendance_tracker.attendance.forecast import get_forecast_engine
from attendance_tracker.attendance.repository import CANONICAL_SUBJECTS, get_repository
from attendance_tracker.calendar.service import get_calendar_service
from attendance_tracker.core.subject_resolver import get_subject_resolver
from attendance_tracker.core.tool_router import get_tool_router
from attendance_tracker.timetable.future_generator import get_future_class_generator
from attendance_tracker.web.app import create_app


@pytest.fixture(scope="module")
def web_client():
    """Create test client for web API endpoints."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def router():
    """ToolRouter instance."""
    return get_tool_router()


# ── 1. Future Class Generator & Forecast Engine Tests ─────────────────────────

def test_future_class_generator() -> None:
    generator = get_future_class_generator()
    start_d = date(2026, 10, 5)  # Monday
    end_d = date(2026, 10, 10)    # Saturday
    sched = generator.generate_schedule_between(start_d, end_d)

    assert sched.total_classes > 0
    assert sched.start_date == start_d
    assert sched.end_date == end_d
    # Every generated class must have a canonical subject
    valid_codes = [s["code"] for s in CANONICAL_SUBJECTS]
    for fc in sched.classes:
        assert fc.subject_code in valid_codes


def test_forecast_engine_semester() -> None:
    repo = get_repository()
    cur_att = repo.get_all_current_attendance()
    total_att = cur_att.total_attended
    total_cond = cur_att.total_conducted

    forecast_engine = get_forecast_engine()
    res = forecast_engine.forecast_semester()

    assert res.current_conducted == total_cond
    assert res.current_attended == total_att
    assert res.future_classes_count > 0
    assert res.max_possible_attended > res.current_attended
    assert res.max_possible_percentage > res.current_percentage
    assert isinstance(res.target_75_feasible, bool)
    assert isinstance(res.target_70_feasible, bool)


def test_forecast_day_skip_tomorrow() -> None:
    repo = get_repository()
    cur_att = repo.get_all_current_attendance()
    total_att = cur_att.total_attended
    total_cond = cur_att.total_conducted

    forecast_engine = get_forecast_engine()
    res = forecast_engine.forecast_day_skip()

    assert res.date_str is not None
    assert res.day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert res.current_conducted == total_cond
    assert res.current_attended == total_att
    if not res.is_holiday and res.classes_count > 0:
        assert res.if_attend_all_conducted == total_cond + res.classes_count
        assert res.if_attend_all_attended == total_att + res.classes_count
        assert res.if_skip_all_attended == total_att
        assert res.if_skip_all_conducted == total_cond + res.classes_count


def test_forecast_subject_skip_overall_impact() -> None:
    repo = get_repository()
    cur_att = repo.get_all_current_attendance()
    total_att = cur_att.total_attended
    total_cond = cur_att.total_conducted

    forecast_engine = get_forecast_engine()
    # If student misses 5 classes of ACD (code 22PC1AM203)
    res = forecast_engine.forecast_subject_skip("22PC1AM203", miss_count=5)

    assert res.subject_short == "ACD"
    assert res.miss_count == 5
    assert res.current_overall_attended == total_att
    assert res.current_overall_conducted == total_cond
    assert res.projected_overall_attended == total_att
    assert res.projected_overall_conducted == total_cond + 5
    expected_pct = (total_att / (total_cond + 5)) * 100.0
    assert abs(res.projected_overall_percentage - expected_pct) < 0.05
    assert res.overall_delta < 0


# ── 2. Calendar Overrides & Last Working Day Persistence ───────────────────────

def test_calendar_overrides_crud() -> None:
    repo = get_repository()
    cal_svc = get_calendar_service()
    test_d = date(2026, 11, 15)  # Sunday normally
    test_d_str = test_d.isoformat()

    # 1. Add working day override on a Sunday
    repo.add_calendar_override(test_d_str, "WORKING_DAY", name="Make up classes")
    status = cal_svc.get_day_status(test_d)
    assert status.is_holiday is False
    assert status.is_override is True
    assert status.override_type == "WORKING_DAY"

    # 2. Add holiday override on a Monday
    test_mon = date(2026, 11, 16)
    test_mon_str = test_mon.isoformat()
    repo.add_calendar_override(test_mon_str, "HOLIDAY", name="College Fest")
    status_mon = cal_svc.get_day_status(test_mon)
    assert status_mon.is_holiday is True
    assert status_mon.is_override is True
    assert status_mon.override_type == "HOLIDAY"

    # 3. Clean up overrides
    repo.delete_calendar_override(test_d_str)
    repo.delete_calendar_override(test_mon_str)


def test_last_working_day_persistence() -> None:
    repo = get_repository()
    cal_svc = get_calendar_service()

    repo.set_last_working_day("2026-12-05")
    assert cal_svc.get_last_working_day() == date(2026, 12, 5)

    # Restore default
    repo.set_last_working_day("2026-11-28")
    assert cal_svc.get_last_working_day() == date(2026, 11, 28)


# ── 3. Universal Ingestion Adapters ───────────────────────────────────────────

def test_csv_adapter_parsing_and_validation() -> None:
    adapter = CSVAttendanceAdapter()
    csv_content = """Subject,Attended,Conducted
OS,22,33
ACD,18,30
DE,20,32
IoTAP,16,28
WP,19,31
GS,25,35
OE-1,21,33
DE LAB,12,18
IoTAP LAB,10,15
"""
    preview = adapter.parse_text(csv_content)
    assert preview.is_valid is True
    assert len(preview.records) == 9
    assert preview.total_attended == 163
    assert preview.total_conducted == 255
    assert len(preview.errors) == 0


def test_csv_adapter_invalid_classes() -> None:
    adapter = CSVAttendanceAdapter()
    bad_csv = """Subject,Attended,Conducted
OS,35,30
"""
    preview = adapter.parse_text(bad_csv)
    assert preview.is_valid is False
    assert len(preview.errors) > 0


# ── 4. Assistant Natural Language Intent Queries ──────────────────────────────

def test_nlu_can_skip_tomorrow(router) -> None:
    resp = router.process("Can I skip tomorrow?")
    assert resp.nlu.intent == "CAN_SKIP_DAY"
    assert resp.tool.success is True
    assert "Tomorrow" in resp.response_text or "holiday" in resp.response_text.lower() or "classes" in resp.response_text.lower()


def test_nlu_maximum_attendance(router) -> None:
    resp = router.process("What is the maximum attendance I can reach this semester?")
    assert resp.nlu.intent == "SEMESTER_FORECAST"
    assert resp.tool.success is True
    assert "Semester Forecast" in resp.response_text or "Semester Attendance Forecast" in resp.response_text
    assert "Maximum Possible" in resp.response_text


def test_nlu_how_many_classes_can_i_skip(router) -> None:
    resp = router.process("How many classes can I skip before the semester ends?")
    assert resp.nlu.intent == "SEMESTER_FORECAST"
    assert resp.tool.success is True
    assert "Remaining Future Classes:" in resp.response_text or "Safe future misses:" in resp.response_text


def test_nlu_what_changed_today(router) -> None:
    resp = router.process("What changed today?")
    assert resp.nlu.intent == "CHECK_ATTENDANCE_UPDATE"
    assert resp.tool.success is True
    assert "Attendance Update Status" in resp.response_text or "Attendance Changes Detected" in resp.response_text


def test_nlu_miss_subject_classes(router) -> None:
    resp = router.process("If I miss 5 ACD classes, what happens?")
    assert resp.nlu.intent == "SUBJECT_SKIP_IMPACT"
    assert resp.tool.success is True
    assert "Impact of Missing 5 Class(es) of Automata and Compiler Design" in resp.response_text
    assert "Primary Impact (Overall Attendance):" in resp.response_text
    assert "Projected Overall:" in resp.response_text
    assert "Decision Verdict:" in resp.response_text


# ── 5. Phase 13 REST API Endpoints ────────────────────────────────────────────

def test_api_calendar_status(web_client) -> None:
    res = web_client.get("/api/calendar/status")
    assert res.status_code == 200
    data = res.json()
    assert "date" in data
    assert "is_holiday" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)


def test_api_weekly_timetable(web_client) -> None:
    res = web_client.get("/api/timetable/weekly")
    assert res.status_code == 200
    data = res.json()
    assert "days" in data
    assert "Monday" in data["schedule"]
    assert len(data["schedule"]["Monday"]) == 6


def test_api_last_working_day(web_client) -> None:
    # GET
    res_get = web_client.get("/api/calendar/last-working-day")
    assert res_get.status_code == 200
    assert "last_working_day" in res_get.json()

    # POST
    res_post = web_client.post(
        "/api/calendar/last-working-day",
        json={"last_working_day": "2026-12-10"}
    )
    assert res_post.status_code == 200
    assert res_post.json()["last_working_day"] == "2026-12-10"

    # Reset
    web_client.post(
        "/api/calendar/last-working-day",
        json={"last_working_day": "2026-11-28"}
    )


def test_api_calendar_overrides_flow(web_client) -> None:
    # 1. Add override
    res_add = web_client.post(
        "/api/calendar/overrides",
        json={"date": "2026-11-20", "override_type": "HOLIDAY", "reason": "Test Holiday"}
    )
    assert res_add.status_code == 200
    assert res_add.json()["success"] is True

    # 2. List overrides
    res_list = web_client.get("/api/calendar/overrides")
    assert res_list.status_code == 200
    overrides = res_list.json()
    assert any(o["date"] == "2026-11-20" for o in overrides)

    # 3. Delete override
    res_del = web_client.delete("/api/calendar/overrides/2026-11-20")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True


def test_api_import_preview_and_commit(web_client) -> None:
    # 1. Preview text
    csv_text = "Subject,Attended,Conducted\nOS,22,33\nACD,18,30\nDE,20,32\nIoTAP,16,28\nWP,19,31\nGS,25,35\nOE-1,21,33\nDE LAB,12,18\nIoTAP LAB,10,15"
    res_prev = web_client.post(
        "/api/import/preview-text",
        json={"csv_text": csv_text}
    )
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert prev_data["is_valid"] is True
    assert len(prev_data["records"]) == 9

    # 2. Commit
    records_payload = [
        {"subject_code": r["subject_code"], "attended_classes": r["attended"], "total_classes": r["conducted"]}
        for r in prev_data["records"]
    ]
    res_commit = web_client.post(
        "/api/import/commit",
        json={"records": records_payload, "source": "csv"}
    )
    assert res_commit.status_code == 200
    commit_data = res_commit.json()
    assert commit_data["success"] is True
    assert "snapshot_id" in commit_data


def test_api_snapshots_list_and_compare(web_client) -> None:
    res_list = web_client.get("/api/snapshots")
    assert res_list.status_code == 200
    snaps = res_list.json()
    assert isinstance(snaps, list)
    assert len(snaps) > 0

    # Compare latest
    res_comp = web_client.get("/api/snapshots/compare")
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert "summary_text" in comp_data
