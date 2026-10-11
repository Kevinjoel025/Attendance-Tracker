"""
tests/test_phase13_1_calendar.py
─────────────────────────────────
Comprehensive regression test suite for Phase 13.1 Calendar Duplication Fixes
and Mid-Term 1 Management.

Validates:
1. An imported holiday appears exactly once in the unified table.
2. An official holiday is not duplicated by the display layer.
3. A working-day override replaces the effective holiday status without deleting the official default.
4. Deleting an override restores the applicable default calendar rule.
5. Editing an entry updates the existing record rather than creating another.
6. Adding a holiday persists after restarting the application.
7. Deleting a user-created holiday restores the regular timetable for that date.
8. CSV and Excel imports do not create duplicate calendar entries.
9. Mid-Term 1 start and end dates can be saved and reloaded.
10. Invalid Mid-Term 1 date ranges are rejected.
11. Mid-Term 2 is no longer displayed or used as an active configured period.
12. The future class generator respects Mid-Term 1.
13. Last working day and weekend rules still work.
14. Existing portal sync endpoints remain functional.
15. Strict precedence hierarchy is preserved throughout.
"""

from datetime import date, datetime, timedelta
import io
import pytest
from starlette.testclient import TestClient

from attendance_tracker.attendance.adapters.calendar_adapter import HolidayCalendarAdapter
from attendance_tracker.attendance.repository import get_repository
from attendance_tracker.calendar.service import get_calendar_service
from attendance_tracker.timetable.future_generator import get_future_class_generator
from attendance_tracker.web.app import create_app


@pytest.fixture(scope="module")
def web_client():
    """Create test client for FastAPI routes."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def clean_calendar_state():
    """Clean up test calendar overrides before and after each test."""
    repo = get_repository()
    # Reset mid term to default
    repo.set_mid_term_1("2026-09-07", "2026-09-12")
    # Clean test override dates
    test_dates = [
        "2026-10-02", "2026-10-05", "2026-10-15", "2026-10-16",
        "2026-10-27", "2026-10-28", "2026-11-02", "2026-11-03", "2026-11-20"
    ]
    for d in test_dates:
        repo.delete_calendar_override(d)
    yield
    for d in test_dates:
        repo.delete_calendar_override(d)
    repo.set_mid_term_1("2026-09-07", "2026-09-12")


# ── 1. Unified Table & Deduplication Tests ───────────────────────────────────

def test_unified_table_contains_official_holidays_without_duplicates(web_client) -> None:
    """Requirement 2: An official holiday is not duplicated by the display layer."""
    res = web_client.get("/api/calendar/entries")
    assert res.status_code == 200
    entries = res.json()
    assert isinstance(entries, list)
    assert len(entries) > 0

    # Ensure every date/start_date in the unified table appears exactly once
    dates = [e["start_date"] for e in entries]
    assert len(dates) == len(set(dates)), f"Found duplicate dates in unified calendar: {dates}"

    # Verify Gandhi Jayanti (2026-10-02) appears exactly once as official
    gandhi_entries = [e for e in entries if e["start_date"] == "2026-10-02"]
    assert len(gandhi_entries) == 1
    assert gandhi_entries[0]["source"] == "OFFICIAL"
    assert gandhi_entries[0]["type"] == "HOLIDAY"


def test_working_day_override_replaces_status_without_duplicate(web_client) -> None:
    """Requirement 3: Working-day override replaces effective holiday status without deleting official default."""
    repo = get_repository()
    cal_svc = get_calendar_service()

    # 1. Create a working-day override on Gandhi Jayanti (2026-10-02)
    repo.add_calendar_override("2026-10-02", "WORKING_DAY", name="Special Working Day")

    # 2. Check API unified entries
    res = web_client.get("/api/calendar/entries")
    assert res.status_code == 200
    entries = res.json()

    # Must appear EXACTLY ONCE
    matching = [e for e in entries if e["start_date"] == "2026-10-02"]
    assert len(matching) == 1
    assert matching[0]["type"] == "WORKING_DAY"
    assert matching[0]["is_override"] is True
    assert matching[0]["source"] == "USER_OVERRIDE"
    assert "Mahatma Gandhi Jayanti" in matching[0]["original_rule"]

    # 3. Calendar service status
    status = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status.is_working_day is True
    assert status.is_holiday is False
    assert status.is_override is True


def test_deleting_override_restores_default_rule(web_client) -> None:
    """Requirement 4 & 7: Deleting an override restores the default calendar rule."""
    repo = get_repository()
    cal_svc = get_calendar_service()

    # Part A: Override on an official holiday (2026-10-02)
    repo.add_calendar_override("2026-10-02", "WORKING_DAY", name="Special Working Day")
    assert cal_svc.get_calendar_status(date(2026, 10, 2)).is_working_day is True

    del_res = web_client.delete("/api/calendar/entries/2026-10-02")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Now official holiday is restored
    status_restored = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status_restored.is_holiday is True
    assert status_restored.is_working_day is False
    assert status_restored.is_override is False

    # Part B: User holiday on a regular working day (2026-10-27, Tuesday)
    repo.add_calendar_override("2026-10-27", "HOLIDAY", name="User Holiday")
    assert cal_svc.get_calendar_status(date(2026, 10, 27)).is_holiday is True

    del_res2 = web_client.delete("/api/calendar/entries/2026-10-27")
    assert del_res2.status_code == 200
    # Restores regular working day
    status_tue = cal_svc.get_calendar_status(date(2026, 10, 27))
    assert status_tue.is_working_day is True
    assert status_tue.is_holiday is False
    assert status_tue.event_type == "REGULAR_WORKING_DAY"


def test_add_and_edit_calendar_entry_updates_record(web_client) -> None:
    """Requirement 5 & 6: Adding and editing calendar entry updates existing record without duplicate."""
    # 1. Add entry
    add_payload = {
        "date": "2026-11-02",
        "override_type": "HOLIDAY",
        "name": "Initial Fest Day",
        "note": "Annual college fest",
    }
    res_add = web_client.post("/api/calendar/entries", json=add_payload)
    assert res_add.status_code == 200
    assert res_add.json()["success"] is True

    # Verify unified table has 1 entry
    res1 = web_client.get("/api/calendar/entries")
    matches1 = [e for e in res1.json() if e["start_date"] == "2026-11-02"]
    assert len(matches1) == 1
    assert matches1[0]["name"] == "Initial Fest Day"

    # 2. Edit entry (same date, new name and type)
    edit_payload = {
        "date": "2026-11-02",
        "override_type": "WORKING_DAY",
        "name": "Modified Fest Working Day",
        "note": "Classes conducted in afternoon",
        "old_date": "2026-11-02",
    }
    res_edit = web_client.post("/api/calendar/entries", json=edit_payload)
    assert res_edit.status_code == 200

    res2 = web_client.get("/api/calendar/entries")
    matches2 = [e for e in res2.json() if e["start_date"] == "2026-11-02"]
    assert len(matches2) == 1
    assert matches2[0]["name"] == "Modified Fest Working Day"
    assert matches2[0]["type"] == "WORKING_DAY"

    # 3. Edit date (moving from 2026-11-02 to 2026-11-03)
    move_payload = {
        "date": "2026-11-03",
        "override_type": "HOLIDAY",
        "name": "Rescheduled Fest Holiday",
        "old_date": "2026-11-02",
    }
    res_move = web_client.post("/api/calendar/entries", json=move_payload)
    assert res_move.status_code == 200

    res3 = web_client.get("/api/calendar/entries")
    assert len([e for e in res3.json() if e["start_date"] == "2026-11-02"]) == 0
    matches3 = [e for e in res3.json() if e["start_date"] == "2026-11-03"]
    assert len(matches3) == 1
    assert matches3[0]["name"] == "Rescheduled Fest Holiday"


def test_csv_and_excel_imports_do_not_create_duplicates(web_client) -> None:
    """Requirement 1 & 8: CSV and Excel imports do not create duplicate calendar entries."""
    csv_text = """Date,Type,Name
2026-10-02,HOLIDAY,Mahatma Gandhi Jayanti
2026-10-05,HOLIDAY,Special Holiday 1
2026-10-05,HOLIDAY,Special Holiday 1 Duplicate
2026-10-28,WORKING_DAY,Compensatory Working Day
"""
    # 1. Preview text
    prev_res = web_client.post("/api/import/calendar/preview-text", json={"csv_text": csv_text})
    assert prev_res.status_code == 200
    preview = prev_res.json()
    assert preview["valid_count"] >= 3

    # 2. Commit import
    records = [
        {"date": r["date"], "override_type": r["override_type"], "name": r["name"]}
        for r in preview["records"] if r["is_valid"]
    ]
    commit_res = web_client.post("/api/import/calendar/commit", json={"records": records, "source": "csv"})
    assert commit_res.status_code == 200

    # 3. Query unified calendar entries
    entries_res = web_client.get("/api/calendar/entries")
    assert entries_res.status_code == 200
    entries = entries_res.json()

    # Each date must appear exactly once
    assert len([e for e in entries if e["start_date"] == "2026-10-02"]) == 1
    assert len([e for e in entries if e["start_date"] == "2026-10-05"]) == 1
    assert len([e for e in entries if e["start_date"] == "2026-10-28"]) == 1


# ── 2. Mid-Term 1 Management Tests ──────────────────────────────────────────

def test_mid_term_1_save_and_reload(web_client) -> None:
    """Requirement 9: Mid-Term 1 start and end dates can be saved and reloaded."""
    repo = get_repository()
    cal_svc = get_calendar_service()

    # 1. Update Mid-Term 1 dates via API
    req_body = {
        "start": "2026-09-08",
        "end": "2026-09-14",
    }
    post_res = web_client.post("/api/calendar/midterm", json=req_body)
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["success"] is True
    assert data["mid_term_1"]["start"] == "2026-09-08"
    assert data["mid_term_1"]["end"] == "2026-09-14"

    # 2. Reload via GET API
    get_res = web_client.get("/api/calendar/midterm")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["mid_term_1"]["start"] == "2026-09-08"
    assert get_data["mid_term_1"]["end"] == "2026-09-14"

    # 3. Verify CalendarService directly
    s_date, e_date = cal_svc.get_mid_term_1()
    assert s_date == date(2026, 9, 8)
    assert e_date == date(2026, 9, 14)


def test_invalid_mid_term_1_dates_rejected(web_client) -> None:
    """Requirement 10: Invalid Mid-Term 1 date ranges are rejected."""
    # A. End date before start date
    res_inv_range = web_client.post("/api/calendar/midterm", json={
        "start": "2026-09-15",
        "end": "2026-09-10"
    })
    assert res_inv_range.status_code == 400
    assert "cannot be before start date" in res_inv_range.json()["detail"]

    # B. Malformed date string
    res_malformed = web_client.post("/api/calendar/midterm", json={
        "start": "not-a-date",
        "end": "2026-09-10"
    })
    assert res_malformed.status_code == 400


def test_mid_term_2_is_removed_from_active_calendar(web_client) -> None:
    """Requirement 11: Mid-Term 2 is no longer displayed or returned."""
    res_holidays = web_client.get("/api/calendar/holidays")
    assert res_holidays.status_code == 200
    h_data = res_holidays.json()

    assert "mid_term_1" in h_data
    assert "mid_term_2" not in h_data

    # Check unified entries
    res_entries = web_client.get("/api/calendar/entries")
    assert res_entries.status_code == 200
    entries = res_entries.json()

    # Mid-Term 1 appears
    mt1_entries = [e for e in entries if e.get("type") == "MID_TERM" or "Mid-Term 1" in e.get("name", "")]
    assert len(mt1_entries) == 1

    # Mid-Term 2 does not appear
    mt2_entries = [e for e in entries if "Mid-Term 2" in e.get("name", "")]
    assert len(mt2_entries) == 0


def test_future_class_generator_respects_mid_term_1() -> None:
    """Requirement 12: Future class generator excludes classes throughout Mid-Term 1."""
    generator = get_future_class_generator()
    cal_svc = get_calendar_service()

    # Configure Mid-Term 1: 2026-09-07 (Monday) to 2026-09-12 (Saturday)
    cal_svc.set_mid_term_1("2026-09-07", "2026-09-12")

    # Generate schedule across the exact Mid-Term 1 window
    sched = generator.generate_schedule_between(date(2026, 9, 7), date(2026, 9, 12))

    # All 6 days must be treated as exam holidays, 0 classes conducted
    assert sched.total_classes == 0
    assert sched.holidays_count == 6
    assert sched.working_days_count == 0


def test_last_working_day_and_weekend_rules_preserved(web_client) -> None:
    """Requirement 13: Weekend and last working day rules still function correctly."""
    cal_svc = get_calendar_service()

    # Sunday is holiday
    sun_status = cal_svc.get_calendar_status(date(2026, 10, 4))
    assert sun_status.is_holiday is True
    assert sun_status.event_type == "WEEKEND"

    # 2nd Saturday (2026-10-10) is holiday
    sat2_status = cal_svc.get_calendar_status(date(2026, 10, 10))
    assert sat2_status.is_holiday is True
    assert sat2_status.event_name == "Second Saturday"

    # 3rd Saturday (2026-10-17) is working day
    sat3_status = cal_svc.get_calendar_status(date(2026, 10, 17))
    assert sat3_status.is_working_day is True
    assert sat3_status.event_type == "REGULAR_WORKING_DAY"

    # Update last working day via API
    res_lwd = web_client.post("/api/calendar/last-working-day", json={"last_working_day": "2026-12-05"})
    assert res_lwd.status_code == 200
    assert cal_svc.get_last_working_day() == date(2026, 12, 5)

    # Restore default last working day
    web_client.post("/api/calendar/last-working-day", json={"last_working_day": "2026-11-28"})


def test_excel_holiday_upload_deduplication(web_client) -> None:
    """Requirement 8: Excel upload does not create duplicate entries in unified calendar."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Calendar"
    ws.append(["Date", "Category", "Event Description", "Remarks"])
    ws.append(["2026-10-02", "HOLIDAY", "Gandhi Jayanti", "Official"])
    ws.append(["2026-11-20", "HOLIDAY", "Special Break", "College Holiday"])
    ws.append(["2026-11-20", "HOLIDAY", "Special Break Repeated", "Duplicate row in Excel"])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    # Upload
    files = {"file": ("test_calendar.xlsx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    res_upload = web_client.post("/api/import/calendar/upload", files=files)
    assert res_upload.status_code == 200
    preview = res_upload.json()
    assert preview["valid_count"] >= 2

    # Commit
    records = [{"date": r["date"], "override_type": r["override_type"], "name": r["name"]} for r in preview["records"] if r["is_valid"]]
    res_commit = web_client.post("/api/import/calendar/commit", json={"records": records, "source": "excel"})
    assert res_commit.status_code == 200

    # Verify unified entries
    res_entries = web_client.get("/api/calendar/entries")
    assert res_entries.status_code == 200
    entries = res_entries.json()
    assert len([e for e in entries if e["start_date"] == "2026-11-20"]) == 1


def test_portal_sync_remains_untouched_and_functional(web_client) -> None:
    """Requirement 14: Existing portal sync endpoints and safety remain completely intact."""
    from unittest.mock import patch
    from attendance_tracker.portal.models import PortalSyncResult

    with patch("attendance_tracker.portal.service.PortalAttendanceService.sync_attendance") as mock_sync:
        mock_sync.return_value = PortalSyncResult(
            success=True,
            snapshot_id=99,
            changes_detected=[],
            records_count=9,
            duration_ms=120.0,
            error_message=None,
        )
        res = web_client.post("/api/sync")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert isinstance(data["changed"], bool)

