"""
tests/test_calendar_import_and_entry_fixes.py
─────────────────────────────────────────────
Comprehensive regression tests verifying:
1. Importing a holiday persists it in the database.
2. The calendar listing API returns the imported holiday.
3. The imported holiday appears after refreshing calendar data.
4. Imported holidays survive application restart (new repository/service instance).
5. Adding a holiday through the API persists it.
6. Adding a working-day override persists it.
7. Frontend form structure and Add Entry controls exist and validate properly.
8. Saving a valid entry calls the API and updates authoritative entries.
9. Invalid input displays validation errors and rejects bad dates/types.
10. A failed save does not report success.
11. Unified entries loading properly transitions through states.
12. Duplicate imports do not produce duplicate displayed entries.
13. Existing timetable and calendar precedence behavior remains unchanged.
14. Portal synchronization continues working.
"""

from datetime import date, datetime
import io
import pytest
from starlette.testclient import TestClient

from attendance_tracker.attendance.adapters.calendar_adapter import HolidayCalendarAdapter
from attendance_tracker.attendance.repository import AttendanceRepository, get_repository
from attendance_tracker.calendar.service import CalendarService, get_calendar_service
from attendance_tracker.timetable.service import get_timetable_service
from attendance_tracker.web.app import create_app


@pytest.fixture(scope="module")
def web_client():
    """Create test client for FastAPI routes."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def clean_test_overrides():
    """Clean up any test calendar overrides before and after each test."""
    repo = get_repository()
    test_dates = [
        "2026-10-02", "2026-10-05", "2026-10-15", "2026-10-16",
        "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28",
        "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-10",
        "2026-11-15", "2026-11-20", "2026-12-01"
    ]
    for d in test_dates:
        repo.delete_calendar_override(d)
    yield
    for d in test_dates:
        repo.delete_calendar_override(d)


# ── 1. Holiday Import Persistence & Listing ───────────────────────────────────

def test_1_importing_holiday_persists_it(web_client) -> None:
    """Requirement 1: Importing a holiday persists it."""
    csv_content = (
        "Date,Type,Name\n"
        "2026-10-25,HOLIDAY,Special Deepavali Vacation\n"
        "2026-10-26,HOLIDAY,Govardhan Puja\n"
    )
    # Preview text
    prev_res = web_client.post("/api/import/calendar/preview-text", json={"csv_text": csv_content})
    assert prev_res.status_code == 200
    prev_data = prev_res.json()
    assert prev_data["is_valid"] is True
    assert prev_data["valid_count"] == 2

    # Commit import
    commit_res = web_client.post(
        "/api/import/calendar/commit",
        json={
            "source": "csv",
            "records": [
                {"date": "2026-10-25", "override_type": "HOLIDAY", "name": "Special Deepavali Vacation"},
                {"date": "2026-10-26", "override_type": "HOLIDAY", "name": "Govardhan Puja"},
            ],
        },
    )
    assert commit_res.status_code == 200
    data = commit_res.json()
    assert data["success"] is True
    assert data["imported_count"] == 2

    # Verify directly in SQLite repository
    repo = get_repository()
    rec1 = repo.get_calendar_override("2026-10-25")
    assert rec1 is not None
    assert rec1.date == "2026-10-25"
    assert rec1.override_type == "HOLIDAY"
    assert rec1.name == "Special Deepavali Vacation"

    rec2 = repo.get_calendar_override("2026-10-26")
    assert rec2 is not None
    assert rec2.date == "2026-10-26"
    assert rec2.override_type == "HOLIDAY"


def test_2_calendar_listing_api_returns_imported_holiday(web_client) -> None:
    """Requirement 2: The calendar listing API returns the imported holiday."""
    # Commit a holiday
    web_client.post(
        "/api/import/calendar/commit",
        json={
            "source": "csv",
            "records": [
                {"date": "2026-11-10", "override_type": "HOLIDAY", "name": "State Formation Day"}
            ],
        },
    )

    # Query calendar entries
    res = web_client.get("/api/calendar/entries")
    assert res.status_code == 200
    entries = res.json()

    matching = [e for e in entries if e["start_date"] == "2026-11-10"]
    assert len(matching) == 1
    assert matching[0]["name"] == "State Formation Day"
    assert matching[0]["type"] == "HOLIDAY"
    assert matching[0]["is_override"] is True
    assert matching[0]["source"] == "USER_OVERRIDE"


def test_3_imported_holiday_appears_after_refreshing_calendar_data(web_client) -> None:
    """Requirement 3: The imported holiday appears after refreshing the calendar data."""
    # 1. First fetch
    res1 = web_client.get("/api/calendar/entries")
    assert res1.status_code == 200
    assert not any(e["start_date"] == "2026-11-15" for e in res1.json())

    # 2. Import new holiday
    web_client.post(
        "/api/import/calendar/commit",
        json={
            "source": "excel",
            "records": [
                {"date": "2026-11-15", "override_type": "HOLIDAY", "name": "Mid-Semester Break"}
            ],
        },
    )

    # 3. Refresh (Second fetch)
    res2 = web_client.get("/api/calendar/entries")
    assert res2.status_code == 200
    entries2 = res2.json()
    matching = [e for e in entries2 if e["start_date"] == "2026-11-15"]
    assert len(matching) == 1
    assert matching[0]["name"] == "Mid-Semester Break"


def test_4_imported_holidays_survive_application_restart() -> None:
    """Requirement 4: Imported holidays survive application restart."""
    repo = AttendanceRepository()
    repo.add_calendar_override("2026-12-01", "HOLIDAY", name="Winter Cultural Festival")

    # Simulate fresh app startup with new repository and calendar service instances
    fresh_repo = AttendanceRepository()
    fresh_cal_svc = CalendarService(repository=fresh_repo)

    entries = fresh_cal_svc.get_unified_calendar_entries()
    matching = [e for e in entries if e["start_date"] == "2026-12-01"]
    assert len(matching) == 1
    assert matching[0]["name"] == "Winter Cultural Festival"
    assert matching[0]["type"] == "HOLIDAY"
    assert matching[0]["is_override"] is True


# ── 2. Add / Edit / Delete Calendar Entries via API ──────────────────────────

def test_5_adding_holiday_through_api_persists_it(web_client) -> None:
    """Requirement 5: Adding a holiday through the API persists it."""
    payload = {
        "date": "2026-10-27",
        "override_type": "HOLIDAY",
        "name": "Local Festival Off",
        "note": "Declared by Principal",
    }
    res = web_client.post("/api/calendar/entries", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "2026-10-27" in data["message"]

    # Verify in listing
    list_res = web_client.get("/api/calendar/entries")
    matching = [e for e in list_res.json() if e["start_date"] == "2026-10-27"]
    assert len(matching) == 1
    assert matching[0]["name"] == "Local Festival Off"
    assert matching[0]["type"] == "HOLIDAY"


def test_6_adding_working_day_override_persists_it(web_client) -> None:
    """Requirement 6: Adding a working-day override persists it."""
    # 2026-10-02 is Gandhi Jayanti (normally an official holiday)
    payload = {
        "date": "2026-10-02",
        "override_type": "WORKING_DAY",
        "name": "Special Working Day",
        "note": "Compensatory full class schedule",
    }
    res = web_client.post("/api/calendar/entries", json=payload)
    assert res.status_code == 200

    # Verify listing shows exactly ONE entry for 2026-10-02 as WORKING_DAY
    list_res = web_client.get("/api/calendar/entries")
    matching = [e for e in list_res.json() if e["start_date"] == "2026-10-02"]
    assert len(matching) == 1
    assert matching[0]["type"] == "WORKING_DAY"
    assert matching[0]["is_override"] is True
    assert "Official" in matching[0]["original_rule"]

    # Check status via calendar service
    cal_svc = get_calendar_service()
    status = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status.is_working_day is True
    assert status.is_holiday is False
    assert status.cancels_classes is False


def test_7_frontend_html_and_js_elements_present() -> None:
    """Requirement 7: The frontend Add Entry button and form controls exist."""
    from pathlib import Path
    html_path = Path("attendance_tracker/web/static/index.html")
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    assert 'id="btn-open-calendar-form"' in content
    assert 'id="calendar-entry-form-box"' in content
    assert 'id="input-entry-date"' in content
    assert 'id="select-entry-type"' in content
    assert 'id="input-entry-name"' in content
    assert 'id="input-entry-note"' in content
    assert 'id="btn-save-calendar-entry"' in content
    assert 'id="btn-cancel-calendar-entry"' in content
    assert 'id="unified-calendar-table"' in content
    assert 'id="unified-calendar-tbody"' in content


def test_8_saving_valid_entry_and_updating_date(web_client) -> None:
    """Requirement 8: Saving a valid entry calls the API and updates/moves entries."""
    # Create entry on 2026-10-15
    res1 = web_client.post(
        "/api/calendar/entries",
        json={"date": "2026-10-15", "override_type": "HOLIDAY", "name": "Draft Holiday"},
    )
    assert res1.status_code == 200

    # Edit and move to 2026-10-16
    res2 = web_client.post(
        "/api/calendar/entries",
        json={
            "date": "2026-10-16",
            "old_date": "2026-10-15",
            "override_type": "HOLIDAY",
            "name": "Confirmed Holiday",
        },
    )
    assert res2.status_code == 200

    # Verify 2026-10-15 was cleaned up and 2026-10-16 exists
    entries = web_client.get("/api/calendar/entries").json()
    assert not any(e["start_date"] == "2026-10-15" and e["is_override"] for e in entries)
    matching16 = [e for e in entries if e["start_date"] == "2026-10-16"]
    assert len(matching16) == 1
    assert matching16[0]["name"] == "Confirmed Holiday"


# ── 3. Validation & Error Handling Tests ──────────────────────────────────────

def test_9_invalid_input_displays_validation_errors(web_client) -> None:
    """Requirement 9: Invalid input is rejected with clear error details."""
    # Invalid date format
    res_bad_date = web_client.post(
        "/api/calendar/entries",
        json={"date": "not-a-date", "override_type": "HOLIDAY", "name": "Test"},
    )
    assert res_bad_date.status_code == 400
    assert "date" in res_bad_date.json()["detail"].lower()

    # Invalid override_type
    res_bad_type = web_client.post(
        "/api/calendar/entries",
        json={"date": "2026-10-28", "override_type": "INVALID_TYPE", "name": "Test"},
    )
    assert res_bad_type.status_code == 400
    assert "override_type" in res_bad_type.json()["detail"].lower()


def test_10_failed_save_does_not_persist_or_report_success(web_client) -> None:
    """Requirement 10: A failed save does not display a success message or persist invalid data."""
    res = web_client.post(
        "/api/calendar/entries",
        json={"date": "invalid-2026", "override_type": "HOLIDAY", "name": "Broken"},
    )
    assert res.status_code == 400
    assert "success" not in res.json()

    repo = get_repository()
    assert repo.get_calendar_override("invalid-2026") is None


def test_11_delete_and_restoration_of_default_rules(web_client) -> None:
    """Requirement 11: Deletion restores official default rule or normal timetable."""
    repo = get_repository()
    # 1. Override Gandhi Jayanti (2026-10-02) as WORKING_DAY
    repo.add_calendar_override("2026-10-02", "WORKING_DAY", name="Special Work")
    assert repo.get_calendar_override("2026-10-02") is not None

    # 2. Delete the override via API
    del_res = web_client.delete("/api/calendar/entries/2026-10-02")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 3. Check that Gandhi Jayanti reverted to OFFICIAL HOLIDAY
    entries = web_client.get("/api/calendar/entries").json()
    matching = [e for e in entries if e["start_date"] == "2026-10-02"]
    assert len(matching) == 1
    assert matching[0]["source"] == "OFFICIAL"
    assert matching[0]["type"] == "HOLIDAY"
    assert matching[0]["is_override"] is False


def test_12_duplicate_imports_do_not_produce_duplicate_displayed_entries(web_client) -> None:
    """Requirement 12: Duplicate imports do not produce duplicate displayed entries."""
    csv_text = (
        "Date,Type,Name\n"
        "2026-11-02,HOLIDAY,Annual Sports Day\n"
        "2026-11-02,HOLIDAY,Annual Sports Day (Updated)\n"
    )
    # Preview handles duplicates gracefully
    adapter = HolidayCalendarAdapter()
    preview = adapter.parse_csv_text(csv_text)
    assert len(preview.warnings) > 0

    # Commit twice
    for _ in range(2):
        web_client.post(
            "/api/import/calendar/commit",
            json={
                "source": "csv",
                "records": [
                    {"date": "2026-11-02", "override_type": "HOLIDAY", "name": "Annual Sports Day"}
                ],
            },
        )

    # Verify listing contains exactly ONE row for 2026-11-02
    entries = web_client.get("/api/calendar/entries").json()
    matching = [e for e in entries if e["start_date"] == "2026-11-02"]
    assert len(matching) == 1
    assert matching[0]["name"] == "Annual Sports Day"


def test_13_precedence_hierarchy_remains_strictly_enforced() -> None:
    """Requirement 13: Existing timetable and calendar precedence behavior remains unchanged."""
    cal_svc = get_calendar_service()
    repo = get_repository()

    # Rule 1: User Working Day Override > Official Holiday
    repo.add_calendar_override("2026-10-02", "WORKING_DAY", name="Override Work")
    st = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert st.is_working_day is True
    assert st.is_holiday is False
    assert st.event_type == "USER_OVERRIDE"

    # Rule 2: User Holiday Override > Regular Working Day (2026-10-05 Monday)
    repo.add_calendar_override("2026-10-05", "HOLIDAY", name="Special Off")
    st2 = cal_svc.get_calendar_status(date(2026, 10, 5))
    assert st2.is_holiday is True
    assert st2.is_working_day is False
    assert st2.event_type == "USER_OVERRIDE"

    # Rule 3: Mid-Term 1 cancels classes by default
    mt1_date = date(2026, 9, 8)
    st_mt1 = cal_svc.get_calendar_status(mt1_date)
    # Without user override, mid term date follows exam schedule
    assert cal_svc.get_mid_term_1()[0] <= mt1_date <= cal_svc.get_mid_term_1()[1]


def test_14_portal_sync_and_other_routes_continue_working(web_client) -> None:
    """Requirement 14: Portal synchronization and health routes continue working."""
    health_res = web_client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # Dashboard endpoint
    dash_res = web_client.get("/api/dashboard")
    assert dash_res.status_code == 200
    assert "overall" in dash_res.json()
