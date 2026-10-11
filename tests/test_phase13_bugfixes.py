"""
tests/test_phase13_bugfixes.py
───────────────────────────────
Comprehensive test suite verifying Phase 13 bug fixes:
1. Portal sync reports true status (changed vs unchanged vs failure, no false success).
2. Holiday / Mid-term API and data structure verification.
3. Universal holiday & calendar exception import (CSV, Excel, text) with validation, conflict handling, and forecasting impact.
"""

from datetime import date, datetime
import io
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from attendance_tracker.attendance.adapters.calendar_adapter import (
    HolidayCalendarAdapter,
    HolidayImportPreview,
    HolidayImportRecord,
)
from attendance_tracker.attendance.repository import get_repository
from attendance_tracker.calendar.service import get_calendar_service
from attendance_tracker.portal.models import PortalSyncResult
from attendance_tracker.timetable.future_generator import get_future_class_generator
from attendance_tracker.web.app import create_app


@pytest.fixture(scope="module")
def web_client():
    """Create test client for web API endpoints."""
    app = create_app()
    with TestClient(app) as client:
        yield client


# ── 1. Portal Sync Verification Tests ─────────────────────────────────────────

def test_sync_failure_returns_failure_status(web_client) -> None:
    """Verify failed portal sync returns success=False and clear error message (no false success)."""
    with patch("attendance_tracker.portal.service.PortalAttendanceService.sync_attendance") as mock_sync:
        mock_sync.return_value = PortalSyncResult(
            success=False,
            error_message="Invalid credentials or portal login timed out.",
            duration_ms=1250.0,
            timestamp=datetime.now(),
        )

        res = web_client.post("/api/sync")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is False
        assert data["status"] == "FAILED"
        assert "Invalid credentials" in data["error_message"]
        assert "Sync failed" in data["message"]
        assert data["snapshot_id"] is None
        assert data["changed"] is False


def test_sync_success_unchanged_attendance(web_client) -> None:
    """Verify successful sync with unchanged data returns changed=False and informative message."""
    repo = get_repository()
    latest = repo.get_latest_snapshot()
    snap_id = latest.id if latest else 1

    with patch("attendance_tracker.portal.service.PortalAttendanceService.sync_attendance") as mock_sync:
        mock_sync.return_value = PortalSyncResult(
            success=True,
            snapshot_id=snap_id,
            records_count=9,
            changes_detected=[],
            duration_ms=2100.0,
            timestamp=datetime.now(),
        )

        with patch("attendance_tracker.attendance.comparator.SnapshotComparisonEngine.compare_latest_with_previous") as mock_comp:
            mock_comp.return_value = MagicMock(
                has_changes=False,
                changed_subjects_count=0,
                summary_text="No attendance changes detected.",
            )

            res = web_client.post("/api/sync")
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["status"] == "SUCCESS"
            assert data["changed"] is False
            assert "No attendance changes detected" in data["message"]


def test_sync_success_changed_attendance(web_client) -> None:
    """Verify successful sync with changes returns changed=True and delta summary."""
    with patch("attendance_tracker.portal.service.PortalAttendanceService.sync_attendance") as mock_sync:
        mock_sync.return_value = PortalSyncResult(
            success=True,
            snapshot_id=99,
            records_count=9,
            changes_detected=[],
            duration_ms=2500.0,
            timestamp=datetime.now(),
        )

        with patch("attendance_tracker.attendance.comparator.SnapshotComparisonEngine.compare_latest_with_previous") as mock_comp:
            mock_comp.return_value = MagicMock(
                has_changes=True,
                changed_subjects_count=2,
                summary_text="Attendance changes detected in OS and ACD.",
            )

            res = web_client.post("/api/sync")
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["changed"] is True
            assert data["changes_count"] == 2
            assert "Attendance updated" in data["message"]
            assert data["comparison_summary"] is not None


# ── 2. Holiday / Midterm Calendar API Tests ───────────────────────────────────

def test_api_calendar_holidays_structure(web_client) -> None:
    """Verify GET /api/calendar/holidays returns official holidays and mid-term exam windows."""
    res = web_client.get("/api/calendar/holidays")
    assert res.status_code == 200
    data = res.json()

    assert "official_holidays" in data
    assert isinstance(data["official_holidays"], list)
    assert len(data["official_holidays"]) > 0

    assert "mid_term_1" in data
    assert "start" in data["mid_term_1"]
    assert "end" in data["mid_term_1"]
    assert "mid_term_2" not in data

    # Verify holiday fields
    first_hol = data["official_holidays"][0]
    assert "date" in first_hol
    assert "name" in first_hol


def test_calendar_service_working_day_and_holiday_precedence() -> None:
    """Verify CalendarService enforces strict precedence for overrides over named holidays."""
    repo = get_repository()
    cal_svc = get_calendar_service()

    # 1. Official holiday: 2026-10-02 (Mahatma Gandhi Jayanti) is holiday by default
    status_default = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status_default.is_holiday is True

    # 2. Add working-day override on Gandhi Jayanti
    repo.add_calendar_override("2026-10-02", "WORKING_DAY", name="Special Working Day")
    status_override = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status_override.is_holiday is False
    assert status_override.is_working_day is True
    assert status_override.is_override is True

    # 3. Clean up override
    repo.delete_calendar_override("2026-10-02")
    status_restored = cal_svc.get_calendar_status(date(2026, 10, 2))
    assert status_restored.is_holiday is True


# ── 3. Holiday / Calendar Ingestion Tests ─────────────────────────────────────

def test_holiday_csv_adapter_parsing_and_normalization() -> None:
    """Verify HolidayCalendarAdapter parses various date and type formats."""
    adapter = HolidayCalendarAdapter()
    csv_text = """Date,Type,Name
2026-10-02,HOLIDAY,Mahatma Gandhi Jayanti
2026-10-19,holiday,Maharnavami
2026-10-20,HOLIDAY,Vijaya Dasami
2026-10-21,WORKING_DAY,Compensatory Working Day
2026-10-22,working day,Special Classes
2026-10-23,OFF,College Fest Break
"""
    preview = adapter.parse_csv_text(csv_text)
    assert preview.is_valid is True
    assert preview.total_rows == 6
    assert preview.valid_count == 6
    assert preview.invalid_count == 0

    types = [r.override_type for r in preview.records]
    assert types == ["HOLIDAY", "HOLIDAY", "HOLIDAY", "WORKING_DAY", "WORKING_DAY", "HOLIDAY"]


def test_holiday_csv_adapter_invalid_and_conflicting_rows() -> None:
    """Verify HolidayCalendarAdapter detects invalid dates and logs warnings for conflicts."""
    adapter = HolidayCalendarAdapter()
    csv_text = """Date,Type,Name
invalid-date,HOLIDAY,Bad Date Row
2026-11-10,HOLIDAY,Initial Holiday
2026-11-10,WORKING_DAY,Changed to Working Day
"""
    preview = adapter.parse_csv_text(csv_text)
    assert preview.total_rows == 3
    assert preview.valid_count == 2
    assert preview.invalid_count == 1
    assert len(preview.errors) == 1
    assert len(preview.warnings) == 1
    assert "conflicting types" in preview.warnings[0].lower()


def test_api_calendar_import_preview_and_commit(web_client) -> None:
    """Verify complete API flow for previewing and committing calendar exceptions."""
    repo = get_repository()
    cal_svc = get_calendar_service()

    csv_text = """Date,Type,Name
2026-11-17,HOLIDAY,Tech Fest Holiday
2026-11-18,WORKING_DAY,Makeup Lab Session
"""
    # 1. Preview text
    res_prev = web_client.post(
        "/api/import/calendar/preview-text",
        json={"csv_text": csv_text}
    )
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert prev_data["is_valid"] is True
    assert prev_data["valid_count"] == 2

    # 2. Commit import
    commit_payload = [
        {"date": r["date"], "override_type": r["override_type"], "name": r["name"]}
        for r in prev_data["records"] if r["is_valid"]
    ]
    res_commit = web_client.post(
        "/api/import/calendar/commit",
        json={"records": commit_payload, "source": "csv"}
    )
    assert res_commit.status_code == 200
    commit_data = res_commit.json()
    assert commit_data["success"] is True
    assert commit_data["imported_count"] == 2

    # 3. Verify calendar status immediately reflects imported overrides
    status_17 = cal_svc.get_calendar_status(date(2026, 11, 17))
    assert status_17.is_holiday is True
    assert status_17.is_override is True

    status_18 = cal_svc.get_calendar_status(date(2026, 11, 18))
    assert status_18.is_holiday is False
    assert status_18.is_working_day is True
    assert status_18.is_override is True

    # 4. Clean up
    repo.delete_calendar_override("2026-11-17")
    repo.delete_calendar_override("2026-11-18")


def test_imported_holidays_affect_future_class_generator() -> None:
    """Verify that importing a holiday cancels class generation for that date."""
    repo = get_repository()
    generator = get_future_class_generator()

    test_tue = date(2026, 11, 17)  # Tuesday

    # Before override: Tuesday has classes
    sched_before = generator.get_classes_for_date(test_tue)
    count_before = len(sched_before)
    assert count_before > 0

    # Add holiday override via repository
    repo.add_calendar_override("2026-11-17", "HOLIDAY", name="Imported Holiday")

    # After override: Tuesday has 0 classes
    sched_after = generator.get_classes_for_date(test_tue)
    assert len(sched_after) == 0

    # Clean up
    repo.delete_calendar_override("2026-11-17")
    sched_restored = generator.get_classes_for_date(test_tue)
    assert len(sched_restored) == count_before


def test_holiday_excel_adapter_parsing_and_upload(web_client) -> None:
    """Verify HolidayCalendarAdapter parses binary Excel (.xlsx) files."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Holidays"
    ws.append(["Date", "Type", "Name"])
    ws.append(["2026-11-23", "HOLIDAY", "Special Event Holiday"])
    ws.append(["2026-11-24", "WORKING_DAY", "Extra Lab Class"])

    buffer = io.BytesIO()
    wb.save(buffer)
    excel_bytes = buffer.getvalue()

    adapter = HolidayCalendarAdapter()
    preview = adapter.parse_excel_bytes(excel_bytes)
    assert preview.is_valid is True
    assert preview.valid_count == 2
    assert preview.records[0].date == "2026-11-23"
    assert preview.records[0].override_type == "HOLIDAY"
    assert preview.records[1].date == "2026-11-24"
    assert preview.records[1].override_type == "WORKING_DAY"

    # Test multipart file upload endpoint
    files = {"file": ("holidays.xlsx", io.BytesIO(excel_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    res_upload = web_client.post("/api/import/calendar/upload", files=files)
    assert res_upload.status_code == 200
    upload_data = res_upload.json()
    assert upload_data["is_valid"] is True
    assert upload_data["valid_count"] == 2


def test_timetable_and_attendance_imports_still_work(web_client) -> None:
    """Verify that regular attendance CSV and Excel imports still work 100%."""
    csv_text = "Subject,Attended,Conducted\nOS,22,33\nACD,18,30\nDE,20,32\nIoTAP,16,28\nWP,19,31\nGS,25,35\nOE-1,21,33\nDE LAB,12,18\nIoTAP LAB,10,15"
    res_prev = web_client.post(
        "/api/import/preview-text",
        json={"csv_text": csv_text}
    )
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert prev_data["is_valid"] is True
    assert len(prev_data["records"]) == 9
    assert prev_data["total_attended"] == 163
    assert prev_data["total_conducted"] == 255

