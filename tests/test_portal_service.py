"""
tests/test_portal_service.py
────────────────────────────
Integration and unit tests for PortalAttendanceService and PortalAttendanceScraper.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from attendance_tracker.attendance.change_detector import AttendanceChangeDetector
from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.portal.auth import PortalAuthenticator
from attendance_tracker.portal.attendance import AttendanceParser, PortalAttendanceScraper
from attendance_tracker.portal.models import RawSubjectRecord, ScrapedAttendanceData
from attendance_tracker.portal.service import PortalAttendanceService


MOCK_9_SUBJECT_RECORDS = [
    RawSubjectRecord(subject_name_raw="22MN6HS302", attended=4, conducted=12, subject_code="22MN6HS302"),
    RawSubjectRecord(subject_name_raw="22OE1EN301", attended=12, conducted=26, subject_code="22OE1EN301"),
    RawSubjectRecord(subject_name_raw="22PC1AM202", attended=16, conducted=34, subject_code="22PC1AM202"),
    RawSubjectRecord(subject_name_raw="22PC1AM203", attended=26, conducted=39, subject_code="22PC1AM203"),
    RawSubjectRecord(subject_name_raw="22PC1IN301", attended=21, conducted=33, subject_code="22PC1IN301"),
    RawSubjectRecord(subject_name_raw="22PC1IT202", attended=22, conducted=33, subject_code="22PC1IT202"),
    RawSubjectRecord(subject_name_raw="22PC2AM202", attended=24, conducted=36, subject_code="22PC2AM202"),
    RawSubjectRecord(subject_name_raw="22PC2IN301", attended=15, conducted=33, subject_code="22PC2IN301"),
    RawSubjectRecord(subject_name_raw="22PE1AM301", attended=23, conducted=42, subject_code="22PE1AM301"),
]


@pytest.fixture
def test_env(tmp_path: Path):
    db_file = tmp_path / "test_service.db"
    repo = AttendanceRepository(db_path=db_file)
    repo.seed_canonical_subjects()
    repo.seed_aliases()
    repo.seed_initial_snapshot()

    resolver = SubjectResolver(repository=repo)
    change_detector = AttendanceChangeDetector()
    authenticator = MagicMock(spec=PortalAuthenticator)
    scraper = MagicMock(spec=PortalAttendanceScraper)

    service = PortalAttendanceService(
        repository=repo,
        authenticator=authenticator,
        scraper=scraper,
        change_detector=change_detector,
    )
    return repo, authenticator, scraper, service


def test_sync_attendance_without_credentials_fails_safely(test_env):
    repo, auth, scraper, service = test_env

    with patch("attendance_tracker.portal.service.has_portal_credentials", return_value=False):
        result = service.sync_attendance(username="24071A6947")
        assert result.success is False
        assert "No portal credentials configured" in result.error_message

        # Verify initial snapshot is untouched
        latest = repo.get_latest_snapshot()
        assert latest.source == "seed"

        # Verify scrape_log recorded failure
        with repo.get_connection() as conn:
            logs = conn.execute("SELECT status, error_msg FROM scrape_log;").fetchall()
            assert len(logs) == 1
            assert logs[0]["status"] == "FAILED"
            assert "No portal credentials" in logs[0]["error_msg"]


def test_sync_attendance_successful_flow(test_env):
    repo, auth, scraper, service = test_env

    scraped_data = ScrapedAttendanceData(
        scraped_at=datetime(2026, 10, 4, 23, 30, 0),
        records=MOCK_9_SUBJECT_RECORDS,
    )

    scraper.scrape_attendance.return_value = scraped_data

    # Mock Playwright context inside sync_attendance
    mock_page = MagicMock()
    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page
    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    with patch("attendance_tracker.portal.service.sync_playwright") as mock_pw_ctx, \
         patch("attendance_tracker.portal.service.has_portal_credentials", return_value=True):
        
        mock_pw_ctx.return_value.__enter__.return_value = mock_playwright

        result = service.sync_attendance(username="24071A6947", password="ValidPassword")

        assert result.success is True
        assert result.snapshot_id is not None
        assert result.records_count == 9

        # Verify snapshot created in repository
        latest = repo.get_latest_snapshot()
        assert latest.id == result.snapshot_id
        assert latest.source == "portal"

        # Verify OS record in new snapshot (from seed 17/28 to 22/33)
        os_att = repo.get_attendance_for_subject("OS", snapshot_id=result.snapshot_id)
        assert os_att.attended == 22
        assert os_att.conducted == 33

        # Verify changes were detected and persisted
        os_changes = repo.get_changes_for_subject("OS")
        assert len(os_changes) == 1
        assert os_changes[0].delta_attended == 5
        assert os_changes[0].delta_conducted == 5

        # Verify scrape_log recorded SUCCESS
        with repo.get_connection() as conn:
            logs = conn.execute("SELECT status, snapshot_id, error_msg FROM scrape_log;").fetchall()
            assert len(logs) == 1
            assert logs[0]["status"] == "SUCCESS"
            assert logs[0]["snapshot_id"] == result.snapshot_id


def test_sync_attendance_failure_preserves_database(test_env):
    repo, auth, scraper, service = test_env

    auth.login.side_effect = Exception("Network connection lost")

    mock_page = MagicMock()
    mock_context = MagicMock()
    mock_context.new_page.return_value = mock_page
    mock_browser = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    with patch("attendance_tracker.portal.service.sync_playwright") as mock_pw_ctx, \
         patch("attendance_tracker.portal.service.has_portal_credentials", return_value=True):
        
        mock_pw_ctx.return_value.__enter__.return_value = mock_playwright

        result = service.sync_attendance(username="24071A6947", password="ValidPassword")

        assert result.success is False
        assert "Network connection lost" in result.error_message

        # Ensure database is still on the initial seed snapshot
        latest = repo.get_latest_snapshot()
        assert latest.source == "seed"

        # Check scrape_log
        with repo.get_connection() as conn:
            logs = conn.execute("SELECT status, error_msg FROM scrape_log;").fetchall()
            assert len(logs) == 1
            assert logs[0]["status"] == "FAILED"


def test_portal_scraper_navigation_and_click(tmp_path: Path):
    """Test that PortalAttendanceScraper waits for #attp, clicks it, and waits for #dynamicModal table."""
    db_file = tmp_path / "test_scraper.db"
    repo = AttendanceRepository(db_path=db_file)
    repo.seed_canonical_subjects()
    repo.seed_aliases()
    resolver = SubjectResolver(repository=repo)
    parser = AttendanceParser(resolver=resolver, expected_subject_count=9)
    scraper = PortalAttendanceScraper(parser=parser, resolver=resolver)

    mock_page = MagicMock()
    mock_tile = MagicMock()
    mock_modal = MagicMock()
    mock_modal.inner_html.return_value = """
    <table>
        <tr><td></td><td>October, 2026</td><td>Cumulative</td></tr>
        <tr><th>22MN6HS302</th><td>2 / 2</td><td>4 / 12</td></tr>
        <tr><th>22OE1EN301</th><td>1 / 1</td><td>12 / 26</td></tr>
        <tr><th>22PC1AM202</th><td>1 / 1</td><td>16 / 34</td></tr>
        <tr><th>22PC1AM203</th><td>0 / 0</td><td>26 / 39</td></tr>
        <tr><th>22PC1IN301</th><td>1 / 1</td><td>21 / 33</td></tr>
        <tr><th>22PC1IT202</th><td>1 / 1</td><td>22 / 33</td></tr>
        <tr><th>22PC2AM202</th><td>3 / 3</td><td>24 / 36</td></tr>
        <tr><th>22PC2IN301</th><td>0 / 0</td><td>15 / 33</td></tr>
        <tr><th>22PE1AM301</th><td>0 / 1</td><td>23 / 42</td></tr>
        <tr><th>Total</th><td>9 / 10</td><td>163 / 288</td></tr>
    </table>
    """

    def wait_selector_side_effect(selector, timeout=None, state=None):
        if "#attp" in selector:
            return mock_tile
        if "#dynamicModal" in selector:
            return mock_modal
        return MagicMock()

    mock_page.wait_for_selector.side_effect = wait_selector_side_effect

    data = scraper.scrape_attendance(mock_page)

    # Verify tile was waited on and clicked
    mock_tile.click.assert_called_once()
    assert len(data.records) == 9
    os_rec = next(r for r in data.records if r.subject_code == "22PC1IT202")
    assert os_rec.attended == 22
    assert os_rec.conducted == 33
