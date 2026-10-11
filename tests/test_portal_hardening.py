"""
tests/test_portal_hardening.py
──────────────────────────────
Comprehensive portal failure safety, lockout protection, and DOM resilience tests for Phase 10.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.subject_resolver import SubjectResolver
from attendance_tracker.portal.auth import PortalAuthenticator
from attendance_tracker.portal.attendance import AttendanceParser, PortalAttendanceScraper
from attendance_tracker.portal.exceptions import (
    PortalAuthError,
    PortalNavigationError,
    PortalScrapeError,
    PortalValidationError,
)
from attendance_tracker.portal.models import ScrapedAttendanceData
from attendance_tracker.portal.service import PortalAttendanceService


@pytest.fixture
def test_repo(tmp_path: Path) -> AttendanceRepository:
    db_file = tmp_path / "portal_hardening.db"
    repo = AttendanceRepository(db_path=db_file)
    repo.seed_canonical_subjects()
    repo.seed_aliases()
    repo.seed_initial_snapshot()
    return repo


def test_auth_captcha_challenge_triggers_immediate_abort():
    """Verify that when a CAPTCHA or security challenge is displayed, login aborts immediately with PortalAuthError."""
    authenticator = PortalAuthenticator(timeout_ms=5000)
    mock_page = MagicMock()
    mock_page.url = "https://automation.vnrvjiet.ac.in/Login"

    # Form inputs found
    mock_user_input = MagicMock()
    mock_pass_input = MagicMock()
    mock_submit_btn = MagicMock()
    mock_page.wait_for_selector.side_effect = [mock_user_input, mock_pass_input, mock_submit_btn]

    # CAPTCHA challenge element found
    mock_challenge_el = MagicMock()
    mock_challenge_el.is_visible.return_value = True
    mock_form = MagicMock()

    def query_selector_mock(selector):
        if selector == "#frm-login":
            return mock_form
        if selector == "#loginChallenge":
            return mock_challenge_el
        return None

    mock_page.query_selector.side_effect = query_selector_mock

    with pytest.raises(PortalAuthError) as exc_info:
        authenticator.login(mock_page, username="24071A6947", password="SamplePassword")

    assert "CAPTCHA or DOB required" in str(exc_info.value)
    # Ensure submit was clicked at most once (ZERO automated retries)
    assert mock_submit_btn.click.call_count == 1


def test_auth_account_lockout_detection():
    """Verify account lockout message is cleanly caught and reported."""
    authenticator = PortalAuthenticator(timeout_ms=5000)
    mock_page = MagicMock()
    mock_page.url = "https://automation.vnrvjiet.ac.in/Login"

    mock_user_input = MagicMock()
    mock_pass_input = MagicMock()
    mock_submit_btn = MagicMock()
    mock_page.wait_for_selector.side_effect = [mock_user_input, mock_pass_input, mock_submit_btn]

    mock_form = MagicMock()
    mock_page.query_selector.side_effect = lambda sel: mock_form if sel == "#frm-login" else None

    # Simulate AJAX response reporting account block
    response_handler = None
    def on_mock(event_name, handler):
        nonlocal response_handler
        if event_name == "response":
            response_handler = handler

    mock_page.on.side_effect = on_mock

    def wait_for_timeout_mock(ms):
        if response_handler:
            mock_resp = MagicMock()
            mock_resp.url = "https://automation.vnrvjiet.ac.in/Login"
            mock_resp.request.method = "POST"
            mock_resp.json.return_value = {
                "Status": 0,
                "Data": {
                    "Message": "Invalid credentials",
                    "BlockedUntil": "2026-10-05 12:00:00",
                    "CaptchaRequired": True,
                },
            }
            response_handler(mock_resp)

    mock_page.wait_for_timeout.side_effect = wait_for_timeout_mock

    with pytest.raises(PortalAuthError) as exc_info:
        authenticator.login(mock_page, username="24071A6947", password="WrongPassword")

    assert "Account blocked until" in str(exc_info.value)
    assert "CAPTCHA verification triggered" in str(exc_info.value)


def test_auth_network_timeout_triggers_portal_navigation_error():
    """Verify network timeout on page load raises PortalNavigationError."""
    authenticator = PortalAuthenticator(timeout_ms=1000)
    mock_page = MagicMock()
    mock_page.goto.side_effect = PlaywrightTimeoutError("Navigation timeout")

    with pytest.raises(PortalNavigationError) as exc_info:
        authenticator.login(mock_page, username="24071A6947", password="SomePassword")

    assert "Timeout navigating to portal URL" in str(exc_info.value)


def test_scraper_missing_launcher_tile_raises_navigation_error(test_repo: AttendanceRepository):
    """Verify missing #attp action tile raises PortalNavigationError."""
    resolver = SubjectResolver(repository=test_repo)
    scraper = PortalAttendanceScraper(resolver=resolver, timeout_ms=1000)

    mock_page = MagicMock()
    mock_page.url = "https://automation.vnrvjiet.ac.in/App"
    mock_page.wait_for_selector.side_effect = PlaywrightTimeoutError("Timeout on #attp")

    with pytest.raises(PortalNavigationError) as exc_info:
        scraper.scrape_attendance(mock_page)

    assert "Timed out waiting for attendance action tile (#attp)" in str(exc_info.value)


def test_scraper_modal_timeout_raises_navigation_error(test_repo: AttendanceRepository):
    """Verify failure of #dynamicModal to open raises PortalNavigationError."""
    resolver = SubjectResolver(repository=test_repo)
    scraper = PortalAttendanceScraper(resolver=resolver, timeout_ms=1000)

    mock_page = MagicMock()
    mock_page.url = "https://automation.vnrvjiet.ac.in/App"
    mock_tile = MagicMock()

    def wait_selector_mock(selector, timeout=None, state=None):
        if "#attp" in selector:
            return mock_tile
        if "#dynamicModal" in selector:
            raise PlaywrightTimeoutError("Modal timeout")
        return MagicMock()

    mock_page.wait_for_selector.side_effect = wait_selector_mock

    with pytest.raises(PortalNavigationError) as exc_info:
        scraper.scrape_attendance(mock_page)

    assert "Timed out waiting for attendance modal (#dynamicModal)" in str(exc_info.value)


def test_parser_duplicate_subjects_rejected(test_repo: AttendanceRepository):
    """Verify duplicate subject codes in modal HTML raise PortalValidationError."""
    resolver = SubjectResolver(repository=test_repo)
    parser = AttendanceParser(resolver=resolver, expected_subject_count=9)

    html_duplicate = """
    <table>
        <tr><td></td><td>October, 2026</td><td>Cumulative</td></tr>
        <tr><th>22MN6HS302</th><td>2 / 2</td><td>4 / 12</td></tr>
        <tr><th>22MN6HS302</th><td>2 / 2</td><td>4 / 12</td></tr>
        <tr><th>22OE1EN301</th><td>1 / 1</td><td>12 / 26</td></tr>
        <tr><th>22PC1AM202</th><td>1 / 1</td><td>16 / 34</td></tr>
        <tr><th>22PC1AM203</th><td>0 / 0</td><td>26 / 39</td></tr>
        <tr><th>22PC1IN301</th><td>1 / 1</td><td>21 / 33</td></tr>
        <tr><th>22PC1IT202</th><td>1 / 1</td><td>22 / 33</td></tr>
        <tr><th>22PC2AM202</th><td>3 / 3</td><td>24 / 36</td></tr>
        <tr><th>22PC2IN301</th><td>0 / 0</td><td>15 / 33</td></tr>
    </table>
    """
    with pytest.raises(PortalValidationError) as exc_info:
        parser.parse_html(html_duplicate, strict_count=False)

    assert "Duplicate subject entries found" in str(exc_info.value)


def test_parser_extra_unexpected_subjects_rejected(test_repo: AttendanceRepository):
    """Verify extra unexpected subjects raise PortalScrapeError."""
    # Add a 10th temporary subject to repository so resolver knows it
    test_repo.add_subject("22PC9ZZ999", "Extra Subject", "ES", 75.0)
    resolver = SubjectResolver(repository=test_repo)
    parser = AttendanceParser(resolver=resolver, expected_subject_count=9)

    html_extra = """
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
        <tr><th>22PC9ZZ999</th><td>1 / 1</td><td>5 / 5</td></tr>
    </table>
    """
    with pytest.raises(PortalScrapeError) as exc_info:
        parser.parse_html(html_extra, strict_count=True)

    assert "Unexpected extra subjects extracted: expected 9, got 10" in str(exc_info.value)
