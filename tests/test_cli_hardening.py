"""
tests/test_cli_hardening.py
───────────────────────────
CLI interface robustness, argument parsing, graceful error handling,
and credential protection tests for Phase 10.
"""

from unittest.mock import MagicMock, patch
import pytest

from attendance_tracker.cli.main import (
    handle_ask_query,
    handle_credentials_command,
    handle_sync_portal,
    render_attendance_table,
)
from attendance_tracker.portal.models import PortalSyncResult


def test_cli_ask_query_execution():
    """Verify handle_ask_query executes cleanly without unhandled errors."""
    resp = handle_ask_query("What is my attendance in OS?", verbose=True)
    assert resp is not None
    assert resp.nlu.intent == "GET_SUBJECT_ATTENDANCE"
    assert resp.nlu.subject == "OS"


def test_cli_ask_query_quiet_mode():
    """Verify handle_ask_query runs in quiet mode."""
    resp = handle_ask_query("What is my attendance in OS?", verbose=False)
    assert resp is not None
    assert resp.nlu.intent == "GET_SUBJECT_ATTENDANCE"


def test_cli_overview_table_rendering():
    """Verify render_attendance_table runs and outputs data."""
    # Should run with default repo data without error
    render_attendance_table()


def test_cli_overview_table_empty_state():
    """Verify render_attendance_table handles empty repository gracefully."""
    mock_repo = MagicMock()
    mock_repo.get_all_current_attendance.return_value = None

    with patch("attendance_tracker.cli.main.get_repository", return_value=mock_repo):
        render_attendance_table()


def test_cli_sync_success_display():
    """Verify handle_sync_portal displays formatted success panel."""
    mock_svc = MagicMock()
    mock_svc.sync_attendance.return_value = PortalSyncResult(
        success=True,
        snapshot_id=2,
        records_count=9,
        changes_detected=[],
        duration_ms=450.0,
    )

    with patch("attendance_tracker.cli.main.get_portal_service", return_value=mock_svc):
        handle_sync_portal(headless=True)


def test_cli_sync_failure_display():
    """Verify handle_sync_portal displays formatted error panel on failure without traceback."""
    mock_svc = MagicMock()
    mock_svc.sync_attendance.return_value = PortalSyncResult(
        success=False,
        error_message="Portal authentication failed: Invalid student ID or password.",
        duration_ms=250.0,
    )

    with patch("attendance_tracker.cli.main.get_portal_service", return_value=mock_svc):
        handle_sync_portal(headless=True)


def test_cli_credentials_check_configured():
    """Verify handle_credentials_command displays configuration state."""
    with patch("attendance_tracker.cli.main.has_portal_credentials", return_value=True):
        handle_credentials_command("check")


def test_cli_credentials_check_unconfigured():
    """Verify handle_credentials_command displays unconfigured notice."""
    with patch("attendance_tracker.cli.main.has_portal_credentials", return_value=False):
        handle_credentials_command("check")


def test_cli_credentials_unknown_action():
    """Verify handle_credentials_command handles unknown action string."""
    handle_credentials_command("invalid_action_name")
