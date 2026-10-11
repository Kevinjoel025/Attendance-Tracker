"""
portal/service.py
─────────────────
High-level attendance synchronization service.

Coordinates:
1. Credential retrieval (from Windows Credential Manager / keyring)
2. Safe single-attempt browser authentication
3. Attendance scraping and table extraction
4. Atomic SQLite snapshot creation
5. Deterministic change detection & persistence
6. Audit logging in scrape_log
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo
from playwright.sync_api import sync_playwright

from attendance_tracker.attendance.change_detector import (
    AttendanceChangeDetector,
    get_change_detector,
)
from attendance_tracker.attendance.repository import (
    AttendanceRepository,
    get_repository,
)
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.portal.auth import PortalAuthenticator
from attendance_tracker.portal.attendance import AttendanceParser, PortalAttendanceScraper
from attendance_tracker.portal.credentials import (
    get_portal_credentials,
    has_portal_credentials,
)
from attendance_tracker.portal.exceptions import (
    CredentialNotFoundError,
    PortalAuthError,
    PortalError,
    PortalNavigationError,
    PortalScrapeError,
    PortalValidationError,
)
from attendance_tracker.portal.models import PortalSyncResult, ScrapedAttendanceData

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


class PortalAttendanceService:
    """
    High-level orchestrator for syncing attendance from the college portal into SQLite.
    """

    def __init__(
        self,
        repository: Optional[AttendanceRepository] = None,
        authenticator: Optional[PortalAuthenticator] = None,
        scraper: Optional[PortalAttendanceScraper] = None,
        change_detector: Optional[AttendanceChangeDetector] = None,
    ) -> None:
        self.repo = repository or get_repository()
        self.authenticator = authenticator or PortalAuthenticator()
        self.scraper = scraper or PortalAttendanceScraper()
        self.change_detector = change_detector or get_change_detector()

    def sync_attendance(
        self,
        headless: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_ms: int = 45000,
    ) -> PortalSyncResult:
        """
        Execute a complete end-to-end attendance synchronization.

        SAFETY RULES:
        - Exactly ONE login attempt is executed.
        - On scrape failure, existing snapshots remain untouched.
        - All attempts are recorded in `scrape_log`.
        """
        start_time = time.perf_counter()
        attempted_at = datetime.now(_TZ)
        attempted_at_str = attempted_at.isoformat()

        user = username or config.portal.username

        # 1. Check credentials if password is not explicitly passed
        if password is None:
            if not has_portal_credentials(user):
                err_msg = (
                    f"No portal credentials configured for student ID '{user}'. "
                    f"Run 'python -m attendance_tracker.portal.setup_credentials' first."
                )
                logger.warning(err_msg)
                self._record_scrape_log(attempted_at_str, None, "FAILED", None, err_msg)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return PortalSyncResult(
                    success=False,
                    error_message=err_msg,
                    duration_ms=duration_ms,
                    timestamp=attempted_at,
                )

        # 2. Execute browser workflow with Playwright
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context()
                page = context.new_page()

                try:
                    # Single-attempt login
                    self.authenticator.login(
                        page,
                        username=username,
                        password=password,
                    )

                    # Extract attendance data
                    scraped_data: ScrapedAttendanceData = self.scraper.scrape_attendance(page)

                finally:
                    context.close()
                    browser.close()

            # 3. Process scraped records and prepare database payload
            db_records: list[dict[str, Any]] = []
            unmapped_records: list[str] = []

            for r in scraped_data.records:
                if r.subject_code:
                    db_records.append(
                        {
                            "subject_code": r.subject_code,
                            "attended": r.attended,
                            "conducted": r.conducted,
                        }
                    )
                else:
                    unmapped_records.append(r.subject_name_raw)

            if unmapped_records:
                logger.warning(f"Unmapped subjects found during scrape: {unmapped_records}")

            if not db_records:
                raise PortalScrapeError(
                    "None of the scraped records could be mapped to known subjects."
                )

            # Look up previous snapshot for delta calculation before writing new snapshot
            prev_snapshot = self.repo.get_latest_snapshot()

            # 4. Atomically persist new snapshot and records
            raw_payload = json.dumps(
                {
                    "scraped_at": attempted_at_str,
                    "source": "portal",
                    "student_id": user,
                    "records_count": len(db_records),
                    "raw_records": [
                        {
                            "subject_raw": r.subject_name_raw,
                            "subject_code": r.subject_code,
                            "attended": r.attended,
                            "conducted": r.conducted,
                        }
                        for r in scraped_data.records
                    ],
                }
            )

            new_snapshot = self.repo.create_snapshot(
                scraped_at=attempted_at_str,
                source="portal",
                raw_json=raw_payload,
                records=db_records,
            )

            # 5. Detect and persist attendance changes
            detected_changes: list[dict[str, Any]] = []

            for rec in db_records:
                s_code = rec["subject_code"]
                curr_att = rec["attended"]
                curr_cond = rec["conducted"]

                prev_att = None
                prev_cond = None
                prev_time = None

                if prev_snapshot is not None:
                    prev_subj_att = self.repo.get_attendance_for_subject(
                        s_code, snapshot_id=prev_snapshot.id
                    )
                    if prev_subj_att is not None:
                        prev_att = prev_subj_att.attended
                        prev_cond = prev_subj_att.conducted
                        try:
                            prev_time = datetime.fromisoformat(prev_snapshot.scraped_at)
                        except Exception:
                            pass

                change_res = self.change_detector.detect_change(
                    subject_code=s_code,
                    curr_attended=curr_att,
                    curr_conducted=curr_cond,
                    prev_attended=prev_att,
                    prev_conducted=prev_cond,
                    prev_snapshot_time=prev_time,
                    curr_snapshot_time=attempted_at,
                    scrape_status="SUCCESS",
                )

                # Save change to database if there was a previous snapshot
                if prev_snapshot is not None and prev_att is not None and prev_cond is not None:
                    self.repo.save_change(
                        detected_at=attempted_at_str,
                        subject_code=s_code,
                        prev_snapshot_id=prev_snapshot.id,
                        curr_snapshot_id=new_snapshot.id,
                        prev_attended=prev_att,
                        prev_conducted=prev_cond,
                        curr_attended=curr_att,
                        curr_conducted=curr_cond,
                        delta_attended=change_res.delta_attended,
                        delta_conducted=change_res.delta_conducted,
                        status=change_res.status.value,
                        explanation=change_res.explanation,
                    )

                detected_changes.append(
                    {
                        "subject_code": s_code,
                        "status": change_res.status.value,
                        "delta_attended": change_res.delta_attended,
                        "delta_conducted": change_res.delta_conducted,
                        "explanation": change_res.explanation,
                    }
                )

            # 6. Log success in scrape_log
            completed_at_str = datetime.now(_TZ).isoformat()
            self._record_scrape_log(
                attempted_at_str, completed_at_str, "SUCCESS", new_snapshot.id, None
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"Portal sync completed successfully in {duration_ms:.1f}ms. Snapshot #{new_snapshot.id} saved."
            )

            return PortalSyncResult(
                success=True,
                snapshot_id=new_snapshot.id,
                records_count=len(db_records),
                changes_detected=detected_changes,
                duration_ms=duration_ms,
                timestamp=attempted_at,
            )

        except (
            PortalAuthError,
            PortalNavigationError,
            PortalScrapeError,
            PortalValidationError,
            CredentialNotFoundError,
            PortalError,
        ) as e:
            err_str = str(e)
            logger.error(f"Portal sync error: {err_str}")
            completed_at_str = datetime.now(_TZ).isoformat()
            self._record_scrape_log(
                attempted_at_str, completed_at_str, "FAILED", None, err_str
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PortalSyncResult(
                success=False,
                error_message=err_str,
                duration_ms=duration_ms,
                timestamp=attempted_at,
            )

        except Exception as e:
            err_str = f"Unexpected portal sync error: {type(e).__name__}: {e}"
            logger.exception(err_str)
            completed_at_str = datetime.now(_TZ).isoformat()
            self._record_scrape_log(
                attempted_at_str, completed_at_str, "FAILED", None, err_str
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PortalSyncResult(
                success=False,
                error_message=err_str,
                duration_ms=duration_ms,
                timestamp=attempted_at,
            )

    def _record_scrape_log(
        self,
        attempted_at: str,
        completed_at: Optional[str],
        status: str,
        snapshot_id: Optional[int],
        error_msg: Optional[str],
    ) -> None:
        """Helper to insert an audit record in scrape_log."""
        try:
            with self.repo.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO scrape_log (attempted_at, completed_at, status, snapshot_id, error_msg)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (attempted_at, completed_at, status, snapshot_id, error_msg),
                )
        except Exception as e:
            logger.error(f"Failed to record scrape_log entry: {e}")


# ── Global Singleton Accessor ─────────────────────────────────────────────────

_default_service: Optional[PortalAttendanceService] = None


def get_portal_service() -> PortalAttendanceService:
    """Return the singleton PortalAttendanceService."""
    global _default_service
    if _default_service is None:
        _default_service = PortalAttendanceService()
    return _default_service
