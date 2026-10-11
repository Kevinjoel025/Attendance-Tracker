"""
web/routes/sync.py
──────────────────
Portal synchronization API endpoint.
Invokes the existing single-attempt Playwright portal synchronization service safely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from attendance_tracker.attendance.comparator import get_comparison_engine
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.portal.models import PortalSyncResult
from attendance_tracker.web.state import get_app_services

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Portal Synchronization"])


class SyncResponse(BaseModel):
    success: bool
    status: str
    changed: bool = False
    snapshot_id: Optional[int] = None
    records_count: int = 0
    changes_count: int = 0
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    comparison_summary: Optional[str] = None


@router.post("/sync", response_model=SyncResponse)
def sync_portal_attendance() -> SyncResponse:
    """
    Trigger real portal attendance retrieval and database update.
    Never exposes credentials or session tokens.
    """
    services = get_app_services()
    logger.info("Web API initiated portal synchronization...")

    try:
        res: PortalSyncResult = services.portal_service.sync_attendance(headless=True)
        if not res.success:
            err = res.error_message or "Portal synchronization failed."
            logger.warning(f"Portal sync reported failure: {err}")
            return SyncResponse(
                success=False,
                status="FAILED",
                changed=False,
                snapshot_id=None,
                records_count=0,
                changes_count=0,
                duration_ms=round(res.duration_ms, 2),
                error_message=err,
                message=f"Sync failed: {err}",
                timestamp=res.timestamp.isoformat() if res.timestamp else None,
            )

        # Check comparison with previous snapshot
        comp_engine = get_comparison_engine(
            repository=services.repository,
            subject_resolver=services.subject_resolver,
        )
        comp_result = comp_engine.compare_latest_with_previous()
        has_changes = comp_result.has_changes

        if has_changes:
            msg = f"Portal synced successfully. Attendance updated with {comp_result.changed_subjects_count} subject change(s)."
        else:
            msg = "Portal synced successfully. No attendance changes detected."

        return SyncResponse(
            success=True,
            status="SUCCESS",
            changed=has_changes,
            snapshot_id=res.snapshot_id,
            records_count=res.records_count,
            changes_count=comp_result.changed_subjects_count,
            duration_ms=round(res.duration_ms, 2),
            error_message=None,
            message=msg,
            timestamp=res.timestamp.isoformat() if res.timestamp else None,
            comparison_summary=comp_result.summary_text if has_changes else None,
        )

    except Exception as e:
        logger.error(f"Unexpected error during portal sync: {e}", exc_info=True)
        return SyncResponse(
            success=False,
            status="FAILED",
            changed=False,
            snapshot_id=None,
            records_count=0,
            changes_count=0,
            duration_ms=0.0,
            error_message="Portal synchronization encountered an unexpected error. Please verify credentials or retry shortly.",
            message="Sync failed due to an unexpected server error.",
            timestamp=None,
        )
