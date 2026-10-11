"""
web/routes/updates.py
─────────────────────
API endpoint for attendance change history and timetable schedule context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from fastapi import APIRouter
from pydantic import BaseModel

from attendance_tracker.core.config import config
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api", tags=["Updates & Schedule"])
_TZ = ZoneInfo(config.timezone)


class ChangeItem(BaseModel):
    id: int
    detected_at: str
    timestamp: str
    subject_code: str
    subject_name: Optional[str] = None
    prev_attended: int
    prev_conducted: int
    curr_attended: int
    curr_conducted: int
    delta_attended: int
    delta_conducted: int
    status: str
    explanation: Optional[str] = None
    description: Optional[str] = None


class ScheduledClassItem(BaseModel):
    period_num: int
    period: str
    start_time: str
    end_time: str
    time: str
    subject: str
    subject_name: str
    subject_code: str
    is_attendance_bearing: bool


class UpdatesResponse(BaseModel):
    has_changes: bool
    recent_changes: List[ChangeItem]
    recent_logs: List[ChangeItem]
    today_date: str
    today_day: str
    scheduled_classes: List[ScheduledClassItem]
    today_schedule: List[ScheduledClassItem]
    snapshots_count: int
    latest_snapshot_at: Optional[str] = None


@router.get("/updates", response_model=UpdatesResponse)
def get_attendance_updates() -> UpdatesResponse:
    """Fetch detected attendance changes, audit records, and today's timetable schedule."""
    services = get_app_services()
    repo = services.repository

    now = datetime.now(_TZ)
    today_date_str = now.strftime("%Y-%m-%d")
    today_day_str = now.strftime("%A")

    # 1. Fetch recent changes from attendance_changes table
    with repo.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.detected_at, c.subject_code, s.short_name, s.full_name,
                   c.prev_attended, c.prev_conducted, c.curr_attended, c.curr_conducted,
                   c.delta_attended, c.delta_conducted, c.status, c.explanation
            FROM attendance_changes c
            LEFT JOIN subjects s ON c.subject_code = s.code
            ORDER BY c.detected_at DESC, c.id DESC
            LIMIT 20;
            """
        ).fetchall()

    changes: list[ChangeItem] = []
    for r in rows:
        subj_display = f"{r['full_name']} ({r['short_name']})" if r["short_name"] else r["subject_code"]
        changes.append(
            ChangeItem(
                id=r["id"],
                detected_at=r["detected_at"],
                timestamp=r["detected_at"],
                subject_code=r["subject_code"],
                subject_name=subj_display,
                prev_attended=r["prev_attended"],
                prev_conducted=r["prev_conducted"],
                curr_attended=r["curr_attended"],
                curr_conducted=r["curr_conducted"],
                delta_attended=r["delta_attended"],
                delta_conducted=r["delta_conducted"],
                status=r["status"],
                explanation=r["explanation"],
                description=r["explanation"] or f"+{r['delta_attended']}/+{r['delta_conducted']} ({r['status']})",
            )
        )

    # 2. Fetch today's scheduled classes from timetable
    classes_today = services.timetable_service.get_classes_for_date(now.date())
    scheduled_items: list[ScheduledClassItem] = []
    for c in classes_today:
        time_str = f"{c.start_time} - {c.end_time}"
        scheduled_items.append(
            ScheduledClassItem(
                period_num=c.period_num,
                period=f"Period {c.period_num}",
                start_time=c.start_time,
                end_time=c.end_time,
                time=time_str,
                subject=c.subject,
                subject_name=c.subject,
                subject_code=c.subject_code,
                is_attendance_bearing=c.is_attendance_bearing,
            )
        )

    # 3. Snapshot count
    snapshots = repo.list_snapshots(limit=10)
    latest_snap = snapshots[0] if snapshots else None

    return UpdatesResponse(
        has_changes=len(changes) > 0,
        recent_changes=changes,
        recent_logs=changes,
        today_date=today_date_str,
        today_day=today_day_str,
        scheduled_classes=scheduled_items,
        today_schedule=scheduled_items,
        snapshots_count=len(snapshots),
        latest_snapshot_at=latest_snap.scraped_at if latest_snap else None,
    )
