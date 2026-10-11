"""
web/routes/calendar.py
──────────────────────
API endpoints for Timetable, Academic Calendar, Overrides, and Last Working Day.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from attendance_tracker.calendar.service import CalendarService, get_calendar_service
from attendance_tracker.core.config import config
from attendance_tracker.timetable.service import TimetableService, get_timetable_service
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api", tags=["Calendar & Timetable"])

_TZ = ZoneInfo(config.timezone)


# ── Schemas ───────────────────────────────────────────────────────────────────

class OverrideCreateRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    override_type: str = Field(..., description="HOLIDAY or WORKING_DAY")
    reason: Optional[str] = Field(default="", description="Reason for override")
    name: Optional[str] = Field(default="", description="Event name / reason")
    note: Optional[str] = Field(default="", description="Optional note")
    old_date: Optional[str] = Field(default=None, description="Original date if editing date")


class CalendarEntryRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    override_type: str = Field(..., description="HOLIDAY or WORKING_DAY")
    name: Optional[str] = Field(default="", description="Event name / reason")
    reason: Optional[str] = Field(default="", description="Reason / note")
    note: Optional[str] = Field(default="", description="Optional note")
    old_date: Optional[str] = Field(default=None, description="Original date if editing date")


class MidTerm1Request(BaseModel):
    start: str = Field(..., description="Start date in YYYY-MM-DD format")
    end: str = Field(..., description="End date in YYYY-MM-DD format")


class LastWorkingDayRequest(BaseModel):
    last_working_day: str = Field(..., description="Date in YYYY-MM-DD format")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/calendar/entries")
def get_unified_calendar_entries() -> List[Dict[str, Any]]:
    """Return the authoritative single unified list of all calendar entries."""
    services = get_app_services()
    return services.calendar_service.get_unified_calendar_entries()


@router.post("/calendar/entries")
def save_calendar_entry(req: CalendarEntryRequest) -> Dict[str, Any]:
    """Add or edit a calendar entry/override."""
    date_str = req.date.strip()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    override_type = req.override_type.strip().upper()
    if override_type not in ("HOLIDAY", "WORKING_DAY"):
        raise HTTPException(
            status_code=400,
            detail="override_type must be either 'HOLIDAY' or 'WORKING_DAY'."
        )

    entry_name = (req.name or req.reason or req.note or "").strip()
    if not entry_name:
        entry_name = "User Holiday Override" if override_type == "HOLIDAY" else "User Working Day Override"
    services = get_app_services()
    repo = services.repository

    # If old_date was specified and differs from new date, remove the old override
    if req.old_date:
        old_d_str = req.old_date.strip()
        if old_d_str and old_d_str != date_str:
            repo.delete_calendar_override(old_d_str)

    record = repo.add_calendar_override(
        date_str=date_str,
        override_type=override_type,
        name=entry_name,
    )

    return {
        "success": True,
        "message": f"Calendar entry for {date_str} saved as {override_type}.",
        "entry": {
            "id": f"override_{record.id}",
            "db_id": record.id,
            "date": record.date,
            "override_type": record.override_type,
            "name": record.name,
            "created_at": record.created_at,
        },
    }


@router.delete("/calendar/entries/{date_str}")
def delete_calendar_entry(date_str: str) -> Dict[str, Any]:
    """Delete a calendar entry / override and restore the underlying default rule."""
    clean_date = date_str.strip()
    services = get_app_services()
    repo = services.repository
    cal_svc = services.calendar_service

    # Check if a user override exists in SQLite
    override = repo.get_calendar_override(clean_date)
    if override is not None:
        repo.delete_calendar_override(clean_date)
        return {
            "success": True,
            "message": f"Calendar override for {clean_date} deleted. Default rule restored.",
        }

    # If no override, check if it's an official holiday
    try:
        d = datetime.strptime(clean_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    status = cal_svc.get_calendar_status(d)
    if status.is_holiday:
        # Create a working-day override to suppress the official holiday
        repo.add_calendar_override(
            date_str=clean_date,
            override_type="WORKING_DAY",
            name="Official Holiday Removed / Working Day",
        )
        return {
            "success": True,
            "message": f"Official holiday {clean_date} removed. Working day restored.",
        }

    return {
        "success": True,
        "message": f"Calendar entry for {clean_date} cleared.",
    }


@router.get("/calendar/midterm")
def get_mid_term_1() -> Dict[str, Any]:
    """Get the currently configured Mid-Term 1 date range."""
    services = get_app_services()
    s_date, e_date = services.calendar_service.get_mid_term_1()
    return {
        "mid_term_1": {
            "start": s_date.isoformat(),
            "end": e_date.isoformat(),
        }
    }


@router.post("/calendar/midterm")
def save_mid_term_1(req: MidTerm1Request) -> Dict[str, Any]:
    """Update the Mid-Term 1 start and end dates."""
    start_str = req.start.strip()
    end_str = req.end.strip()

    try:
        s_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start date format. Use YYYY-MM-DD.")

    try:
        e_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid end date format. Use YYYY-MM-DD.")

    if e_date < s_date:
        raise HTTPException(status_code=400, detail="Mid-Term 1 end date cannot be before start date.")

    services = get_app_services()
    services.calendar_service.set_mid_term_1(s_date, e_date)

    return {
        "success": True,
        "mid_term_1": {
            "start": start_str,
            "end": end_str,
        },
        "message": f"Mid-Term 1 examination window updated to {start_str} → {end_str}.",
    }


@router.get("/calendar/status")
def get_calendar_day_status(
    target_date: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD date")
) -> Dict[str, Any]:
    """Get holiday, working day, and scheduled class status for a specific date."""
    services = get_app_services()
    cal_svc = services.calendar_service
    tt_svc = services.timetable_service
    resolver = services.subject_resolver

    if target_date:
        try:
            d = datetime.strptime(target_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        d = datetime.now(_TZ).date()

    day_name = d.strftime("%A")
    day_status = cal_svc.get_day_status(d)
    classes = tt_svc.get_classes_for_date(d)

    formatted_classes = []
    for c in classes:
        sub = resolver.resolve(c.subject_code)
        formatted_classes.append({
            "period_num": c.period_num,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "code": c.subject_code,
            "short_name": sub.short_name if sub else c.subject,
            "full_name": sub.full_name if sub else c.subject,
            "is_lab": ("LAB" in (sub.short_name if sub else c.subject).upper()),
            "is_attendance_bearing": c.is_attendance_bearing,
        })

    return {
        "date": d.isoformat(),
        "day_name": day_name,
        "is_holiday": day_status.is_holiday,
        "holiday_name": day_status.holiday_name,
        "is_override": day_status.is_override,
        "override_type": day_status.override_type,
        "classes_count": len(formatted_classes),
        "classes": formatted_classes,
    }


@router.get("/calendar/overrides")
def list_calendar_overrides() -> List[Dict[str, Any]]:
    """List all user-defined holiday and working-day calendar overrides."""
    services = get_app_services()
    overrides = services.repository.list_calendar_overrides()
    return [
        {
            "id": o.id,
            "date": o.date,
            "override_type": o.override_type,
            "reason": o.name,
            "created_at": o.created_at,
        }
        for o in overrides
    ]


@router.post("/calendar/overrides")
def set_calendar_override(req: OverrideCreateRequest) -> Dict[str, Any]:
    """Add or update a user-defined calendar override (HOLIDAY or WORKING_DAY)."""
    date_str = req.date.strip()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    override_type = req.override_type.strip().upper()
    if override_type not in ("HOLIDAY", "WORKING_DAY"):
        raise HTTPException(
            status_code=400,
            detail="override_type must be either 'HOLIDAY' or 'WORKING_DAY'."
        )

    entry_name = (req.name or req.reason or req.note or "").strip()
    services = get_app_services()
    repo = services.repository

    if req.old_date:
        old_d_str = req.old_date.strip()
        if old_d_str and old_d_str != date_str:
            repo.delete_calendar_override(old_d_str)

    record = repo.add_calendar_override(
        date_str=date_str,
        override_type=override_type,
        name=entry_name,
    )
    return {
        "success": True,
        "message": f"Calendar override set for {date_str} as {override_type}.",
        "override": {
            "id": record.id,
            "date": record.date,
            "override_type": record.override_type,
            "reason": record.name,
            "created_at": record.created_at,
        },
    }


@router.delete("/calendar/overrides/{date_str}")
def delete_calendar_override(date_str: str) -> Dict[str, Any]:
    """Delete a user-defined calendar override for a date."""
    services = get_app_services()
    deleted = services.repository.delete_calendar_override(date_str.strip())
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No override found for date {date_str}.")
    return {
        "success": True,
        "message": f"Calendar override for {date_str} deleted successfully.",
    }


@router.get("/calendar/last-working-day")
def get_last_working_day() -> Dict[str, Any]:
    """Get the currently configured semester last working day / end date."""
    services = get_app_services()
    last_day = services.calendar_service.get_last_working_day()
    return {
        "last_working_day": last_day.isoformat(),
        "is_configured": True,
    }


@router.post("/calendar/last-working-day")
def set_last_working_day(req: LastWorkingDayRequest) -> Dict[str, Any]:
    """Update the semester last working day / end date."""
    date_str = req.last_working_day.strip()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    today = datetime.now(_TZ).date()
    if d < today:
        raise HTTPException(status_code=400, detail="Last working day cannot be in the past.")

    services = get_app_services()
    services.repository.set_last_working_day(date_str)
    return {
        "success": True,
        "last_working_day": date_str,
        "message": f"Semester last working day updated to {date_str}.",
    }


@router.get("/calendar/holidays")
def list_semester_holidays() -> Dict[str, Any]:
    """List official semester holidays, examination periods, and date boundaries."""
    services = get_app_services()
    cal_svc = services.calendar_service

    named_holidays = cal_svc.list_named_holidays()
    last_working_day = cal_svc.get_last_working_day().isoformat()
    s_date, e_date = cal_svc.get_mid_term_1()

    return {
        "semester_start": "2026-07-16",
        "semester_end": last_working_day,
        "official_holidays": named_holidays,
        "exceptions": cal_svc.list_exceptions(),
        "mid_term_1": {
            "start": s_date.isoformat(),
            "end": e_date.isoformat(),
            "type": "EXAM_HOLIDAY",
        },
    }


@router.get("/timetable/weekly")
def get_weekly_timetable() -> Dict[str, Any]:
    """Get the full Monday-Saturday weekly timetable schedule."""
    services = get_app_services()
    tt_svc = services.timetable_service
    resolver = services.subject_resolver

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekly_schedule = {}

    for day in days:
        slots = tt_svc.get_schedule_for_day(day)
        formatted_slots = []
        for s in slots:
            sub = resolver.resolve(s.subject_code)
            formatted_slots.append({
                "period_num": s.period_num,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "subject_code": s.subject_code,
                "short_name": sub.short_name if sub else s.subject,
                "full_name": sub.full_name if sub else s.subject,
                "is_lab": ("LAB" in (sub.short_name if sub else s.subject).upper()),
                "is_attendance_bearing": s.is_attendance_bearing,
                "faculty": getattr(s, "faculty", ""),
                "room": getattr(s, "room", ""),
            })
        weekly_schedule[day] = formatted_slots

    return {
        "days": days,
        "schedule": weekly_schedule,
    }
