"""
web/routes/import_data.py
─────────────────────────
API endpoints for universal attendance and holiday/calendar ingestion (CSV text, CSV file, Excel file).
Includes multi-stage validation, previewing, and database committing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from attendance_tracker.attendance.adapters.base import IngestionPreview, IngestionRecord
from attendance_tracker.attendance.adapters.calendar_adapter import (
    HolidayCalendarAdapter,
    HolidayImportPreview,
)
from attendance_tracker.attendance.adapters.csv_adapter import CSVAttendanceAdapter
from attendance_tracker.attendance.adapters.excel_adapter import ExcelAttendanceAdapter
from attendance_tracker.attendance.adapters.screenshot_adapter import ScreenshotAttendanceAdapter
from attendance_tracker.attendance.comparator import get_comparison_engine
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api/import", tags=["Data Ingestion & Imports"])


# ── Request Schemas ───────────────────────────────────────────────────────────

class CSVTextPreviewRequest(BaseModel):
    csv_text: str = Field(..., min_length=3, description="Raw CSV attendance text")


class CommitRecordItem(BaseModel):
    subject_code: str
    attended_classes: int
    total_classes: int


class CommitRequest(BaseModel):
    source: str = Field(default="manual", description="Source identifier (csv, excel, manual, screenshot)")
    records: List[CommitRecordItem] = Field(..., min_length=1, description="List of subject attendance records to commit")


class CalendarPreviewTextRequest(BaseModel):
    csv_text: str = Field(..., min_length=3, description="Raw CSV holiday/calendar text")


class CalendarCommitItem(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    override_type: str = Field(..., description="HOLIDAY or WORKING_DAY")
    name: Optional[str] = Field(default="", description="Name or reason")


class CalendarCommitRequest(BaseModel):
    source: str = Field(default="csv", description="Source identifier (csv, excel, manual)")
    records: List[CalendarCommitItem] = Field(..., min_length=1, description="List of calendar exceptions to commit")


# ── Attendance Ingestion Endpoints ───────────────────────────────────────────

@router.post("/preview-text")
def preview_csv_text(req: CSVTextPreviewRequest) -> Dict[str, Any]:
    """Parse raw CSV text and generate a structured attendance ingestion preview."""
    services = get_app_services()
    adapter = CSVAttendanceAdapter(
        repository=services.repository,
        subject_resolver=services.subject_resolver,
    )
    preview = adapter.parse_text(req.csv_text)
    return preview.to_dict()


@router.post("/upload")
async def upload_attendance_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a .csv or .xlsx attendance file and generate a structured ingestion preview."""
    filename = file.filename or "unknown"
    content = await file.read()

    services = get_app_services()

    if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
        adapter = ExcelAttendanceAdapter(
            repository=services.repository,
            subject_resolver=services.subject_resolver,
        )
        preview = adapter.parse_bytes(content, filename=filename)
    elif filename.lower().endswith(".csv") or filename.lower().endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        adapter = CSVAttendanceAdapter(
            repository=services.repository,
            subject_resolver=services.subject_resolver,
        )
        preview = adapter.parse_text(text)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .csv or .xlsx attendance file."
        )

    return preview.to_dict()


@router.post("/commit")
def commit_imported_attendance(req: CommitRequest) -> Dict[str, Any]:
    """Commit a previewed attendance snapshot into the database."""
    services = get_app_services()
    repo = services.repository
    resolver = services.subject_resolver

    valid_sources = ("csv", "excel", "manual", "screenshot", "portal", "seed")
    source = req.source.lower().strip()
    if source not in valid_sources:
        source = "manual"

    # Validate records
    subject_tuples = []
    for item in req.records:
        sub = resolver.resolve(item.subject_code)
        if sub is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown subject code '{item.subject_code}'."
            )
        if item.attended_classes < 0 or item.total_classes < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Classes cannot be negative for {sub.short_name}."
            )
        if item.attended_classes > item.total_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Attended classes ({item.attended_classes}) cannot exceed conducted classes ({item.total_classes}) for {sub.short_name}."
            )
        subject_tuples.append((sub.code, item.attended_classes, item.total_classes))

    # Save to database
    snapshot_id = repo.save_snapshot(
        subject_records=subject_tuples,
        source=source,
    )

    # Compare with previous snapshot
    comp_engine = get_comparison_engine(repository=repo, subject_resolver=resolver)
    comp_result = comp_engine.compare_latest_with_previous()

    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "source": source,
        "records_count": len(subject_tuples),
        "message": f"Successfully committed snapshot #{snapshot_id} from {source}.",
        "has_changes": comp_result.has_changes,
        "comparison_summary": comp_result.summary_text,
    }


# ── Holiday / Calendar Ingestion Endpoints ────────────────────────────────────

@router.post("/calendar/preview-text")
def preview_calendar_csv_text(req: CalendarPreviewTextRequest) -> Dict[str, Any]:
    """Parse raw CSV text containing holiday / calendar overrides and generate a preview."""
    adapter = HolidayCalendarAdapter()
    preview = adapter.parse_csv_text(req.csv_text)
    return preview.to_dict()


@router.post("/calendar/upload")
async def upload_calendar_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a .csv or .xlsx file containing holidays / working day overrides and preview."""
    filename = file.filename or "unknown"
    content = await file.read()
    adapter = HolidayCalendarAdapter()

    if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
        preview = adapter.parse_excel_bytes(content, filename=filename)
    elif filename.lower().endswith(".csv") or filename.lower().endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        preview = adapter.parse_csv_text(text, source="csv")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .csv or .xlsx holiday/calendar file."
        )

    return preview.to_dict()


@router.post("/calendar/commit")
def commit_imported_calendar(req: CalendarCommitRequest) -> Dict[str, Any]:
    """Commit validated holiday / working-day overrides into SQLite calendar_overrides table."""
    services = get_app_services()
    repo = services.repository

    committed_count = 0
    saved_records = []

    for item in req.records:
        date_str = item.date.strip()
        type_norm = item.override_type.strip().upper()
        if type_norm not in ("HOLIDAY", "WORKING_DAY"):
            type_norm = "HOLIDAY"

        rec = repo.add_calendar_override(
            date_str=date_str,
            override_type=type_norm,
            name=item.name or "",
        )
        saved_records.append({
            "id": rec.id,
            "date": rec.date,
            "override_type": rec.override_type,
            "name": rec.name,
        })
        committed_count += 1

    return {
        "success": True,
        "imported_count": committed_count,
        "source": req.source,
        "message": f"Successfully imported {committed_count} holiday/calendar exception(s).",
        "overrides": saved_records,
    }
