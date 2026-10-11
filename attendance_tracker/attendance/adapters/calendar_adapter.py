"""
attendance/adapters/calendar_adapter.py
───────────────────────────────────────
Universal holiday and calendar exception ingestion adapter.
Supports CSV text, CSV files, and Excel (.xlsx) spreadsheets.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
import io
import re
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)
_TZ = ZoneInfo(config.timezone)


@dataclass(frozen=True)
class HolidayImportRecord:
    """A validated holiday or calendar override entry."""
    date: str  # ISO YYYY-MM-DD
    override_type: str  # "HOLIDAY" | "WORKING_DAY"
    name: str  # Name or reason
    note: str
    raw_date: str
    is_valid: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "override_type": self.override_type,
            "name": self.name,
            "note": self.note,
            "raw_date": self.raw_date,
            "is_valid": self.is_valid,
            "error": self.error,
        }


@dataclass(frozen=True)
class HolidayImportPreview:
    """Pre-commit validation preview of imported holiday/calendar data."""
    source: str
    timestamp: str
    total_rows: int
    valid_count: int
    invalid_count: int
    records: list[HolidayImportRecord]
    errors: list[str]
    warnings: list[str]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "records": [r.to_dict() for r in self.records],
            "errors": self.errors,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }


def _parse_and_normalize_date(raw_date_str: str) -> Optional[str]:
    """
    Parse a raw date string in multiple common formats and return ISO 'YYYY-MM-DD'.
    Returns None if date cannot be parsed.
    """
    if not raw_date_str:
        return None

    cleaned = str(raw_date_str).strip()
    # Check ISO format first YYYY-MM-DD
    iso_match = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", cleaned)
    if iso_match:
        y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            return date(y, m, d).isoformat()
        except ValueError:
            return None

    # Check DD-MM-YYYY or DD/MM/YYYY or MM-DD-YYYY or MM/DD/YYYY
    dmy_match = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", cleaned)
    if dmy_match:
        p1, p2, y = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
        # Try DD/MM/YYYY first (standard in India)
        try:
            return date(y, p2, p1).isoformat()
        except ValueError:
            pass
        # Try MM/DD/YYYY next (US format)
        try:
            return date(y, p1, p2).isoformat()
        except ValueError:
            pass

    # Fallback to datetime.strptime for textual and two-digit year date formats
    for fmt in (
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d-%b-%Y",
        "%d %B %Y", "%d-%b-%y", "%d/%m/%y", "%d-%m-%y"
    ):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            pass

    return None


def _normalize_override_type(raw_type: str) -> Optional[str]:
    """
    Normalize arbitrary type strings into either 'HOLIDAY' or 'WORKING_DAY'.
    """
    if not raw_type:
        return "HOLIDAY"

    t = str(raw_type).strip().upper()
    # Clean whitespace and underscores
    t_clean = re.sub(r"[\s_]+", " ", t)

    if t_clean in ("HOLIDAY", "OFF", "VACATION", "LEAVE", "BREAK", "FEST"):
        return "HOLIDAY"
    if t_clean in ("WORKING DAY", "WORKING", "WORK", "WORKINGDAY", "CLASS", "COMPENSATORY"):
        return "WORKING_DAY"

    if "WORK" in t_clean:
        return "WORKING_DAY"
    if "HOLI" in t_clean or "OFF" in t_clean or "LEAVE" in t_clean:
        return "HOLIDAY"

    return None


class HolidayCalendarAdapter:
    """
    Parses and validates CSV or Excel calendar exceptions and holiday lists.
    """

    def parse_csv_text(self, csv_text: str, source: str = "csv_text") -> HolidayImportPreview:
        """Parse raw CSV text containing holiday/calendar records."""
        now_str = datetime.now(_TZ).isoformat()
        if not csv_text or not csv_text.strip():
            return HolidayImportPreview(
                source=source,
                timestamp=now_str,
                total_rows=0,
                valid_count=0,
                invalid_count=0,
                records=[],
                errors=["CSV content is empty."],
                warnings=[],
                is_valid=False,
            )

        stream = io.StringIO(csv_text.strip())
        reader = csv.reader(stream)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not rows:
            return HolidayImportPreview(
                source=source,
                timestamp=now_str,
                total_rows=0,
                valid_count=0,
                invalid_count=0,
                records=[],
                errors=["No data rows found in CSV."],
                warnings=[],
                is_valid=False,
            )

        return self._process_raw_rows(rows, source=source, timestamp=now_str)

    def parse_excel_bytes(self, data: bytes, filename: str = "calendar.xlsx") -> HolidayImportPreview:
        """Parse binary Excel (.xlsx) file containing holiday/calendar records."""
        now_str = datetime.now(_TZ).isoformat()
        try:
            import openpyxl
        except ImportError:
            return HolidayImportPreview(
                source="excel",
                timestamp=now_str,
                total_rows=0,
                valid_count=0,
                invalid_count=0,
                records=[],
                errors=["openpyxl is not installed for Excel import."],
                warnings=[],
                is_valid=False,
            )

        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            sheet = wb.active
            if sheet is None:
                return HolidayImportPreview(
                    source="excel",
                    timestamp=now_str,
                    total_rows=0,
                    valid_count=0,
                    invalid_count=0,
                    records=[],
                    errors=["No active sheet found in Excel workbook."],
                    warnings=[],
                    is_valid=False,
                )

            rows: list[list[str]] = []
            for row in sheet.iter_rows(values_only=True):
                if row and any(cell is not None and str(cell).strip() for cell in row):
                    row_str = []
                    for cell in row:
                        if isinstance(cell, (datetime, date)):
                            row_str.append(cell.strftime("%Y-%m-%d"))
                        elif cell is None:
                            row_str.append("")
                        else:
                            row_str.append(str(cell).strip())
                    rows.append(row_str)

            if not rows:
                return HolidayImportPreview(
                    source="excel",
                    timestamp=now_str,
                    total_rows=0,
                    valid_count=0,
                    invalid_count=0,
                    records=[],
                    errors=["Excel file contains no data."],
                    warnings=[],
                    is_valid=False,
                )

            return self._process_raw_rows(rows, source="excel", timestamp=now_str)

        except Exception as e:
            logger.error(f"Failed to parse Excel calendar file: {e}", exc_info=True)
            return HolidayImportPreview(
                source="excel",
                timestamp=now_str,
                total_rows=0,
                valid_count=0,
                invalid_count=0,
                records=[],
                errors=[f"Failed to read Excel file: {e}"],
                warnings=[],
                is_valid=False,
            )

    def _process_raw_rows(
        self,
        rows: list[list[str]],
        source: str,
        timestamp: str,
    ) -> HolidayImportPreview:
        """Process extracted rows, detect headers, normalize values, and validate."""
        errors: list[str] = []
        warnings: list[str] = []
        records: list[HolidayImportRecord] = []

        header = [col.strip().lower() for col in rows[0]]
        has_header = False
        col_date = -1
        col_type = -1
        col_name = -1
        col_note = -1

        for idx, col in enumerate(header):
            if any(k in col for k in ("date", "day")):
                col_date = idx
                has_header = True
            elif any(k in col for k in ("type", "override", "status", "category")):
                col_type = idx
                has_header = True
            elif any(k in col for k in ("name", "holiday", "event", "title", "reason", "description")):
                if col_name == -1:
                    col_name = idx
                    has_header = True
                elif col_note == -1:
                    col_note = idx
            elif any(k in col for k in ("note", "remarks", "comment")):
                col_note = idx
                has_header = True

        data_rows = rows[1:] if has_header else rows

        # Positional fallback if headers not found
        if col_date == -1:
            col_date = 0
        if col_type == -1:
            col_type = 1 if len(rows[0]) > 1 else -1
        if col_name == -1:
            col_name = 2 if len(rows[0]) > 2 else (1 if col_type == -1 and len(rows[0]) > 1 else -1)

        seen_dates: dict[str, str] = {}  # date_str -> override_type

        for row_idx, row in enumerate(data_rows, start=2 if has_header else 1):
            if not row or not any(cell.strip() for cell in row):
                continue

            raw_d = row[col_date].strip() if col_date < len(row) else ""
            raw_t = row[col_type].strip() if (col_type != -1 and col_type < len(row)) else "HOLIDAY"
            raw_n = row[col_name].strip() if (col_name != -1 and col_name < len(row)) else ""
            raw_note = row[col_note].strip() if (col_note != -1 and col_note < len(row)) else ""

            # Check if raw_t is actually a name if only 2 columns provided (e.g. Date, Name)
            if col_type != -1 and col_name == -1:
                norm_t = _normalize_override_type(raw_t)
                if norm_t is None:
                    # User provided Date, Name
                    raw_n = raw_t
                    raw_t = "HOLIDAY"

            parsed_d = _parse_and_normalize_date(raw_d)
            normalized_t = _normalize_override_type(raw_t) or "HOLIDAY"

            row_error = None
            if not parsed_d:
                row_error = f"Row {row_idx}: Invalid date '{raw_d}'."
            elif normalized_t not in ("HOLIDAY", "WORKING_DAY"):
                row_error = f"Row {row_idx}: Unrecognized calendar type '{raw_t}'."

            name_val = raw_n or raw_note or ("Holiday" if normalized_t == "HOLIDAY" else "Working Day Override")
            note_val = raw_note

            if row_error:
                errors.append(row_error)
                records.append(
                    HolidayImportRecord(
                        date=parsed_d or "",
                        override_type=normalized_t,
                        name=name_val,
                        note=note_val,
                        raw_date=raw_d,
                        is_valid=False,
                        error=row_error,
                    )
                )
            else:
                # Check duplicate dates in import batch
                if parsed_d in seen_dates:
                    prev_t = seen_dates[parsed_d]
                    if prev_t != normalized_t:
                        warnings.append(
                            f"Row {row_idx}: Date {parsed_d} had conflicting types ({prev_t} vs {normalized_t}). Updated to {normalized_t}."
                        )
                    else:
                        warnings.append(f"Row {row_idx}: Duplicate date {parsed_d} overridden.")
                seen_dates[parsed_d] = normalized_t

                records.append(
                    HolidayImportRecord(
                        date=parsed_d,
                        override_type=normalized_t,
                        name=name_val,
                        note=note_val,
                        raw_date=raw_d,
                        is_valid=True,
                        error=None,
                    )
                )

        valid_count = sum(1 for r in records if r.is_valid)
        invalid_count = len(records) - valid_count
        is_valid = valid_count > 0

        return HolidayImportPreview(
            source=source,
            timestamp=timestamp,
            total_rows=len(records),
            valid_count=valid_count,
            invalid_count=invalid_count,
            records=records,
            errors=errors,
            warnings=warnings,
            is_valid=is_valid,
        )
