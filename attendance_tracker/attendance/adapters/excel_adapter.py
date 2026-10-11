"""
attendance/adapters/excel_adapter.py
────────────────────────────────────
Excel (.xlsx) import adapter for attendance snapshot ingestion.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo
import openpyxl

from attendance_tracker.attendance.adapters.base import (
    BaseAttendanceAdapter,
    IngestionPreview,
    IngestionRecord,
)
from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)
_TZ = ZoneInfo(config.timezone)


class ExcelAttendanceAdapter(BaseAttendanceAdapter):
    """
    Parses Excel (.xlsx) workbooks into normalized AttendanceSnapshot data.
    """

    def parse_preview(
        self,
        input_data: bytes | io.BytesIO | str,
        timestamp: Optional[datetime | str] = None,
    ) -> IngestionPreview:
        """Parse Excel workbook bytes and validate all subject rows."""
        if timestamp is None:
            ts_str = datetime.now(_TZ).isoformat()
        elif isinstance(timestamp, datetime):
            ts_str = timestamp.isoformat()
        else:
            ts_str = str(timestamp).strip()

        if isinstance(input_data, bytes):
            file_stream = io.BytesIO(input_data)
        elif isinstance(input_data, str):
            with open(input_data, "rb") as f:
                file_stream = io.BytesIO(f.read())
        else:
            file_stream = input_data

        errors: list[str] = []
        warnings: list[str] = []
        records: list[IngestionRecord] = []
        seen_codes: set[str] = set()

        try:
            wb = openpyxl.load_workbook(file_stream, data_only=True)
            sheet = wb.active
            if sheet is None:
                raise ValueError("Workbook has no active worksheet.")
        except Exception as e:
            return IngestionPreview(
                source="excel",
                timestamp=ts_str,
                records_count=0,
                total_attended=0,
                total_conducted=0,
                overall_percentage=0.0,
                records=[],
                errors=[f"Failed to open Excel workbook: {e}"],
                warnings=[],
                is_valid=False,
            )

        rows = []
        for r in sheet.iter_rows(values_only=True):
            if r and any(cell is not None for cell in r):
                rows.append([str(c).strip() if c is not None else "" for c in r])

        if not rows:
            return IngestionPreview(
                source="excel",
                timestamp=ts_str,
                records_count=0,
                total_attended=0,
                total_conducted=0,
                overall_percentage=0.0,
                records=[],
                errors=["Excel sheet is empty."],
                warnings=[],
                is_valid=False,
            )

        # Detect header row
        header = [col.lower() for col in rows[0]]
        has_header = False
        col_subj = -1
        col_att = -1
        col_cond = -1

        for idx, col in enumerate(header):
            if any(k in col for k in ("subject", "course", "code")):
                col_subj = idx
                has_header = True
            elif any(k in col for k in ("attended", "present", "att")):
                col_att = idx
                has_header = True
            elif any(k in col for k in ("conducted", "total", "held", "classes")):
                col_cond = idx
                has_header = True

        data_rows = rows[1:] if has_header else rows

        if col_subj == -1 or col_att == -1 or col_cond == -1:
            col_subj = 0
            col_att = 1
            col_cond = 2

        for row_idx, row in enumerate(data_rows, start=2 if has_header else 1):
            if len(row) < 3:
                continue

            raw_subj = row[col_subj].strip()
            raw_att = row[col_att].strip()
            raw_cond = row[col_cond].strip()

            if not raw_subj or (not raw_att and not raw_cond):
                continue

            subj = self.resolver.resolve(raw_subj)
            if subj is None:
                errors.append(f"Row {row_idx}: Unknown subject '{raw_subj}'. Must match one of 9 canonical subjects.")
                continue

            if subj.code in seen_codes:
                warnings.append(f"Row {row_idx}: Duplicate entry for {subj.short_name} ({subj.code}). Overwriting earlier row.")
                records = [r for r in records if r.subject_code != subj.code]

            try:
                # Handle numeric float strings like "22.0"
                att_val = int(float(raw_att))
            except ValueError:
                errors.append(f"Row {row_idx}: Attended classes '{raw_att}' must be an integer.")
                continue

            try:
                cond_val = int(float(raw_cond))
            except ValueError:
                errors.append(f"Row {row_idx}: Conducted classes '{raw_cond}' must be an integer.")
                continue

            if att_val < 0 or cond_val < 0:
                errors.append(f"Row {row_idx}: Counts cannot be negative (got {att_val}/{cond_val}).")
                continue

            if att_val > cond_val:
                errors.append(f"Row {row_idx}: Attended ({att_val}) cannot exceed conducted ({cond_val}) for {subj.short_name}.")
                continue

            pct = current_percentage(att_val, cond_val)
            rec = IngestionRecord(
                subject_code=subj.code,
                short_name=subj.short_name,
                full_name=subj.full_name,
                attended=att_val,
                conducted=cond_val,
                percentage=pct,
                raw_identifier=raw_subj,
            )
            records.append(rec)
            seen_codes.add(subj.code)

        total_att = sum(r.attended for r in records)
        total_cond = sum(r.conducted for r in records)
        overall_pct = current_percentage(total_att, total_cond)
        is_valid = len(errors) == 0 and len(records) > 0

        return IngestionPreview(
            source="excel",
            timestamp=ts_str,
            records_count=len(records),
            total_attended=total_att,
            total_conducted=total_cond,
            overall_percentage=overall_pct,
            records=records,
            errors=errors,
            warnings=warnings,
            is_valid=is_valid,
        )
