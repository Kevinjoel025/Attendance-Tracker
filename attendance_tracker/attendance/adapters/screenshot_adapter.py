"""
attendance/adapters/screenshot_adapter.py
──────────────────────────────────────────
Screenshot / manual dictionary adapter for attendance snapshot ingestion.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

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


class ScreenshotAttendanceAdapter(BaseAttendanceAdapter):
    """
    Ingests attendance extracted from screenshots or direct dictionary payloads.
    """

    def parse_preview(
        self,
        input_data: list[dict[str, Any]] | dict[str, Any],
        timestamp: Optional[datetime | str] = None,
    ) -> IngestionPreview:
        """Parse structured dictionary/OCR extracted rows and validate all subjects."""
        if timestamp is None:
            ts_str = datetime.now(_TZ).isoformat()
        elif isinstance(timestamp, datetime):
            ts_str = timestamp.isoformat()
        else:
            ts_str = str(timestamp).strip()

        errors: list[str] = []
        warnings: list[str] = []
        records: list[IngestionRecord] = []
        seen_codes: set[str] = set()

        raw_list = input_data if isinstance(input_data, list) else input_data.get("records", [])

        if not raw_list:
            return IngestionPreview(
                source="screenshot",
                timestamp=ts_str,
                records_count=0,
                total_attended=0,
                total_conducted=0,
                overall_percentage=0.0,
                records=[],
                errors=["No attendance records provided."],
                warnings=[],
                is_valid=False,
            )

        for idx, item in enumerate(raw_list, start=1):
            raw_subj = item.get("subject_code") or item.get("subject") or item.get("code") or ""
            raw_att = item.get("attended")
            raw_cond = item.get("conducted") or item.get("total")

            if not raw_subj:
                errors.append(f"Record {idx}: Missing subject identifier.")
                continue

            subj = self.resolver.resolve(str(raw_subj))
            if subj is None:
                errors.append(f"Record {idx}: Unknown subject '{raw_subj}'. Must match one of 9 canonical subjects.")
                continue

            if subj.code in seen_codes:
                warnings.append(f"Record {idx}: Duplicate entry for {subj.short_name}. Overwriting earlier entry.")
                records = [r for r in records if r.subject_code != subj.code]

            try:
                att_val = int(raw_att)
                cond_val = int(raw_cond)
            except (ValueError, TypeError):
                errors.append(f"Record {idx}: Attended and conducted values must be integers (got {raw_att}/{raw_cond}).")
                continue

            if att_val < 0 or cond_val < 0:
                errors.append(f"Record {idx}: Counts cannot be negative.")
                continue

            if att_val > cond_val:
                errors.append(f"Record {idx}: Attended ({att_val}) cannot exceed conducted ({cond_val}) for {subj.short_name}.")
                continue

            pct = current_percentage(att_val, cond_val)
            rec = IngestionRecord(
                subject_code=subj.code,
                short_name=subj.short_name,
                full_name=subj.full_name,
                attended=att_val,
                conducted=cond_val,
                percentage=pct,
                raw_identifier=str(raw_subj),
            )
            records.append(rec)
            seen_codes.add(subj.code)

        total_att = sum(r.attended for r in records)
        total_cond = sum(r.conducted for r in records)
        overall_pct = current_percentage(total_att, total_cond)
        is_valid = len(errors) == 0 and len(records) > 0

        return IngestionPreview(
            source="screenshot",
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
