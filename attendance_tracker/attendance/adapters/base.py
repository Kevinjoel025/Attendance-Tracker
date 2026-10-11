"""
attendance/adapters/base.py
────────────────────────────
Universal attendance snapshot ingestion models and base adapter interface.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.attendance.repository import AttendanceRepository, AttendanceSnapshot, get_repository
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


@dataclass(frozen=True)
class IngestionRecord:
    """A validated subject record ready for snapshot persistence."""
    subject_code: str
    short_name: str
    full_name: str
    attended: int
    conducted: int
    percentage: float
    raw_identifier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_code": self.subject_code,
            "short_name": self.short_name,
            "full_name": self.full_name,
            "attended": self.attended,
            "conducted": self.conducted,
            "percentage": self.percentage,
            "raw_identifier": self.raw_identifier,
        }


@dataclass(frozen=True)
class IngestionPreview:
    """Pre-commit validation preview of imported attendance data."""
    source: str
    timestamp: str
    records_count: int
    total_attended: int
    total_conducted: int
    overall_percentage: float
    records: list[IngestionRecord]
    errors: list[str]
    warnings: list[str]
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "records_count": self.records_count,
            "total_attended": self.total_attended,
            "total_conducted": self.total_conducted,
            "overall_percentage": self.overall_percentage,
            "records": [r.to_dict() for r in self.records],
            "errors": self.errors,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }


class BaseAttendanceAdapter(ABC):
    """Abstract base class for all attendance input adapters."""

    def __init__(
        self,
        repository: Optional[AttendanceRepository] = None,
        subject_resolver: Optional[SubjectResolver] = None,
    ) -> None:
        self.repo = repository or get_repository()
        self.resolver = subject_resolver or get_subject_resolver()

    @abstractmethod
    def parse_preview(self, input_data: Any, timestamp: Optional[datetime | str] = None) -> IngestionPreview:
        """Parse raw input into an IngestionPreview without persisting to SQLite."""
        pass

    def parse_text(self, text: str, timestamp: Optional[datetime | str] = None) -> IngestionPreview:
        """Convenience method to parse text content."""
        return self.parse_preview(text, timestamp=timestamp)

    def parse_bytes(self, data: bytes, filename: Optional[str] = None, timestamp: Optional[datetime | str] = None) -> IngestionPreview:
        """Convenience method to parse byte content."""
        return self.parse_preview(data, timestamp=timestamp)

    def commit(self, preview: IngestionPreview) -> AttendanceSnapshot:
        """Atomically persist a valid IngestionPreview into SQLite as a new snapshot."""
        if not preview.is_valid:
            raise ValueError(f"Cannot commit invalid snapshot. Errors: {'; '.join(preview.errors)}")

        raw_payload = json.dumps(
            {
                "source": preview.source,
                "timestamp": preview.timestamp,
                "records_count": preview.records_count,
                "total_attended": preview.total_attended,
                "total_conducted": preview.total_conducted,
                "overall_percentage": preview.overall_percentage,
                "records": [
                    {
                        "subject_code": r.subject_code,
                        "short_name": r.short_name,
                        "full_name": r.full_name,
                        "attended": r.attended,
                        "conducted": r.conducted,
                        "percentage": r.percentage,
                    }
                    for r in preview.records
                ],
            }
        )

        db_records = [
            {
                "subject_code": r.subject_code,
                "attended": r.attended,
                "conducted": r.conducted,
            }
            for r in preview.records
        ]

        return self.repo.create_snapshot(
            scraped_at=preview.timestamp,
            source=preview.source,
            raw_json=raw_payload,
            records=db_records,
        )
