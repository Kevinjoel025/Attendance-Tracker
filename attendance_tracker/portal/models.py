"""
portal/models.py
────────────────
Data structures for portal scraping, subject mapping, and sync results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional


@dataclass(frozen=True)
class RawSubjectRecord:
    """Raw record extracted directly from the portal."""
    subject_name_raw: str
    attended: int
    conducted: int
    percentage_raw: Optional[float] = None
    subject_code: Optional[str] = None  # Resolved canonical subject code


@dataclass(frozen=True)
class ScrapedAttendanceData:
    """Container for attendance data extracted during a scrape session."""
    scraped_at: datetime
    source: str = "portal"
    records: list[RawSubjectRecord] = field(default_factory=list)
    raw_html_snippet: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortalSyncResult:
    """Summary of a portal synchronization operation."""
    success: bool
    snapshot_id: Optional[int] = None
    records_count: int = 0
    changes_detected: list[dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: Optional[datetime] = None
