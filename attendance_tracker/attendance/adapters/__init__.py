"""
attendance/adapters/__init__.py
───────────────────────────────
Package exposing universal attendance snapshot ingestion adapters.
"""

from attendance_tracker.attendance.adapters.base import (
    BaseAttendanceAdapter,
    IngestionPreview,
    IngestionRecord,
)
from attendance_tracker.attendance.adapters.csv_adapter import CSVAttendanceAdapter
from attendance_tracker.attendance.adapters.excel_adapter import ExcelAttendanceAdapter
from attendance_tracker.attendance.adapters.screenshot_adapter import ScreenshotAttendanceAdapter

__all__ = [
    "BaseAttendanceAdapter",
    "IngestionPreview",
    "IngestionRecord",
    "CSVAttendanceAdapter",
    "ExcelAttendanceAdapter",
    "ScreenshotAttendanceAdapter",
]
