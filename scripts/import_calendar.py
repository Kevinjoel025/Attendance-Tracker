"""
scripts/import_calendar.py
──────────────────────────
Validates data/holidays.json and imports holidays & exceptions into the SQLite database.

Usage:
    python scripts/import_calendar.py
    python -m scripts.import_calendar
"""

import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attendance_tracker.calendar.service import CalendarService
from attendance_tracker.core.database import connection, init_db
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)


def import_calendar(data_path: Path | str | None = None) -> int:
    """
    Load calendar holidays from JSON and populate the database academic_calendar table.
    Returns the number of imported entries.
    """
    init_db()
    service = CalendarService(data_path)

    imported_count = 0
    with connection() as conn:
        conn.execute("DELETE FROM academic_calendar;")

        # 1. Named holidays
        for d, hol in service._named_holidays.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO academic_calendar (
                    date,
                    event_type,
                    name,
                    cancels_classes
                ) VALUES (?, ?, ?, ?);
                """,
                (
                    d.isoformat(),
                    hol["event_type"],
                    hol["name"],
                    1 if hol["cancels_classes"] else 0,
                ),
            )
            imported_count += 1

        # 2. Exceptions (e.g. 2026-02-14)
        for d, exc in service._exceptions.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO academic_calendar (
                    date,
                    event_type,
                    name,
                    cancels_classes
                ) VALUES (?, ?, ?, ?);
                """,
                (
                    d.isoformat(),
                    exc["event_type"],
                    exc["name"],
                    1 if exc["cancels_classes"] else 0,
                ),
            )
            imported_count += 1

    print(f"Successfully imported {imported_count} academic calendar entries into SQLite.")
    return imported_count


if __name__ == "__main__":
    import_calendar()
