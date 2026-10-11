"""
scripts/import_timetable.py
───────────────────────────
Validates data/timetable.json and imports the weekly schedule into the SQLite database.

Usage:
    python scripts/import_timetable.py
    python -m scripts.import_timetable
"""

import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attendance_tracker.core.database import connection, init_db
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.timetable.service import TimetableService

logger = get_logger(__name__)

DAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def import_timetable(data_path: Path | str | None = None) -> int:
    """
    Load timetable from JSON and populate the database timetable table.
    Returns the number of imported class slots.
    """
    init_db()
    service = TimetableService(data_path)

    imported_count = 0
    with connection() as conn:
        # Clear existing timetable entries
        conn.execute("DELETE FROM timetable;")

        for day_name, day_idx in DAY_MAP.items():
            classes = service._schedule.get(day_name, [])
            for c in classes:
                conn.execute(
                    """
                    INSERT INTO timetable (
                        day_of_week,
                        period_num,
                        start_time,
                        end_time,
                        subject_code,
                        is_attendance_bearing
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        day_idx,
                        c.period_num,
                        c.start_time,
                        c.end_time,
                        c.subject_code,
                        1 if c.is_attendance_bearing else 0,
                    ),
                )
                imported_count += 1

    print(f"Successfully imported {imported_count} timetable periods into SQLite.")
    return imported_count


if __name__ == "__main__":
    import_timetable()
