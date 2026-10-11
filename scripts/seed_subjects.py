"""
scripts/seed_subjects.py
────────────────────────
Database seeding script.

Seeds:
1. 9 canonical subjects
2. Subject aliases
3. Initial development mock attendance snapshot (2026-09-19)
4. Timetable schedule from data/timetable.json
5. Academic calendar holidays from data/holidays.json

Usage:
    python scripts/seed_subjects.py
    python -m scripts.seed_subjects
"""

import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attendance_tracker.attendance.repository import get_repository
from attendance_tracker.core.logging_setup import get_logger
from scripts.import_calendar import import_calendar
from scripts.import_timetable import import_timetable

logger = get_logger(__name__)


def seed_database(force_snapshot: bool = False) -> None:
    """Run full database seeding."""
    repo = get_repository()
    result = repo.seed_all(force=force_snapshot)
    print(f"Seeded {result['subjects_seeded']} subjects.")
    print(f"Seeded {result['aliases_seeded']} subject aliases.")
    print(f"Initial mock attendance snapshot ID: {result['snapshot_id']}.")

    # Also import timetable and calendar
    import_timetable()
    import_calendar()
    print("Database seeding completed successfully.")


if __name__ == "__main__":
    seed_database()
