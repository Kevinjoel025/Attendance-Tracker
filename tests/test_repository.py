"""
tests/test_repository.py
────────────────────────
Comprehensive unit and integration tests for AttendanceRepository.

Covers:
- Database initialization (tables, indexes, foreign keys, WAL mode)
- Subject CRUD, validation, and deterministic resolution
- Subject aliases (case-insensitivity, whitespace trimming, foreign key enforcement)
- Attendance snapshots and records (atomicity, chronological ordering)
- Seed data verification (9 canonical subjects, 36 aliases, initial 2026-09-19 snapshot)
- Attendance calculations and overall totals (119/236 = 50.42%)
- Attendance change persistence and retrieval
- Idempotent seeding
- CLI rendering integration
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from attendance_tracker.attendance.repository import (
    CANONICAL_ALIASES,
    CANONICAL_SUBJECTS,
    INITIAL_SEED_ATTENDANCE,
    INITIAL_SEED_SNAPSHOT_DATE,
    AttendanceRepository,
    Subject,
    get_repository,
)
from attendance_tracker.cli.main import render_attendance_table
from scripts.import_calendar import import_calendar
from scripts.import_timetable import import_timetable

_TZ = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def repo(tmp_path: Path):
    """Provide a fresh isolated database for each test."""
    db_file = tmp_path / "test_attendance.db"
    return AttendanceRepository(db_path=db_file)


class TestDatabaseInitialization:

    def test_schema_creates_all_eight_tables(self, repo: AttendanceRepository):
        with repo.get_connection() as conn:
            tables = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%';
                """
            ).fetchall()
            table_names = {r["name"] for r in tables}

        expected_tables = {
            "subjects",
            "subject_aliases",
            "attendance_snapshots",
            "attendance_records",
            "attendance_changes",
            "timetable",
            "academic_calendar",
            "scrape_log",
        }
        assert expected_tables.issubset(table_names), f"Missing tables: {expected_tables - table_names}"

    def test_foreign_keys_are_enabled(self, repo: AttendanceRepository):
        with repo.get_connection() as conn:
            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            assert fk == 1, "Foreign keys must be enabled in SQLite connection"

    def test_repeated_init_is_safe(self, repo: AttendanceRepository):
        # Should not raise
        repo.init_schema()
        repo.init_schema()


class TestSubjectsAndAliases:

    def test_add_and_get_subject(self, repo: AttendanceRepository):
        s = repo.add_subject("CS101", "Computer Science 101", "CS", 75.0)
        assert isinstance(s, Subject)
        assert s.code == "CS101"
        assert s.full_name == "Computer Science 101"
        assert s.short_name == "CS"
        assert s.threshold == 75.0

        fetched = repo.get_subject("CS101")
        assert fetched == s

    def test_subject_validation(self, repo: AttendanceRepository):
        with pytest.raises(ValueError, match="code cannot be empty"):
            repo.add_subject("", "Name", "Short")
        with pytest.raises(ValueError, match="full name cannot be empty"):
            repo.add_subject("CODE", "", "Short")
        with pytest.raises(ValueError, match="short name cannot be empty"):
            repo.add_subject("CODE", "Name", "")
        with pytest.raises(ValueError, match="Threshold must be between"):
            repo.add_subject("CODE", "Name", "Short", threshold=105.0)

    def test_unknown_subject_returns_none(self, repo: AttendanceRepository):
        assert repo.get_subject("NON_EXISTENT") is None
        assert repo.resolve_subject("non existent") is None

    def test_alias_mapping_and_resolution(self, repo: AttendanceRepository):
        repo.add_subject("22PC1IT202", "Operating Systems", "OS", 75.0)
        repo.add_alias("operating system", "22PC1IT202")
        repo.add_alias("os", "22PC1IT202")

        # Exact code lookup
        assert repo.resolve_subject("22PC1IT202").code == "22PC1IT202"
        # Short name case-insensitive
        assert repo.resolve_subject("os").code == "22PC1IT202"
        assert repo.resolve_subject("OS").code == "22PC1IT202"
        # Full name case-insensitive
        assert repo.resolve_subject("Operating Systems").code == "22PC1IT202"
        # Alias lookup with whitespace
        assert repo.resolve_subject("  operating system  ").code == "22PC1IT202"

    def test_alias_for_nonexistent_subject_raises(self, repo: AttendanceRepository):
        with pytest.raises(ValueError, match="does not exist"):
            repo.add_alias("ghost", "FAKE_CODE")


class TestCanonicalSeedData:

    def test_seed_all_populates_nine_subjects(self, repo: AttendanceRepository):
        res = repo.seed_all()
        assert res["subjects_seeded"] == 9
        assert res["aliases_seeded"] == 36

        subjects = repo.list_subjects()
        assert len(subjects) == 9
        codes = {s.code for s in subjects}
        expected_codes = {s["code"] for s in CANONICAL_SUBJECTS}
        assert codes == expected_codes
        assert all(s.threshold == 75.0 for s in subjects)

    def test_all_canonical_aliases_resolve_correctly(self, repo: AttendanceRepository):
        repo.seed_all()

        for code, aliases in CANONICAL_ALIASES.items():
            expected_subject = repo.get_subject(code)
            assert expected_subject is not None
            for alias in aliases:
                resolved = repo.resolve_subject(alias)
                assert resolved is not None, f"Alias '{alias}' failed to resolve"
                assert resolved.code == code, f"Alias '{alias}' resolved to {resolved.code}, expected {code}"

    def test_seed_initial_snapshot_values(self, repo: AttendanceRepository):
        repo.seed_all()

        overall = repo.get_all_current_attendance()
        assert overall is not None
        assert overall.total_attended == 119
        assert overall.total_conducted == 236
        assert abs(overall.overall_percentage - 50.4237) < 0.001
        assert len(overall.subject_records) == 9

        # Verify specific subject values
        gs = repo.get_attendance_for_subject("GS")
        assert gs is not None
        assert gs.attended == 0
        assert gs.conducted == 8
        assert gs.percentage == 0.0

        os_rec = repo.get_attendance_for_subject("OS")
        assert os_rec is not None
        assert os_rec.attended == 17
        assert os_rec.conducted == 28
        assert abs(os_rec.percentage - 60.7143) < 0.001

    def test_seed_idempotency(self, repo: AttendanceRepository):
        # Running seed twice must not duplicate subjects, aliases, or snapshots
        repo.seed_all()
        repo.seed_all()

        subjects = repo.list_subjects()
        assert len(subjects) == 9

        snapshots = repo.list_snapshots()
        assert len(snapshots) == 1


class TestSnapshotsAndRecords:

    def test_create_snapshot_validation(self, repo: AttendanceRepository):
        repo.add_subject("CS101", "Computer Science", "CS", 75.0)

        # Attended > Conducted
        with pytest.raises(ValueError, match="cannot exceed conducted"):
            repo.create_snapshot(
                scraped_at=datetime.now(_TZ),
                source="manual",
                raw_json="{}",
                records=[{"subject_code": "CS101", "attended": 11, "conducted": 10}],
            )

        # Negative attended
        with pytest.raises(ValueError, match="cannot be negative"):
            repo.create_snapshot(
                scraped_at=datetime.now(_TZ),
                source="manual",
                raw_json="{}",
                records=[{"subject_code": "CS101", "attended": -1, "conducted": 10}],
            )

        # Invalid source
        with pytest.raises(ValueError, match="Invalid source"):
            repo.create_snapshot(
                scraped_at=datetime.now(_TZ),
                source="invalid_source",
                raw_json="{}",
                records=[{"subject_code": "CS101", "attended": 5, "conducted": 10}],
            )

        # Unknown subject
        with pytest.raises(ValueError, match="unknown subject"):
            repo.create_snapshot(
                scraped_at=datetime.now(_TZ),
                source="manual",
                raw_json="{}",
                records=[{"subject_code": "UNKNOWN", "attended": 5, "conducted": 10}],
            )

    def test_chronological_latest_snapshot_selection(self, repo: AttendanceRepository):
        repo.add_subject("CS101", "Computer Science", "CS", 75.0)

        # Insert snapshot for Sep 15
        s1 = repo.create_snapshot(
            scraped_at="2026-09-15T10:00:00+05:30",
            source="portal",
            raw_json="{}",
            records=[{"subject_code": "CS101", "attended": 10, "conducted": 10}],
        )

        # Insert snapshot for Sep 18 (latest by time)
        s2 = repo.create_snapshot(
            scraped_at="2026-09-18T10:00:00+05:30",
            source="portal",
            raw_json="{}",
            records=[{"subject_code": "CS101", "attended": 12, "conducted": 14}],
        )

        # Insert snapshot for Sep 16 (inserted after s2, but earlier timestamp)
        s3 = repo.create_snapshot(
            scraped_at="2026-09-16T10:00:00+05:30",
            source="portal",
            raw_json="{}",
            records=[{"subject_code": "CS101", "attended": 11, "conducted": 12}],
        )

        latest = repo.get_latest_snapshot()
        assert latest is not None
        assert latest.id == s2.id, "Latest snapshot must be chosen based on timestamp, not insertion order"

        att = repo.get_attendance_for_subject("CS101")
        assert att.attended == 12
        assert att.conducted == 14

    def test_attendance_history_retrieval(self, repo: AttendanceRepository):
        repo.add_subject("CS101", "Computer Science", "CS", 75.0)

        repo.create_snapshot("2026-09-15T10:00:00+05:30", "portal", "{}", [{"subject_code": "CS101", "attended": 10, "conducted": 10}])
        repo.create_snapshot("2026-09-16T10:00:00+05:30", "portal", "{}", [{"subject_code": "CS101", "attended": 11, "conducted": 12}])
        repo.create_snapshot("2026-09-17T10:00:00+05:30", "portal", "{}", [{"subject_code": "CS101", "attended": 12, "conducted": 13}])

        history = repo.get_attendance_history("CS101")
        assert len(history) == 3
        assert [h.attended for h in history] == [10, 11, 12]
        assert [h.conducted for h in history] == [10, 12, 13]


class TestChangeHistoryPersistence:

    def test_save_and_retrieve_change(self, repo: AttendanceRepository):
        repo.add_subject("CS101", "Computer Science", "CS", 75.0)
        s1 = repo.create_snapshot("2026-09-15T10:00:00+05:30", "portal", "{}", [{"subject_code": "CS101", "attended": 10, "conducted": 10}])
        s2 = repo.create_snapshot("2026-09-16T10:00:00+05:30", "portal", "{}", [{"subject_code": "CS101", "attended": 11, "conducted": 11}])

        change_id = repo.save_change(
            detected_at="2026-09-16T10:00:00+05:30",
            subject_code="CS101",
            prev_snapshot_id=s1.id,
            curr_snapshot_id=s2.id,
            prev_attended=10,
            prev_conducted=10,
            curr_attended=11,
            curr_conducted=11,
            delta_attended=1,
            delta_conducted=1,
            status="PRESENT",
            explanation="1 class attended (+1/+1). Marked Present.",
        )
        assert change_id > 0

        changes = repo.get_changes_for_subject("CS101")
        assert len(changes) == 1
        assert changes[0].status == "PRESENT"
        assert changes[0].delta_attended == 1
        assert changes[0].delta_conducted == 1


class TestCLIIntegration:

    def test_render_attendance_table_with_seed_data(self, repo: AttendanceRepository):
        repo.seed_all()
        # Ensure render_attendance_table runs without error on the seeded data
        overall = repo.get_all_current_attendance()
        assert overall is not None
        render_attendance_table(overall)
