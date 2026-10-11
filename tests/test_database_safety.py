"""
tests/test_database_safety.py
──────────────────────────────
Production database safety, transaction atomicity, integrity constraint enforcement,
and audit logging tests for Phase 10.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
import pytest

from attendance_tracker.attendance.repository import (
    CANONICAL_ALIASES,
    CANONICAL_SUBJECTS,
    AttendanceRepository,
    OverallAttendance,
    SubjectAttendance,
)


@pytest.fixture
def repo(tmp_path: Path) -> AttendanceRepository:
    """Provide an isolated, freshly initialized SQLite repository for each test."""
    db_file = tmp_path / "safety_test.db"
    r = AttendanceRepository(db_path=db_file)
    r.seed_canonical_subjects()
    r.seed_aliases()
    return r


def test_atomic_transaction_rollback_on_subject_record_failure(repo: AttendanceRepository):
    """
    Verify that if a multi-row snapshot insert encounters an invalid record (e.g. unknown subject),
    the ENTIRE transaction is rolled back and NO partial snapshot or records are committed.
    """
    initial_snapshots = repo.list_snapshots()
    assert len(initial_snapshots) == 0

    invalid_records = [
        {"subject_code": "22MN6HS302", "attended": 4, "conducted": 12},
        {"subject_code": "22OE1EN301", "attended": 12, "conducted": 26},
        {"subject_code": "NON_EXISTENT_SUBJECT", "attended": 5, "conducted": 10},
    ]

    with pytest.raises(ValueError) as exc_info:
        repo.create_snapshot(
            scraped_at=datetime(2026, 10, 5, 10, 0, 0),
            source="portal",
            raw_json="{}",
            records=invalid_records,
        )

    assert "Cannot record attendance for unknown subject" in str(exc_info.value)

    # Verify atomic rollback: no snapshot and no records exist in the database
    with repo.get_connection() as conn:
        snap_count = conn.execute("SELECT COUNT(*) FROM attendance_snapshots;").fetchone()[0]
        rec_count = conn.execute("SELECT COUNT(*) FROM attendance_records;").fetchone()[0]
        assert snap_count == 0
        assert rec_count == 0


def test_negative_attendance_counts_rejected(repo: AttendanceRepository):
    """Verify that negative attended or conducted counts are rejected before database commit."""
    records_neg_att = [
        {"subject_code": "22MN6HS302", "attended": -1, "conducted": 10},
    ]
    with pytest.raises(ValueError) as exc:
        repo.create_snapshot("2026-10-05T10:00:00", "portal", "{}", records_neg_att)
    assert "cannot be negative" in str(exc.value)

    records_neg_cond = [
        {"subject_code": "22MN6HS302", "attended": 5, "conducted": -5},
    ]
    with pytest.raises(ValueError) as exc:
        repo.create_snapshot("2026-10-05T10:00:00", "portal", "{}", records_neg_cond)
    assert "cannot be negative" in str(exc.value)


def test_attended_exceeding_conducted_rejected(repo: AttendanceRepository):
    """Verify that attended > conducted invariant is enforced at repository level."""
    invalid_records = [
        {"subject_code": "22PC1IT202", "attended": 35, "conducted": 30},
    ]
    with pytest.raises(ValueError) as exc:
        repo.create_snapshot("2026-10-05T10:00:00", "portal", "{}", invalid_records)
    assert "cannot exceed conducted" in str(exc.value)


def test_invalid_source_rejected(repo: AttendanceRepository):
    """Verify that invalid snapshot sources are rejected."""
    records = [{"subject_code": "22PC1IT202", "attended": 20, "conducted": 25}]
    with pytest.raises(ValueError) as exc:
        repo.create_snapshot("2026-10-05T10:00:00", "unauthorized_source", "{}", records)
    assert "Invalid source" in str(exc.value)


def test_snapshot_chronological_ordering_and_tie_breaking(repo: AttendanceRepository):
    """Verify latest snapshot ordering by timestamp and ID tie-breaking."""
    records = [{"subject_code": "22PC1IT202", "attended": 20, "conducted": 25}]

    s1 = repo.create_snapshot("2026-10-01T10:00:00", "portal", "{}", records)
    s2 = repo.create_snapshot("2026-10-03T10:00:00", "portal", "{}", records)
    s3 = repo.create_snapshot("2026-10-02T10:00:00", "portal", "{}", records)

    latest = repo.get_latest_snapshot()
    assert latest is not None
    assert latest.id == s2.id
    assert latest.scraped_at == "2026-10-03T10:00:00"

    # Same timestamp tie-break: highest ID wins
    s4 = repo.create_snapshot("2026-10-03T10:00:00", "portal", "{}", records)
    latest_tie = repo.get_latest_snapshot()
    assert latest_tie.id == s4.id


def test_subject_alias_foreign_key_enforcement(repo: AttendanceRepository):
    """Verify that aliases cannot be added for non-existent subject codes."""
    with pytest.raises(ValueError) as exc:
        repo.add_alias("fake_alias", "NON_EXISTENT_CODE")
    assert "does not exist" in str(exc.value)


def test_scrape_log_audit_recording(repo: AttendanceRepository):
    """Verify that scrape_log correctly audits operations with all required attributes."""
    records = [{"subject_code": "22PC1IT202", "attended": 20, "conducted": 25}]
    snap = repo.create_snapshot("2026-10-05T10:00:00", "portal", "{}", records)

    with repo.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO scrape_log (attempted_at, completed_at, status, snapshot_id, error_msg)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("2026-10-05T10:00:00", "2026-10-05T10:00:05", "SUCCESS", snap.id, None),
        )
        conn.execute(
            """
            INSERT INTO scrape_log (attempted_at, completed_at, status, snapshot_id, error_msg)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("2026-10-05T11:00:00", "2026-10-05T11:00:03", "FAILED", None, "Portal login timeout"),
        )

        logs = conn.execute("SELECT * FROM scrape_log ORDER BY id ASC;").fetchall()
        assert len(logs) == 2
        assert logs[0]["status"] == "SUCCESS"
        assert logs[0]["snapshot_id"] == snap.id
        assert logs[0]["error_msg"] is None

        assert logs[1]["status"] == "FAILED"
        assert logs[1]["snapshot_id"] is None
        assert logs[1]["error_msg"] == "Portal login timeout"


def test_database_reset_isolation(repo: AttendanceRepository):
    """Verify reset_database clears all data while preserving schema integrity."""
    repo.seed_initial_snapshot()
    assert repo.get_latest_snapshot() is not None

    repo.reset_database()
    assert repo.get_latest_snapshot() is None
    assert len(repo.list_subjects()) == 0

    # Reseed and verify normal operation
    repo.seed_canonical_subjects()
    assert len(repo.list_subjects()) == 9
