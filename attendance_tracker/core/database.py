"""
core/database.py
────────────────
SQLite database connection and schema initialization.

- One SQLite file: data/attendance.db
- Schema is created on first run (CREATE TABLE IF NOT EXISTS).
- No ORM — plain sqlite3 with typed Python wrappers.
- WAL journal mode for better concurrency and crash safety.

SECURITY: Credentials are NEVER stored in this database.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from attendance_tracker.core.config import get_db_path
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- ─────────────────────────────────────────────────────────
-- 1. SUBJECTS
-- Canonical subject list. All other tables reference subject_code.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS subjects (
    code        TEXT PRIMARY KEY,   -- e.g. "22PC1IT202"
    full_name   TEXT NOT NULL,      -- "Operating Systems"
    short_name  TEXT NOT NULL,      -- "OS"
    threshold   REAL NOT NULL DEFAULT 75.0  -- College threshold per subject
);

-- ─────────────────────────────────────────────────────────
-- 2. SUBJECT ALIASES
-- Maps any lowercase alias → canonical subject code.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS subject_aliases (
    alias        TEXT PRIMARY KEY,   -- "operating system"
    subject_code TEXT NOT NULL REFERENCES subjects(code)
);

-- ─────────────────────────────────────────────────────────
-- 3. ATTENDANCE SNAPSHOTS
-- One row per successful portal scrape, CSV/Excel/Screenshot/manual import.
-- Stores the raw JSON for auditability.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at  TEXT NOT NULL,   -- ISO 8601 with tz, e.g. "2026-09-19T10:00:00+05:30"
    source      TEXT NOT NULL    -- "portal" | "manual" | "seed" | "csv" | "excel" | "screenshot"
        CHECK (source IN ('portal', 'manual', 'seed', 'csv', 'excel', 'screenshot')),
    raw_json    TEXT NOT NULL    -- Full JSON blob
);

-- ─────────────────────────────────────────────────────────
-- 4. ATTENDANCE RECORDS
-- Per-subject values extracted from each snapshot.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id  INTEGER NOT NULL REFERENCES attendance_snapshots(id),
    scraped_at   TEXT NOT NULL,
    subject_code TEXT NOT NULL REFERENCES subjects(code),
    attended     INTEGER NOT NULL CHECK (attended >= 0),
    conducted    INTEGER NOT NULL CHECK (conducted >= 0),
    percentage   REAL NOT NULL
);

-- ─────────────────────────────────────────────────────────
-- 5. ATTENDANCE CHANGES
-- Detected delta between consecutive snapshots.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance_changes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at         TEXT NOT NULL,
    subject_code        TEXT NOT NULL REFERENCES subjects(code),
    prev_snapshot_id    INTEGER NOT NULL REFERENCES attendance_snapshots(id),
    curr_snapshot_id    INTEGER NOT NULL REFERENCES attendance_snapshots(id),
    prev_attended       INTEGER NOT NULL,
    prev_conducted      INTEGER NOT NULL,
    curr_attended       INTEGER NOT NULL,
    curr_conducted      INTEGER NOT NULL,
    delta_attended      INTEGER NOT NULL,
    delta_conducted     INTEGER NOT NULL,
    status              TEXT NOT NULL,  -- see AttendanceStatus enum
    explanation         TEXT            -- human-readable explanation
);

-- ─────────────────────────────────────────────────────────
-- 6. TIMETABLE
-- Regular weekly schedule.
-- day_of_week: 0=Monday … 5=Saturday (college has no Sunday classes)
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS timetable (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    day_of_week         INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    period_num          INTEGER NOT NULL CHECK (period_num BETWEEN 1 AND 6),
    start_time          TEXT NOT NULL,   -- "10:00" (24h)
    end_time            TEXT NOT NULL,   -- "11:00" (24h)
    subject_code        TEXT NOT NULL,   -- may be a non-attendance activity
    is_attendance_bearing INTEGER NOT NULL DEFAULT 1  -- 0 for LIB/SPORTS/etc.
        CHECK (is_attendance_bearing IN (0, 1)),
    UNIQUE (day_of_week, period_num)
);

-- ─────────────────────────────────────────────────────────
-- 7. ACADEMIC CALENDAR
-- Official holidays and special working days.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS academic_calendar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,  -- "2026-10-02"
    event_type      TEXT NOT NULL          -- "HOLIDAY" | "WORKING_DAY_EXCEPTION" | "EXAM" | "EVENT"
        CHECK (event_type IN ('HOLIDAY', 'WORKING_DAY_EXCEPTION', 'EXAM', 'EVENT', 'BREAK')),
    name            TEXT NOT NULL,         -- "Gandhi Jayanti"
    cancels_classes INTEGER NOT NULL DEFAULT 1  -- 1=no classes, 0=classes still occur
        CHECK (cancels_classes IN (0, 1))
);

-- ─────────────────────────────────────────────────────────
-- 8. SCRAPE LOG
-- Records every scrape attempt (success or failure).
-- REQUIRED for the attendance change detection decision tree.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scrape_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attempted_at  TEXT NOT NULL,   -- ISO 8601
    completed_at  TEXT,            -- NULL if failed
    status        TEXT NOT NULL    -- "SUCCESS" | "FAILED" | "PARTIAL"
        CHECK (status IN ('SUCCESS', 'FAILED', 'PARTIAL')),
    snapshot_id   INTEGER REFERENCES attendance_snapshots(id),  -- NULL if failed
    error_msg     TEXT             -- Details if failed. NEVER log credentials here.
);

-- ─────────────────────────────────────────────────────────
-- 9. CALENDAR OVERRIDES
-- User-defined working-day or holiday overrides.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS calendar_overrides (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,  -- "YYYY-MM-DD"
    override_type   TEXT NOT NULL CHECK (override_type IN ('WORKING_DAY', 'HOLIDAY')),
    name            TEXT,                  -- Reason / description
    cancels_classes INTEGER NOT NULL DEFAULT 1 CHECK (cancels_classes IN (0, 1)),
    created_at      TEXT NOT NULL          -- ISO 8601
);

-- ─────────────────────────────────────────────────────────
-- 10. SEMESTER CONFIG
-- Key-value settings such as last_working_day and semester_start_date.
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS semester_config (
    key         TEXT PRIMARY KEY,  -- e.g. "last_working_day"
    value       TEXT NOT NULL,     -- e.g. "2026-11-28"
    updated_at  TEXT NOT NULL
);

-- ─────────────────────────────────────────────────────────
-- INDEXES
-- ─────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ar_subject_time
    ON attendance_records(subject_code, scraped_at);

CREATE INDEX IF NOT EXISTS idx_ac_subject_time
    ON attendance_changes(subject_code, detected_at);

CREATE INDEX IF NOT EXISTS idx_timetable_day
    ON timetable(day_of_week, start_time);

CREATE INDEX IF NOT EXISTS idx_calendar_date
    ON academic_calendar(date);

CREATE INDEX IF NOT EXISTS idx_scrape_log_time
    ON scrape_log(attempted_at);

CREATE INDEX IF NOT EXISTS idx_calendar_overrides_date
    ON calendar_overrides(date);
"""


# ── Connection management ─────────────────────────────────────────────────────

def _get_db_path() -> Path:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection with WAL mode and foreign keys enabled.

    Callers are responsible for closing the connection.
    Prefer using the `connection()` context manager instead.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row  # rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that opens a connection and auto-commits or rolls back.

    Usage:
        with connection() as conn:
            conn.execute("INSERT INTO ...")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialization ─────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables if they don't already exist.
    Safe to call on every startup.
    """
    db_path = _get_db_path()
    logger.info(f"Initializing database at {db_path}")
    with connection() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database schema ready.")


def db_exists() -> bool:
    """Return True if the database file already exists."""
    return _get_db_path().exists()
