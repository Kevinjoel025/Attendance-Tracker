BEGIN TRANSACTION;
CREATE TABLE academic_calendar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,  -- "2026-10-02"
    event_type      TEXT NOT NULL          -- "HOLIDAY" | "WORKING_DAY_EXCEPTION" | "EXAM" | "EVENT"
        CHECK (event_type IN ('HOLIDAY', 'WORKING_DAY_EXCEPTION', 'EXAM', 'EVENT', 'BREAK')),
    name            TEXT NOT NULL,         -- "Gandhi Jayanti"
    cancels_classes INTEGER NOT NULL DEFAULT 1  -- 1=no classes, 0=classes still occur
        CHECK (cancels_classes IN (0, 1))
);
CREATE TABLE attendance_changes (
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
CREATE TABLE attendance_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id  INTEGER NOT NULL REFERENCES attendance_snapshots(id),
    scraped_at   TEXT NOT NULL,
    subject_code TEXT NOT NULL REFERENCES subjects(code),
    attended     INTEGER NOT NULL CHECK (attended >= 0),
    conducted    INTEGER NOT NULL CHECK (conducted >= 0),
    percentage   REAL NOT NULL
);
CREATE TABLE attendance_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at  TEXT NOT NULL,   -- ISO 8601 with tz, e.g. "2026-09-19T10:00:00+05:30"
    source      TEXT NOT NULL    -- "portal" | "manual" | "seed"
        CHECK (source IN ('portal', 'manual', 'seed')),
    raw_json    TEXT NOT NULL    -- Full JSON blob
);
CREATE TABLE calendar_overrides (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,  -- "YYYY-MM-DD"
    override_type   TEXT NOT NULL CHECK (override_type IN ('WORKING_DAY', 'HOLIDAY')),
    name            TEXT,                  -- Reason / description
    cancels_classes INTEGER NOT NULL DEFAULT 1 CHECK (cancels_classes IN (0, 1)),
    created_at      TEXT NOT NULL          -- ISO 8601
);
CREATE TABLE scrape_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attempted_at  TEXT NOT NULL,   -- ISO 8601
    completed_at  TEXT,            -- NULL if failed
    status        TEXT NOT NULL    -- "SUCCESS" | "FAILED" | "PARTIAL"
        CHECK (status IN ('SUCCESS', 'FAILED', 'PARTIAL')),
    snapshot_id   INTEGER REFERENCES attendance_snapshots(id),  -- NULL if failed
    error_msg     TEXT             -- Details if failed. NEVER log credentials here.
);
CREATE TABLE semester_config (
    key         TEXT PRIMARY KEY,  -- e.g. "last_working_day"
    value       TEXT NOT NULL,     -- e.g. "2026-11-28"
    updated_at  TEXT NOT NULL
);
CREATE TABLE subject_aliases (
    alias        TEXT PRIMARY KEY,   -- "operating system"
    subject_code TEXT NOT NULL REFERENCES subjects(code)
);
CREATE TABLE subjects (
    code        TEXT PRIMARY KEY,   -- e.g. "22PC1IT202"
    full_name   TEXT NOT NULL,      -- "Operating Systems"
    short_name  TEXT NOT NULL,      -- "OS"
    threshold   REAL NOT NULL DEFAULT 75.0  -- College threshold per subject
);
CREATE TABLE timetable (
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
CREATE INDEX idx_ar_subject_time
    ON attendance_records(subject_code, scraped_at);
CREATE INDEX idx_ac_subject_time
    ON attendance_changes(subject_code, detected_at);
CREATE INDEX idx_timetable_day
    ON timetable(day_of_week, start_time);
CREATE INDEX idx_calendar_date
    ON academic_calendar(date);
CREATE INDEX idx_scrape_log_time
    ON scrape_log(attempted_at);
CREATE INDEX idx_calendar_overrides_date
    ON calendar_overrides(date);
DELETE FROM "sqlite_sequence";
COMMIT;
