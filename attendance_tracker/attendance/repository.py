"""
attendance/repository.py
────────────────────────
Persistent SQLite repository layer for subjects, aliases, attendance snapshots,
records, and changes.

RULES:
- SQLite WAL mode and foreign keys enabled.
- Pure database persistence/retrieval (no SLM, no NLU, no scraping, no NLG).
- Atomic transactions for multi-row writes.
- Deterministic calculation of percentages via attendance.engine.
- Validates attended >= 0, conducted >= 0, attended <= conducted.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.core.config import config, get_db_path
from attendance_tracker.core.database import _SCHEMA_SQL
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Canonical Seed Data ───────────────────────────────────────────────────────

CANONICAL_SUBJECTS = [
    {"code": "22MN6HS302", "full_name": "Gender Sensitization", "short_name": "GS", "threshold": 75.0},
    {"code": "22OE1EN301", "full_name": "Storytelling for Effective Communication", "short_name": "OE-1", "threshold": 75.0},
    {"code": "22PC1AM202", "full_name": "Data Engineering", "short_name": "DE", "threshold": 75.0},
    {"code": "22PC1AM203", "full_name": "Automata and Compiler Design", "short_name": "ACD", "threshold": 75.0},
    {"code": "22PC1IN301", "full_name": "IoT Architecture and Protocols", "short_name": "IoTAP", "threshold": 75.0},
    {"code": "22PC1IT202", "full_name": "Operating Systems", "short_name": "OS", "threshold": 75.0},
    {"code": "22PC2AM202", "full_name": "Data Engineering Laboratory", "short_name": "DE LAB", "threshold": 75.0},
    {"code": "22PC2IN301", "full_name": "IoT Architecture and Protocols Laboratory", "short_name": "IoTAP LAB", "threshold": 75.0},
    {"code": "22PE1AM301", "full_name": "Web Programming", "short_name": "WP", "threshold": 75.0},
]

CANONICAL_ALIASES = {
    "22MN6HS302": ["gender", "gender sensitization", "gs"],
    "22OE1EN301": ["storytelling", "storytelling for effective communication", "storytelling communication", "oe", "oe-1", "oe1"],
    "22PC1AM202": ["de", "data", "data engineering", "data eng"],
    "22PC1AM203": ["acd", "automata", "automata and compiler design", "compiler design", "compiler"],
    "22PC1IN301": ["iot", "iotap", "iot architecture", "iot architecture and protocols"],
    "22PC1IT202": ["os", "operating system", "operating systems"],
    "22PC2AM202": ["de lab", "data engineering lab", "data engineering laboratory"],
    "22PC2IN301": ["iot lab", "iotap lab", "iot architecture lab", "iot architecture and protocols lab"],
    "22PE1AM301": ["wp", "web", "web programming", "web prog"],
}



INITIAL_SEED_SNAPSHOT_DATE = "2026-09-19T00:00:00+05:30"

INITIAL_SEED_ATTENDANCE = [
    {"subject_code": "22MN6HS302", "attended": 0, "conducted": 8},
    {"subject_code": "22OE1EN301", "attended": 9, "conducted": 20},
    {"subject_code": "22PC1AM202", "attended": 11, "conducted": 28},
    {"subject_code": "22PC1AM203", "attended": 21, "conducted": 33},
    {"subject_code": "22PC1IN301", "attended": 15, "conducted": 27},
    {"subject_code": "22PC1IT202", "attended": 17, "conducted": 28},
    {"subject_code": "22PC2AM202", "attended": 18, "conducted": 30},
    {"subject_code": "22PC2IN301", "attended": 9, "conducted": 27},
    {"subject_code": "22PE1AM301", "attended": 19, "conducted": 35},
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Subject:
    code: str
    full_name: str
    short_name: str
    threshold: float = 75.0


@dataclass(frozen=True)
class SubjectAlias:
    alias: str
    subject_code: str


@dataclass(frozen=True)
class AttendanceSnapshot:
    id: int
    scraped_at: str
    source: str
    raw_json: str


@dataclass(frozen=True)
class AttendanceRecord:
    id: int
    snapshot_id: int
    scraped_at: str
    subject_code: str
    attended: int
    conducted: int
    percentage: float


@dataclass(frozen=True)
class SubjectAttendance:
    subject: Subject
    attended: int
    conducted: int
    percentage: float
    scraped_at: str
    snapshot_id: int


@dataclass(frozen=True)
class OverallAttendance:
    total_attended: int
    total_conducted: int
    overall_percentage: float
    subject_records: list[SubjectAttendance]
    scraped_at: str
    snapshot_id: int


@dataclass(frozen=True)
class CalendarOverrideRecord:
    id: int
    date: str
    override_type: str  # "WORKING_DAY" | "HOLIDAY"
    name: Optional[str]
    cancels_classes: bool
    created_at: str


@dataclass(frozen=True)
class AttendanceChangeRecord:
    id: int
    detected_at: str
    subject_code: str
    prev_snapshot_id: int
    curr_snapshot_id: int
    prev_attended: int
    prev_conducted: int
    curr_attended: int
    curr_conducted: int
    delta_attended: int
    delta_conducted: int
    status: str
    explanation: Optional[str]


# ── Repository Class ──────────────────────────────────────────────────────────

class AttendanceRepository:
    """
    Data access object for managing SQLite attendance tables.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = get_db_path()
        self.db_path = Path(db_path)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_schema(self) -> None:
        """Create all database tables and indexes if they do not exist."""
        with self.get_connection() as conn:
            conn.executescript(_SCHEMA_SQL)

    def reset_database(self) -> None:
        """Drop all table contents safely (primarily used for test isolation)."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys=OFF;")
            conn.execute("DELETE FROM calendar_overrides;")
            conn.execute("DELETE FROM semester_config;")
            conn.execute("DELETE FROM attendance_changes;")
            conn.execute("DELETE FROM attendance_records;")
            conn.execute("DELETE FROM scrape_log;")
            conn.execute("DELETE FROM attendance_snapshots;")
            conn.execute("DELETE FROM subject_aliases;")
            conn.execute("DELETE FROM timetable;")
            conn.execute("DELETE FROM academic_calendar;")
            conn.execute("DELETE FROM subjects;")
            conn.execute("PRAGMA foreign_keys=ON;")

    # ── Subject Management ────────────────────────────────────────────────────

    def add_subject(
        self,
        code: str,
        full_name: str,
        short_name: str,
        threshold: float = 75.0,
    ) -> Subject:
        """Insert or update a canonical subject."""
        code_clean = code.strip()
        full_name_clean = full_name.strip()
        short_name_clean = short_name.strip()

        if not code_clean:
            raise ValueError("Subject code cannot be empty.")
        if not full_name_clean:
            raise ValueError("Subject full name cannot be empty.")
        if not short_name_clean:
            raise ValueError("Subject short name cannot be empty.")
        if threshold < 0.0 or threshold > 100.0:
            raise ValueError(f"Threshold must be between 0 and 100, got {threshold}.")

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO subjects (code, full_name, short_name, threshold)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    full_name=excluded.full_name,
                    short_name=excluded.short_name,
                    threshold=excluded.threshold;
                """,
                (code_clean, full_name_clean, short_name_clean, threshold),
            )
        return Subject(
            code=code_clean,
            full_name=full_name_clean,
            short_name=short_name_clean,
            threshold=threshold,
        )

    def get_subject(self, code: str) -> Optional[Subject]:
        """Look up a subject by its exact canonical code."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT code, full_name, short_name, threshold FROM subjects WHERE code = ?;",
                (code.strip(),),
            ).fetchone()
        if row is None:
            return None
        return Subject(
            code=row["code"],
            full_name=row["full_name"],
            short_name=row["short_name"],
            threshold=float(row["threshold"]),
        )

    def list_subjects(self) -> list[Subject]:
        """List all canonical subjects ordered by code."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT code, full_name, short_name, threshold FROM subjects ORDER BY code ASC;"
            ).fetchall()
        return [
            Subject(
                code=r["code"],
                full_name=r["full_name"],
                short_name=r["short_name"],
                threshold=float(r["threshold"]),
            )
            for r in rows
        ]

    def add_alias(self, alias: str, subject_code: str) -> SubjectAlias:
        """Insert or replace an alias mapping to a canonical subject."""
        alias_clean = alias.strip().lower()
        code_clean = subject_code.strip()

        if not alias_clean:
            raise ValueError("Alias cannot be empty.")

        with self.get_connection() as conn:
            # Verify subject exists first (enforcing foreign key behavior)
            subj = conn.execute(
                "SELECT code FROM subjects WHERE code = ?;", (code_clean,)
            ).fetchone()
            if subj is None:
                raise ValueError(f"Subject code '{code_clean}' does not exist.")

            conn.execute(
                """
                INSERT INTO subject_aliases (alias, subject_code)
                VALUES (?, ?)
                ON CONFLICT(alias) DO UPDATE SET subject_code=excluded.subject_code;
                """,
                (alias_clean, code_clean),
            )
        return SubjectAlias(alias=alias_clean, subject_code=code_clean)

    def get_subject_by_alias(self, alias: str) -> Optional[Subject]:
        """Look up a subject using a lowercase alias."""
        alias_clean = alias.strip().lower()
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.code, s.full_name, s.short_name, s.threshold
                FROM subject_aliases a
                JOIN subjects s ON a.subject_code = s.code
                WHERE a.alias = ?;
                """,
                (alias_clean,),
            ).fetchone()
        if row is None:
            return None
        return Subject(
            code=row["code"],
            full_name=row["full_name"],
            short_name=row["short_name"],
            threshold=float(row["threshold"]),
        )

    def resolve_subject(self, query: str) -> Optional[Subject]:
        """
        Deterministic subject resolution:
        1. Exact canonical code match
        2. Exact short name match (case-insensitive)
        3. Exact full name match (case-insensitive)
        4. Alias match (case-insensitive, whitespace-trimmed)
        """
        if not query:
            return None
        q = query.strip()
        q_lower = q.lower()

        # 1. Exact code
        s = self.get_subject(q)
        if s:
            return s

        # 2. Case-insensitive short name or full name
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT code, full_name, short_name, threshold
                FROM subjects
                WHERE LOWER(short_name) = ? OR LOWER(full_name) = ? OR LOWER(code) = ?;
                """,
                (q_lower, q_lower, q_lower),
            ).fetchone()
        if row:
            return Subject(
                code=row["code"],
                full_name=row["full_name"],
                short_name=row["short_name"],
                threshold=float(row["threshold"]),
            )

        # 3. Alias lookup
        return self.get_subject_by_alias(q_lower)

    # ── Snapshot & Attendance Management ──────────────────────────────────────

    def create_snapshot(
        self,
        scraped_at: datetime | str,
        source: str,
        raw_json: str,
        records: list[dict[str, Any]],
    ) -> AttendanceSnapshot:
        """
        Atomically create an attendance snapshot and insert all subject records.

        Each record in `records` must contain:
        - subject_code: str
        - attended: int
        - conducted: int
        """
        if isinstance(scraped_at, datetime):
            scraped_at_str = scraped_at.isoformat()
        else:
            scraped_at_str = str(scraped_at).strip()

        source_clean = source.strip().lower()
        if source_clean not in ("portal", "manual", "seed", "csv", "excel", "screenshot"):
            raise ValueError(f"Invalid source '{source}'. Must be 'portal', 'manual', 'seed', 'csv', 'excel', or 'screenshot'.")

        # Validate all records before opening transaction
        validated_records = []
        for r in records:
            s_code = r["subject_code"].strip()
            att = int(r["attended"])
            cond = int(r["conducted"])

            if att < 0:
                raise ValueError(f"Attended count cannot be negative: {att}")
            if cond < 0:
                raise ValueError(f"Conducted count cannot be negative: {cond}")
            if att > cond:
                raise ValueError(f"Attended ({att}) cannot exceed conducted ({cond}) for {s_code}.")

            pct = current_percentage(att, cond)
            validated_records.append((s_code, att, cond, pct))

        with self.get_connection() as conn:
            # 1. Insert snapshot
            try:
                cur = conn.execute(
                    """
                    INSERT INTO attendance_snapshots (scraped_at, source, raw_json)
                    VALUES (?, ?, ?);
                    """,
                    (scraped_at_str, source_clean, raw_json),
                )
            except sqlite3.IntegrityError:
                cur = conn.execute(
                    """
                    INSERT INTO attendance_snapshots (scraped_at, source, raw_json)
                    VALUES (?, 'manual', ?);
                    """,
                    (scraped_at_str, raw_json),
                )
            snapshot_id = cur.lastrowid

            # 2. Insert records
            for s_code, att, cond, pct in validated_records:
                # Verify foreign key subject existence explicitly for clearer error messages
                subj = conn.execute("SELECT code FROM subjects WHERE code = ?;", (s_code,)).fetchone()
                if subj is None:
                    raise ValueError(f"Cannot record attendance for unknown subject '{s_code}'.")

                conn.execute(
                    """
                    INSERT INTO attendance_records (
                        snapshot_id, scraped_at, subject_code, attended, conducted, percentage
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (snapshot_id, scraped_at_str, s_code, att, cond, pct),
                )

        return AttendanceSnapshot(
            id=snapshot_id,
            scraped_at=scraped_at_str,
            source=source_clean,
            raw_json=raw_json,
        )

    def save_snapshot(
        self,
        subject_records: list[dict[str, Any]] | list[tuple[str, int, int]],
        source: str = "manual",
        scraped_at: Optional[datetime | str] = None,
        raw_json: Optional[str] = None,
    ) -> int:
        """Convenience method to save a snapshot and return its integer ID."""
        if scraped_at is None:
            scraped_at = datetime.now(_TZ).isoformat()

        formatted_records = []
        for r in subject_records:
            if isinstance(r, dict):
                formatted_records.append({
                    "subject_code": r.get("subject_code") or r.get("code"),
                    "attended": int(r.get("attended") or r.get("attended_classes", 0)),
                    "conducted": int(r.get("conducted") or r.get("total_classes", 0)),
                })
            elif isinstance(r, (tuple, list)) and len(r) >= 3:
                formatted_records.append({
                    "subject_code": str(r[0]),
                    "attended": int(r[1]),
                    "conducted": int(r[2]),
                })

        if raw_json is None:
            raw_json = json.dumps({
                "source": source,
                "timestamp": str(scraped_at),
                "records": formatted_records,
            })

        snap = self.create_snapshot(
            scraped_at=scraped_at,
            source=source,
            raw_json=raw_json,
            records=formatted_records,
        )
        return snap.id

    def get_snapshot(self, snapshot_id: int) -> Optional[AttendanceSnapshot]:
        """Look up a snapshot by its ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT id, scraped_at, source, raw_json FROM attendance_snapshots WHERE id = ?;",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return AttendanceSnapshot(
            id=row["id"],
            scraped_at=row["scraped_at"],
            source=row["source"],
            raw_json=row["raw_json"],
        )

    def get_latest_snapshot(self) -> Optional[AttendanceSnapshot]:
        """
        Return the latest snapshot ordered by timestamp (scraped_at),
        falling back to snapshot id on timestamp collision.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, scraped_at, source, raw_json
                FROM attendance_snapshots
                ORDER BY scraped_at DESC, id DESC
                LIMIT 1;
                """
            ).fetchone()
        if row is None:
            return None
        return AttendanceSnapshot(
            id=row["id"],
            scraped_at=row["scraped_at"],
            source=row["source"],
            raw_json=row["raw_json"],
        )

    def list_snapshots(self, limit: int = 50) -> list[AttendanceSnapshot]:
        """List historical snapshots ordered from newest to oldest."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, scraped_at, source, raw_json
                FROM attendance_snapshots
                ORDER BY scraped_at DESC, id DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [
            AttendanceSnapshot(
                id=r["id"],
                scraped_at=r["scraped_at"],
                source=r["source"],
                raw_json=r["raw_json"],
            )
            for r in rows
        ]

    def get_attendance_for_subject(
        self,
        subject_query: str,
        snapshot_id: Optional[int] = None,
    ) -> Optional[SubjectAttendance]:
        """
        Retrieve attendance for a specific subject from a snapshot
        (defaults to the latest snapshot).
        """
        subj = self.resolve_subject(subject_query)
        if subj is None:
            return None

        with self.get_connection() as conn:
            if snapshot_id is None:
                # Latest record for this subject
                row = conn.execute(
                    """
                    SELECT r.snapshot_id, r.scraped_at, r.attended, r.conducted, r.percentage
                    FROM attendance_records r
                    WHERE r.subject_code = ?
                    ORDER BY r.scraped_at DESC, r.id DESC
                    LIMIT 1;
                    """,
                    (subj.code,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT r.snapshot_id, r.scraped_at, r.attended, r.conducted, r.percentage
                    FROM attendance_records r
                    WHERE r.subject_code = ? AND r.snapshot_id = ?;
                    """,
                    (subj.code, snapshot_id),
                ).fetchone()

        if row is None:
            return None

        att = int(row["attended"])
        cond = int(row["conducted"])
        pct = current_percentage(att, cond)

        return SubjectAttendance(
            subject=subj,
            attended=att,
            conducted=cond,
            percentage=pct,
            scraped_at=row["scraped_at"],
            snapshot_id=row["snapshot_id"],
        )

    def get_all_current_attendance(
        self,
        snapshot_id: Optional[int] = None,
    ) -> Optional[OverallAttendance]:
        """
        Retrieve attendance for all subjects in a snapshot (defaults to latest)
        and compute aggregate totals.
        """
        with self.get_connection() as conn:
            if snapshot_id is None:
                latest = self.get_latest_snapshot()
                if latest is None:
                    return None
                target_snapshot_id = latest.id
                scraped_at = latest.scraped_at
            else:
                snap = self.get_snapshot(snapshot_id)
                if snap is None:
                    return None
                target_snapshot_id = snap.id
                scraped_at = snap.scraped_at

            rows = conn.execute(
                """
                SELECT r.attended, r.conducted, r.percentage,
                       s.code, s.full_name, s.short_name, s.threshold
                FROM attendance_records r
                JOIN subjects s ON r.subject_code = s.code
                WHERE r.snapshot_id = ?
                ORDER BY s.code ASC;
                """,
                (target_snapshot_id,),
            ).fetchall()

        subject_records: list[SubjectAttendance] = []
        total_attended = 0
        total_conducted = 0

        for r in rows:
            subj = Subject(
                code=r["code"],
                full_name=r["full_name"],
                short_name=r["short_name"],
                threshold=float(r["threshold"]),
            )
            att = int(r["attended"])
            cond = int(r["conducted"])
            pct = current_percentage(att, cond)
            total_attended += att
            total_conducted += cond

            subject_records.append(
                SubjectAttendance(
                    subject=subj,
                    attended=att,
                    conducted=cond,
                    percentage=pct,
                    scraped_at=scraped_at,
                    snapshot_id=target_snapshot_id,
                )
            )

        overall_pct = current_percentage(total_attended, total_conducted)

        return OverallAttendance(
            total_attended=total_attended,
            total_conducted=total_conducted,
            overall_percentage=overall_pct,
            subject_records=subject_records,
            scraped_at=scraped_at,
            snapshot_id=target_snapshot_id,
        )

    def get_attendance_history(self, subject_code: str) -> list[AttendanceRecord]:
        """Return historical records for a specific subject ordered chronologically."""
        subj = self.resolve_subject(subject_code)
        if subj is None:
            return []

        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, snapshot_id, scraped_at, subject_code, attended, conducted, percentage
                FROM attendance_records
                WHERE subject_code = ?
                ORDER BY scraped_at ASC, id ASC;
                """,
                (subj.code,),
            ).fetchall()

        return [
            AttendanceRecord(
                id=r["id"],
                snapshot_id=r["snapshot_id"],
                scraped_at=r["scraped_at"],
                subject_code=r["subject_code"],
                attended=r["attended"],
                conducted=r["conducted"],
                percentage=r["percentage"],
            )
            for r in rows
        ]

    # ── Change History Management ─────────────────────────────────────────────

    def save_change(
        self,
        detected_at: datetime | str,
        subject_code: str,
        prev_snapshot_id: int,
        curr_snapshot_id: int,
        prev_attended: int,
        prev_conducted: int,
        curr_attended: int,
        curr_conducted: int,
        delta_attended: int,
        delta_conducted: int,
        status: str,
        explanation: Optional[str] = None,
    ) -> int:
        """Persist a detected change result into the database."""
        if isinstance(detected_at, datetime):
            detected_at_str = detected_at.isoformat()
        else:
            detected_at_str = str(detected_at).strip()

        with self.get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO attendance_changes (
                    detected_at,
                    subject_code,
                    prev_snapshot_id,
                    curr_snapshot_id,
                    prev_attended,
                    prev_conducted,
                    curr_attended,
                    curr_conducted,
                    delta_attended,
                    delta_conducted,
                    status,
                    explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    detected_at_str,
                    subject_code.strip(),
                    prev_snapshot_id,
                    curr_snapshot_id,
                    prev_attended,
                    prev_conducted,
                    curr_attended,
                    curr_conducted,
                    delta_attended,
                    delta_conducted,
                    status.strip(),
                    explanation,
                ),
            )
            return cur.lastrowid

    def get_changes_for_subject(self, subject_code: str) -> list[AttendanceChangeRecord]:
        """Return historical change records for a subject."""
        subj = self.resolve_subject(subject_code)
        if subj is None:
            return []

        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, detected_at, subject_code, prev_snapshot_id, curr_snapshot_id,
                       prev_attended, prev_conducted, curr_attended, curr_conducted,
                       delta_attended, delta_conducted, status, explanation
                FROM attendance_changes
                WHERE subject_code = ?
                ORDER BY detected_at DESC, id DESC;
                """,
                (subj.code,),
            ).fetchall()

        return [
            AttendanceChangeRecord(
                id=r["id"],
                detected_at=r["detected_at"],
                subject_code=r["subject_code"],
                prev_snapshot_id=r["prev_snapshot_id"],
                curr_snapshot_id=r["curr_snapshot_id"],
                prev_attended=r["prev_attended"],
                prev_conducted=r["prev_conducted"],
                curr_attended=r["curr_attended"],
                curr_conducted=r["curr_conducted"],
                delta_attended=r["delta_attended"],
                delta_conducted=r["delta_conducted"],
                status=r["status"],
                explanation=r["explanation"],
            )
            for r in rows
        ]

    # ── Calendar Overrides Management ─────────────────────────────────────────

    def add_calendar_override(
        self,
        target_date: Optional[date | str] = None,
        override_type: str = "HOLIDAY",
        name: Optional[str] = None,
        date_str: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> CalendarOverrideRecord:
        """Add or update a user-defined calendar override (WORKING_DAY or HOLIDAY)."""
        raw_d = target_date if target_date is not None else date_str
        if raw_d is None:
            raise ValueError("Target date must be provided for calendar override.")

        if isinstance(raw_d, (date, datetime)):
            d_str = raw_d.strftime("%Y-%m-%d")
        else:
            d_str = str(raw_d).strip()

        type_norm = override_type.strip().upper()
        if type_norm not in ("WORKING_DAY", "HOLIDAY"):
            raise ValueError(f"Invalid override_type '{override_type}'. Must be 'WORKING_DAY' or 'HOLIDAY'.")

        override_name = name if name is not None else (reason or "")
        cancels = 1 if type_norm == "HOLIDAY" else 0
        now_str = datetime.now(_TZ).isoformat()

        with self.get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO calendar_overrides (date, override_type, name, cancels_classes, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    override_type=excluded.override_type,
                    name=excluded.name,
                    cancels_classes=excluded.cancels_classes,
                    created_at=excluded.created_at;
                """,
                (d_str, type_norm, override_name, cancels, now_str),
            )
            row_id = cur.lastrowid

        return CalendarOverrideRecord(
            id=row_id,
            date=d_str,
            override_type=type_norm,
            name=override_name,
            cancels_classes=bool(cancels),
            created_at=now_str,
        )

    def get_calendar_override(self, target_date: date | str) -> Optional[CalendarOverrideRecord]:
        """Look up an override for a specific date."""
        if isinstance(target_date, (date, datetime)):
            date_str = target_date.strftime("%Y-%m-%d")
        else:
            date_str = str(target_date).strip()

        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, date, override_type, name, cancels_classes, created_at
                FROM calendar_overrides
                WHERE date = ?;
                """,
                (date_str,),
            ).fetchone()

        if row is None:
            return None

        return CalendarOverrideRecord(
            id=row["id"],
            date=row["date"],
            override_type=row["override_type"],
            name=row["name"],
            cancels_classes=bool(row["cancels_classes"]),
            created_at=row["created_at"],
        )

    def list_calendar_overrides(self) -> list[CalendarOverrideRecord]:
        """List all user-defined calendar overrides ordered by date."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, date, override_type, name, cancels_classes, created_at
                FROM calendar_overrides
                ORDER BY date ASC;
                """
            ).fetchall()

        return [
            CalendarOverrideRecord(
                id=r["id"],
                date=r["date"],
                override_type=r["override_type"],
                name=r["name"],
                cancels_classes=bool(r["cancels_classes"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete_calendar_override(self, target_date: date | str) -> bool:
        """Remove a calendar override for a given date."""
        if isinstance(target_date, (date, datetime)):
            date_str = target_date.strftime("%Y-%m-%d")
        else:
            date_str = str(target_date).strip()

        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM calendar_overrides WHERE date = ?;", (date_str,))
            return cur.rowcount > 0

    # ── Semester Configuration Management ─────────────────────────────────────

    def set_semester_config(self, key: str, value: str) -> str:
        """Set a configuration parameter (e.g. last_working_day)."""
        key_clean = key.strip()
        val_clean = str(value).strip()
        now_str = datetime.now(_TZ).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO semester_config (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at;
                """,
                (key_clean, val_clean, now_str),
            )
        return val_clean

    def get_semester_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a semester configuration value."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT value FROM semester_config WHERE key = ?;", (key.strip(),)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def get_last_working_day(self, default: str = "2026-11-28") -> date:
        """Return the configured last working day of the semester."""
        val = self.get_semester_config("last_working_day", default)
        try:
            return datetime.strptime(val, "%Y-%m-%d").date()
        except Exception:
            return datetime.strptime(default, "%Y-%m-%d").date()

    def set_last_working_day(self, target_date: date | str) -> date:
        """Update the configured last working day of the semester."""
        if isinstance(target_date, (date, datetime)):
            d_str = target_date.strftime("%Y-%m-%d")
            d_obj = target_date if isinstance(target_date, date) else target_date.date()
        else:
            d_str = str(target_date).strip()
            d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()

        self.set_semester_config("last_working_day", d_str)
        return d_obj

    def get_mid_term_1(
        self,
        default_start: str = "2026-09-07",
        default_end: str = "2026-09-12",
    ) -> tuple[date, date]:
        """Return the configured (start_date, end_date) for Mid-Term 1."""
        start_str = self.get_semester_config("mid_term_1_start", default_start) or default_start
        end_str = self.get_semester_config("mid_term_1_end", default_end) or default_end
        try:
            d_start = datetime.strptime(start_str, "%Y-%m-%d").date()
        except Exception:
            d_start = datetime.strptime(default_start, "%Y-%m-%d").date()
        try:
            d_end = datetime.strptime(end_str, "%Y-%m-%d").date()
        except Exception:
            d_end = datetime.strptime(default_end, "%Y-%m-%d").date()
        return d_start, d_end

    def set_mid_term_1(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> tuple[date, date]:
        """Set the configured (start_date, end_date) for Mid-Term 1."""
        if isinstance(start_date, (date, datetime)):
            s_str = start_date.strftime("%Y-%m-%d")
            s_obj = start_date if isinstance(start_date, date) else start_date.date()
        else:
            s_str = str(start_date).strip()
            s_obj = datetime.strptime(s_str, "%Y-%m-%d").date()

        if isinstance(end_date, (date, datetime)):
            e_str = end_date.strftime("%Y-%m-%d")
            e_obj = end_date if isinstance(end_date, date) else end_date.date()
        else:
            e_str = str(end_date).strip()
            e_obj = datetime.strptime(e_str, "%Y-%m-%d").date()

        if e_obj < s_obj:
            raise ValueError(f"Mid-Term 1 end date ({e_str}) cannot be before start date ({s_str}).")

        self.set_semester_config("mid_term_1_start", s_str)
        self.set_semester_config("mid_term_1_end", e_str)
        return s_obj, e_obj

    def get_previous_snapshot(self, before_snapshot_id: Optional[int] = None) -> Optional[AttendanceSnapshot]:
        """Look up the snapshot immediately preceding before_snapshot_id (or latest)."""
        with self.get_connection() as conn:
            if before_snapshot_id is None:
                rows = conn.execute(
                    """
                    SELECT id, scraped_at, source, raw_json
                    FROM attendance_snapshots
                    ORDER BY scraped_at DESC, id DESC
                    LIMIT 2;
                    """
                ).fetchall()
                if len(rows) < 2:
                    return None
                return AttendanceSnapshot(
                    id=rows[1]["id"],
                    scraped_at=rows[1]["scraped_at"],
                    source=rows[1]["source"],
                    raw_json=rows[1]["raw_json"],
                )
            else:
                curr = self.get_snapshot(before_snapshot_id)
                if curr is None:
                    return None
                row = conn.execute(
                    """
                    SELECT id, scraped_at, source, raw_json
                    FROM attendance_snapshots
                    WHERE (scraped_at < ? OR (scraped_at = ? AND id < ?))
                    ORDER BY scraped_at DESC, id DESC
                    LIMIT 1;
                    """,
                    (curr.scraped_at, curr.scraped_at, curr.id),
                ).fetchone()
                if row is None:
                    return None
                return AttendanceSnapshot(
                    id=row["id"],
                    scraped_at=row["scraped_at"],
                    source=row["source"],
                    raw_json=row["raw_json"],
                )

    # ── Seeding ───────────────────────────────────────────────────────────────

    def seed_canonical_subjects(self) -> int:
        """Seed the 9 canonical VNR VJIET subjects idempotently."""
        count = 0
        for subj_data in CANONICAL_SUBJECTS:
            self.add_subject(
                code=subj_data["code"],
                full_name=subj_data["full_name"],
                short_name=subj_data["short_name"],
                threshold=subj_data["threshold"],
            )
            count += 1
        return count

    def seed_aliases(self) -> int:
        """Seed the canonical subject aliases idempotently."""
        count = 0
        for code, alias_list in CANONICAL_ALIASES.items():
            for alias in alias_list:
                self.add_alias(alias, code)
                count += 1
        return count

    def seed_initial_snapshot(self, force: bool = False) -> Optional[AttendanceSnapshot]:
        """
        Seed the initial 2026-09-19 development snapshot idempotently.
        If a seed snapshot already exists and force is False, returns existing snapshot.
        """
        with self.get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM attendance_snapshots WHERE source = 'seed' LIMIT 1;"
            ).fetchone()

        if existing is not None and not force:
            return self.get_snapshot(existing["id"])

        raw_payload = json.dumps(
            {
                "snapshot_date": INITIAL_SEED_SNAPSHOT_DATE,
                "source": "seed",
                "note": "Initial development seed data as of 2026-09-19",
                "records": INITIAL_SEED_ATTENDANCE,
            }
        )

        return self.create_snapshot(
            scraped_at=INITIAL_SEED_SNAPSHOT_DATE,
            source="seed",
            raw_json=raw_payload,
            records=INITIAL_SEED_ATTENDANCE,
        )

    def seed_all(self, force: bool = False) -> dict[str, Any]:
        """
        Comprehensive database seeding:
        1. 9 canonical subjects
        2. 30+ canonical aliases
        3. Initial mock attendance snapshot
        """
        num_subjects = self.seed_canonical_subjects()
        num_aliases = self.seed_aliases()
        snapshot = self.seed_initial_snapshot(force=force)
        return {
            "subjects_seeded": num_subjects,
            "aliases_seeded": num_aliases,
            "snapshot_id": snapshot.id if snapshot else None,
        }


# ── Global Singleton and Helper Functions ─────────────────────────────────────

_default_repo: Optional[AttendanceRepository] = None


def get_repository() -> AttendanceRepository:
    """Return the singleton AttendanceRepository instance."""
    global _default_repo
    if _default_repo is None:
        _default_repo = AttendanceRepository()
    return _default_repo
