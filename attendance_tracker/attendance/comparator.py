"""
attendance/comparator.py
─────────────────────────
Authoritative deterministic attendance snapshot comparison engine.

Answers queries like:
- "What changed today?"
- "Did my attendance update?"
- "What classes were registered today?"
- "Which subjects changed?"
- "How much did my attendance increase?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.attendance.engine import current_percentage
from attendance_tracker.attendance.repository import AttendanceRepository, AttendanceSnapshot, get_repository
from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubjectDelta:
    """Attendance delta for a single subject between two snapshots."""
    subject_code: str
    short_name: str
    full_name: str
    prev_attended: Optional[int]
    prev_conducted: Optional[int]
    curr_attended: int
    curr_conducted: int
    delta_attended: int
    delta_conducted: int
    prev_percentage: Optional[float]
    curr_percentage: float
    status: str  # "PRESENT" | "ABSENT" | "MULTIPLE" | "UNCHANGED" | "NEW"
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_code": self.subject_code,
            "short_name": self.short_name,
            "full_name": self.full_name,
            "prev_attended": self.prev_attended,
            "prev_conducted": self.prev_conducted,
            "curr_attended": self.curr_attended,
            "curr_conducted": self.curr_conducted,
            "delta_attended": self.delta_attended,
            "delta_conducted": self.delta_conducted,
            "prev_percentage": self.prev_percentage,
            "curr_percentage": self.curr_percentage,
            "status": self.status,
            "description": self.description,
        }


@dataclass(frozen=True)
class SnapshotComparisonResult:
    """Structured result of comparing two attendance snapshots."""
    curr_snapshot_id: Optional[int]
    prev_snapshot_id: Optional[int]
    curr_timestamp: str
    prev_timestamp: Optional[str]
    has_changes: bool
    changed_subjects_count: int
    overall_prev_attended: Optional[int]
    overall_prev_conducted: Optional[int]
    overall_curr_attended: int
    overall_curr_conducted: int
    overall_delta_attended: int
    overall_delta_conducted: int
    overall_prev_percentage: Optional[float]
    overall_curr_percentage: float
    overall_delta_percentage: float
    subject_deltas: list[SubjectDelta]
    summary_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "curr_snapshot_id": self.curr_snapshot_id,
            "prev_snapshot_id": self.prev_snapshot_id,
            "curr_timestamp": self.curr_timestamp,
            "prev_timestamp": self.prev_timestamp,
            "has_changes": self.has_changes,
            "changed_subjects_count": self.changed_subjects_count,
            "overall_prev_attended": self.overall_prev_attended,
            "overall_prev_conducted": self.overall_prev_conducted,
            "overall_curr_attended": self.overall_curr_attended,
            "overall_curr_conducted": self.overall_curr_conducted,
            "overall_delta_attended": self.overall_delta_attended,
            "overall_delta_conducted": self.overall_delta_conducted,
            "overall_prev_percentage": self.overall_prev_percentage,
            "overall_curr_percentage": self.overall_curr_percentage,
            "overall_delta_percentage": self.overall_delta_percentage,
            "subject_deltas": [d.to_dict() for d in self.subject_deltas],
            "summary_text": self.summary_text,
        }


# ── Comparison Engine ─────────────────────────────────────────────────────────

class SnapshotComparisonEngine:
    """
    Compares consecutive or specified attendance snapshots.
    """

    def __init__(
        self,
        repository: Optional[AttendanceRepository] = None,
        subject_resolver: Optional[SubjectResolver] = None,
    ) -> None:
        self.repo = repository or get_repository()
        self.resolver = subject_resolver or get_subject_resolver()

    def compare_latest_with_previous(self) -> SnapshotComparisonResult:
        """Compare the most recent snapshot against the one before it."""
        latest = self.repo.get_latest_snapshot()
        if latest is None:
            return SnapshotComparisonResult(
                curr_snapshot_id=None,
                prev_snapshot_id=None,
                curr_timestamp=datetime.now(_TZ).isoformat(),
                prev_timestamp=None,
                has_changes=False,
                changed_subjects_count=0,
                overall_prev_attended=None,
                overall_prev_conducted=None,
                overall_curr_attended=0,
                overall_curr_conducted=0,
                overall_delta_attended=0,
                overall_delta_conducted=0,
                overall_prev_percentage=None,
                overall_curr_percentage=0.0,
                overall_delta_percentage=0.0,
                subject_deltas=[],
                summary_text="No attendance records available in the database yet.",
            )

        prev = self.repo.get_previous_snapshot(before_snapshot_id=latest.id)
        return self.compare_snapshots(
            curr_snapshot_id=latest.id,
            prev_snapshot_id=prev.id if prev else None,
        )

    def compare_snapshots(
        self,
        curr_snapshot_id: int,
        prev_snapshot_id: Optional[int] = None,
    ) -> SnapshotComparisonResult:
        """Compare two specific snapshots."""
        curr_overall = self.repo.get_all_current_attendance(snapshot_id=curr_snapshot_id)
        if curr_overall is None:
            raise ValueError(f"Snapshot #{curr_snapshot_id} not found.")

        prev_overall = None
        if prev_snapshot_id is not None:
            prev_overall = self.repo.get_all_current_attendance(snapshot_id=prev_snapshot_id)

        # Build maps of subject codes
        curr_map = {r.subject.code: r for r in curr_overall.subject_records}
        prev_map = {r.subject.code: r for r in prev_overall.subject_records} if prev_overall else {}

        all_codes = sorted(set(curr_map.keys()) | set(prev_map.keys()))
        deltas: list[SubjectDelta] = []
        changed_count = 0

        for code in all_codes:
            curr_rec = curr_map.get(code)
            prev_rec = prev_map.get(code)

            subj = self.resolver.resolve(code)
            short = subj.short_name if subj else code
            full = subj.full_name if subj else code

            if curr_rec and not prev_rec:
                # Newly tracked
                deltas.append(
                    SubjectDelta(
                        subject_code=code,
                        short_name=short,
                        full_name=full,
                        prev_attended=None,
                        prev_conducted=None,
                        curr_attended=curr_rec.attended,
                        curr_conducted=curr_rec.conducted,
                        delta_attended=curr_rec.attended,
                        delta_conducted=curr_rec.conducted,
                        prev_percentage=None,
                        curr_percentage=curr_rec.percentage,
                        status="NEW",
                        description=f"Initial record: {curr_rec.attended}/{curr_rec.conducted} ({curr_rec.percentage:.2f}%)",
                    )
                )
            elif curr_rec and prev_rec:
                d_att = curr_rec.attended - prev_rec.attended
                d_cond = curr_rec.conducted - prev_rec.conducted

                if d_cond == 0:
                    status = "UNCHANGED"
                    desc = f"Unchanged: {curr_rec.attended}/{curr_rec.conducted} ({curr_rec.percentage:.2f}%)"
                elif d_cond == 1 and d_att == 1:
                    status = "PRESENT"
                    changed_count += 1
                    desc = f"Increased from {prev_rec.attended}/{prev_rec.conducted} to {curr_rec.attended}/{curr_rec.conducted} (+1 attended, +1 conducted). Marked Present."
                elif d_cond == 1 and d_att == 0:
                    status = "ABSENT"
                    changed_count += 1
                    desc = f"Changed from {prev_rec.attended}/{prev_rec.conducted} to {curr_rec.attended}/{curr_rec.conducted} (+0 attended, +1 conducted). Marked Absent."
                else:
                    status = "MULTIPLE"
                    changed_count += 1
                    desc = f"Changed from {prev_rec.attended}/{prev_rec.conducted} to {curr_rec.attended}/{curr_rec.conducted} (+{d_att} attended, +{d_cond} conducted)."

                deltas.append(
                    SubjectDelta(
                        subject_code=code,
                        short_name=short,
                        full_name=full,
                        prev_attended=prev_rec.attended,
                        prev_conducted=prev_rec.conducted,
                        curr_attended=curr_rec.attended,
                        curr_conducted=curr_rec.conducted,
                        delta_attended=d_att,
                        delta_conducted=d_cond,
                        prev_percentage=prev_rec.percentage,
                        curr_percentage=curr_rec.percentage,
                        status=status,
                        description=desc,
                    )
                )

        has_changes = changed_count > 0

        # Overall totals
        overall_prev_att = prev_overall.total_attended if prev_overall else None
        overall_prev_cond = prev_overall.total_conducted if prev_overall else None
        overall_prev_pct = prev_overall.overall_percentage if prev_overall else None

        overall_curr_att = curr_overall.total_attended
        overall_curr_cond = curr_overall.total_conducted
        overall_curr_pct = curr_overall.overall_percentage

        if prev_overall:
            overall_d_att = overall_curr_att - prev_overall.total_attended
            overall_d_cond = overall_curr_cond - prev_overall.total_conducted
            overall_d_pct = round(overall_curr_pct - prev_overall.overall_percentage, 2)
        else:
            overall_d_att = 0
            overall_d_cond = 0
            overall_d_pct = 0.0

        # Construct human-readable summary
        if not prev_overall:
            summary_text = (
                f"Attendance Update Status:\n"
                f"Initial attendance snapshot established: {overall_curr_att}/{overall_curr_cond} ({overall_curr_pct:.2f}%). "
                f"No previous baseline snapshot to compare against."
            )
        elif not has_changes:
            summary_text = (
                f"Attendance Update Status:\n"
                f"No attendance changes detected. Your attendance remains at "
                f"{overall_curr_att}/{overall_curr_cond} ({overall_curr_pct:.2f}%)."
            )
        else:
            changed_items = [d for d in deltas if d.status in ("PRESENT", "ABSENT", "MULTIPLE")]
            change_details = "\n".join(f"- {d.short_name}: {d.description}" for d in changed_items)
            sign = "+" if overall_d_pct >= 0 else ""
            summary_text = (
                f"Attendance Changes Detected across {len(changed_items)} subject(s):\n"
                f"{change_details}\n\n"
                f"Overall Attendance: {overall_prev_att}/{overall_prev_cond} ({overall_prev_pct:.2f}%) → "
                f"{overall_curr_att}/{overall_curr_cond} ({overall_curr_pct:.2f}%) ({sign}{overall_d_pct:.2f}%)."
            )

        return SnapshotComparisonResult(
            curr_snapshot_id=curr_snapshot_id,
            prev_snapshot_id=prev_snapshot_id,
            curr_timestamp=curr_overall.scraped_at,
            prev_timestamp=prev_overall.scraped_at if prev_overall else None,
            has_changes=has_changes,
            changed_subjects_count=changed_count,
            overall_prev_attended=overall_prev_att,
            overall_prev_conducted=overall_prev_cond,
            overall_curr_attended=overall_curr_att,
            overall_curr_conducted=overall_curr_cond,
            overall_delta_attended=overall_d_att,
            overall_delta_conducted=overall_d_cond,
            overall_prev_percentage=overall_prev_pct,
            overall_curr_percentage=overall_curr_pct,
            overall_delta_percentage=overall_d_pct,
            subject_deltas=deltas,
            summary_text=summary_text,
        )


# ── Global Singleton Accessor ─────────────────────────────────────────────────

_default_comparator: Optional[SnapshotComparisonEngine] = None


def get_comparison_engine(
    repository: Optional[AttendanceRepository] = None,
    subject_resolver: Optional[SubjectResolver] = None,
) -> SnapshotComparisonEngine:
    """Return the singleton SnapshotComparisonEngine instance."""
    global _default_comparator
    if _default_comparator is None:
        _default_comparator = SnapshotComparisonEngine(
            repository=repository,
            subject_resolver=subject_resolver,
        )
    return _default_comparator
