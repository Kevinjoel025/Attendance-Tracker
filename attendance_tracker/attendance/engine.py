"""
attendance/engine.py
────────────────────
Deterministic attendance calculation engine.

RULES:
- No I/O, no database, no SLM, no network.
- Every function is a pure calculation.
- The SLM NEVER calls these functions directly.
- The tool router calls them with validated parameters.
- All arithmetic is done here, not by the SLM.

Formulas:
    current_percentage      = attended / conducted × 100
    project_after_attending = (attended + n) / (conducted + n) × 100
    project_after_missing   = attended / (conducted + n) × 100
    classes_required        = ceil((T·C - 100·A) / (100 - T))  where T=target%
    max_can_miss            = floor((100·A / T) - C)            where T=threshold%

Verified examples (32/40 = 80%):
    To reach 85%:  classes_required = 14   (NOT 4)
    Max can miss at 75% threshold: max_can_miss = 2   (NOT 5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttendanceStats:
    """Current attendance statistics for one subject."""
    subject_code: str
    attended: int
    conducted: int
    percentage: float
    college_threshold: float
    personal_target: float
    above_college_threshold: bool
    above_personal_target: bool
    deficit_to_college_threshold: float   # pp below threshold (0 if above)
    deficit_to_personal_target: float     # pp below target (0 if above)


@dataclass(frozen=True)
class ProjectionResult:
    """Result of an attendance projection calculation."""
    subject_code: str
    current_attended: int
    current_conducted: int
    current_percentage: float
    action: str           # "ATTEND" | "MISS"
    count: int
    projected_attended: int
    projected_conducted: int
    projected_percentage: float
    college_threshold: float
    above_college_threshold_after: bool
    threshold_breached: bool   # True if projection drops below college threshold


@dataclass(frozen=True)
class ClassesRequiredResult:
    """Result of classes_required_for_target calculation."""
    subject_code: str
    current_attended: int
    current_conducted: int
    current_percentage: float
    target_percentage: float
    classes_required: int            # 0 if already at/above target; inf if impossible
    already_above_target: bool
    impossible: bool                 # True only when target == 100% and attended < conducted


@dataclass(frozen=True)
class MaxCanMissResult:
    """Result of max_classes_can_miss calculation."""
    subject_code: str
    current_attended: int
    current_conducted: int
    current_percentage: float
    threshold_percentage: float
    max_can_miss: int                # 0 means no classes can be missed
    already_below_threshold: bool


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_attendance(attended: int, conducted: int) -> None:
    """Raise ValueError for invalid attendance inputs."""
    if not isinstance(attended, int) or not isinstance(conducted, int):
        raise TypeError(
            f"attended and conducted must be integers, "
            f"got {type(attended).__name__} and {type(conducted).__name__}."
        )
    if attended < 0:
        raise ValueError(f"attended cannot be negative: {attended}")
    if conducted < 0:
        raise ValueError(f"conducted cannot be negative: {conducted}")
    if attended > conducted:
        raise ValueError(
            f"attended ({attended}) cannot exceed conducted ({conducted}). "
            f"This is a data anomaly."
        )


def _validate_count(n: int, param_name: str = "n") -> None:
    """Raise ValueError if n is not a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"{param_name} must be an integer, got {type(n).__name__}.")
    if n < 0:
        raise ValueError(f"{param_name} must be non-negative, got {n}.")


def _validate_percentage(pct: float, param_name: str = "percentage") -> None:
    """Raise ValueError if percentage is out of reasonable range."""
    if not isinstance(pct, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(pct).__name__}.")


# ── Core calculation functions ────────────────────────────────────────────────

def current_percentage(attended: int, conducted: int) -> float:
    """
    Calculate current attendance percentage.

    Returns 0.0 if conducted == 0 (no classes held yet).

    >>> current_percentage(32, 40)
    80.0
    >>> current_percentage(0, 0)
    0.0
    """
    _validate_attendance(attended, conducted)
    if conducted == 0:
        return 0.0
    return round((attended / conducted) * 100, 4)


def project_after_attending(attended: int, conducted: int, n: int) -> float:
    """
    Project attendance percentage after attending n additional classes.

    Both attended and conducted increase by n.

    >>> project_after_attending(32, 40, 14)
    85.185...
    >>> project_after_attending(32, 40, 0)
    80.0
    """
    _validate_attendance(attended, conducted)
    _validate_count(n, "n")
    if conducted + n == 0:
        return 0.0
    return round(((attended + n) / (conducted + n)) * 100, 4)


def project_after_missing(attended: int, conducted: int, n: int) -> float:
    """
    Project attendance percentage after missing n classes.

    Only conducted increases by n; attended stays the same.

    >>> project_after_missing(32, 40, 2)
    76.1905...
    >>> project_after_missing(32, 40, 0)
    80.0
    """
    _validate_attendance(attended, conducted)
    _validate_count(n, "n")
    new_conducted = conducted + n
    if new_conducted == 0:
        return 0.0
    return round((attended / new_conducted) * 100, 4)


def classes_required_for_target(
    attended: int,
    conducted: int,
    target_pct: float,
) -> ClassesRequiredResult:
    """
    Calculate how many consecutive classes must be attended to reach target_pct.

    Mathematical derivation:
        (attended + x) / (conducted + x) >= target_pct / 100
        attended + x >= (target_pct/100) * (conducted + x)
        attended + x >= (target_pct/100)*conducted + (target_pct/100)*x
        x - (target_pct/100)*x >= (target_pct/100)*conducted - attended
        x * (1 - target_pct/100) >= (target_pct/100)*conducted - attended
        x >= ((target_pct * conducted) - (100 * attended)) / (100 - target_pct)

    Edge cases:
        - conducted == 0, attended == 0: any target requires ceil(target/(100-target)) classes
        - already above target: returns 0
        - target == 100% and attended == conducted: already there, returns 0
        - target == 100% and attended < conducted: mathematically impossible, impossible=True
        - target <= 0: returns 0 (trivially satisfied)
        - target >= 100 and attended < conducted: impossible

    Verified example:
        classes_required_for_target(32, 40, 85.0).classes_required == 14

    >>> classes_required_for_target(32, 40, 85.0).classes_required
    14
    >>> classes_required_for_target(32, 40, 80.0).already_above_target
    True
    """
    _validate_attendance(attended, conducted)
    _validate_percentage(target_pct, "target_pct")

    curr_pct = current_percentage(attended, conducted)

    # Trivially satisfied
    if target_pct <= 0.0:
        return ClassesRequiredResult(
            subject_code="",
            current_attended=attended,
            current_conducted=conducted,
            current_percentage=curr_pct,
            target_percentage=target_pct,
            classes_required=0,
            already_above_target=True,
            impossible=False,
        )

    # 100% edge case
    if target_pct >= 100.0:
        if attended == conducted:
            # Already at 100%
            return ClassesRequiredResult(
                subject_code="",
                current_attended=attended,
                current_conducted=conducted,
                current_percentage=curr_pct,
                target_percentage=target_pct,
                classes_required=0,
                already_above_target=True,
                impossible=False,
            )
        else:
            # attended < conducted → (attended+x)/(conducted+x) → 1 as x→∞ but never =1
            return ClassesRequiredResult(
                subject_code="",
                current_attended=attended,
                current_conducted=conducted,
                current_percentage=curr_pct,
                target_percentage=target_pct,
                classes_required=0,
                already_above_target=False,
                impossible=True,
            )

    # Already at or above target
    if curr_pct >= target_pct:
        return ClassesRequiredResult(
            subject_code="",
            current_attended=attended,
            current_conducted=conducted,
            current_percentage=curr_pct,
            target_percentage=target_pct,
            classes_required=0,
            already_above_target=True,
            impossible=False,
        )

    # General case
    # x = ((target_pct * conducted) - (100 * attended)) / (100 - target_pct)
    numerator = (target_pct * conducted) - (100.0 * attended)
    denominator = 100.0 - target_pct

    # denominator > 0 guaranteed since target_pct < 100
    x = numerator / denominator
    required = max(0, math.ceil(x))

    # Edge case: 0/0 degenerate base.
    # The algebraic formula gives 0 because numerator and denominator both reduce.
    # But from 0/0, attending 1 class gives 1/1=100% which satisfies any target<100%.
    # Therefore the minimum is always 1.
    if conducted == 0 and attended == 0 and required == 0:
        required = 1

    return ClassesRequiredResult(
        subject_code="",
        current_attended=attended,
        current_conducted=conducted,
        current_percentage=curr_pct,
        target_percentage=target_pct,
        classes_required=required,
        already_above_target=False,
        impossible=False,
    )


def max_classes_can_miss(
    attended: int,
    conducted: int,
    threshold_pct: float,
) -> MaxCanMissResult:
    """
    Calculate the maximum number of future classes that can be missed
    while staying at or above threshold_pct.

    Mathematical derivation:
        attended / (conducted + x) >= threshold_pct / 100
        100 * attended >= threshold_pct * (conducted + x)
        100 * attended >= threshold_pct * conducted + threshold_pct * x
        100 * attended - threshold_pct * conducted >= threshold_pct * x
        x <= (100 * attended - threshold_pct * conducted) / threshold_pct
        x <= (100 * attended / threshold_pct) - conducted

    Edge cases:
        - already below threshold: returns 0, already_below_threshold=True
        - threshold == 0: returns infinity (can miss infinite classes)
        - threshold == 100: can miss 0 classes unless attended == conducted
        - conducted == 0: degenerate case, returns 0

    Verified example:
        max_classes_can_miss(32, 40, 75.0).max_can_miss == 2

    >>> max_classes_can_miss(32, 40, 75.0).max_can_miss
    2
    """
    _validate_attendance(attended, conducted)
    _validate_percentage(threshold_pct, "threshold_pct")

    curr_pct = current_percentage(attended, conducted)

    # Threshold of 0% — can miss anything
    if threshold_pct <= 0.0:
        return MaxCanMissResult(
            subject_code="",
            current_attended=attended,
            current_conducted=conducted,
            current_percentage=curr_pct,
            threshold_percentage=threshold_pct,
            max_can_miss=10**9,   # effectively infinite; use a large sentinel
            already_below_threshold=False,
        )

    # Already below threshold
    if curr_pct < threshold_pct:
        return MaxCanMissResult(
            subject_code="",
            current_attended=attended,
            current_conducted=conducted,
            current_percentage=curr_pct,
            threshold_percentage=threshold_pct,
            max_can_miss=0,
            already_below_threshold=True,
        )

    # Threshold of 100% — can only miss 0 if already at 100%, else 0
    if threshold_pct >= 100.0:
        return MaxCanMissResult(
            subject_code="",
            current_attended=attended,
            current_conducted=conducted,
            current_percentage=curr_pct,
            threshold_percentage=threshold_pct,
            max_can_miss=0,
            already_below_threshold=(curr_pct < 100.0),
        )

    # conducted == 0 edge case
    if conducted == 0:
        return MaxCanMissResult(
            subject_code="",
            current_attended=0,
            current_conducted=0,
            current_percentage=0.0,
            threshold_percentage=threshold_pct,
            max_can_miss=0,
            already_below_threshold=True,
        )

    # General case
    # x <= (100 * attended / threshold_pct) - conducted
    x = (100.0 * attended / threshold_pct) - conducted
    can_miss = max(0, math.floor(x))

    return MaxCanMissResult(
        subject_code="",
        current_attended=attended,
        current_conducted=conducted,
        current_percentage=curr_pct,
        threshold_percentage=threshold_pct,
        max_can_miss=can_miss,
        already_below_threshold=False,
    )


# ── Combined stats ────────────────────────────────────────────────────────────

def compute_stats(
    subject_code: str,
    attended: int,
    conducted: int,
    college_threshold: float = 75.0,
    personal_target: float = 70.0,
) -> AttendanceStats:
    """
    Compute full attendance statistics for one subject.

    Returns an AttendanceStats dataclass with all derived fields.
    """
    _validate_attendance(attended, conducted)
    pct = current_percentage(attended, conducted)
    return AttendanceStats(
        subject_code=subject_code,
        attended=attended,
        conducted=conducted,
        percentage=pct,
        college_threshold=college_threshold,
        personal_target=personal_target,
        above_college_threshold=(pct >= college_threshold),
        above_personal_target=(pct >= personal_target),
        deficit_to_college_threshold=max(0.0, round(college_threshold - pct, 4)),
        deficit_to_personal_target=max(0.0, round(personal_target - pct, 4)),
    )


def compute_projection(
    subject_code: str,
    attended: int,
    conducted: int,
    action: str,
    count: int,
    college_threshold: float = 75.0,
) -> ProjectionResult:
    """
    Compute an attendance projection.

    action: "ATTEND" | "MISS"
    count:  number of future classes
    """
    action = action.upper()
    if action not in ("ATTEND", "MISS"):
        raise ValueError(f"action must be 'ATTEND' or 'MISS', got '{action}'.")
    _validate_attendance(attended, conducted)
    _validate_count(count, "count")

    curr_pct = current_percentage(attended, conducted)

    if action == "ATTEND":
        proj_attended = attended + count
        proj_conducted = conducted + count
    else:  # MISS
        proj_attended = attended
        proj_conducted = conducted + count

    proj_pct = round((proj_attended / proj_conducted) * 100, 4) if proj_conducted > 0 else 0.0
    above_after = proj_pct >= college_threshold

    return ProjectionResult(
        subject_code=subject_code,
        current_attended=attended,
        current_conducted=conducted,
        current_percentage=curr_pct,
        action=action,
        count=count,
        projected_attended=proj_attended,
        projected_conducted=proj_conducted,
        projected_percentage=proj_pct,
        college_threshold=college_threshold,
        above_college_threshold_after=above_after,
        threshold_breached=(curr_pct >= college_threshold and not above_after),
    )
