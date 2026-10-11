"""
tests/test_engine.py
────────────────────
Comprehensive unit tests for the attendance calculation engine.

Tests include:
- Standard cases
- Edge cases (conducted=0, attended=0, target=100%, threshold=0/100)
- Brute-force verification against exhaustive search
- Verified examples from the architecture docs
- ValueError / TypeError cases

BRUTE-FORCE VERIFICATION:
For classes_required_for_target and max_classes_can_miss,
we cross-check the algebraic result against an exhaustive integer search.
This ensures the formulas are correct for all small integer cases.
"""

import math
import pytest

from attendance_tracker.attendance.engine import (
    AttendanceStats,
    ClassesRequiredResult,
    MaxCanMissResult,
    ProjectionResult,
    classes_required_for_target,
    compute_projection,
    compute_stats,
    current_percentage,
    max_classes_can_miss,
    project_after_attending,
    project_after_missing,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. current_percentage
# ══════════════════════════════════════════════════════════════════════════════

class TestCurrentPercentage:

    def test_standard(self):
        assert current_percentage(32, 40) == 80.0

    def test_zero_zero(self):
        # conducted=0 → return 0.0, no division error
        assert current_percentage(0, 0) == 0.0

    def test_zero_attended_nonzero_conducted(self):
        assert current_percentage(0, 10) == 0.0

    def test_perfect_attendance(self):
        assert current_percentage(40, 40) == 100.0

    def test_one_of_one(self):
        assert current_percentage(1, 1) == 100.0

    def test_rounding(self):
        result = current_percentage(1, 3)
        assert abs(result - 33.3333) < 0.001

    def test_large_values(self):
        result = current_percentage(750, 1000)
        assert result == 75.0

    def test_invalid_attended_gt_conducted(self):
        with pytest.raises(ValueError, match="cannot exceed conducted"):
            current_percentage(41, 40)

    def test_invalid_negative_attended(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            current_percentage(-1, 40)

    def test_invalid_negative_conducted(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            current_percentage(10, -1)

    def test_invalid_type_attended(self):
        with pytest.raises(TypeError):
            current_percentage(32.5, 40)

    def test_invalid_type_conducted(self):
        with pytest.raises(TypeError):
            current_percentage(32, "40")

    def test_real_data_os(self):
        # OS: 17/28
        result = current_percentage(17, 28)
        assert abs(result - 60.7143) < 0.001

    def test_real_data_iotap_lab(self):
        # IoTAP LAB: 9/27
        result = current_percentage(9, 27)
        assert abs(result - 33.3333) < 0.001

    def test_real_data_overall(self):
        # Overall: 119/236
        result = current_percentage(119, 236)
        assert abs(result - 50.4237) < 0.001


# ══════════════════════════════════════════════════════════════════════════════
# 2. project_after_attending
# ══════════════════════════════════════════════════════════════════════════════

class TestProjectAfterAttending:

    def test_standard(self):
        # 32/40 + attend 14 = 46/54 ≈ 85.185%
        result = project_after_attending(32, 40, 14)
        assert abs(result - 85.1852) < 0.001

    def test_zero_extra(self):
        # Attending 0 more classes → same percentage
        result = project_after_attending(32, 40, 0)
        assert result == 80.0

    def test_attend_large_number(self):
        # Attending many classes should push percentage toward 100
        result = project_after_attending(0, 10, 1000)
        assert result > 99.0

    def test_attend_from_zero_base(self):
        # Start from 0/0, attend 10 → 100%
        result = project_after_attending(0, 0, 10)
        assert result == 100.0

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            project_after_attending(32, 40, -1)

    def test_invalid_n_type(self):
        with pytest.raises(TypeError):
            project_after_attending(32, 40, 2.5)

    def test_monotonically_increasing(self):
        # More attended classes → higher percentage
        prev = current_percentage(17, 28)
        for n in range(1, 20):
            curr = project_after_attending(17, 28, n)
            assert curr > prev, f"Non-monotonic at n={n}"
            prev = curr


# ══════════════════════════════════════════════════════════════════════════════
# 3. project_after_missing
# ══════════════════════════════════════════════════════════════════════════════

class TestProjectAfterMissing:

    def test_standard_miss_2(self):
        # 32/40, miss 2 → 32/42 ≈ 76.19%
        result = project_after_missing(32, 40, 2)
        assert abs(result - 76.1905) < 0.001

    def test_standard_miss_3(self):
        # 32/40, miss 3 → 32/43 ≈ 74.42%
        result = project_after_missing(32, 40, 3)
        assert abs(result - 74.4186) < 0.001

    def test_zero_miss(self):
        result = project_after_missing(32, 40, 0)
        assert result == 80.0

    def test_miss_many(self):
        # Missing many classes → percentage approaches 0
        result = project_after_missing(1, 10, 10000)
        assert result < 0.1

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            project_after_missing(32, 40, -1)

    def test_monotonically_decreasing(self):
        prev = current_percentage(17, 28)
        for n in range(1, 20):
            curr = project_after_missing(17, 28, n)
            assert curr < prev, f"Non-monotonic at n={n}"
            prev = curr


# ══════════════════════════════════════════════════════════════════════════════
# 4. classes_required_for_target — BRUTE FORCE VERIFIED
# ══════════════════════════════════════════════════════════════════════════════

def _brute_force_classes_required(attended: int, conducted: int, target_pct: float) -> int:
    """Exhaustive search for the minimum classes needed to reach target_pct."""
    if target_pct >= 100.0 and attended < conducted:
        return -1  # impossible sentinel
    if target_pct >= 100.0 and attended == conducted:
        return 0
    if target_pct <= 0.0:
        return 0
    curr = (attended / conducted * 100) if conducted > 0 else 0.0
    if curr >= target_pct:
        return 0
    for x in range(1, 10_000):
        new_pct = ((attended + x) / (conducted + x)) * 100
        if new_pct >= target_pct:
            return x
    return -1  # not found in search range


class TestClassesRequiredForTarget:

    # ── Architecture-specified verified examples ──────────────────────────────

    def test_documented_example_85(self):
        """The architecture document states: 32/40 to reach 85% requires 14 classes."""
        result = classes_required_for_target(32, 40, 85.0)
        assert result.classes_required == 14, (
            f"Expected 14, got {result.classes_required}. "
            "This is a documented verified example."
        )
        assert not result.already_above_target
        assert not result.impossible

    def test_documented_example_80_already_there(self):
        """32/40 = 80%, already at target 80%."""
        result = classes_required_for_target(32, 40, 80.0)
        assert result.already_above_target
        assert result.classes_required == 0

    def test_already_above_target(self):
        result = classes_required_for_target(32, 40, 75.0)
        assert result.already_above_target
        assert result.classes_required == 0

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_target_zero(self):
        result = classes_required_for_target(0, 10, 0.0)
        assert result.classes_required == 0
        assert result.already_above_target

    def test_target_100_with_perfect_attendance(self):
        result = classes_required_for_target(10, 10, 100.0)
        assert result.classes_required == 0
        assert result.already_above_target
        assert not result.impossible

    def test_target_100_with_imperfect_attendance(self):
        result = classes_required_for_target(9, 10, 100.0)
        assert result.impossible
        assert not result.already_above_target

    def test_zero_zero_to_75(self):
        # 0/0 → need 1 class (1/1 = 100% ≥ 75%)
        result = classes_required_for_target(0, 0, 75.0)
        assert result.classes_required == 1

    def test_zero_attended_to_75(self):
        # 0/10 = 0% → need classes to reach 75%
        result = classes_required_for_target(0, 10, 75.0)
        bf = _brute_force_classes_required(0, 10, 75.0)
        assert result.classes_required == bf

    def test_negative_target_raises_no_error_returns_trivial(self):
        # Negative target is trivially satisfied
        result = classes_required_for_target(5, 10, -10.0)
        assert result.classes_required == 0
        assert result.already_above_target

    # ── Real subject data ─────────────────────────────────────────────────────

    def test_real_de_to_75(self):
        # DE: 11/28 = 39.29% → to reach 75%
        result = classes_required_for_target(11, 28, 75.0)
        bf = _brute_force_classes_required(11, 28, 75.0)
        assert result.classes_required == bf

    def test_real_acd_to_75(self):
        # ACD: 21/33 = 63.64% → to reach 75%
        result = classes_required_for_target(21, 33, 75.0)
        bf = _brute_force_classes_required(21, 33, 75.0)
        assert result.classes_required == bf

    # ── Brute-force verification across integer grid ──────────────────────────

    @pytest.mark.parametrize("attended,conducted", [
        (0, 0), (0, 5), (0, 10), (5, 10), (7, 10), (10, 10),
        (17, 28), (21, 33), (11, 28), (9, 27), (19, 35),
        (32, 40), (15, 27), (18, 30),
    ])
    @pytest.mark.parametrize("target", [50.0, 60.0, 70.0, 75.0, 80.0, 85.0])
    def test_brute_force_grid(self, attended, conducted, target):
        if attended > conducted:
            return  # Invalid input, skip
        result = classes_required_for_target(attended, conducted, target)
        expected = _brute_force_classes_required(attended, conducted, target)
        assert result.classes_required == expected, (
            f"MISMATCH: {attended}/{conducted} to {target}% → "
            f"formula={result.classes_required}, brute_force={expected}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. max_classes_can_miss — BRUTE FORCE VERIFIED
# ══════════════════════════════════════════════════════════════════════════════

def _brute_force_max_can_miss(attended: int, conducted: int, threshold_pct: float) -> int:
    """Exhaustive search for the maximum classes that can be missed."""
    if threshold_pct <= 0.0:
        return 10**9  # effectively infinite
    curr = (attended / conducted * 100) if conducted > 0 else 0.0
    if curr < threshold_pct:
        return 0
    for x in range(0, 10_000):
        new_pct = (attended / (conducted + x + 1)) * 100
        if new_pct < threshold_pct:
            return x
    return 10_000  # large number


class TestMaxClassesCanMiss:

    # ── Architecture-specified verified examples ──────────────────────────────

    def test_documented_example_75(self):
        """The architecture document states: 32/40 at 75% threshold → max 2 can miss."""
        result = max_classes_can_miss(32, 40, 75.0)
        assert result.max_can_miss == 2, (
            f"Expected 2, got {result.max_can_miss}. "
            "This is a documented verified example."
        )
        assert not result.already_below_threshold

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_already_below_threshold(self):
        result = max_classes_can_miss(9, 27, 75.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    def test_threshold_zero(self):
        result = max_classes_can_miss(0, 0, 0.0)
        assert result.max_can_miss >= 10**9  # effectively infinite

    def test_threshold_100_perfect_attendance(self):
        # At exactly 100% with threshold 100% → can miss 0
        result = max_classes_can_miss(10, 10, 100.0)
        assert result.max_can_miss == 0

    def test_threshold_100_imperfect(self):
        result = max_classes_can_miss(9, 10, 100.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    def test_conducted_zero(self):
        result = max_classes_can_miss(0, 0, 75.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    def test_personal_target_70(self):
        # At 80%, threshold 70% — how many can miss?
        result = max_classes_can_miss(32, 40, 70.0)
        bf = _brute_force_max_can_miss(32, 40, 70.0)
        assert result.max_can_miss == bf

    # ── Real subject data ─────────────────────────────────────────────────────

    def test_real_gs_cannot_miss(self):
        # GS: 0/8 = 0% < 75% → cannot miss any
        result = max_classes_can_miss(0, 8, 75.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    def test_real_de_lab(self):
        # DE LAB: 18/30 = 60% < 75% → cannot miss any
        result = max_classes_can_miss(18, 30, 75.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    def test_real_acd_at_threshold(self):
        # ACD: 21/33 = 63.64% < 75%
        result = max_classes_can_miss(21, 33, 75.0)
        assert result.max_can_miss == 0
        assert result.already_below_threshold

    # ── Brute-force verification across integer grid ──────────────────────────

    @pytest.mark.parametrize("attended,conducted", [
        (0, 0), (0, 5), (5, 10), (7, 10), (10, 10),
        (17, 28), (21, 33), (11, 28), (9, 27), (19, 35),
        (32, 40), (15, 27), (18, 30),
    ])
    @pytest.mark.parametrize("threshold", [50.0, 60.0, 70.0, 75.0, 80.0])
    def test_brute_force_grid(self, attended, conducted, threshold):
        if attended > conducted:
            return
        if conducted == 0 and threshold > 0:
            return  # 0/0 is below threshold, already tested
        result = max_classes_can_miss(attended, conducted, threshold)
        expected = _brute_force_max_can_miss(attended, conducted, threshold)
        if expected >= 10_000:
            assert result.max_can_miss >= 9999, (
                f"Expected large value, got {result.max_can_miss}"
            )
        else:
            assert result.max_can_miss == expected, (
                f"MISMATCH: {attended}/{conducted} threshold={threshold}% → "
                f"formula={result.max_can_miss}, brute_force={expected}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. compute_projection
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeProjection:

    def test_miss_projection(self):
        result = compute_projection("OS", 17, 28, "MISS", 3, college_threshold=75.0)
        assert isinstance(result, ProjectionResult)
        assert result.projected_attended == 17
        assert result.projected_conducted == 31
        expected_pct = round(17 / 31 * 100, 4)
        assert abs(result.projected_percentage - expected_pct) < 0.001
        assert result.above_college_threshold_after is False
        assert result.threshold_breached is False  # was already below

    def test_attend_projection(self):
        result = compute_projection("ACD", 21, 33, "ATTEND", 5, college_threshold=75.0)
        assert result.projected_attended == 26
        assert result.projected_conducted == 38
        expected_pct = round(26 / 38 * 100, 4)
        assert abs(result.projected_percentage - expected_pct) < 0.001

    def test_invalid_action(self):
        with pytest.raises(ValueError, match="action must be"):
            compute_projection("OS", 17, 28, "SKIP", 3)

    def test_zero_count(self):
        result = compute_projection("OS", 17, 28, "MISS", 0)
        assert abs(result.projected_percentage - current_percentage(17, 28)) < 0.001

    def test_threshold_breached_flag(self):
        # 32/40 = 80% (above 75%). Miss 3 → 32/43 = 74.4% (below 75%) → breached!
        result = compute_projection("TEST", 32, 40, "MISS", 3, college_threshold=75.0)
        assert result.threshold_breached is True
        assert not result.above_college_threshold_after

    def test_case_insensitive_action(self):
        # Should accept lowercase
        result = compute_projection("OS", 17, 28, "miss", 3)
        assert result.action == "MISS"


# ══════════════════════════════════════════════════════════════════════════════
# 7. compute_stats
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeStats:

    def test_all_real_subjects(self):
        subjects = [
            ("22MN6HS302", 0, 8),
            ("22OE1EN301", 9, 20),
            ("22PC1AM202", 11, 28),
            ("22PC1AM203", 21, 33),
            ("22PC1IN301", 15, 27),
            ("22PC1IT202", 17, 28),
            ("22PC2AM202", 18, 30),
            ("22PC2IN301", 9, 27),
            ("22PE1AM301", 19, 35),
        ]
        for code, attended, conducted in subjects:
            stats = compute_stats(code, attended, conducted, 75.0, 70.0)
            assert isinstance(stats, AttendanceStats)
            expected_pct = current_percentage(attended, conducted)
            assert abs(stats.percentage - expected_pct) < 0.001
            assert stats.above_college_threshold == (expected_pct >= 75.0)
            assert stats.above_personal_target == (expected_pct >= 70.0)
            # None of the current subjects are above threshold
            assert not stats.above_college_threshold
            assert not stats.above_personal_target

    def test_deficit_calculation(self):
        stats = compute_stats("TEST", 32, 40, 75.0, 70.0)
        # 80% is above both thresholds
        assert stats.above_college_threshold
        assert stats.above_personal_target
        assert stats.deficit_to_college_threshold == 0.0
        assert stats.deficit_to_personal_target == 0.0

    def test_deficit_below_threshold(self):
        stats = compute_stats("TEST", 17, 28, 75.0, 70.0)
        # 60.71% — deficit to 75% is 14.29%, deficit to 70% is 9.29%
        assert stats.deficit_to_college_threshold > 0
        assert stats.deficit_to_personal_target > 0
        assert abs(stats.deficit_to_college_threshold - 14.2857) < 0.01
