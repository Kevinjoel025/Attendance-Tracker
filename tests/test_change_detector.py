"""
tests/test_change_detector.py
─────────────────────────────
Comprehensive unit tests for AttendanceChangeDetector.

Covers all status codes:
1. PRESENT (+1/+1)
2. ABSENT (+0/+1)
3. MULTIPLE_CLASSES_REGISTERED (conducted delta > 1)
4. DATA_ANOMALY (negative delta, attended > conducted, delta_att > delta_cond)
5. NO_PREVIOUS_SNAPSHOT (baseline snapshot)
6. SCRAPE_FAILED (failed scrape status)
7. HOLIDAY (delta 0/0 on holiday)
8. NO_CLASS_SCHEDULED (delta 0/0 when subject has no scheduled class)
9. CLASS_NOT_YET_OCCURRED (delta 0/0 before class ends)
10. CLASS_MAY_NOT_BE_REGISTERED_YET (delta 0/0 post-class but before 23:00 IST)
11. POTENTIALLY_UNREGISTERED (delta 0/0 at or after 23:00 IST with fresh scrape)
12. DATA_MAY_BE_STALE (delta 0/0 post-class, but scrape timestamp is pre-class)
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo
import pytest

from attendance_tracker.attendance.change_detector import (
    AttendanceChangeDetector,
    AttendanceStatus,
    detect_change,
    get_change_detector,
)

_TZ = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def detector():
    return get_change_detector()


class TestDirectAttendanceTransitions:

    def test_present_single_class(self, detector: AttendanceChangeDetector):
        # 32/40 -> 33/41 on Monday for DE (P1: 10:00-11:00)
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=33,
            curr_conducted=41,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),  # Monday
        )
        assert res.status == AttendanceStatus.PRESENT
        assert res.delta_attended == 1
        assert res.delta_conducted == 1
        assert "Present" in res.explanation

    def test_absent_single_class(self, detector: AttendanceChangeDetector):
        # 32/40 -> 32/41
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=32,
            curr_conducted=41,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),  # Monday
        )
        assert res.status == AttendanceStatus.ABSENT
        assert res.delta_attended == 0
        assert res.delta_conducted == 1
        assert "Absent" in res.explanation

    def test_multiple_classes_registered(self, detector: AttendanceChangeDetector):
        # 32/40 -> 34/43 (+2 attended / +3 conducted)
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=34,
            curr_conducted=43,
            prev_attended=32,
            prev_conducted=40,
        )
        assert res.status == AttendanceStatus.MULTIPLE_CLASSES_REGISTERED
        assert res.delta_attended == 2
        assert res.delta_conducted == 3

    def test_no_previous_snapshot(self, detector: AttendanceChangeDetector):
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=11,
            curr_conducted=28,
            prev_attended=None,
            prev_conducted=None,
        )
        assert res.status == AttendanceStatus.NO_PREVIOUS_SNAPSHOT
        assert res.prev_attended is None
        assert res.prev_conducted is None

    def test_scrape_failed(self, detector: AttendanceChangeDetector):
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            scrape_status="FAILED",
        )
        assert res.status == AttendanceStatus.SCRAPE_FAILED


class TestDataAnomalies:

    def test_negative_conducted_delta(self, detector: AttendanceChangeDetector):
        # Conducted dropped from 40 to 39
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=32,
            curr_conducted=39,
            prev_attended=32,
            prev_conducted=40,
        )
        assert res.status == AttendanceStatus.DATA_ANOMALY
        assert res.is_anomalous is True

    def test_negative_attended_delta(self, detector: AttendanceChangeDetector):
        # Attended dropped from 32 to 31
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=31,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
        )
        assert res.status == AttendanceStatus.DATA_ANOMALY
        assert res.is_anomalous is True

    def test_attended_exceeds_conducted(self, detector: AttendanceChangeDetector):
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=42,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
        )
        assert res.status == AttendanceStatus.DATA_ANOMALY
        assert res.is_anomalous is True

    def test_delta_attended_exceeds_delta_conducted(self, detector: AttendanceChangeDetector):
        # Attended +2 but conducted +1
        res = detector.detect_change(
            subject_code="22PC1AM202",
            curr_attended=34,
            curr_conducted=41,
            prev_attended=32,
            prev_conducted=40,
        )
        assert res.status == AttendanceStatus.DATA_ANOMALY
        assert res.is_anomalous is True


class TestZeroDeltaReasoning:

    def test_holiday_status(self, detector: AttendanceChangeDetector):
        # 2026-10-02 is Gandhi Jayanti (Friday). DE has a class on Friday normally.
        # But Gandhi Jayanti cancels classes.
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 10, 2),
            current_time_override=datetime(2026, 10, 2, 12, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.HOLIDAY
        assert "Gandhi Jayanti" in res.explanation

    def test_second_saturday_holiday(self, detector: AttendanceChangeDetector):
        # 2026-09-12 is 2nd Saturday. Normally Saturday has classes (OE-1, DE, OS, IoTAP, GS).
        # On 2nd Saturday, it's a holiday.
        res = detector.detect_change(
            subject_code="22MN6HS302",  # GS
            curr_attended=0,
            curr_conducted=8,
            prev_attended=0,
            prev_conducted=8,
            reference_date=date(2026, 9, 12),
            current_time_override=datetime(2026, 9, 12, 15, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.HOLIDAY

    def test_no_class_scheduled_on_day(self, detector: AttendanceChangeDetector):
        # Monday 2026-09-21: GS (Gender Sensitization) has NO class on Monday (only Saturday)
        res = detector.detect_change(
            subject_code="22MN6HS302",  # GS
            curr_attended=0,
            curr_conducted=8,
            prev_attended=0,
            prev_conducted=8,
            reference_date=date(2026, 9, 21),  # Monday
            current_time_override=datetime(2026, 9, 21, 15, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.NO_CLASS_SCHEDULED

    def test_non_attendance_activity_does_not_create_expectation(self, detector: AttendanceChangeDetector):
        # Wednesday 2026-09-23: ECA/CCA is period 6, but is non-attendance
        res = detector.detect_change(
            subject_code="ECA/CCA",
            curr_attended=0,
            curr_conducted=0,
            prev_attended=0,
            prev_conducted=0,
            reference_date=date(2026, 9, 23),
            current_time_override=datetime(2026, 9, 23, 17, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.NO_CLASS_SCHEDULED


class TestRegistrationTimingAndObservationWindow:

    def test_case_a_before_class_end(self, detector: AttendanceChangeDetector):
        # Monday 2026-09-21: DE is Period 1 (10:00 to 11:00)
        # Current evaluation time: 10:30 (class has not ended yet)
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),
            curr_snapshot_time=datetime(2026, 9, 21, 10, 30, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 21, 10, 30, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.CLASS_NOT_YET_OCCURRED
        assert "not yet concluded" in res.explanation

    def test_case_b_after_class_end_before_2300(self, detector: AttendanceChangeDetector):
        # Monday 2026-09-21: DE (P1: 10:00–11:00) ended.
        # Evaluation time: 15:30. Fresh scrape at 15:30. Observation window open until 23:00.
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),
            curr_snapshot_time=datetime(2026, 9, 21, 15, 30, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 21, 15, 30, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.CLASS_MAY_NOT_BE_REGISTERED_YET
        assert "may not have been registered" in res.explanation

    def test_case_c_after_2300_potentially_unregistered(self, detector: AttendanceChangeDetector):
        # Monday 2026-09-21: DE (P1: 10:00–11:00) ended.
        # Evaluation time: 23:15 (after 23:00). Fresh scrape at 23:15.
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),
            curr_snapshot_time=datetime(2026, 9, 21, 23, 15, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 21, 23, 15, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.POTENTIALLY_UNREGISTERED
        assert "Attendance may not have been registered" in res.explanation

    def test_case_d_stale_data_pre_class_scrape(self, detector: AttendanceChangeDetector):
        # Monday 2026-09-21: DE is P1 (10:00–11:00).
        # Current evaluation time: 14:00 (after class).
        # BUT current snapshot was taken at 09:30 (before class ended).
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 21),
            curr_snapshot_time=datetime(2026, 9, 21, 9, 30, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 21, 14, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.DATA_MAY_BE_STALE

    def test_past_date_evaluation(self, detector: AttendanceChangeDetector):
        # Evaluating past date Monday 2026-09-14 was a holiday (Vinayaka Chavithi).
        # Evaluating Monday 2026-09-07 (past date, DE scheduled, scrape taken next day)
        res = detector.detect_change(
            subject_code="22PC1AM202",  # DE
            curr_attended=32,
            curr_conducted=40,
            prev_attended=32,
            prev_conducted=40,
            reference_date=date(2026, 9, 7),  # Past Monday
            curr_snapshot_time=datetime(2026, 9, 8, 10, 0, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 8, 10, 0, tzinfo=_TZ),
        )
        assert res.status == AttendanceStatus.POTENTIALLY_UNREGISTERED

    def test_repeated_period_class_session_end(self, detector: AttendanceChangeDetector):
        # Thursday 2026-09-24: DE LAB runs across P1 (10:00–11:00), P2 (11:00–12:00), P3 (12:00–13:00)
        # Session ends at 13:00.
        # At 11:30 (during P2): CLASS_NOT_YET_OCCURRED
        res_during = detector.detect_change(
            subject_code="22PC2AM202",  # DE LAB
            curr_attended=18,
            curr_conducted=30,
            prev_attended=18,
            prev_conducted=30,
            reference_date=date(2026, 9, 24),
            curr_snapshot_time=datetime(2026, 9, 24, 11, 30, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 24, 11, 30, tzinfo=_TZ),
        )
        assert res_during.status == AttendanceStatus.CLASS_NOT_YET_OCCURRED

        # At 14:00 (after 13:00): CLASS_MAY_NOT_BE_REGISTERED_YET
        res_after = detector.detect_change(
            subject_code="22PC2AM202",  # DE LAB
            curr_attended=18,
            curr_conducted=30,
            prev_attended=18,
            prev_conducted=30,
            reference_date=date(2026, 9, 24),
            curr_snapshot_time=datetime(2026, 9, 24, 14, 0, tzinfo=_TZ),
            current_time_override=datetime(2026, 9, 24, 14, 0, tzinfo=_TZ),
        )
        assert res_after.status == AttendanceStatus.CLASS_MAY_NOT_BE_REGISTERED_YET
