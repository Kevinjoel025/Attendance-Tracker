"""
tests/test_timetable.py
───────────────────────
Comprehensive unit tests for TimetableService.

Covers:
- Schedule for all weekdays (Monday–Friday) and Saturday
- Sunday (no classes)
- Pre-effective date handling (< 2026-07-16)
- Period times and period object properties
- Lunch break detection (13:00–13:40)
- Non-attendance activities (LIB, SPORTS, ECA/CCA, SL, MTP)
- Attendance-bearing subject detection
- Repeated periods (OE-1, DE LAB, IoTAP LAB, GS)
- Datetime lookup (exact start, mid-period, boundary, lunch, outside hours)
- Subject query filtering
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo
import pytest

from attendance_tracker.timetable.service import (
    PeriodInfo,
    ScheduledClass,
    TimetableService,
    get_timetable_service,
)

_TZ = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def service():
    return get_timetable_service()


class TestTimetableStructure:

    def test_effective_date(self, service: TimetableService):
        assert service.get_effective_date() == date(2026, 7, 16)
        assert not service.is_effective(date(2026, 7, 15))
        assert service.is_effective(date(2026, 7, 16))
        assert service.is_effective(date(2026, 9, 19))

    def test_all_six_periods_defined(self, service: TimetableService):
        for p in range(1, 7):
            info = service.get_period_info(p)
            assert info is not None
            assert info.period_num == p

        assert service.get_period_info(1).start_time == "10:00"
        assert service.get_period_info(1).end_time == "11:00"
        assert service.get_period_info(2).start_time == "11:00"
        assert service.get_period_info(2).end_time == "12:00"
        assert service.get_period_info(3).start_time == "12:00"
        assert service.get_period_info(3).end_time == "13:00"
        assert service.get_period_info(4).start_time == "13:40"
        assert service.get_period_info(4).end_time == "14:40"
        assert service.get_period_info(5).start_time == "14:40"
        assert service.get_period_info(5).end_time == "15:40"
        assert service.get_period_info(6).start_time == "15:40"
        assert service.get_period_info(6).end_time == "16:40"

    def test_lunch_break_detection(self, service: TimetableService):
        # Lunch is 13:00 to 13:40
        assert service.is_lunch_break(time(13, 0)) is True
        assert service.is_lunch_break(time(13, 20)) is True
        assert service.is_lunch_break(time(13, 39, 59)) is True
        assert service.is_lunch_break(time(13, 40)) is False
        assert service.is_lunch_break(time(12, 59)) is False
        assert service.is_lunch_break(time(10, 30)) is False

    def test_sunday_returns_no_classes(self, service: TimetableService):
        # 2026-09-20 is a Sunday
        sunday = date(2026, 9, 20)
        assert sunday.weekday() == 6
        assert service.get_classes_for_date(sunday) == []
        assert service.get_attendance_bearing_classes(sunday) == []

    def test_date_before_effective_returns_empty(self, service: TimetableService):
        # 2026-07-13 is a Monday before 2026-07-16
        pre_date = date(2026, 7, 13)
        assert service.get_classes_for_date(pre_date) == []


class TestDailySchedules:

    def test_monday_schedule(self, service: TimetableService):
        # 2026-09-21 is a Monday
        mon = date(2026, 9, 21)
        classes = service.get_classes_for_date(mon)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["DE", "WP", "OS", "ACD", "OE-1", "OE-1"]
        assert all(c.is_attendance_bearing for c in classes)

        # Verify repeated OE-1
        oe1_classes = [c for c in classes if c.subject == "OE-1"]
        assert len(oe1_classes) == 2
        assert oe1_classes[0].period_num == 5
        assert oe1_classes[1].period_num == 6

    def test_tuesday_schedule(self, service: TimetableService):
        # 2026-09-22 is a Tuesday
        tue = date(2026, 9, 22)
        classes = service.get_classes_for_date(tue)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["WP", "IoTAP", "ACD", "IoTAP LAB", "IoTAP LAB", "IoTAP LAB"]
        assert all(c.is_attendance_bearing for c in classes)

        # Verify 3 consecutive IoTAP LAB periods
        lab_classes = [c for c in classes if c.subject == "IoTAP LAB"]
        assert len(lab_classes) == 3
        assert [c.period_num for c in lab_classes] == [4, 5, 6]

    def test_wednesday_schedule(self, service: TimetableService):
        # 2026-09-23 is a Wednesday
        wed = date(2026, 9, 23)
        classes = service.get_classes_for_date(wed)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["DE", "WP", "IoTAP", "ACD", "OS", "ECA/CCA"]

        # ECA/CCA is non-attendance
        att_classes = service.get_attendance_bearing_classes(wed)
        assert len(att_classes) == 5
        assert [c.subject for c in att_classes] == ["DE", "WP", "IoTAP", "ACD", "OS"]
        assert classes[5].is_attendance_bearing is False

    def test_thursday_schedule(self, service: TimetableService):
        # 2026-09-24 is a Thursday
        thu = date(2026, 9, 24)
        classes = service.get_classes_for_date(thu)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["DE LAB", "DE LAB", "DE LAB", "WP", "LIB", "SPORTS"]

        # DE LAB 3 periods, WP 1 period, LIB & SPORTS non-attendance
        att_classes = service.get_attendance_bearing_classes(thu)
        assert len(att_classes) == 4
        assert [c.subject for c in att_classes] == ["DE LAB", "DE LAB", "DE LAB", "WP"]
        assert classes[4].is_attendance_bearing is False  # LIB
        assert classes[5].is_attendance_bearing is False  # SPORTS

    def test_friday_schedule(self, service: TimetableService):
        # 2026-09-25 is a Friday
        fri = date(2026, 9, 25)
        classes = service.get_classes_for_date(fri)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["IoTAP", "OS", "ACD", "DE", "SL", "MTP"]

        # SL and MTP non-attendance
        att_classes = service.get_attendance_bearing_classes(fri)
        assert len(att_classes) == 4
        assert [c.subject for c in att_classes] == ["IoTAP", "OS", "ACD", "DE"]
        assert classes[4].is_attendance_bearing is False  # SL
        assert classes[5].is_attendance_bearing is False  # MTP

    def test_saturday_schedule(self, service: TimetableService):
        # 2026-09-26 is a Saturday
        sat = date(2026, 9, 26)
        classes = service.get_classes_for_date(sat)
        assert len(classes) == 6
        assert [c.subject for c in classes] == ["OE-1", "DE", "OS", "IoTAP", "GS", "GS"]
        assert all(c.is_attendance_bearing for c in classes)

        # GS has 2 consecutive periods
        gs_classes = [c for c in classes if c.subject == "GS"]
        assert len(gs_classes) == 2
        assert [c.period_num for c in gs_classes] == [5, 6]


class TestSubjectQueries:

    def test_get_classes_for_subject_by_short_name(self, service: TimetableService):
        mon = date(2026, 9, 21)
        # Monday has DE in P1
        de_classes = service.get_classes_for_subject(mon, "DE")
        assert len(de_classes) == 1
        assert de_classes[0].period_num == 1

        # Monday has OE-1 in P5 & P6
        oe_classes = service.get_classes_for_subject(mon, "OE-1")
        assert len(oe_classes) == 2

    def test_get_classes_for_subject_by_code(self, service: TimetableService):
        mon = date(2026, 9, 21)
        # OS code is 22PC1IT202
        os_classes = service.get_classes_for_subject(mon, "22PC1IT202")
        assert len(os_classes) == 1
        assert os_classes[0].subject == "OS"
        assert os_classes[0].period_num == 3

    def test_has_class_today(self, service: TimetableService):
        mon = date(2026, 9, 21)
        assert service.has_class_today("DE", mon) is True
        assert service.has_class_today("OS", mon) is True
        assert service.has_class_today("GS", mon) is False  # GS is only on Saturday
        assert service.has_class_today("IoTAP LAB", mon) is False  # Only on Tuesday

        wed = date(2026, 9, 23)
        # ECA/CCA is on Wednesday, but is NOT attendance-bearing
        assert service.has_class_today("ECA/CCA", wed) is False

    def test_non_attendance_activities_list(self, service: TimetableService):
        for act in ["LIB", "SPORTS", "ECA/CCA", "SL", "MTP"]:
            assert service.is_attendance_bearing(act) is False

        for subj in ["DE", "WP", "OS", "ACD", "OE-1", "IoTAP", "IoTAP LAB", "DE LAB", "GS"]:
            assert service.is_attendance_bearing(subj) is True


class TestDatetimeLookups:

    def test_exact_start_time(self, service: TimetableService):
        # Monday at 10:00:00 -> P1 DE
        dt = datetime(2026, 9, 21, 10, 0, 0, tzinfo=_TZ)
        c = service.get_class_at_datetime(dt)
        assert c is not None
        assert c.subject == "DE"
        assert c.period_num == 1

    def test_mid_period_time(self, service: TimetableService):
        # Monday at 10:35:00 -> P1 DE
        dt = datetime(2026, 9, 21, 10, 35, 0, tzinfo=_TZ)
        c = service.get_class_at_datetime(dt)
        assert c is not None
        assert c.subject == "DE"

    def test_period_boundary_transition(self, service: TimetableService):
        # Monday at 11:00:00 -> P2 WP (since P1 is [10:00, 11:00) and P2 is [11:00, 12:00))
        dt = datetime(2026, 9, 21, 11, 0, 0, tzinfo=_TZ)
        c = service.get_class_at_datetime(dt)
        assert c is not None
        assert c.subject == "WP"
        assert c.period_num == 2

    def test_lunch_time_returns_none(self, service: TimetableService):
        # Monday at 13:20:00 -> Lunch
        dt = datetime(2026, 9, 21, 13, 20, 0, tzinfo=_TZ)
        assert service.get_class_at_datetime(dt) is None

    def test_before_college_hours_returns_none(self, service: TimetableService):
        dt = datetime(2026, 9, 21, 9, 30, 0, tzinfo=_TZ)
        assert service.get_class_at_datetime(dt) is None

    def test_after_college_hours_returns_none(self, service: TimetableService):
        dt = datetime(2026, 9, 21, 17, 0, 0, tzinfo=_TZ)
        assert service.get_class_at_datetime(dt) is None

    def test_sunday_returns_none(self, service: TimetableService):
        dt = datetime(2026, 9, 20, 10, 30, 0, tzinfo=_TZ)
        assert service.get_class_at_datetime(dt) is None
