"""
tests/test_calendar.py
──────────────────────
Comprehensive unit tests for CalendarService.

Covers:
- All 20 official 2026 named holidays
- 2nd and 4th Saturday recurring holidays
- 1st, 3rd, and 5th Saturday working days
- Explicit working-day exception: 2026-02-14 (overrides 2nd Saturday)
- Calendar precedence rules
- Sundays (weekend / no classes)
- Regular working weekdays
- Upcoming holidays retrieval
"""

from datetime import date
import pytest

from attendance_tracker.calendar.service import (
    CalendarService,
    get_calendar_service,
)


@pytest.fixture
def service():
    return get_calendar_service()


class TestNamedHolidays2026:
    """Verify all 20 official holidays from the VNR VJIET 2026 calendar."""

    @pytest.mark.parametrize("d_str,expected_name", [
        ("2026-01-01", "New Year Day"),
        ("2026-01-14", "Bhogi"),
        ("2026-01-15", "Sankranti/Pongal"),
        ("2026-01-16", "Kanumu"),
        ("2026-01-26", "Republic Day"),
        ("2026-03-03", "Holi"),
        ("2026-03-19", "Ugadi"),
        ("2026-03-21", "Eid-ul-Fitr (Ramzan)"),
        ("2026-03-27", "Sri Rama Navami"),
        ("2026-04-03", "Good Friday"),
        ("2026-04-14", "Dr. B. R. Ambedkar's Birthday"),
        ("2026-05-27", "Eid-ul-Azha (Bakrid)"),
        ("2026-08-15", "Independence Day"),
        ("2026-08-21", "Varalakshmi Vratham"),
        ("2026-08-26", "Eid-Milad-un-Nabi"),
        ("2026-09-14", "Vinayaka Chavithi"),
        ("2026-10-02", "Mahatma Gandhi Jayanti"),
        ("2026-10-19", "Maharnavami"),
        ("2026-10-20", "Vijaya Dasami / Dushera"),
        ("2026-12-25", "Christmas"),
    ])
    def test_official_holidays(self, service: CalendarService, d_str: str, expected_name: str):
        d = date.fromisoformat(d_str)
        status = service.get_calendar_status(d)
        assert status.is_holiday is True
        assert status.is_working_day is False
        assert status.cancels_classes is True
        assert status.event_name == expected_name
        assert service.is_holiday(d) is True
        assert service.is_working_day(d) is False


class TestSaturdayRulesAndExceptions:

    def test_2026_02_14_working_day_exception(self, service: CalendarService):
        """
        CRITICAL TEST: 2026-02-14 is the 2nd Saturday of February 2026.
        By official VNR VJIET rule, it is explicitly a WORKING DAY
        in lieu of New Year's Day.
        """
        d = date(2026, 2, 14)
        assert d.weekday() == 5  # Saturday
        assert service.is_second_or_fourth_saturday(d) is True

        status = service.get_calendar_status(d)
        assert status.is_holiday is False, "2026-02-14 must NOT be a holiday"
        assert status.is_working_day is True, "2026-02-14 MUST be a working day"
        assert status.cancels_classes is False
        assert status.event_type == "WORKING_DAY_EXCEPTION"
        assert "Working Day" in (status.event_name or "")

        assert service.is_working_day(d) is True
        assert service.is_holiday(d) is False
        assert service.cancels_classes(d) is False

    @pytest.mark.parametrize("d,nth", [
        (date(2026, 1, 10), 2),
        (date(2026, 1, 24), 4),
        (date(2026, 3, 14), 2),
        (date(2026, 3, 28), 4),
        (date(2026, 4, 11), 2),
        (date(2026, 4, 25), 4),
        (date(2026, 5, 9), 2),
        (date(2026, 5, 23), 4),
        (date(2026, 6, 13), 2),
        (date(2026, 6, 27), 4),
        (date(2026, 7, 11), 2),
        (date(2026, 7, 25), 4),
        (date(2026, 8, 8), 2),
        (date(2026, 8, 22), 4),
        (date(2026, 9, 12), 2),
        (date(2026, 9, 26), 4),
        (date(2026, 10, 10), 2),
        (date(2026, 10, 24), 4),
        (date(2026, 11, 14), 2),
        (date(2026, 11, 28), 4),
        (date(2026, 12, 12), 2),
        (date(2026, 12, 26), 4),
    ])
    def test_second_and_fourth_saturdays_are_holidays(self, service: CalendarService, d: date, nth: int):
        assert d.weekday() == 5
        assert service.is_second_or_fourth_saturday(d) is True
        status = service.get_calendar_status(d)
        assert status.is_holiday is True
        assert status.is_working_day is False
        assert status.cancels_classes is True
        expected_name = "Second Saturday" if nth == 2 else "Fourth Saturday"
        assert status.event_name == expected_name

    @pytest.mark.parametrize("d", [
        date(2026, 1, 3),   # 1st Sat
        date(2026, 1, 17),  # 3rd Sat
        date(2026, 1, 31),  # 5th Sat
        date(2026, 2, 7),   # 1st Sat
        date(2026, 2, 21),  # 3rd Sat
        date(2026, 5, 2),   # 1st Sat
        date(2026, 5, 16),  # 3rd Sat
        date(2026, 5, 30),  # 5th Sat
        date(2026, 8, 1),   # 1st Sat
        date(2026, 8, 29),  # 5th Sat
        date(2026, 10, 31), # 5th Sat
    ])
    def test_first_third_fifth_saturdays_are_working_days(self, service: CalendarService, d: date):
        # Unless it is a named holiday (e.g. 2026-08-15 is 3rd Sat and Independence Day)
        if service.get_calendar_status(d).event_type == "HOLIDAY":
            pytest.skip("Falls on a named holiday")
        assert d.weekday() == 5
        assert service.is_second_or_fourth_saturday(d) is False
        status = service.get_calendar_status(d)
        assert status.is_holiday is False
        assert status.is_working_day is True


class TestRegularDaysAndSundays:

    def test_regular_weekday_working_day(self, service: CalendarService):
        # Wednesday 2026-09-16 (not a holiday)
        d = date(2026, 9, 16)
        assert d.weekday() == 2  # Wednesday
        status = service.get_calendar_status(d)
        assert status.is_holiday is False
        assert status.is_working_day is True
        assert status.cancels_classes is False
        assert status.event_type == "REGULAR_WORKING_DAY"

    def test_sunday_is_weekend_holiday(self, service: CalendarService):
        # Sunday 2026-09-20
        d = date(2026, 9, 20)
        assert d.weekday() == 6
        status = service.get_calendar_status(d)
        assert status.is_holiday is True
        assert status.is_working_day is False
        assert status.cancels_classes is True
        assert status.event_type == "WEEKEND"
        assert status.event_name == "Sunday"

    def test_upcoming_holidays(self, service: CalendarService):
        start = date(2026, 9, 1)
        upcoming = service.get_upcoming_holidays(start_date=start, count=3)
        assert len(upcoming) == 3
        # In September 2026:
        # Sep 12 (2nd Sat), Sep 14 (Vinayaka Chavithi), Sep 26 (4th Sat)
        assert upcoming[0].date == date(2026, 9, 12)
        assert upcoming[1].date == date(2026, 9, 14)
        assert upcoming[2].date == date(2026, 9, 26)
