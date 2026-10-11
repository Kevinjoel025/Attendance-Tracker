"""
calendar/service.py
───────────────────
Academic calendar service for holiday, weekend, and working-day determination.

PRECEDENCE RULES (Strict):
1. Explicit user working-day override (e.g. 2026-10-19 declared as working day).
2. Explicit user holiday override (e.g. 2026-10-21 declared as holiday).
3. Explicit working-day exception (e.g. 2026-02-14 Working Day in lieu of New Year's Day).
4. Explicit named official holiday (e.g. 2026-10-02 Mahatma Gandhi Jayanti).
5. Mid-semester examination holidays.
6. Recurring 2nd/4th Saturday holiday rule.
7. Regular timetable (Monday–Friday & 1st/3rd/5th Saturday are working days; Sunday is weekend/no classes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from zoneinfo import ZoneInfo

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.time_service import parse_date

if TYPE_CHECKING:
    from attendance_tracker.attendance.repository import AttendanceRepository

logger = get_logger(__name__)

_TZ = ZoneInfo(config.timezone)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalendarStatus:
    """Detailed calendar status for a specific calendar date."""
    date: date
    is_holiday: bool
    is_working_day: bool
    event_name: Optional[str]
    event_type: str  # "HOLIDAY" | "WORKING_DAY_EXCEPTION" | "USER_OVERRIDE" | "REGULAR_WORKING_DAY" | "WEEKEND"
    cancels_classes: bool

    @property
    def holiday_name(self) -> Optional[str]:
        return self.event_name

    @property
    def is_override(self) -> bool:
        return self.event_type == "USER_OVERRIDE"

    @property
    def override_type(self) -> Optional[str]:
        if self.event_type == "USER_OVERRIDE":
            return "HOLIDAY" if self.is_holiday else "WORKING_DAY"
        return None


# ── Service Class ─────────────────────────────────────────────────────────────

class CalendarService:
    """
    Academic calendar service managing holidays, recurring rules, and exceptions.
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        repository: Optional[AttendanceRepository] = None,
    ) -> None:
        if data_path is None:
            data_path = Path(config.paths.holidays)
        self.data_path = Path(data_path)
        self.repository = repository
        self._named_holidays: dict[date, dict] = {}
        self._exceptions: dict[date, dict] = {}
        self._recurring_second_fourth_sat: bool = True
        self._load()

    def _get_repository(self) -> Optional[AttendanceRepository]:
        if self.repository is not None:
            return self.repository
        try:
            from attendance_tracker.attendance.repository import get_repository
            return get_repository()
        except Exception:
            return None

    def _load(self) -> None:
        """Load holidays, exceptions, and rules from holidays.json."""
        if not self.data_path.exists():
            logger.warning(f"Holidays file not found at {self.data_path}")
            return

        with self.data_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Recurring rules
        rules = raw.get("recurring_rules", {})
        self._recurring_second_fourth_sat = bool(
            rules.get("second_and_fourth_saturday_holiday", True)
        )

        # Named holidays
        for item in raw.get("named_holidays", []):
            d = parse_date(item["date"])
            self._named_holidays[d] = {
                "name": item["name"],
                "event_type": item.get("event_type", "HOLIDAY"),
                "cancels_classes": item.get("cancels_classes", True),
            }

        # Exceptions (e.g. 2026-02-14)
        for item in raw.get("exceptions", []):
            d = parse_date(item["date"])
            self._exceptions[d] = {
                "name": item["name"],
                "event_type": item.get("event_type", "WORKING_DAY_EXCEPTION"),
                "cancels_classes": item.get("cancels_classes", False),
            }

    def list_named_holidays(self) -> list[dict[str, Any]]:
        """Return all pre-configured named holidays as dictionaries."""
        results = []
        for d, item in sorted(self._named_holidays.items(), key=lambda x: x[0]):
            results.append({
                "date": d.isoformat(),
                "name": item["name"],
                "event_type": item.get("event_type", "HOLIDAY"),
                "type": item.get("event_type", "HOLIDAY"),
                "cancels_classes": item.get("cancels_classes", True),
            })
        return results

    def list_exceptions(self) -> list[dict[str, Any]]:
        """Return all pre-configured working-day exceptions."""
        results = []
        for d, item in sorted(self._exceptions.items(), key=lambda x: x[0]):
            results.append({
                "date": d.isoformat(),
                "name": item["name"],
                "event_type": item.get("event_type", "WORKING_DAY_EXCEPTION"),
                "type": item.get("event_type", "WORKING_DAY_EXCEPTION"),
                "cancels_classes": item.get("cancels_classes", False),
            })
        return results

    @staticmethod
    def is_second_or_fourth_saturday(target_date: date) -> bool:
        """
        Return True if target_date is the 2nd or 4th Saturday of its month.
        Day of week 5 = Saturday.
        N-th Saturday of month = ((day - 1) // 7) + 1.
        """
        if target_date.weekday() != 5:
            return False
        nth_saturday = ((target_date.day - 1) // 7) + 1
        return nth_saturday in (2, 4)

    @staticmethod
    def get_saturday_order(target_date: date) -> Optional[int]:
        """Return 1, 2, 3, 4, 5 if Saturday, else None."""
        if target_date.weekday() != 5:
            return None
        return ((target_date.day - 1) // 7) + 1

    def get_calendar_status(self, target_date: date) -> CalendarStatus:
        """
        Determine the full calendar status of target_date according to strict precedence:

        1. Explicit user working-day override
        2. Explicit user holiday override
        3. Explicit working-day exception (e.g. 2026-02-14)
        4. Explicit named official holiday (e.g. 2026-10-02)
        5. Recurring 2nd/4th Saturday rule
        6. Sunday rule -> Weekend / no classes
        7. Regular timetable (Monday–Friday or 1st/3rd/5th Saturday)
        """
        # Level 1 & 2: User-defined overrides from SQLite repository
        repo = self._get_repository()
        if repo is not None:
            try:
                user_override = repo.get_calendar_override(target_date)
                if user_override is not None:
                    if user_override.override_type == "WORKING_DAY":
                        return CalendarStatus(
                            date=target_date,
                            is_holiday=False,
                            is_working_day=True,
                            event_name=user_override.name or "User Working Day Override",
                            event_type="USER_OVERRIDE",
                            cancels_classes=False,
                        )
                    elif user_override.override_type == "HOLIDAY":
                        return CalendarStatus(
                            date=target_date,
                            is_holiday=True,
                            is_working_day=False,
                            event_name=user_override.name or "User Holiday Override",
                            event_type="USER_OVERRIDE",
                            cancels_classes=True,
                        )
            except Exception:
                pass

        # Level 3: Explicit working-day exception from json
        if target_date in self._exceptions:
            exc = self._exceptions[target_date]
            return CalendarStatus(
                date=target_date,
                is_holiday=False,
                is_working_day=True,
                event_name=exc["name"],
                event_type=exc["event_type"],
                cancels_classes=exc["cancels_classes"],
            )

        # Level 4: Explicit named holiday from json
        if target_date in self._named_holidays:
            hol = self._named_holidays[target_date]
            return CalendarStatus(
                date=target_date,
                is_holiday=True,
                is_working_day=False,
                event_name=hol["name"],
                event_type=hol["event_type"],
                cancels_classes=hol["cancels_classes"],
            )

        # Level 5: Recurring 2nd/4th Saturday rule
        if self._recurring_second_fourth_sat and self.is_second_or_fourth_saturday(target_date):
            nth = self.get_saturday_order(target_date)
            name = "Second Saturday" if nth == 2 else "Fourth Saturday"
            return CalendarStatus(
                date=target_date,
                is_holiday=True,
                is_working_day=False,
                event_name=name,
                event_type="HOLIDAY",
                cancels_classes=True,
            )

        # Level 6: Regular Sunday rule
        if target_date.weekday() == 6:
            return CalendarStatus(
                date=target_date,
                is_holiday=True,
                is_working_day=False,
                event_name="Sunday",
                event_type="WEEKEND",
                cancels_classes=True,
            )

        # Level 7: Regular working day (Monday–Friday or 1st/3rd/5th Saturday)
        return CalendarStatus(
            date=target_date,
            is_holiday=False,
            is_working_day=True,
            event_name=None,
            event_type="REGULAR_WORKING_DAY",
            cancels_classes=False,
        )

    def is_holiday(self, target_date: date) -> bool:
        """Return True if target_date is a holiday or non-working weekend."""
        return self.get_calendar_status(target_date).is_holiday

    def is_working_day(self, target_date: date) -> bool:
        """Return True if target_date is a college working day."""
        return self.get_calendar_status(target_date).is_working_day

    def get_day_status(self, target_date: date) -> CalendarStatus:
        """Alias for get_calendar_status."""
        return self.get_calendar_status(target_date)

    def cancels_classes(self, target_date: date) -> bool:
        """Return True if classes are cancelled on target_date."""
        return self.get_calendar_status(target_date).cancels_classes

    def get_holiday_name(self, target_date: date) -> Optional[str]:
        """Return the name of the holiday/event if any, else None."""
        return self.get_calendar_status(target_date).event_name

    def get_last_working_day(self, default: str = "2026-11-28") -> date:
        """Return the configured last working day of the semester."""
        repo = self._get_repository()
        if repo is not None:
            return repo.get_last_working_day(default=default)
        try:
            return datetime.strptime(default, "%Y-%m-%d").date()
        except Exception:
            return date(2026, 11, 28)

    def set_last_working_day(self, target_date: date | str) -> date:
        """Set the configured last working day of the semester."""
        repo = self._get_repository()
        if repo is not None:
            return repo.set_last_working_day(target_date)
        if isinstance(target_date, (date, datetime)):
            return target_date if isinstance(target_date, date) else target_date.date()
        return datetime.strptime(str(target_date).strip(), "%Y-%m-%d").date()

    def get_mid_term_1(
        self,
        default_start: str = "2026-09-07",
        default_end: str = "2026-09-12",
    ) -> tuple[date, date]:
        """Return the configured (start_date, end_date) for Mid-Term 1."""
        repo = self._get_repository()
        if repo is not None:
            return repo.get_mid_term_1(default_start=default_start, default_end=default_end)
        try:
            return datetime.strptime(default_start, "%Y-%m-%d").date(), datetime.strptime(default_end, "%Y-%m-%d").date()
        except Exception:
            return date(2026, 9, 7), date(2026, 9, 12)

    def set_mid_term_1(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> tuple[date, date]:
        """Set the configured (start_date, end_date) for Mid-Term 1."""
        repo = self._get_repository()
        if repo is not None:
            return repo.set_mid_term_1(start_date, end_date)
        if isinstance(start_date, (date, datetime)):
            s_obj = start_date if isinstance(start_date, date) else start_date.date()
        else:
            s_obj = datetime.strptime(str(start_date).strip(), "%Y-%m-%d").date()
        if isinstance(end_date, (date, datetime)):
            e_obj = end_date if isinstance(end_date, date) else end_date.date()
        else:
            e_obj = datetime.strptime(str(end_date).strip(), "%Y-%m-%d").date()
        if e_obj < s_obj:
            raise ValueError(f"Mid-Term 1 end date cannot be before start date.")
        return s_obj, e_obj

    def get_unified_calendar_entries(self) -> list[dict[str, Any]]:
        """
        Produce the authoritative single unified list of all calendar entries.
        Combines:
        - Pre-configured institutional named holidays
        - Institutional working-day exceptions
        - Configured Mid-Term 1 date range
        - User-defined and imported overrides from SQLite database

        Guarantees:
        - Every calendar event or date range appears exactly once.
        - Active user overrides modify/replace the default status with source indication.
        - Entries are chronologically sorted by date.
        """
        entries_by_date: dict[str, dict[str, Any]] = {}

        # 1. Base official named holidays
        for d, item in sorted(self._named_holidays.items(), key=lambda x: x[0]):
            d_str = d.isoformat()
            entries_by_date[d_str] = {
                "id": f"official_{d_str}",
                "date": d_str,
                "start_date": d_str,
                "end_date": d_str,
                "name": item["name"],
                "type": "HOLIDAY",
                "type_label": "Holiday",
                "source": "OFFICIAL",
                "source_label": "Official Holiday",
                "is_override": False,
                "original_rule": None,
                "note": "",
                "cancels_classes": item.get("cancels_classes", True),
                "can_edit": True,
                "can_delete": True,
            }

        # 2. Base official exceptions
        for d, item in sorted(self._exceptions.items(), key=lambda x: x[0]):
            d_str = d.isoformat()
            entries_by_date[d_str] = {
                "id": f"exception_{d_str}",
                "date": d_str,
                "start_date": d_str,
                "end_date": d_str,
                "name": item["name"],
                "type": "WORKING_DAY",
                "type_label": "Working Day",
                "source": "OFFICIAL",
                "source_label": "Official Exception",
                "is_override": False,
                "original_rule": None,
                "note": "",
                "cancels_classes": item.get("cancels_classes", False),
                "can_edit": True,
                "can_delete": True,
            }

        # 3. Mid-Term 1
        try:
            mt1_start, mt1_end = self.get_mid_term_1()
            mt1_entry = {
                "id": "mid_term_1",
                "date": f"{mt1_start.isoformat()} – {mt1_end.isoformat()}",
                "start_date": mt1_start.isoformat(),
                "end_date": mt1_end.isoformat(),
                "name": "Mid-Term 1 Examinations",
                "type": "MID_TERM",
                "type_label": "Mid-Term",
                "source": "OFFICIAL",
                "source_label": "Institutional Exams",
                "is_override": False,
                "original_rule": "Exam Period (Classes Suspended)",
                "note": "Classes suspended during Mid-Term 1",
                "cancels_classes": True,
                "can_edit": True,
                "can_delete": False,
            }
        except Exception:
            mt1_entry = None

        # 4. Overlay user overrides from repository
        repo = self._get_repository()
        if repo is not None:
            try:
                overrides = repo.list_calendar_overrides()
                for o in overrides:
                    d_str = o.date
                    ov_type = o.override_type.upper()
                    type_lbl = "Holiday" if ov_type == "HOLIDAY" else "Working Day"
                    cancels = (ov_type == "HOLIDAY")

                    if d_str in entries_by_date:
                        orig = entries_by_date[d_str]
                        orig_rule = f"Official: {orig['name']} ({orig['type_label']})"
                        name_display = o.name if o.name else orig["name"]
                        entries_by_date[d_str] = {
                            "id": f"override_{o.id}",
                            "db_id": o.id,
                            "date": d_str,
                            "start_date": d_str,
                            "end_date": d_str,
                            "name": name_display,
                            "type": ov_type,
                            "type_label": type_lbl,
                            "source": "USER_OVERRIDE",
                            "source_label": "User Override",
                            "is_override": True,
                            "original_rule": orig_rule,
                            "note": o.name or "",
                            "cancels_classes": cancels,
                            "can_edit": True,
                            "can_delete": True,
                        }
                    else:
                        name_display = o.name if o.name else ("User Holiday Override" if ov_type == "HOLIDAY" else "User Working Day Override")
                        entries_by_date[d_str] = {
                            "id": f"override_{o.id}",
                            "db_id": o.id,
                            "date": d_str,
                            "start_date": d_str,
                            "end_date": d_str,
                            "name": name_display,
                            "type": ov_type,
                            "type_label": type_lbl,
                            "source": "USER_OVERRIDE",
                            "source_label": "User Override",
                            "is_override": True,
                            "original_rule": "Regular Timetable",
                            "note": o.name or "",
                            "cancels_classes": cancels,
                            "can_edit": True,
                            "can_delete": True,
                        }
            except Exception as e:
                logger.error(f"Error reading calendar overrides for unified table: {e}")

        all_entries = list(entries_by_date.values())
        if mt1_entry is not None:
            all_entries.append(mt1_entry)

        all_entries.sort(key=lambda x: x["start_date"])
        return all_entries

    def get_upcoming_holidays(
        self,
        start_date: date | None = None,
        count: int = 5,
    ) -> list[CalendarStatus]:
        """
        Return the next `count` calendar holidays/exceptions starting from start_date.
        """
        if start_date is None:
            start_date = datetime.now(_TZ).date()

        results = []
        curr = start_date
        for _ in range(365):
            status = self.get_calendar_status(curr)
            if status.is_holiday and status.event_type != "WEEKEND":
                results.append(status)
                if len(results) >= count:
                    break
            curr += timedelta(days=1)

        return results


# ── Global Singleton and Convenience Functions ────────────────────────────────

_default_service: Optional[CalendarService] = None


def get_calendar_service(repository: Optional[AttendanceRepository] = None) -> CalendarService:
    """Return the singleton CalendarService instance."""
    global _default_service
    if _default_service is None:
        _default_service = CalendarService(repository=repository)
    elif repository is not None:
        _default_service.repository = repository
    return _default_service


def is_holiday(target_date: date) -> bool:
    return get_calendar_service().is_holiday(target_date)


def is_working_day(target_date: date) -> bool:
    return get_calendar_service().is_working_day(target_date)


def cancels_classes(target_date: date) -> bool:
    return get_calendar_service().cancels_classes(target_date)


def get_calendar_status(target_date: date) -> CalendarStatus:
    return get_calendar_service().get_calendar_status(target_date)


def get_holiday_name(target_date: date) -> Optional[str]:
    return get_calendar_service().get_holiday_name(target_date)


def is_second_or_fourth_saturday(target_date: date) -> bool:
    return get_calendar_service().is_second_or_fourth_saturday(target_date)


def get_upcoming_holidays(
    start_date: date | None = None,
    count: int = 5,
) -> list[CalendarStatus]:
    return get_calendar_service().get_upcoming_holidays(start_date, count)
