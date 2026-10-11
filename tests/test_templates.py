"""
tests/test_templates.py
───────────────────────
Unit tests for response templates (response/templates.py).
"""

import pytest

from attendance_tracker.attendance.engine import (
    ClassesRequiredResult,
    MaxCanMissResult,
    ProjectionResult,
)
from attendance_tracker.attendance.repository import (
    OverallAttendance,
    Subject,
    SubjectAttendance,
)
from attendance_tracker.response.templates import (
    format_change_updates,
    format_classes_required,
    format_error,
    format_help,
    format_max_can_miss,
    format_overall_attendance,
    format_projection,
    format_subject_attendance,
    format_unknown,
)


@pytest.fixture
def sample_subject() -> Subject:
    return Subject(
        code="22PC1IT202",
        full_name="Operating Systems",
        short_name="OS",
        threshold=75.0,
    )


@pytest.fixture
def sample_subject_attendance(sample_subject: Subject) -> SubjectAttendance:
    return SubjectAttendance(
        subject=sample_subject,
        attended=17,
        conducted=28,
        percentage=60.714285714285715,
        scraped_at="2026-09-19T00:00:00+05:30",
        snapshot_id=1,
    )


@pytest.fixture
def sample_overall_attendance(sample_subject_attendance: SubjectAttendance) -> OverallAttendance:
    return OverallAttendance(
        total_attended=119,
        total_conducted=236,
        overall_percentage=50.42372881355932,
        subject_records=[sample_subject_attendance],
        scraped_at="2026-09-19T00:00:00+05:30",
        snapshot_id=1,
    )


def test_format_overall_attendance(sample_overall_attendance: OverallAttendance) -> None:
    text = format_overall_attendance(sample_overall_attendance)
    assert "Overall Attendance: 50.42%" in text
    assert "119 / 236 classes" in text
    assert "OS (Operating Systems)" in text


def test_format_subject_attendance(sample_subject_attendance: SubjectAttendance) -> None:
    text = format_subject_attendance(sample_subject_attendance)
    assert "Operating Systems (OS - 22PC1IT202)" in text
    assert "60.71%" in text
    assert "17 attended / 28 conducted" in text
    assert "College Threshold (75.0%)" in text


def test_format_projection_attending(sample_subject: Subject) -> None:
    proj = ProjectionResult(
        subject_code=sample_subject.code,
        current_attended=17,
        current_conducted=28,
        current_percentage=60.71,
        action="ATTEND",
        count=3,
        projected_attended=20,
        projected_conducted=31,
        projected_percentage=64.52,
        college_threshold=75.0,
        above_college_threshold_after=False,
        threshold_breached=False,
    )
    text = format_projection(proj, subject_name=sample_subject.full_name)
    assert "Attendance Projection for Operating Systems" in text
    assert "Attending next 3 class(es)" in text
    assert "20/31" in text
    assert "64.52%" in text


def test_format_projection_threshold_breached(sample_subject: Subject) -> None:
    proj = ProjectionResult(
        subject_code=sample_subject.code,
        current_attended=20,
        current_conducted=25,
        current_percentage=80.0,
        action="MISS",
        count=3,
        projected_attended=20,
        projected_conducted=28,
        projected_percentage=71.43,
        college_threshold=75.0,
        above_college_threshold_after=False,
        threshold_breached=True,
    )
    text = format_projection(proj, subject_name=sample_subject.full_name)
    assert "Missing next 3 class(es)" in text
    assert "Warning: Missing these classes will cause attendance to breach" in text


def test_format_classes_required() -> None:
    req = ClassesRequiredResult(
        subject_code="22PC1AM202",
        current_attended=11,
        current_conducted=28,
        current_percentage=39.29,
        target_percentage=75.0,
        classes_required=40,
        already_above_target=False,
        impossible=False,
    )
    text = format_classes_required(req, subject_name="Data Engineering")
    assert "Target Goal for Data Engineering" in text
    assert "must attend 40 consecutive class(es)" in text
    assert "75.0%" in text


def test_format_classes_required_already_above() -> None:
    req = ClassesRequiredResult(
        subject_code="22PC1AM202",
        current_attended=30,
        current_conducted=32,
        current_percentage=93.75,
        target_percentage=75.0,
        classes_required=0,
        already_above_target=True,
        impossible=False,
    )
    text = format_classes_required(req, subject_name="Data Engineering")
    assert "already at or above your target" in text


def test_format_classes_required_impossible() -> None:
    req = ClassesRequiredResult(
        subject_code="22PC1AM202",
        current_attended=10,
        current_conducted=12,
        current_percentage=83.33,
        target_percentage=100.0,
        classes_required=999999,
        already_above_target=False,
        impossible=True,
    )
    text = format_classes_required(req, subject_name="Data Engineering")
    assert "mathematically impossible" in text


def test_format_max_can_miss() -> None:
    res = MaxCanMissResult(
        subject_code="22PC1IT202",
        current_attended=30,
        current_conducted=32,
        current_percentage=93.75,
        threshold_percentage=75.0,
        max_can_miss=8,
        already_below_threshold=False,
    )
    text = format_max_can_miss(res, subject_name="Operating Systems")
    assert "Bunk Margin for Operating Systems" in text
    assert "safely miss up to 8 class(es)" in text


def test_format_max_can_miss_below_threshold() -> None:
    res = MaxCanMissResult(
        subject_code="22PC1IT202",
        current_attended=17,
        current_conducted=28,
        current_percentage=60.71,
        threshold_percentage=75.0,
        max_can_miss=0,
        already_below_threshold=True,
    )
    text = format_max_can_miss(res, subject_name="Operating Systems")
    assert "cannot miss any classes" in text
    assert "below the 75.0% threshold" in text


def test_format_change_updates_empty() -> None:
    text = format_change_updates([])
    assert "No attendance changes or new classes recorded" in text


def test_format_help(sample_subject: Subject) -> None:
    text = format_help([sample_subject])
    assert "College Attendance Tracker -- Help & Capabilities" in text
    assert "Overall Attendance" in text
    assert "OS" in text


def test_format_unknown() -> None:
    text = format_unknown("what is the weather today")
    assert "couldn't understand your request" in text


def test_format_error() -> None:
    text = format_error("Subject not found")
    assert "Error: Subject not found" in text
