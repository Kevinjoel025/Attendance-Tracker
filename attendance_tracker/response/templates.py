"""
response/templates.py
─────────────────────
Natural language response formatting templates for deterministic tool results.

RULES:
- Zero arithmetic is done here; all numbers come pre-computed from the ToolResult.
- Formats structured tool results into clean, robust, ASCII-compatible text.
- Handles success, warning, deficit, and error conditions gracefully.
- OVERALL ATTENDANCE is prioritized as the authoritative decision metric.
"""

from __future__ import annotations

from typing import Any, List, Optional

from attendance_tracker.attendance.change_detector import AttendanceChangeResult
from attendance_tracker.attendance.engine import (
    ClassesRequiredResult,
    MaxCanMissResult,
    ProjectionResult,
)
from attendance_tracker.attendance.repository import (
    AttendanceChangeRecord,
    OverallAttendance,
    Subject,
    SubjectAttendance,
)
from attendance_tracker.core.config import config


def format_overall_attendance(overall: OverallAttendance) -> str:
    """Format overall attendance overview and subject breakdown."""
    college_thresh = config.attendance.college_threshold
    personal_tgt = config.attendance.personal_target
    pct = overall.overall_percentage

    status_college = (
        f"[OK] {pct - college_thresh:+.2f}% above college threshold ({college_thresh:.1f}%)"
        if pct >= college_thresh
        else f"[WARNING] {college_thresh - pct:.2f}% below college threshold ({college_thresh:.1f}%)"
    )
    status_target = (
        f"[OK] {pct - personal_tgt:+.2f}% above personal target ({personal_tgt:.1f}%)"
        if pct >= personal_tgt
        else f"[WARNING] {personal_tgt - pct:.2f}% below personal target ({personal_tgt:.1f}%)"
    )

    lines = [
        f"Overall Attendance: {pct:.2f}% ({overall.total_attended} / {overall.total_conducted} classes)",
        f"- {status_college}",
        f"- {status_target}",
        "",
        "Subject Breakdown:",
    ]

    for rec in overall.subject_records:
        s = rec.subject
        tag = "[OK]" if rec.percentage >= college_thresh else ("[WARN]" if rec.percentage >= personal_tgt else "[CRIT]")
        lines.append(
            f"  {tag:<7} {s.short_name} ({s.full_name}): {rec.attended}/{rec.conducted} ({rec.percentage:.2f}%)"
        )

    return "\n".join(lines)


def format_subject_attendance(record: SubjectAttendance) -> str:
    """Format single subject attendance report."""
    s = record.subject
    pct = record.percentage
    college_thresh = s.threshold
    personal_tgt = config.attendance.personal_target

    status_college = (
        f"[OK] Meeting requirement ({pct - college_thresh:+.2f}% above threshold)"
        if pct >= college_thresh
        else f"[WARNING] Shortage ({college_thresh - pct:.2f}% below threshold)"
    )
    status_target = (
        f"[OK] Above target ({pct - personal_tgt:+.2f}%)"
        if pct >= personal_tgt
        else f"[WARNING] Below target ({personal_tgt - pct:.2f}%)"
    )

    lines = [
        f"Subject Attendance: {s.full_name} ({s.short_name} - {s.code})",
        f"- Current Attendance: {pct:.2f}% ({record.attended} attended / {record.conducted} conducted)",
        f"- College Threshold ({college_thresh:.1f}%): {status_college}",
        f"- Personal Target ({personal_tgt:.1f}%): {status_target}",
    ]
    return "\n".join(lines)


def format_projection(result: ProjectionResult, subject_name: Optional[str] = None) -> str:
    """Format attendance projection result."""
    name = subject_name or result.subject_code
    action_text = "attending" if result.action.upper() == "ATTEND" else "missing"
    delta_pct = result.projected_percentage - result.current_percentage
    delta_sign = "+" if delta_pct >= 0 else ""

    status_text = (
        f"[OK] Meets college threshold ({result.college_threshold:.1f}%)"
        if result.above_college_threshold_after
        else f"[WARNING] Below college threshold ({result.college_threshold:.1f}%)"
    )

    if result.threshold_breached:
        warning = f"\n[ALERT] Warning: Missing these classes will cause attendance to breach the {result.college_threshold:.1f}% threshold!"
    else:
        warning = ""

    lines = [
        f"Attendance Projection for {name}",
        f"- Action: {action_text.capitalize()} next {result.count} class(es)",
        f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)",
        f"- Projected Attendance: {result.projected_attended}/{result.projected_conducted} ({result.projected_percentage:.2f}%) [{delta_sign}{delta_pct:.2f}%]",
        f"- Threshold Status: {status_text}{warning}",
    ]
    return "\n".join(lines)


def format_classes_required(result: ClassesRequiredResult, subject_name: Optional[str] = None) -> str:
    """Format classes required for target calculation."""
    name = subject_name or result.subject_code

    if result.already_above_target:
        return (
            f"Target Goal for {name}\n"
            f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)\n"
            f"- Target: {result.target_percentage:.1f}%\n"
            f"- Status: [OK] You are already at or above your target! No additional required classes."
        )

    if result.impossible:
        return (
            f"Target Goal for {name}\n"
            f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)\n"
            f"- Target: {result.target_percentage:.1f}%\n"
            f"- Status: [IMPOSSIBLE] It is mathematically impossible to achieve exactly 100.0% attendance once a class has been missed."
        )

    projected_att = result.current_attended + result.classes_required
    projected_cond = result.current_conducted + result.classes_required
    projected_pct = (projected_att / projected_cond) * 100.0 if projected_cond > 0 else 0.0

    return (
        f"Target Goal for {name}\n"
        f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)\n"
        f"- Target: {result.target_percentage:.1f}%\n"
        f"- Classes Required: You must attend {result.classes_required} consecutive class(es) to reach {result.target_percentage:.1f}%.\n"
        f"- Projected after {result.classes_required} classes: {projected_att}/{projected_cond} ({projected_pct:.2f}%)"
    )


def format_max_can_miss(result: MaxCanMissResult, subject_name: Optional[str] = None) -> str:
    """Format maximum classes that can be missed."""
    name = subject_name or result.subject_code

    if result.already_below_threshold:
        deficit = result.threshold_percentage - result.current_percentage
        return (
            f"Bunk Margin for {name}\n"
            f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)\n"
            f"- Threshold: {result.threshold_percentage:.1f}%\n"
            f"- Safe Misses: 0 classes. [WARNING] You are currently {deficit:.2f}% below the {result.threshold_percentage:.1f}% threshold and cannot miss any classes."
        )

    projected_cond = result.current_conducted + result.max_can_miss
    projected_pct = (result.current_attended / projected_cond) * 100.0 if projected_cond > 0 else 0.0

    return (
        f"Bunk Margin for {name}\n"
        f"- Current Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)\n"
        f"- Threshold: {result.threshold_percentage:.1f}%\n"
        f"- Safe Misses: [OK] You can safely miss up to {result.max_can_miss} class(es) while maintaining >= {result.threshold_percentage:.1f}%.\n"
        f"- Projected after missing {result.max_can_miss} class(es): {result.current_attended}/{projected_cond} ({projected_pct:.2f}%)"
    )


def format_change_updates(
    changes: list[Any],
    snapshot_time: Optional[str] = None,
    subject_filter: Optional[str] = None,
) -> str:
    """Format attendance change detection results."""
    time_str = f" as of {snapshot_time}" if snapshot_time else ""
    subj_str = f" for {subject_filter}" if subject_filter else ""

    if not changes:
        return (
            f"Attendance Update Status{time_str}{subj_str}\n"
            f"- No attendance changes or new classes recorded since the last observation.\n"
            f"- Attendance records are up to date."
        )

    lines = [f"Attendance Changes Detected{time_str}{subj_str}:", ""]
    for c in changes:
        if isinstance(c, (AttendanceChangeRecord, AttendanceChangeResult)):
            status = c.status.value if hasattr(c.status, "value") else str(c.status)
            expl = c.explanation or "No explanation provided"
            lines.append(f"- [{c.subject_code}] {status}: {expl}")
        else:
            lines.append(f"- {c}")

    return "\n".join(lines)


# ── Phase 13 Assistant Response Templates ─────────────────────────────────────

def format_day_skip_forecast(result: Any) -> str:
    """Format the impact of skipping classes on a specific date (e.g. tomorrow)."""
    if result.is_holiday:
        return (
            f"{result.day_name} ({result.date_str}) is a holiday ({result.holiday_name}).\n"
            f"No attendance-bearing classes are scheduled, so your attendance is unaffected."
        )

    if result.classes_count == 0:
        return (
            f"No attendance-bearing classes are scheduled on {result.day_name} ({result.date_str}).\n"
            f"Your attendance remains at {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)."
        )

    subjects_list = ", ".join(f"{c.short_name}" for c in result.scheduled_classes)

    lines = [
        f"Tomorrow ({result.day_name}) you have {result.classes_count} attendance-bearing classes: {subjects_list}.",
        "",
        f"If you attend all {result.classes_count}:",
        f"  Overall Attendance → {result.if_attend_all_attended}/{result.if_attend_all_conducted} ({result.if_attend_all_percentage:.2f}%)",
        "",
        f"If you skip all {result.classes_count}:",
        f"  Overall Attendance → {result.if_skip_all_attended}/{result.if_skip_all_conducted} ({result.if_skip_all_percentage:.2f}%) [{result.delta_if_skipped:+.2f}%]",
        "",
        f"Current overall attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%).",
    ]

    if result.can_safely_skip_75:
        lines.append("[OK] You can safely skip tomorrow without dropping below the 75.0% college threshold.")
    elif result.can_safely_skip_70:
        lines.append("[WARNING] Skipping tomorrow will keep you above your 70.0% personal target, but drops below the 75.0% college threshold.")
    else:
        lines.append("[ALERT] At your current overall attendance, you have no safe classes to miss.")

    return "\n".join(lines)


def format_semester_forecast(result: Any) -> str:
    """Format full semester forecast and boundary calculations."""
    lines = [
        f"Semester Attendance Forecast (Until Last Working Day: {result.last_working_day})",
        f"- Current Overall Attendance: {result.current_attended}/{result.current_conducted} ({result.current_percentage:.2f}%)",
        f"- Remaining Future Classes: {result.future_classes_count} classes across {result.working_days_remaining} working days",
        f"- Final Conducted Total: {result.final_conducted_total} classes",
        "",
        "Achievable Attendance Range:",
        f"1. Maximum Possible (Attend 100% of remaining classes):",
        f"   → {result.max_possible_attended}/{result.final_conducted_total} ({result.max_possible_percentage:.2f}%)",
        f"2. Minimum Possible (Miss 100% of remaining classes):",
        f"   → {result.current_attended}/{result.final_conducted_total} ({result.min_possible_percentage:.2f}%)",
        "",
        "Goal Feasibility:",
    ]

    if result.target_75_feasible:
        lines.append(
            f"- 75.0% College Requirement: [FEASIBLE] You must attend at least {result.classes_needed_for_75} of {result.future_classes_count} remaining classes. "
            f"(Safe future misses: {result.max_future_misses_for_75} classes)"
        )
    else:
        lines.append(
            f"- 75.0% College Requirement: [IMPOSSIBLE] Even attending 100% of remaining classes reaches a maximum of {result.max_possible_percentage:.2f}%."
        )

    if result.target_70_feasible:
        lines.append(
            f"- 70.0% Personal Target: [FEASIBLE] You must attend at least {result.classes_needed_for_70} of {result.future_classes_count} remaining classes. "
            f"(Safe future misses: {result.max_future_misses_for_70} classes)"
        )
    else:
        lines.append(
            f"- 70.0% Personal Target: [IMPOSSIBLE] Maximum possible attendance is {result.max_possible_percentage:.2f}%."
        )

    return "\n".join(lines)


def format_subject_skip_impact(result: Any) -> str:
    """Format the impact of missing subject classes on overall attendance."""
    lines = [
        f"Impact of Missing {result.miss_count} Class(es) of {result.subject_full} ({result.subject_short}):",
        "",
        f"Primary Impact (Overall Attendance):",
        f"- Current Overall:   {result.current_overall_attended}/{result.current_overall_conducted} ({result.current_overall_percentage:.2f}%)",
        f"- Projected Overall: {result.projected_overall_attended}/{result.projected_overall_conducted} ({result.projected_overall_percentage:.2f}%) [{result.overall_delta:+.2f}%]",
        "",
        f"Secondary Impact ({result.subject_short} Subject):",
        f"- Current {result.subject_short}:   {result.current_subject_attended}/{result.current_subject_conducted} ({result.current_subject_percentage:.2f}%)",
        f"- Projected {result.subject_short}: {result.projected_subject_attended}/{result.projected_subject_conducted} ({result.projected_subject_percentage:.2f}%) [{result.subject_delta:+.2f}%]",
        "",
        f"Decision Verdict:",
        f"- {result.verdict}",
    ]
    return "\n".join(lines)


def format_snapshot_comparison(result: Any) -> str:
    """Format daily / snapshot-to-snapshot change comparison."""
    return result.summary_text


def format_date_schedule(
    date_str: str,
    day_name: str,
    is_holiday: bool,
    holiday_name: Optional[str],
    classes: list[Any],
) -> str:
    """Format daily timetable schedule and calendar status."""
    if is_holiday:
        return f"Schedule for {day_name} ({date_str}):\n- [HOLIDAY] {holiday_name or 'Holiday'}. No classes scheduled."

    if not classes:
        return f"Schedule for {day_name} ({date_str}):\n- No attendance-bearing classes scheduled."

    lines = [f"Schedule for {day_name} ({date_str}) -- {len(classes)} attendance-bearing class(es):", ""]
    for c in classes:
        subj = getattr(c, "short_name", None) or getattr(c, "subject", "")
        name = getattr(c, "full_name", "") or subj
        p_num = getattr(c, "period_num", "?")
        st = getattr(c, "start_time", "")
        et = getattr(c, "end_time", "")
        time_span = f"({st}–{et})" if st and et else ""
        lines.append(f"Period {p_num} {time_span}: {subj} - {name}")

    return "\n".join(lines)


def format_help(canonical_subjects: Optional[list[Subject]] = None) -> str:
    """Format interactive help message."""
    lines = [
        "College Attendance Tracker -- Help & Capabilities",
        "",
        "I am your personal attendance assistant with full knowledge of your attendance, timetable, holidays, and semester end date.",
        "",
        "Example Questions You Can Ask Me:",
        "1. Planning & Skips:        'Can I skip tomorrow?'",
        "2. Semester Forecast:       'What is the maximum attendance I can reach this semester?'",
        "3. Future Misses Margin:    'How many classes can I skip before the semester ends?'",
        "4. Subject-Specific Skip:   'What happens if I miss 5 ACD classes?'",
        "5. Daily Change History:    'What changed today?' / 'Did my attendance update?'",
        "6. Daily Schedule:          'What classes do I have tomorrow?' / 'Is tomorrow a holiday?'",
        "7. Subject Attendance:      'What is my attendance in OS?'",
        "8. Target Goals:            'How many classes do I need for 75% in GS?'",
        "9. Overall Attendance:      'Show my overall attendance summary'",
    ]

    if canonical_subjects:
        lines.append("")
        lines.append("Tracked Subjects:")
        for s in canonical_subjects:
            lines.append(f"- {s.short_name:<10} - {s.full_name} ({s.code})")

    return "\n".join(lines)


def format_unknown(query: str, reason: Optional[str] = None) -> str:
    """Format response when intent is unknown or cannot be understood."""
    reason_msg = f" ({reason})" if reason else ""
    return (
        f"I'm sorry, I couldn't understand your request: '{query}'{reason_msg}.\n\n"
        f"Try asking:\n"
        f"- 'Can I skip tomorrow?'\n"
        f"- 'What is the maximum attendance I can reach this semester?'\n"
        f"- 'What changed today?'\n"
        f"- 'What happens if I miss 5 classes in OS?'\n"
        f"- 'What is my attendance in OS?'\n"
        f"- 'How many classes do I need for 75% in ACD?'\n"
        f"- Or ask 'help' to see all capabilities."
    )


def format_error(message: str) -> str:
    """Format a general user-facing error message."""
    return f"Error: {message}"
