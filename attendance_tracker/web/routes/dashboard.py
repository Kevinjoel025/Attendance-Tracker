"""
web/routes/dashboard.py
───────────────────────
API endpoints for attendance dashboard, overall metrics, and subject details.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from attendance_tracker.attendance.engine import (
    classes_required_for_target,
    compute_projection,
    current_percentage,
    max_classes_can_miss,
)
from attendance_tracker.core.config import config
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api", tags=["Dashboard & Subjects"])


# ── Response Schemas ──────────────────────────────────────────────────────────

class SubjectSummary(BaseModel):
    code: str
    short_name: str
    name: str
    threshold: float
    attended_classes: int
    total_classes: int
    percentage: float
    status_75: str  # "ABOVE" | "BELOW"
    status_70: str  # "ABOVE" | "BELOW"
    classes_needed_for_75: int
    classes_needed_for_70: int
    safe_classes_to_miss_75: int
    safe_classes_to_miss_70: int


class OverallMetrics(BaseModel):
    total_classes: int
    attended_classes: int
    percentage: float
    target_75: float = 75.0
    target_70: float = 70.0
    status_75: str
    status_70: str
    classes_needed_for_75: int
    classes_needed_for_70: int
    safe_classes_to_miss_75: int
    safe_classes_to_miss_70: int


class DashboardResponse(BaseModel):
    overall: OverallMetrics
    subjects: List[SubjectSummary]
    last_synced: Optional[str] = None
    snapshot_id: Optional[int] = None
    source: Optional[str] = "portal"


class ProjectionDetail(BaseModel):
    action: str  # "ATTEND" | "MISS"
    count: int
    simulated_count: int
    projected_attended: int
    projected_total: int
    projected_percentage: float
    delta_percentage: float
    meets_threshold: bool


class SubjectDetailResponse(BaseModel):
    code: str
    short_name: str
    name: str
    threshold: float
    personal_target: float
    attended_classes: int
    total_classes: int
    percentage: float
    status_75: str
    status_70: str
    last_scraped_at: Optional[str] = None
    snapshot_id: Optional[int] = None
    classes_needed_for_75: int
    classes_needed_for_70: int
    safe_classes_to_miss_75: int
    safe_classes_to_miss_70: int
    classes_required_75: Dict[str, Any]
    classes_required_70: Dict[str, Any]
    max_can_miss_75: Dict[str, Any]
    max_can_miss_70: Dict[str, Any]
    projections: List[ProjectionDetail]


class SimulationRequest(BaseModel):
    action: str = Field(default="attend", description="attend or miss")
    count: int = Field(default=1, ge=1, le=100, description="Number of classes")


# ── Route Handlers ────────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard() -> DashboardResponse:
    """Fetch complete overview metrics and all 9 current-semester subjects."""
    services = get_app_services()
    overall = services.repository.get_all_current_attendance()

    college_thresh = config.attendance.college_threshold
    personal_tgt = config.attendance.personal_target

    if overall is None or not overall.subject_records:
        subjects_list = services.repository.list_subjects()
        subject_summaries: list[SubjectSummary] = []
        for s in subjects_list:
            subject_summaries.append(
                SubjectSummary(
                    code=s.code,
                    short_name=s.short_name,
                    name=s.full_name,
                    threshold=s.threshold,
                    attended_classes=0,
                    total_classes=0,
                    percentage=0.0,
                    status_75="BELOW",
                    status_70="BELOW",
                    classes_needed_for_75=0,
                    classes_needed_for_70=0,
                    safe_classes_to_miss_75=0,
                    safe_classes_to_miss_70=0,
                )
            )
        return DashboardResponse(
            overall=OverallMetrics(
                total_classes=0,
                attended_classes=0,
                percentage=0.0,
                target_75=college_thresh,
                target_70=personal_tgt,
                status_75="BELOW",
                status_70="BELOW",
                classes_needed_for_75=0,
                classes_needed_for_70=0,
                safe_classes_to_miss_75=0,
                safe_classes_to_miss_70=0,
            ),
            subjects=subject_summaries,
            last_synced=None,
            snapshot_id=None,
        )

    # Compute overall calculations
    overall_req_75 = classes_required_for_target(overall.total_attended, overall.total_conducted, college_thresh)
    overall_req_70 = classes_required_for_target(overall.total_attended, overall.total_conducted, personal_tgt)
    overall_miss_75 = max_classes_can_miss(overall.total_attended, overall.total_conducted, college_thresh)
    overall_miss_70 = max_classes_can_miss(overall.total_attended, overall.total_conducted, personal_tgt)

    subject_summaries: list[SubjectSummary] = []

    for rec in overall.subject_records:
        s_pct = rec.percentage
        s_75 = "ABOVE" if s_pct >= college_thresh else "BELOW"
        s_70 = "ABOVE" if s_pct >= personal_tgt else "BELOW"

        req_75 = classes_required_for_target(rec.attended, rec.conducted, college_thresh)
        req_70 = classes_required_for_target(rec.attended, rec.conducted, personal_tgt)
        miss_75 = max_classes_can_miss(rec.attended, rec.conducted, college_thresh)
        miss_70 = max_classes_can_miss(rec.attended, rec.conducted, personal_tgt)

        subject_summaries.append(
            SubjectSummary(
                code=rec.subject.code,
                short_name=rec.subject.short_name,
                name=rec.subject.full_name,
                threshold=rec.subject.threshold,
                attended_classes=rec.attended,
                total_classes=rec.conducted,
                percentage=round(s_pct, 2),
                status_75=s_75,
                status_70=s_70,
                classes_needed_for_75=req_75.classes_required if not req_75.already_above_target else 0,
                classes_needed_for_70=req_70.classes_required if not req_70.already_above_target else 0,
                safe_classes_to_miss_75=miss_75.max_can_miss if not miss_75.already_below_threshold else 0,
                safe_classes_to_miss_70=miss_70.max_can_miss if not miss_70.already_below_threshold else 0,
            )
        )

    latest_snap = services.repository.get_latest_snapshot()
    source_val = latest_snap.source if latest_snap else "portal"

    return DashboardResponse(
        overall=OverallMetrics(
            total_classes=overall.total_conducted,
            attended_classes=overall.total_attended,
            percentage=round(overall.overall_percentage, 2),
            target_75=college_thresh,
            target_70=personal_tgt,
            status_75="ABOVE" if overall.overall_percentage >= college_thresh else "BELOW",
            status_70="ABOVE" if overall.overall_percentage >= personal_tgt else "BELOW",
            classes_needed_for_75=overall_req_75.classes_required if not overall_req_75.already_above_target else 0,
            classes_needed_for_70=overall_req_70.classes_required if not overall_req_70.already_above_target else 0,
            safe_classes_to_miss_75=overall_miss_75.max_can_miss if not overall_miss_75.already_below_threshold else 0,
            safe_classes_to_miss_70=overall_miss_70.max_can_miss if not overall_miss_70.already_below_threshold else 0,
        ),
        subjects=subject_summaries,
        last_synced=overall.scraped_at,
        snapshot_id=overall.snapshot_id,
        source=source_val,
    )


@router.get("/subjects", response_model=List[SubjectSummary])
def list_subjects() -> List[SubjectSummary]:
    """List all subjects with current attendance numbers and threshold status."""
    dashboard = get_dashboard()
    return dashboard.subjects


@router.get("/subjects/{subject_id}", response_model=SubjectDetailResponse)
def get_subject_detail(subject_id: str) -> SubjectDetailResponse:
    """
    Retrieve rich detail and deterministic calculations for a single subject.
    Resolves subject by code, short name, or alias.
    """
    services = get_app_services()
    subj = services.subject_resolver.resolve(subject_id)
    if subj is None:
        raise HTTPException(
            status_code=404,
            detail=f"Subject '{subject_id}' not found. Supported subjects: GS, OE-1, DE, ACD, IoTAP, OS, DE LAB, IoTAP LAB, WP.",
        )

    rec = services.repository.get_attendance_for_subject(subj.code)
    college_thresh = config.attendance.college_threshold
    personal_tgt = config.attendance.personal_target

    attended = rec.attended if rec else 0
    conducted = rec.conducted if rec else 0
    pct = rec.percentage if rec else 0.0

    # Deterministic calculations using existing attendance.engine
    req_75 = classes_required_for_target(attended, conducted, college_thresh)
    req_70 = classes_required_for_target(attended, conducted, personal_tgt)
    miss_75 = max_classes_can_miss(attended, conducted, college_thresh)
    miss_70 = max_classes_can_miss(attended, conducted, personal_tgt)

    # Standard projections
    projections: list[ProjectionDetail] = []
    simulation_scenarios = [
        ("ATTEND", 1),
        ("ATTEND", 2),
        ("ATTEND", 3),
        ("ATTEND", 5),
        ("MISS", 1),
        ("MISS", 2),
        ("MISS", 3),
    ]

    for action, count in simulation_scenarios:
        res = compute_projection(
            subject_code=subj.code,
            attended=attended,
            conducted=conducted,
            action=action,
            count=count,
            college_threshold=college_thresh,
        )
        projections.append(
            ProjectionDetail(
                action=action,
                count=count,
                simulated_count=count,
                projected_attended=res.projected_attended,
                projected_total=res.projected_conducted,
                projected_percentage=round(res.projected_percentage, 2),
                delta_percentage=round(res.projected_percentage - res.current_percentage, 2),
                meets_threshold=res.above_college_threshold_after,
            )
        )

    return SubjectDetailResponse(
        code=subj.code,
        short_name=subj.short_name,
        name=subj.full_name,
        threshold=subj.threshold,
        personal_target=personal_tgt,
        attended_classes=attended,
        total_classes=conducted,
        percentage=round(pct, 2),
        status_75="ABOVE" if pct >= college_thresh else "BELOW",
        status_70="ABOVE" if pct >= personal_tgt else "BELOW",
        last_scraped_at=rec.scraped_at if rec else None,
        snapshot_id=rec.snapshot_id if rec else None,
        classes_needed_for_75=req_75.classes_required if not req_75.already_above_target else 0,
        classes_needed_for_70=req_70.classes_required if not req_70.already_above_target else 0,
        safe_classes_to_miss_75=miss_75.max_can_miss if not miss_75.already_below_threshold else 0,
        safe_classes_to_miss_70=miss_70.max_can_miss if not miss_70.already_below_threshold else 0,
        classes_required_75={
            "target": college_thresh,
            "classes_required": req_75.classes_required,
            "already_above_target": req_75.already_above_target,
            "impossible": req_75.impossible,
        },
        classes_required_70={
            "target": personal_tgt,
            "classes_required": req_70.classes_required,
            "already_above_target": req_70.already_above_target,
            "impossible": req_70.impossible,
        },
        max_can_miss_75={
            "threshold": college_thresh,
            "max_can_miss": miss_75.max_can_miss,
            "already_below_threshold": miss_75.already_below_threshold,
        },
        max_can_miss_70={
            "threshold": personal_tgt,
            "max_can_miss": miss_70.max_can_miss,
            "already_below_threshold": miss_70.already_below_threshold,
        },
        projections=projections,
    )


@router.post("/subjects/{subject_id}/simulate", response_model=ProjectionDetail)
def simulate_custom_projection(
    subject_id: str,
    req: SimulationRequest,
) -> ProjectionDetail:
    """Run an on-the-fly attendance projection calculation for any custom count."""
    services = get_app_services()
    subj = services.subject_resolver.resolve(subject_id)
    if subj is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")

    rec = services.repository.get_attendance_for_subject(subj.code)
    attended = rec.attended if rec else 0
    conducted = rec.conducted if rec else 0
    college_thresh = config.attendance.college_threshold

    action_norm = req.action.upper().strip()
    if action_norm not in ["ATTEND", "MISS"]:
        raise HTTPException(status_code=400, detail="Action must be 'ATTEND' or 'MISS'.")

    res = compute_projection(
        subject_code=subj.code,
        attended=attended,
        conducted=conducted,
        action=action_norm,
        count=req.count,
        college_threshold=college_thresh,
    )

    return ProjectionDetail(
        action=action_norm,
        count=req.count,
        simulated_count=req.count,
        projected_attended=res.projected_attended,
        projected_total=res.projected_conducted,
        projected_percentage=round(res.projected_percentage, 2),
        delta_percentage=round(res.projected_percentage - res.current_percentage, 2),
        meets_threshold=res.above_college_threshold_after,
    )
