"""
web/routes/snapshots.py
───────────────────────
API endpoints for listing attendance snapshots and running historical comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from attendance_tracker.attendance.comparator import get_comparison_engine
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api/snapshots", tags=["Snapshots & History"])


@router.get("")
def list_snapshots(limit: int = Query(20, ge=1, le=100)) -> List[Dict[str, Any]]:
    """List historical attendance snapshots with summary metrics."""
    services = get_app_services()
    repo = services.repository

    with repo.get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT s.id, s.source, s.scraped_at,
                   SUM(r.attended) as total_attended,
                   SUM(r.conducted) as total_conducted,
                   COUNT(r.id) as subjects_count
            FROM attendance_snapshots s
            LEFT JOIN attendance_records r ON r.snapshot_id = s.id
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    result = []
    for r in rows:
        total_att = r["total_attended"] or 0
        total_cond = r["total_conducted"] or 0
        pct = (total_att / total_cond * 100.0) if total_cond > 0 else 0.0
        result.append({
            "id": r["id"],
            "source": r["source"],
            "scraped_at": r["scraped_at"],
            "total_attended": total_att,
            "total_conducted": total_cond,
            "percentage": round(pct, 2),
            "subjects_count": r["subjects_count"] or 0,
        })
    return result


@router.get("/compare")
def compare_snapshots(
    curr_id: Optional[int] = Query(None, description="Current / later snapshot ID"),
    prev_id: Optional[int] = Query(None, description="Previous / baseline snapshot ID"),
) -> Dict[str, Any]:
    """Compare two snapshots or compare the latest snapshot with its previous baseline."""
    services = get_app_services()
    comp_engine = get_comparison_engine(
        repository=services.repository,
        subject_resolver=services.subject_resolver,
    )

    if curr_id is None:
        result = comp_engine.compare_latest_with_previous()
    else:
        try:
            result = comp_engine.compare_snapshots(curr_snapshot_id=curr_id, prev_snapshot_id=prev_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return result.to_dict()
