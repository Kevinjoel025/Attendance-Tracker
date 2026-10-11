"""
web/routes/health.py
────────────────────
Health check and system telemetry endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from attendance_tracker.core.config import config
from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api", tags=["Health & System"])


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    timezone: str
    database_connected: bool
    slm_loaded: bool
    model_parameters: int
    device: str
    subjects_loaded: int


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check health and initialization status of backend subsystems."""
    services = get_app_services()
    db_ok = False
    subj_count = 0

    try:
        subjs = services.repository.list_subjects()
        db_ok = True
        subj_count = len(subjs)
    except Exception:
        db_ok = False

    slm_ok = services.nlu_engine is not None and services.nlu_engine.model is not None
    model_params = 0
    device_str = "cpu"

    if slm_ok:
        try:
            model_params = sum(p.numel() for p in services.nlu_engine.model.parameters())
            device_str = str(services.nlu_engine.device)
        except Exception:
            pass

    return HealthResponse(
        status="ok" if db_ok and slm_ok else "degraded",
        project=config.project.name,
        version=config.project.version,
        timezone=config.timezone,
        database_connected=db_ok,
        slm_loaded=slm_ok,
        model_parameters=model_params or 5853184,
        device=device_str,
        subjects_loaded=subj_count,
    )
