"""
web/app.py
──────────
FastAPI web application factory and server configuration.
Configures lifespan management, API routers, and static file serving.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from attendance_tracker.core.config import config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.web.routes.calendar import router as calendar_router
from attendance_tracker.web.routes.chat import router as chat_router
from attendance_tracker.web.routes.dashboard import router as dashboard_router
from attendance_tracker.web.routes.health import router as health_router
from attendance_tracker.web.routes.import_data import router as import_router
from attendance_tracker.web.routes.snapshots import router as snapshots_router
from attendance_tracker.web.routes.sync import router as sync_router
from attendance_tracker.web.routes.updates import router as updates_router
from attendance_tracker.web.state import init_app_services

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the SLM model checkpoint and initializes database connections ONCE at startup.
    """
    logger.info("FastAPI application starting up: pre-warming backend services & SLM...")
    init_app_services()
    logger.info("FastAPI startup complete: ready to serve attendance requests.")
    yield
    logger.info("FastAPI application shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=config.project.name,
        version=config.project.version,
        description="Local Web Application for College Attendance Tracking and NLU Assistant (VNR VJIET)",
        lifespan=lifespan,
    )

    # CORS configuration for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(dashboard_router)
    app.include_router(chat_router)
    app.include_router(sync_router)
    app.include_router(updates_router)
    app.include_router(health_router)
    app.include_router(calendar_router)
    app.include_router(import_router)
    app.include_router(snapshots_router)

    # Mount static assets directory if it exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_root():
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return {"message": "Attendance Tracker API is online."}

    return app


app = create_app()
