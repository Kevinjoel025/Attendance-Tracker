"""
web/__main__.py
───────────────
CLI entry point for launching the local web interface.
Usage:
    python -m attendance_tracker.web
    python -m attendance_tracker.web --port 8000 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Attendance Tracker local web application.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("==================================================================")
    print("  COLLEGE ATTENDANCE TRACKER — LOCAL WEB APPLICATION")
    print("  VNR VJIET Semester NLU Assistant")
    print("==================================================================")
    print(f"  Server URL:  http://{args.host}:{args.port}")
    print(f"  API Docs:    http://{args.host}:{args.port}/docs")
    print("==================================================================\n")

    uvicorn.run(
        "attendance_tracker.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
