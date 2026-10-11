"""
tests/test_web.py
──────────────────
Integration and functional test suite for Phase 12 Web Application & REST API.
"""

import pytest
from starlette.testclient import TestClient
from unittest.mock import patch, MagicMock

from attendance_tracker.web.app import create_app
from attendance_tracker.web.state import get_app_services


@pytest.fixture(scope="module")
def client():
    """Create a test client with application lifespan enabled."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_static_index_html(client):
    """Test that GET / returns the Single Page Application index.html."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "VNR VJIET Attendance Tracker" in res.text
    assert "Overall Attendance" in res.text
    assert "subject-grid" in res.text


def test_static_assets_serving(client):
    """Test that static CSS and JS assets are correctly served."""
    css_res = client.get("/static/css/style.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]
    assert "--bg-main" in css_res.text

    api_js_res = client.get("/static/js/api.js")
    assert api_js_res.status_code == 200
    assert "API = {" in api_js_res.text

    app_js_res = client.get("/static/js/app.js")
    assert app_js_res.status_code == 200
    assert "DOMContentLoaded" in app_js_res.text


def test_health_endpoint(client):
    """Test GET /api/health telemetry and status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_parameters"] == 5853184
    assert data["subjects_loaded"] == 9
    assert data["device"] in ["cuda", "cpu"]


def test_dashboard_endpoint(client):
    """Test GET /api/dashboard returns overall attendance and 9 canonical subjects."""
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()

    assert "overall" in data
    overall = data["overall"]
    assert overall["total_classes"] in (255, 313)
    assert overall["attended_classes"] in (163, 185)
    assert overall["target_75"] == 75.0
    assert overall["target_70"] == 70.0
    assert overall["classes_needed_for_75"] >= 0

    assert "subjects" in data
    subjects = data["subjects"]
    assert len(subjects) == 9

    codes = [s["code"] for s in subjects]
    assert "22PC1IT202" in codes  # OS
    assert "22PC1AM203" in codes  # ACD


def test_get_subjects_list(client):
    """Test GET /api/subjects returns all 9 subjects."""
    res = client.get("/api/subjects")
    assert res.status_code == 200
    subjects = res.json()
    assert len(subjects) == 9
    assert any(s["short_name"] == "OS" for s in subjects)


def test_get_subject_detail_by_code_and_shortname(client):
    """Test GET /api/subjects/{code} by short name and canonical course code."""
    # 1. By short name "OS"
    res_os = client.get("/api/subjects/OS")
    assert res_os.status_code == 200
    sub_os = res_os.json()
    assert sub_os["code"] == "22PC1IT202"
    assert sub_os["short_name"] == "OS"
    assert sub_os["total_classes"] in (33, 36)
    assert sub_os["attended_classes"] in (22, 25)

    # 2. By code "22PC1IT202"
    res_code = client.get("/api/subjects/22PC1IT202")
    assert res_code.status_code == 200
    assert res_code.json()["short_name"] == "OS"

    # 3. Nonexistent subject
    res_unknown = client.get("/api/subjects/NONEXISTENT")
    assert res_unknown.status_code == 404


def test_simulate_projection_endpoint(client):
    """Test POST /api/subjects/{code}/simulate for attend and miss actions."""
    # Simulate attending 3 next classes in OS
    res_att = client.post(
        "/api/subjects/OS/simulate",
        json={"action": "attend", "count": 3}
    )
    assert res_att.status_code == 200
    data_att = res_att.json()
    assert data_att["projected_attended"] in (25, 28)
    assert data_att["projected_total"] in (36, 39)

    # Simulate missing 2 next classes in OS
    res_miss = client.post(
        "/api/subjects/OS/simulate",
        json={"action": "miss", "count": 2}
    )
    assert res_miss.status_code == 200
    data_miss = res_miss.json()
    assert data_miss["projected_attended"] in (22, 25)
    assert data_miss["projected_total"] in (35, 38)

    # Invalid count
    res_inv = client.post(
        "/api/subjects/OS/simulate",
        json={"action": "attend", "count": 0}
    )
    assert res_inv.status_code == 422


def test_ask_assistant_endpoint(client):
    """Test POST /api/ask routes natural language queries through ToolRouter."""
    # Query 1: Subject Attendance
    res1 = client.post("/api/ask", json={"query": "What is my attendance in OS?"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert "Operating Systems" in data1["response"]
    assert data1["intent"] == "GET_SUBJECT_ATTENDANCE"
    assert data1["tool"] == "get_attendance_for_subject"
    assert data1["latency_ms"] >= 0

    # Query 2: Max classes can miss in ACD
    res2 = client.post("/api/ask", json={"query": "How many classes can I miss in ACD?"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert "Automata and Compiler Design" in data2["response"]
    assert data2["subject"] == "ACD"
    assert data2["intent"] == "MAX_CLASSES_CAN_MISS"

    # Query 3: Empty query error
    res_empty = client.post("/api/ask", json={"query": "   "})
    assert res_empty.status_code == 400


def test_updates_endpoint(client):
    """Test GET /api/updates returns change logs and timetable."""
    res = client.get("/api/updates")
    assert res.status_code == 200
    data = res.json()
    assert "recent_logs" in data
    assert "today_schedule" in data
    assert isinstance(data["recent_logs"], list)
    assert isinstance(data["today_schedule"], list)


def test_sync_endpoint_mocked(client):
    """Test POST /api/sync executes safely and does not leak credentials."""
    from attendance_tracker.portal.models import PortalSyncResult
    from datetime import datetime

    with patch("attendance_tracker.portal.service.PortalAttendanceService.sync_attendance") as mock_sync:
        mock_sync.return_value = PortalSyncResult(
            success=True,
            snapshot_id=99,
            records_count=9,
            changes_detected=[],
            duration_ms=150.0,
            timestamp=datetime.now(),
        )
        res = client.post("/api/sync")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["status"] == "SUCCESS"
        assert "password" not in str(data).lower()
        assert "token" not in str(data).lower()
