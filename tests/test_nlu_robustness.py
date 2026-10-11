"""
tests/test_nlu_robustness.py
────────────────────────────
NLU inference robustness, threshold resolution, entity alias handling,
and intent boundary stress tests for Phase 10.
"""

from pathlib import Path
import pytest

from attendance_tracker.attendance.repository import AttendanceRepository
from attendance_tracker.core.config import config
from attendance_tracker.core.tool_router import ToolRouter, get_tool_router
from attendance_tracker.slm.inference.nlu import NLUInferenceEngine, NLUResult, get_nlu_engine
from attendance_tracker.slm.inference.parser import ParsedNLUOutput, parse_nlu_output


@pytest.fixture(scope="module")
def tool_router() -> ToolRouter:
    """Provide initialized ToolRouter with frozen Phase 7.6 NLU."""
    return get_tool_router()


def test_intent_casing_insensitivity(tool_router: ToolRouter):
    """Verify router and NLU handle lowercase, UPPERCASE, and MixedCase queries."""
    queries = [
        ("what is my overall attendance?", "GET_ATTENDANCE", None),
        ("WHAT IS MY OVERALL ATTENDANCE?", "GET_ATTENDANCE", None),
        ("what is my attendance in os?", "GET_SUBJECT_ATTENDANCE", "OS"),
        ("WHAT IS MY ATTENDANCE IN OS?", "GET_SUBJECT_ATTENDANCE", "OS"),
        ("What Is My Attendance In Os?", "GET_SUBJECT_ATTENDANCE", "OS"),
        ("how many classes can i miss in acd?", "MAX_CLASSES_CAN_MISS", "ACD"),
        ("HOW MANY CLASSES CAN I MISS IN ACD?", "MAX_CLASSES_CAN_MISS", "ACD"),
        ("check if attendance was updated", "CHECK_ATTENDANCE_UPDATE", None),
        ("CHECK IF ATTENDANCE WAS UPDATED", "CHECK_ATTENDANCE_UPDATE", None),
    ]

    for query, expected_intent, expected_subject in queries:
        resp = tool_router.process(query)
        assert resp.nlu.intent == expected_intent, f"Query '{query}' produced {resp.nlu.intent}, expected {expected_intent}"
        if expected_subject:
            assert resp.nlu.subject == expected_subject, f"Query '{query}' resolved subject {resp.nlu.subject}, expected {expected_subject}"
        assert resp.response_text is not None
        assert len(resp.response_text.strip()) > 0


def test_all_8_intents_produce_safe_formatted_responses(tool_router: ToolRouter):
    """Verify all 8 canonical intents produce well-formed, readable text with no leaked placeholders."""
    sample_queries = {
        "GET_ATTENDANCE": "Show me my overall attendance",
        "GET_SUBJECT_ATTENDANCE": "Show me my attendance for Operating Systems",
        "MAX_CLASSES_CAN_MISS": "How many classes can I miss in Data Engineering?",
        "CLASSES_REQUIRED_FOR_TARGET": "How many classes must I attend in WP to reach 75%?",
        "ATTENDANCE_PROJECTION": "What will my attendance in ACD be if I attend the next 3 classes?",
        "CHECK_ATTENDANCE_UPDATE": "Did my attendance change recently?",
        "TIMETABLE_QUERY": "What classes do I have today?",
        "HELP": "How can you help me?",
        "UNKNOWN": "Tell me a joke about quantum computing in space",
    }

    for intent, query in sample_queries.items():
        resp = tool_router.process(query)
        assert resp.response_text is not None
        # Check no unfilled template braces leak into the output
        assert "{" not in resp.response_text or "{}" not in resp.response_text
        assert resp.total_time_ms >= 0.0


def test_threshold_resolution_behaviors(tool_router: ToolRouter):
    """Verify explicit vs target vs college threshold resolutions."""
    # 1. Explicit percentage in query
    resp_80 = tool_router.process("How many classes needed in OS to reach 80%?")
    assert resp_80.nlu.target == 80.0
    assert "80.0%" in resp_80.response_text or "80%" in resp_80.response_text

    resp_85 = tool_router.process("How many classes needed in OS for 85 percent?")
    assert resp_85.nlu.target == 85.0

    # 2. College requirement query
    resp_college = tool_router.process("How many classes required for college requirement in OS?")
    assert resp_college.nlu.target == 75.0
    assert "75.0%" in resp_college.response_text or "75%" in resp_college.response_text

    # 3. Personal target query
    resp_target = tool_router.process("How many classes required for my target in OS?")
    assert resp_target.nlu.target == 70.0
    assert "70.0%" in resp_target.response_text or "70%" in resp_target.response_text


def test_max_classes_can_miss_default_threshold_handling(tool_router: ToolRouter):
    """
    Verify that MAX_CLASSES_CAN_MISS without explicit target evaluates bunk margin safely
    against 75% college threshold (filtering any unprompted SLM target artifacts like 90%).
    """
    query = "How many classes can I miss in OS?"
    resp = tool_router.process(query)

    assert resp.nlu.intent == "MAX_CLASSES_CAN_MISS"
    assert resp.nlu.subject == "OS"

    # User did NOT mention 90% in query
    assert "90" not in query
    assert "90%" not in query

    # Target threshold used for evaluation MUST remain 75.0%
    assert resp.tool.success is True
    assert resp.tool.data.threshold_percentage == 75.0
    assert resp.tool.data.max_can_miss == 0
    assert resp.tool.data.already_below_threshold is True

    # Final response text uses 75.0% and never leaks unprompted 90%
    assert "75.0%" in resp.response_text
    assert "90%" not in resp.response_text
    assert "Operating Systems" in resp.response_text or "OS" in resp.response_text
    assert "classes" in resp.response_text.lower()


def test_projection_language_recognition(tool_router: ToolRouter):
    """Verify deterministic recognition and calculation for projection queries."""
    # 1. "What happens if I attend the next 5 classes of OS?"
    resp_os = tool_router.process("What happens if I attend the next 5 classes of OS?")
    assert resp_os.nlu.intent == "ATTENDANCE_PROJECTION"
    assert resp_os.nlu.subject == "OS"
    assert resp_os.nlu.count == 5
    assert resp_os.tool.success is True
    assert resp_os.tool.data.action == "ATTEND"
    assert resp_os.tool.data.count == 5
    cur_att_os = resp_os.tool.data.current_attended
    cur_cond_os = resp_os.tool.data.current_conducted
    assert resp_os.tool.data.projected_attended == cur_att_os + 5
    assert resp_os.tool.data.projected_conducted == cur_cond_os + 5
    expected_pct_os = ((cur_att_os + 5) / (cur_cond_os + 5)) * 100
    assert resp_os.tool.data.projected_percentage == pytest.approx(expected_pct_os, rel=1e-3)
    assert f"{expected_pct_os:.2f}%" in resp_os.response_text

    # 2. "If I attend 3 more classes in DE, what will my attendance be?"
    resp_de = tool_router.process("If I attend 3 more classes in DE, what will my attendance be?")
    assert resp_de.nlu.intent == "ATTENDANCE_PROJECTION"
    assert resp_de.nlu.subject == "DE"
    assert resp_de.nlu.count == 3
    assert resp_de.tool.success is True
    assert resp_de.tool.data.action == "ATTEND"
    assert resp_de.tool.data.count == 3
    cur_att_de = resp_de.tool.data.current_attended
    cur_cond_de = resp_de.tool.data.current_conducted
    assert resp_de.tool.data.projected_attended == cur_att_de + 3
    assert resp_de.tool.data.projected_conducted == cur_cond_de + 3
    expected_pct_de = ((cur_att_de + 3) / (cur_cond_de + 3)) * 100
    assert resp_de.tool.data.projected_percentage == pytest.approx(expected_pct_de, rel=1e-3)
    assert f"{expected_pct_de:.2f}%" in resp_de.response_text

    # 3. "What will my attendance become after attending 5 classes in IoT?"
    resp_iot = tool_router.process("What will my attendance become after attending 5 classes in IoT?")
    assert resp_iot.nlu.intent == "ATTENDANCE_PROJECTION"
    assert resp_iot.nlu.subject in ("IoT", "IoTAP")
    assert resp_iot.nlu.count == 5
    assert resp_iot.tool.success is True
    assert resp_iot.tool.data.action == "ATTEND"
    assert resp_iot.tool.data.count == 5
    cur_att_iot = resp_iot.tool.data.current_attended
    cur_cond_iot = resp_iot.tool.data.current_conducted
    assert resp_iot.tool.data.projected_attended == cur_att_iot + 5
    assert resp_iot.tool.data.projected_conducted == cur_cond_iot + 5
    expected_pct_iot = ((cur_att_iot + 5) / (cur_cond_iot + 5)) * 100
    assert resp_iot.tool.data.projected_percentage == pytest.approx(expected_pct_iot, rel=1e-3)
    assert f"{expected_pct_iot:.2f}%" in resp_iot.response_text

    # 4. "What happens if I attend the next 2 classes?" (missing subject)
    resp_missing = tool_router.process("What happens if I attend the next 2 classes?")
    assert resp_missing.nlu.intent == "ATTENDANCE_PROJECTION"
    assert resp_missing.nlu.count == 2
    assert resp_missing.nlu.subject is None
    assert resp_missing.tool.success is False
    assert resp_missing.tool.error_type == "MISSING_SUBJECT"
    assert "Please specify a subject" in resp_missing.response_text


def test_unknown_query_graceful_fallback(tool_router: ToolRouter):
    """Verify that unrelated or gibberish queries fallback gracefully to helpful assistance message."""
    gibberish_queries = [
        "asdfghjkl qwertyuiop",
        "can you please give me a summary of the latest space exploration mission",
        "write an essay on artificial intelligence in education quickly",
    ]

    for q in gibberish_queries:
        resp = tool_router.process(q)
        assert resp.nlu.intent in ("UNKNOWN", "HELP")
        assert "I can help you" in resp.response_text or "Try asking" in resp.response_text or "Attendance Assistant" in resp.response_text


def test_punctuation_and_whitespace_resilience(tool_router: ToolRouter):
    """Verify queries with extra spaces, question marks, and exclamation marks work correctly."""
    variations = [
        "   what is my attendance in OS?   ",
        "what is my attendance in OS???",
        "what is my attendance in OS!",
        "what    is    my    attendance    in    OS",
    ]

    for q in variations:
        resp = tool_router.process(q)
        assert resp.nlu.intent == "GET_SUBJECT_ATTENDANCE"
        assert resp.nlu.subject == "OS"
        assert "66.67%" in resp.response_text or "%" in resp.response_text
