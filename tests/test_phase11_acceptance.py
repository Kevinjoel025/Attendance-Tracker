"""
tests/test_phase11_acceptance.py
────────────────────────────────
Phase 11 Acceptance & Regression Test Suite.

Validates:
1. Raw SLM baseline diagnostics across acceptance dataset.
2. Final application quality (Intent, Subject, Target, Count, Execution).
3. 100% subject resolution across all 9 canonical subjects.
4. 100% domain intent routing for all student attendance operations.
"""

import json
from pathlib import Path
import pytest

from attendance_tracker.attendance.repository import get_repository
from attendance_tracker.core.subject_resolver import get_subject_resolver
from attendance_tracker.core.thresholds import COLLEGE_THRESHOLD, resolve_threshold
from attendance_tracker.core.tool_router import ToolRouter
from attendance_tracker.slm.inference.nlu import get_nlu_engine
from scripts.evaluate_acceptance import evaluate_acceptance


@pytest.fixture(scope="module")
def acceptance_results():
    """Run Phase 11 acceptance evaluation and cache results."""
    data_path = Path("tests/data/phase11_acceptance.json")
    assert data_path.exists(), f"Acceptance data not found at {data_path}"
    return evaluate_acceptance(data_path)


def test_phase11_acceptance_dataset_integrity():
    """Verify acceptance dataset is properly formed with all required fields."""
    data_path = Path("tests/data/phase11_acceptance.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 150, f"Expected at least 150 acceptance cases, got {len(data)}"
    for item in data:
        assert "query" in item
        assert "expected_intent" in item
        assert "category" in item


def test_phase11_raw_slm_quality(acceptance_results):
    """Verify raw SLM meets baseline quality criteria without application overrides."""
    raw = acceptance_results["raw_slm"]
    assert raw["intent_accuracy"] >= 80.0, f"Raw SLM intent accuracy below threshold: {raw['intent_accuracy']}%"
    assert raw["subject_accuracy"] >= 80.0, f"Raw SLM subject accuracy below threshold: {raw['subject_accuracy']}%"
    assert raw["malformed_rate"] <= 10.0, f"Raw SLM malformed rate too high: {raw['malformed_rate']}%"
    assert raw["exact_structured_match"] >= 75.0, f"Raw SLM exact match below threshold: {raw['exact_structured_match']}%"


def test_phase11_pipeline_intent_accuracy(acceptance_results):
    """Verify final pipeline intent accuracy exceeds 98%."""
    pipe = acceptance_results["pipeline"]
    assert pipe["intent_accuracy"] >= 98.0, f"Pipeline intent accuracy below threshold: {pipe['intent_accuracy']}%"


def test_phase11_pipeline_subject_accuracy(acceptance_results):
    """Verify final pipeline achieves 100% subject resolution across all subjects."""
    pipe = acceptance_results["pipeline"]
    assert pipe["subject_accuracy"] == 100.0, f"Pipeline subject accuracy not 100%: {pipe['subject_accuracy']}%"


def test_phase11_pipeline_target_and_count_accuracy(acceptance_results):
    """Verify final pipeline achieves 100% accuracy on target percentages and class counts."""
    pipe = acceptance_results["pipeline"]
    assert pipe["target_accuracy"] == 100.0, f"Pipeline target accuracy not 100%: {pipe['target_accuracy']}%"
    assert pipe["count_accuracy"] == 100.0, f"Pipeline count accuracy not 100%: {pipe['count_accuracy']}%"


def test_phase11_pipeline_execution_success(acceptance_results):
    """Verify 100% of acceptance queries execute without unhandled exceptions."""
    pipe = acceptance_results["pipeline"]
    assert pipe["success_rate"] == 100.0, f"Pipeline execution success rate not 100%: {pipe['success_rate']}%"


def test_phase11_per_subject_perfect_resolution(acceptance_results):
    """Verify every one of the 9 canonical subjects achieves 100% resolution."""
    per_subj = acceptance_results["pipeline"]["per_subject"]
    canonical_subjects = ["GS", "OE-1", "DE", "ACD", "IoTAP", "OS", "DE LAB", "IoTAP LAB", "WP"]
    for subj in canonical_subjects:
        assert subj in per_subj, f"Subject {subj} missing from per-subject evaluation"
        assert per_subj[subj]["accuracy"] == 100.0, f"Subject {subj} resolution not 100%: {per_subj[subj]['accuracy']}%"


def test_phase11_domain_intents_perfect_routing(acceptance_results):
    """Verify all 7 attendance domain intents achieve 100% routing accuracy."""
    per_intent = acceptance_results["pipeline"]["per_intent"]
    domain_intents = [
        "GET_ATTENDANCE",
        "GET_SUBJECT_ATTENDANCE",
        "ATTENDANCE_PROJECTION",
        "CLASSES_REQUIRED_FOR_TARGET",
        "MAX_CLASSES_CAN_MISS",
        "CHECK_ATTENDANCE_UPDATE",
        "HELP",
    ]
    for intent in domain_intents:
        assert intent in per_intent, f"Intent {intent} missing from evaluation"
        assert per_intent[intent]["accuracy"] == 100.0, f"Intent {intent} routing not 100%: {per_intent[intent]['accuracy']}%"
