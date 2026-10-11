"""
tests/test_inference.py
───────────────────────
Unit and integration tests for NLUInferenceEngine (slm/inference/nlu.py).
"""

import pytest

from attendance_tracker.slm.inference.nlu import (
    NLUInferenceEngine,
    NLUResult,
    get_nlu_engine,
)


@pytest.fixture(scope="module")
def nlu_engine() -> NLUInferenceEngine:
    return get_nlu_engine()


def test_nlu_engine_initialization(nlu_engine: NLUInferenceEngine) -> None:
    assert nlu_engine.model is not None
    assert nlu_engine.tokenizer is not None
    assert nlu_engine.sep_token_id == 1098


def test_predict_overall_attendance(nlu_engine: NLUInferenceEngine) -> None:
    query = "What is my overall attendance percentage?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "GET_ATTENDANCE"
    assert res.subject is None
    assert res.is_valid is True
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_intents) == 8
    assert res.top_intents[0][0] == "GET_ATTENDANCE"
    assert res.inference_time_ms > 0.0


def test_predict_subject_attendance(nlu_engine: NLUInferenceEngine) -> None:
    query = "How many classes have I attended in OS?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "GET_SUBJECT_ATTENDANCE"
    assert res.subject == "OS"
    assert res.is_valid is True
    assert 0.0 <= res.confidence <= 1.0


def test_predict_projection(nlu_engine: NLUInferenceEngine) -> None:
    query = "What happens if I attend next 3 classes of IoT?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "ATTENDANCE_PROJECTION"
    assert res.subject == "IoTAP"
    assert res.count == 3
    assert res.is_valid is True


def test_predict_classes_required(nlu_engine: NLUInferenceEngine) -> None:
    query = "How many classes do I need to attend to reach 75% in DE?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert res.subject == "DE"
    assert res.target == 75.0
    assert res.is_valid is True


def test_predict_max_can_miss(nlu_engine: NLUInferenceEngine) -> None:
    query = "How many classes can I miss in ACD?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "MAX_CLASSES_CAN_MISS"
    assert res.subject == "ACD"
    assert res.is_valid is True


def test_predict_check_update(nlu_engine: NLUInferenceEngine) -> None:
    query = "Has my attendance been updated today?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "CHECK_ATTENDANCE_UPDATE"
    assert res.is_valid is True


def test_predict_help(nlu_engine: NLUInferenceEngine) -> None:
    query = "What can you do?"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "HELP"
    assert res.is_valid is True


def test_predict_unknown(nlu_engine: NLUInferenceEngine) -> None:
    query = "Tell me a joke"
    res = nlu_engine.predict(query)

    assert isinstance(res, NLUResult)
    assert res.intent == "UNKNOWN"
    assert res.is_valid is True


def test_predict_batch(nlu_engine: NLUInferenceEngine) -> None:
    queries = [
        "What is my overall attendance?",
        "What is my attendance in OS?",
        "Help me",
    ]
    results = nlu_engine.predict_batch(queries)
    assert len(results) == 3
    assert all(isinstance(r, NLUResult) for r in results)
    assert results[0].intent == "GET_ATTENDANCE"
    assert results[1].intent == "GET_SUBJECT_ATTENDANCE"
    assert results[2].intent == "HELP"


def test_nlu_result_to_dict(nlu_engine: NLUInferenceEngine) -> None:
    res = nlu_engine.predict("What is my attendance in OS?")
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "intent" in d
    assert "subject" in d
    assert "confidence" in d
    assert "top_intents" in d
    assert "inference_time_ms" in d
