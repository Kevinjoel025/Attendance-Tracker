"""
tests/test_parser.py
────────────────────
Unit tests for structured NLU output parser (slm/inference/parser.py).
"""

import pytest

from attendance_tracker.slm.inference.parser import (
    VALID_INTENTS,
    ParsedNLUOutput,
    parse_nlu_output,
)


def test_parse_valid_overall_attendance() -> None:
    raw = "GET_ATTENDANCE|NONE|NONE|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "GET_ATTENDANCE"
    assert res.subject is None
    assert res.target is None
    assert res.count is None
    assert res.date is None
    assert res.malformed_reason is None


def test_parse_valid_subject_attendance() -> None:
    raw = "GET_SUBJECT_ATTENDANCE|OS|NONE|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "GET_SUBJECT_ATTENDANCE"
    assert res.subject == "OS"
    assert res.target is None
    assert res.count is None


def test_parse_valid_projection() -> None:
    raw = "ATTENDANCE_PROJECTION|IoTAP|NONE|3|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "ATTENDANCE_PROJECTION"
    assert res.subject == "IoTAP"
    assert res.count == 3
    assert res.target is None


def test_parse_valid_classes_required() -> None:
    raw = "CLASSES_REQUIRED_FOR_TARGET|DE|75|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "CLASSES_REQUIRED_FOR_TARGET"
    assert res.subject == "DE"
    assert res.target == 75.0
    assert res.count is None


def test_parse_valid_max_can_miss() -> None:
    raw = "MAX_CLASSES_CAN_MISS|ACD|70|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "MAX_CLASSES_CAN_MISS"
    assert res.subject == "ACD"
    assert res.target == 70.0


def test_parse_valid_check_update_with_date() -> None:
    raw = "CHECK_ATTENDANCE_UPDATE|NONE|NONE|NONE|TODAY"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.intent == "CHECK_ATTENDANCE_UPDATE"
    assert res.date == "TODAY"


def test_parse_valid_help_and_unknown() -> None:
    for intent in ["HELP", "UNKNOWN"]:
        raw = f"{intent}|NONE|NONE|NONE|NONE"
        res = parse_nlu_output(raw)
        assert res.is_valid is True
        assert res.intent == intent


def test_parse_target_with_percentage_sign() -> None:
    raw = "CLASSES_REQUIRED_FOR_TARGET|WP|80%|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is True
    assert res.target == 80.0


def test_parse_invalid_field_count_too_few() -> None:
    raw = "GET_ATTENDANCE|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is False
    assert res.intent == "GET_ATTENDANCE"
    assert "Expected 5 pipe-delimited fields" in res.malformed_reason


def test_parse_invalid_field_count_too_many() -> None:
    raw = "GET_ATTENDANCE|NONE|NONE|NONE|NONE|EXTRA"
    res = parse_nlu_output(raw)
    assert res.is_valid is False
    assert "Expected 5 pipe-delimited fields" in res.malformed_reason


def test_parse_unrecognized_intent() -> None:
    raw = "INVALID_INTENT|OS|NONE|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is False
    assert res.intent == "UNKNOWN"
    assert "Unrecognized intent" in res.malformed_reason


def test_parse_invalid_numeric_target() -> None:
    raw = "CLASSES_REQUIRED_FOR_TARGET|DE|NOT_A_NUM|NONE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is False
    assert res.target is None
    assert "Invalid numeric target" in res.malformed_reason


def test_parse_invalid_integer_count() -> None:
    raw = "ATTENDANCE_PROJECTION|OS|NONE|THREE|NONE"
    res = parse_nlu_output(raw)
    assert res.is_valid is False
    assert res.count is None
    assert "Invalid integer count" in res.malformed_reason


def test_parse_empty_string() -> None:
    res = parse_nlu_output("")
    assert res.is_valid is False
    assert res.intent == "UNKNOWN"
    assert res.malformed_reason == "Empty output string"
