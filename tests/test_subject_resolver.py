"""
tests/test_subject_resolver.py
──────────────────────────────
Unit tests for SubjectResolver (core/subject_resolver.py).
"""

import pytest

from attendance_tracker.attendance.repository import CANONICAL_SUBJECTS, Subject
from attendance_tracker.core.subject_resolver import SubjectResolver, get_subject_resolver


@pytest.fixture
def resolver() -> SubjectResolver:
    return get_subject_resolver()


def test_resolver_canonical_subject_count(resolver: SubjectResolver) -> None:
    subjects = resolver.get_canonical_subjects()
    assert len(subjects) == 9


def test_resolve_exact_codes(resolver: SubjectResolver) -> None:
    for item in CANONICAL_SUBJECTS:
        code = item["code"]
        resolved = resolver.resolve(code)
        assert resolved is not None
        assert resolved.code == code
        # Test lowercase
        resolved_lower = resolver.resolve(code.lower())
        assert resolved_lower is not None
        assert resolved_lower.code == code


def test_resolve_short_names(resolver: SubjectResolver) -> None:
    short_names = ["GS", "OE-1", "DE", "ACD", "IoTAP", "OS", "DE LAB", "IoTAP LAB", "WP"]
    for sn in short_names:
        resolved = resolver.resolve(sn)
        assert resolved is not None
        assert resolved.short_name == sn
        # Case insensitive
        resolved_lower = resolver.resolve(sn.lower())
        assert resolved_lower is not None
        assert resolved_lower.short_name == sn


def test_resolve_full_names(resolver: SubjectResolver) -> None:
    for item in CANONICAL_SUBJECTS:
        full = item["full_name"]
        resolved = resolver.resolve(full)
        assert resolved is not None
        assert resolved.code == item["code"]


def test_resolve_common_aliases(resolver: SubjectResolver) -> None:
    alias_expectations = {
        "operating systems": "22PC1IT202",
        "operating system": "22PC1IT202",
        "data engineering": "22PC1AM202",
        "data eng": "22PC1AM202",
        "iot": "22PC1IN301",
        "iot architecture": "22PC1IN301",
        "iot lab": "22PC2IN301",
        "de lab": "22PC2AM202",
        "automata": "22PC1AM203",
        "compiler design": "22PC1AM203",
        "web programming": "22PE1AM301",
        "web": "22PE1AM301",
        "storytelling": "22OE1EN301",
        "gender": "22MN6HS302",
        "gender sensitization": "22MN6HS302",
    }
    for alias, expected_code in alias_expectations.items():
        resolved = resolver.resolve(alias)
        assert resolved is not None, f"Failed to resolve alias '{alias}'"
        assert resolved.code == expected_code, f"Alias '{alias}' resolved to {resolved.code}, expected {expected_code}"


def test_resolve_none_and_empty(resolver: SubjectResolver) -> None:
    assert resolver.resolve(None) is None
    assert resolver.resolve("") is None
    assert resolver.resolve("   ") is None
    assert resolver.resolve("NONE") is None
    assert resolver.resolve("None") is None


def test_resolve_unrecognized(resolver: SubjectResolver) -> None:
    assert resolver.resolve("Quantum Physics") is None
    assert resolver.resolve("Rocket Propulsion 101") is None
    assert resolver.resolve("xyzabc123") is None


def test_resolve_fuzzy_matching(resolver: SubjectResolver) -> None:
    # Minor typos that should resolve via fuzzy string matching
    fuzzy_tests = {
        "operating systm": "22PC1IT202",
        "data engineerng": "22PC1AM202",
        "automata design": "22PC1AM203",
        "story teling": "22OE1EN301",
    }
    for typo, expected_code in fuzzy_tests.items():
        resolved = resolver.resolve(typo)
        assert resolved is not None, f"Fuzzy matching failed for '{typo}'"
        assert resolved.code == expected_code
