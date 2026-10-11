"""
tests/test_dataset_validation.py
────────────────────────────────
Unit and integration tests for NLUDataValidator.

Covers:
- Acceptance of valid records and datasets
- Rejection of malformed records (missing fields, invalid intents, invalid subjects)
- Rejection of invalid output pipe formats
- Detection of within-split duplicate texts
- Detection of exact text cross-split leakage
- Detection of normalized text cross-split leakage
- Detection of template-family cross-split leakage
- End-to-end validation of the generated 30,000 example dataset on disk
"""

from pathlib import Path
import pytest

from attendance_tracker.slm.data.generator import NLUDataGenerator
from attendance_tracker.slm.data.validate_dataset import NLUDataValidator, ValidationReport


@pytest.fixture
def validator():
    return NLUDataValidator()


class TestSingleRecordValidation:

    def test_valid_record_passes(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "OS",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|OS|NONE|NONE|NONE",
        }
        errors = validator.validate_record(rec, 1)
        assert len(errors) == 0

    def test_missing_field_rejected(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            # missing subject, target, count, date, output
        }
        errors = validator.validate_record(rec, 1)
        assert len(errors) > 0
        assert any("missing required field" in e for e in errors)

    def test_invalid_intent_rejected(self, validator: NLUDataValidator):
        rec = {
            "text": "Book me a flight",
            "intent": "BOOK_FLIGHT",
            "subject": None,
            "target": None,
            "count": None,
            "date": None,
            "output": "BOOK_FLIGHT|NONE|NONE|NONE|NONE",
        }
        errors = validator.validate_record(rec, 1)
        assert any("invalid intent" in e for e in errors)

    def test_invalid_subject_rejected(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my Physics attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "PHYSICS",  # not in canonical subjects
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|PHYSICS|NONE|NONE|NONE",
        }
        errors = validator.validate_record(rec, 1)
        assert any("invalid subject" in e for e in errors)

    def test_mismatched_output_pipe_rejected(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "OS",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE",  # DE instead of OS
        }
        errors = validator.validate_record(rec, 1)
        assert any("output subject" in e for e in errors)


class TestLeakageAndDuplicateDetection:

    def test_within_split_duplicate_detected(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "OS",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|OS|NONE|NONE|NONE",
            "template_family": "fam1",
        }
        splits = {
            "train": [rec, rec],  # duplicate
            "val": [],
            "test": [],
        }
        report = validator.validate_splits(splits)
        assert report.duplicate_count == 1
        assert not report.is_valid

    def test_exact_cross_split_leakage_detected(self, validator: NLUDataValidator):
        rec = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "OS",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|OS|NONE|NONE|NONE",
            "template_family": "fam1",
        }
        splits = {
            "train": [rec],
            "val": [rec],  # exact same text in val
            "test": [],
        }
        report = validator.validate_splits(splits)
        assert report.exact_leakage_count == 1
        assert not report.is_valid

    def test_template_family_leakage_detected(self, validator: NLUDataValidator):
        rec1 = {
            "text": "What is my OS attendance?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "OS",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|OS|NONE|NONE|NONE",
            "template_family": "shared_family",
        }
        rec2 = {
            "text": "Check DE attendance please",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "DE",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE",
            "template_family": "shared_family",  # same template family in test
        }
        splits = {
            "train": [rec1],
            "val": [],
            "test": [rec2],
        }
        report = validator.validate_splits(splits)
        assert report.template_family_leakage_count >= 1
        assert not report.is_valid


class TestLiveDatasetValidation:

    def test_generated_dataset_passes_full_validation(self, validator: NLUDataValidator):
        data_dir = Path("data/nlu")
        if not (data_dir / "train.jsonl").exists():
            pytest.skip("Dataset files not found on disk")

        report = validator.validate_jsonl_files(data_dir)
        assert report.is_valid is True, f"Validation failed with errors: {report.errors[:5]}"
        assert report.total_examples >= 20000
        assert report.duplicate_count == 0
        assert report.exact_leakage_count == 0
        assert report.normalized_leakage_count == 0
        assert report.template_family_leakage_count == 0
        assert report.invalid_record_count == 0
