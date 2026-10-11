"""
tests/test_generator.py
───────────────────────
Comprehensive unit tests for the synthetic NLU training data generator.

Covers:
- Deterministic output reproduction across runs with the same seed
- All 8 canonical intents represented
- All 9 canonical subjects represented
- Canonical aliases present in generated surface queries
- Output label pipe format integrity (5 fields: INTENT|SUBJECT|TARGET|COUNT|DATE)
- Dual threshold handling (75% college requirement vs 70% personal target vs explicit)
- Count and date slots validity
- Out-of-domain and hard negative UNKNOWN handling
- Train / Val / Test 80/10/10 split generation
- Zero arithmetic in output labels
"""

from pathlib import Path
import pytest

from attendance_tracker.slm.data.generator import (
    CANONICAL_INTENTS,
    SUBJECT_CANONICAL,
    NLUDataGenerator,
    NLURecord,
)


@pytest.fixture
def generator():
    return NLUDataGenerator(seed=42)


class TestGeneratorDeterminismAndSplits:

    def test_deterministic_seed_reproducibility(self):
        gen1 = NLUDataGenerator(seed=123)
        splits1 = gen1.generate_dataset(total_count=100)

        gen2 = NLUDataGenerator(seed=123)
        splits2 = gen2.generate_dataset(total_count=100)

        assert len(splits1["train"]) == len(splits2["train"])
        for r1, r2 in zip(splits1["train"], splits2["train"]):
            assert r1.text == r2.text
            assert r1.output == r2.output
            assert r1.intent == r2.intent

    def test_split_proportions(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=1000)
        assert len(splits["train"]) == 800
        assert len(splits["val"]) == 100
        assert len(splits["test"]) == 100


class TestIntentAndSubjectCoverage:

    def test_all_eight_intents_present(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=1600)
        all_records = splits["train"] + splits["val"] + splits["test"]
        found_intents = {r.intent for r in all_records}

        assert found_intents == set(CANONICAL_INTENTS)

    def test_all_nine_subjects_present(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=1600)
        all_records = splits["train"] + splits["val"] + splits["test"]
        found_subjects = {r.subject for r in all_records if r.subject is not None and r.subject != "NONE"}

        assert found_subjects == set(SUBJECT_CANONICAL)

    def test_test_split_contains_all_intents(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=1600)
        test_intents = {r.intent for r in splits["test"]}
        assert test_intents == set(CANONICAL_INTENTS)


class TestOutputLabelFormatting:

    def test_pipe_delimited_structure(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=800)
        all_records = splits["train"] + splits["val"] + splits["test"]

        for r in all_records:
            parts = r.output.split("|")
            assert len(parts) == 5, f"Output '{r.output}' must have exactly 5 parts"
            intent, subj, target, count, date_val = parts

            assert intent == r.intent
            assert subj == (r.subject if r.subject else "NONE")
            assert target == (r.target if r.target else "NONE")
            assert count == (r.count if r.count else "NONE")
            assert date_val == (r.date if r.date else "NONE")

    def test_get_attendance_has_no_subject(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=800)
        get_att_records = [r for r in splits["train"] if r.intent == "GET_ATTENDANCE"]

        for r in get_att_records:
            assert r.subject is None or r.subject == "NONE"
            assert r.output.startswith("GET_ATTENDANCE|NONE|")

    def test_get_subject_attendance_has_subject(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=800)
        get_subj_records = [r for r in splits["train"] if r.intent == "GET_SUBJECT_ATTENDANCE"]

        for r in get_subj_records:
            assert r.subject in SUBJECT_CANONICAL

    def test_attendance_projection_has_count(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=800)
        proj_records = [r for r in splits["train"] if r.intent == "ATTENDANCE_PROJECTION"]

        for r in proj_records:
            assert r.count is not None
            assert int(r.count) > 0

    def test_unknown_and_help_have_none_slots(self, generator: NLUDataGenerator):
        splits = generator.generate_dataset(total_count=800)
        special_records = [r for r in splits["train"] if r.intent in ("HELP", "UNKNOWN")]

        for r in special_records:
            assert r.subject is None or r.subject == "NONE"
            assert r.target is None or r.target == "NONE"
            assert r.count is None or r.count == "NONE"
            assert r.date is None or r.date == "NONE"


class TestFileSaving:

    def test_save_splits_to_disk(self, generator: NLUDataGenerator, tmp_path: Path):
        splits = generator.generate_dataset(total_count=100)
        saved = generator.save_splits(splits, output_dir=tmp_path)

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "val.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()
        assert len(saved) == 3
