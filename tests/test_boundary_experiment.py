"""
tests/test_boundary_experiment.py
─────────────────────────────────
Unit tests for Phase 7.5 Boundary Experiment:
1. Verification of phase7_5_train.jsonl size and balance (34,400 records, 4,300 per intent).
2. Zero leakage audit between phase7_5_train.jsonl, val.jsonl, and test.jsonl.
3. Checkpoint metadata and config for checkpoints/phase7_5_boundary/best.pt.
4. Model architecture parameter count verification (exact 5,853,184 parameters).
"""

import json
from pathlib import Path
import pytest
import torch

from attendance_tracker.slm.data.validate_dataset import NLUDataValidator
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM


def test_phase7_5_dataset_size_and_balance() -> None:
    """Verify phase7_5_train.jsonl contains 34,400 balanced records (4,300 per intent)."""
    dataset_path = Path("data/nlu/phase7_5_train.jsonl")
    assert dataset_path.exists(), "phase7_5_train.jsonl must exist"

    counts = {}
    total = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                intent = item["intent"]
                counts[intent] = counts.get(intent, 0) + 1
                total += 1

    assert total == 34400, f"Expected 34,400 records, got {total}"
    assert len(counts) == 8, "Expected 8 canonical intents"
    for intent, count in counts.items():
        assert count == 4300, f"Intent {intent} expected 4,300 records, got {count}"


def test_phase7_5_zero_leakage_audit() -> None:
    """Verify 0 cross-split text, normalized, or template leakage in Phase 7.5 dataset."""
    train_path = Path("data/nlu/phase7_5_train.jsonl")
    val_path = Path("data/nlu/val.jsonl")
    test_path = Path("data/nlu/test.jsonl")

    validator = NLUDataValidator()
    
    splits = {}
    for name, p in [("train", train_path), ("val", val_path), ("test", test_path)]:
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        splits[name] = records

    report = validator.validate_splits(splits)
    assert report.is_valid, f"Validation failed with errors: {report.errors[:5]}"
    assert report.exact_leakage_count == 0
    assert report.normalized_leakage_count == 0
    assert report.template_family_leakage_count == 0
    assert report.invalid_record_count == 0


def test_model_exact_parameter_count() -> None:
    """Verify the exact parameter count of AttendanceSLM with weight tying is 5,853,184."""
    model = AttendanceSLM(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 5853184, f"Expected 5,853,184 parameters, got {total_params}"


def test_phase7_5_checkpoint_metadata() -> None:
    """Verify Phase 7.5 checkpoint metadata."""
    ckpt_path = Path("checkpoints/phase7_5_boundary/best.pt")
    assert ckpt_path.exists(), "Phase 7.5 best.pt must exist"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 21
    assert ckpt["val_loss"] < 0.05
    assert ckpt["config"]["vocab_size"] == 4096
