"""
tests/test_generalization_experiment.py
───────────────────────────────────────
Unit tests for Phase 7.4 Targeted NLU Generalization Experiment:
1. Verification of phase7_4_train.jsonl record integrity and size (32,000 records).
2. Zero leakage audit between phase7_4_train.jsonl, val.jsonl, and test.jsonl.
3. Checkpoint metadata and configuration integrity for checkpoints/phase7_4_generalization/best.pt.
4. Evaluation pipeline compatibility with separator token ID 1098.
"""

import json
import os
from pathlib import Path
import pytest
import torch

from attendance_tracker.slm.data.validate_dataset import NLUDataValidator
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BPETokenizer


def test_phase7_4_dataset_size_and_balance() -> None:
    """Verify phase7_4_train.jsonl contains 32,000 balanced records (4,000 per intent)."""
    dataset_path = Path("data/nlu/phase7_4_train.jsonl")
    assert dataset_path.exists(), "phase7_4_train.jsonl must exist"

    counts = {}
    total = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                intent = item["intent"]
                counts[intent] = counts.get(intent, 0) + 1
                total += 1

    assert total == 32000, f"Expected 32,000 records, got {total}"
    assert len(counts) == 8, "Expected 8 canonical intents"
    for intent, count in counts.items():
        assert count == 4000, f"Intent {intent} expected 4,000 records, got {count}"


def test_phase7_4_zero_leakage_audit() -> None:
    """Verify 0 cross-split text, normalized, or template leakage in Phase 7.4 dataset."""
    train_path = Path("data/nlu/phase7_4_train.jsonl")
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
    assert report.exact_leakage_count == 0, "Exact leakage must be 0"
    assert report.normalized_leakage_count == 0, "Normalized leakage must be 0"
    assert report.template_family_leakage_count == 0, "Template leakage must be 0"
    assert report.invalid_record_count == 0, "Invalid records must be 0"


def test_phase7_4_checkpoint_metadata() -> None:
    """Verify Phase 7.4 best.pt checkpoint loads valid configuration and weights."""
    ckpt_path = Path("checkpoints/phase7_4_generalization/best.pt")
    assert ckpt_path.exists(), "Phase 7.4 best.pt must exist"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "epoch" in ckpt
    assert "val_loss" in ckpt
    assert "model_state_dict" in ckpt
    assert ckpt["epoch"] == 58
    assert ckpt["val_loss"] < 0.10

    cfg_dict = ckpt["config"]
    assert cfg_dict["vocab_size"] == 4096
    assert cfg_dict["context_length"] == 256
    assert cfg_dict["d_model"] == 256
    assert cfg_dict["n_layers"] == 6
    assert cfg_dict["n_heads"] == 4
    assert cfg_dict["d_ff"] == 1024


def test_phase7_4_model_loading_and_inference() -> None:
    """Verify AttendanceSLM instantiates and loads Phase 7.4 weights cleanly."""
    ckpt_path = Path("checkpoints/phase7_4_generalization/best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = ckpt["config"]
    config = ModelConfig(
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        pos_encoding=model_cfg["pos_encoding"],
        activation=model_cfg["activation"],
        layer_norm=model_cfg["layer_norm"],
        weight_tying=model_cfg["weight_tying"],
        initializer_range=model_cfg["initializer_range"],
    )
    model = AttendanceSLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randint(0, 4096, (2, 32), dtype=torch.long)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (2, 32, 4096)
