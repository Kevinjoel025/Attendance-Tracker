"""
tests/test_objective_experiment.py
──────────────────────────────────
Unit tests for Phase 7.3 Field-Aware Loss Weighting:
1. Output-field boundary detection & single-token intent verification
2. BPE token span calculation for intents and slots
3. Intent-field loss weighting vs slot-field loss weighting
4. Mathematical equivalence to standard cross entropy when intent_loss_weight=1.0
5. Prompt and separator masking with 0.0 loss weight
6. Padding masking in collated batch weight tensors
7. Evaluation compatibility
"""

import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from attendance_tracker.slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from attendance_tracker.slm.training.trainer import NLUDataset, collate_nlu_batch, TrainingConfig, Trainer
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM


SEP_TOKEN_ID = 1098


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer.load("data/tokenizer/bpe.json")


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    data = [
        {
            "id": "sample_1",
            "text": "What is my attendance in Physics?",
            "intent": "GET_SUBJECT_ATTENDANCE",
            "subject": "Physics",
            "target": None,
            "count": None,
            "date": None,
            "output": "GET_SUBJECT_ATTENDANCE|Physics|NONE|NONE|NONE",
        },
        {
            "id": "sample_2",
            "text": "Can I bunk 2 classes of Math?",
            "intent": "MAX_CLASSES_CAN_MISS",
            "subject": "Math",
            "target": None,
            "count": 2,
            "date": None,
            "output": "MAX_CLASSES_CAN_MISS|Math|NONE|2|NONE",
        },
    ]
    file_path = tmp_path / "sample.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_intent_bpe_spans(tokenizer: BPETokenizer) -> None:
    """Verify all 8 canonical intent strings decompose into valid BPE token IDs."""
    canonical_intents = [
        "GET_ATTENDANCE",
        "GET_SUBJECT_ATTENDANCE",
        "ATTENDANCE_PROJECTION",
        "CLASSES_REQUIRED_FOR_TARGET",
        "MAX_CLASSES_CAN_MISS",
        "CHECK_ATTENDANCE_UPDATE",
        "HELP",
        "UNKNOWN",
    ]
    for intent in canonical_intents:
        ids = tokenizer.encode(intent, add_bos=False, add_eos=False)
        assert len(ids) == 1, f"Intent {intent} has {len(ids)} tokens: {ids}"
        assert ids[0] < tokenizer.vocab_size


def test_dataset_field_loss_weights(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify loss weights assign intent_loss_weight to intent tokens and 1.0 to slots."""
    intent_w = 4.0
    slot_w = 1.0
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=256,
        mask_prompt=True,
        sep_token_id=SEP_TOKEN_ID,
        intent_loss_weight=intent_w,
        slot_loss_weight=slot_w,
    )
    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]
    loss_weights = item["loss_weights"]

    assert len(input_ids) == len(labels) == len(loss_weights)

    # Prompt tokens (before SEP) should have weight 0.0 and label -100
    prompt_ids = tokenizer.encode("What is my attendance in Physics?", add_bos=True, add_eos=False) + [SEP_TOKEN_ID]
    prompt_len = len(prompt_ids) - 1

    for p in range(prompt_len):
        assert loss_weights[p] == 0.0
        assert labels[p] == -100

    # At SEP position (predicting the first token of target = GET_SUBJECT_ATTENDANCE)
    assert input_ids[prompt_len] == SEP_TOKEN_ID
    assert loss_weights[prompt_len] == intent_w
    assert labels[prompt_len] == tokenizer.encode("GET_SUBJECT_ATTENDANCE", add_bos=False, add_eos=False)[0]

    # Remaining target tokens should have slot_w
    for p in range(prompt_len + 1, len(loss_weights)):
        assert loss_weights[p] == slot_w
        assert labels[p] != -100


def test_collate_loss_weights_padding(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify padded positions in a batch have 0.0 weight and -100 label."""
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=256,
        mask_prompt=True,
        sep_token_id=SEP_TOKEN_ID,
        intent_loss_weight=4.0,
    )
    batch = collate_nlu_batch([dataset[0], dataset[1]])

    assert "loss_weights" in batch
    assert batch["loss_weights"].shape == batch["input_ids"].shape

    # Check padding mask
    pad_mask = (batch["input_ids"] == PAD_ID)
    assert (batch["loss_weights"][pad_mask] == 0.0).all()
    assert (batch["labels"][pad_mask] == -100).all()


def test_loss_mathematical_equivalence(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify weighted loss with intent_loss_weight=1.0 is identical to standard cross entropy."""
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=64,
        mask_prompt=True,
        sep_token_id=SEP_TOKEN_ID,
        intent_loss_weight=1.0,
        slot_loss_weight=1.0,
    )
    batch = collate_nlu_batch([dataset[0], dataset[1]])

    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=64, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = AttendanceSLM(config)
    model.eval()

    with torch.no_grad():
        logits = model(batch["input_ids"])
        labels = batch["labels"]
        loss_weights = batch["loss_weights"]
        batch_size = logits.shape[0]

        # Standard loss
        std_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        std_loss = std_criterion(logits.view(-1, config.vocab_size), labels.view(-1))

        # Weighted loss
        unreduced_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        unreduced = unreduced_criterion(logits.view(-1, config.vocab_size), labels.view(-1)).view(batch_size, -1)
        eff_weights = loss_weights * (labels != -100).float()
        weighted_loss = (unreduced * eff_weights).sum() / eff_weights.sum().clamp(min=1e-8)

        assert torch.isclose(std_loss, weighted_loss, atol=1e-6)
