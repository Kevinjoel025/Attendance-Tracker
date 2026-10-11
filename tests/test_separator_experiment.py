"""
tests/test_separator_experiment.py
──────────────────────────────────
Unit tests for the Phase 7.2 separator-format experiment:
- SEP token ID 1098 validity in BPE vocabulary
- Dataset tokenization and sequence formatting with separator
- Prompt loss masking with separator boundary
- Generation pipeline with separator
- Backward compatibility when sep_token_id=None
"""

import json
import pytest
import torch
from pathlib import Path

from attendance_tracker.slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from attendance_tracker.slm.training.trainer import NLUDataset, collate_nlu_batch, TrainingConfig
from attendance_tracker.slm.training.evaluate import generate_completion, parse_pipe_output
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM


SEP_TOKEN_ID = 1098  # <unused_0> in bpe.json


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
            "slots": {"subject": "Physics"},
            "output": "GET_SUBJECT_ATTENDANCE|Physics|NONE|NONE|NONE",
        },
        {
            "id": "sample_2",
            "text": "Can I bunk 2 classes of Math?",
            "intent": "MAX_CLASSES_CAN_MISS",
            "slots": {"subject": "Math", "count": 2},
            "output": "MAX_CLASSES_CAN_MISS|Math|NONE|2|NONE",
        },
    ]
    file_path = tmp_path / "sample.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_sep_token_id_validity(tokenizer: BPETokenizer) -> None:
    """Verify SEP token ID 1098 exists in vocab and corresponds to <unused_0>."""
    assert SEP_TOKEN_ID < tokenizer.vocab_size
    assert tokenizer.id_to_token[SEP_TOKEN_ID] == "<unused_0>"
    assert tokenizer.token_to_id["<unused_0>"] == SEP_TOKEN_ID


def test_dataset_sequence_with_separator(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify [BOS] + Query + [SEP] + Output + [EOS] structure in NLUDataset."""
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=256,
        mask_prompt=True,
        sep_token_id=SEP_TOKEN_ID,
    )
    assert len(dataset) == 2
    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]

    # Decode prompt and target to check positions
    query = "What is my attendance in Physics?"
    output = "GET_SUBJECT_ATTENDANCE|Physics|NONE|NONE|NONE"

    query_tokens = tokenizer.encode(query, add_bos=True, add_eos=False)
    # query_tokens has [BOS] at index 0
    assert query_tokens[0] == BOS_ID

    prompt_tokens = query_tokens + [SEP_TOKEN_ID]
    target_tokens = tokenizer.encode(output, add_bos=False, add_eos=True)
    full_tokens = prompt_tokens + target_tokens

    # Verify input_ids is full[:-1]
    assert input_ids == full_tokens[:-1]

    # Verify SEP is at input_ids[len(prompt_tokens) - 1]
    sep_pos = len(prompt_tokens) - 1
    assert input_ids[sep_pos] == SEP_TOKEN_ID

    # Verify target label at SEP position is the first token of output
    assert labels[sep_pos] == target_tokens[0]

    # Verify prompt tokens (before SEP) are masked with -100
    for p in range(sep_pos):
        assert labels[p] == -100

    # Verify all target token labels after SEP are unmasked
    for p in range(sep_pos, len(labels)):
        assert labels[p] != -100
        assert labels[p] == full_tokens[p + 1]


def test_backward_compatibility_without_separator(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify that setting sep_token_id=None retains baseline behavior."""
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=256,
        mask_prompt=True,
        sep_token_id=None,
    )
    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]

    query = "What is my attendance in Physics?"
    output = "GET_SUBJECT_ATTENDANCE|Physics|NONE|NONE|NONE"

    query_tokens = tokenizer.encode(query, add_bos=True, add_eos=False)
    target_tokens = tokenizer.encode(output, add_bos=False, add_eos=True)
    full_tokens = query_tokens + target_tokens

    assert input_ids == full_tokens[:-1]
    prompt_len = len(query_tokens) - 1
    for p in range(prompt_len):
        assert labels[p] == -100
    assert labels[prompt_len] == target_tokens[0]


def test_collate_nlu_batch_with_separator(tokenizer: BPETokenizer, sample_jsonl: Path) -> None:
    """Verify collation pads properly with separator-enabled dataset."""
    dataset = NLUDataset(
        jsonl_path=sample_jsonl,
        tokenizer=tokenizer,
        context_length=256,
        mask_prompt=True,
        sep_token_id=SEP_TOKEN_ID,
    )
    batch = collate_nlu_batch([dataset[0], dataset[1]])

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape


def test_generate_completion_with_separator(tokenizer: BPETokenizer) -> None:
    """Verify generate_completion handles sep_token_id properly without crashing."""
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
    )
    model = AttendanceSLM(config)
    model.eval()

    result = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt_text="What is my attendance in Math?",
        max_new_tokens=10,
        device="cpu",
        sep_token_id=SEP_TOKEN_ID,
    )
    assert isinstance(result, str)
