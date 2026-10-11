"""
tests/test_training.py
───────────────────────
Unit tests for SLM Dataset, DataLoader collation, LR scheduler, and Trainer smoke runs.
"""

from pathlib import Path
import json
import pytest
import torch
import torch.nn as nn

from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BPETokenizer
from attendance_tracker.slm.training.trainer import (
    NLUDataset,
    Trainer,
    TrainingConfig,
    collate_nlu_batch,
    get_cosine_lr,
)


class TestNLUDatasetAndCollation:

    def test_dataset_loading_and_item_shapes(self, tmp_path: Path):
        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("BPE tokenizer artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)

        mock_data = [
            {"text": "what is my DE attendance", "intent": "GET_SUBJECT_ATTENDANCE", "output": "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE"},
            {"text": "help me with commands", "intent": "HELP", "output": "HELP|NONE|NONE|NONE|NONE"},
        ]
        jsonl_file = tmp_path / "train_mock.jsonl"
        with jsonl_file.open("w", encoding="utf-8") as f:
            for d in mock_data:
                f.write(json.dumps(d) + "\n")

        dataset = NLUDataset(jsonl_file, tokenizer=tokenizer, context_length=256, mask_prompt=True)
        assert len(dataset) == 2

        item0 = dataset[0]
        assert "input_ids" in item0
        assert "labels" in item0
        assert len(item0["input_ids"]) == len(item0["labels"])

        # Prompt positions must be masked with -100
        labels = item0["labels"]
        assert -100 in labels
        assert any(l != -100 for l in labels)

    def test_collate_dynamic_padding(self):
        batch = [
            {"input_ids": [2, 10, 20], "labels": [-100, -100, 30], "num_tokens": 4},
            {"input_ids": [2, 10, 20, 30, 40], "labels": [-100, -100, 30, 40, 50], "num_tokens": 6},
        ]
        collated = collate_nlu_batch(batch)

        assert collated["input_ids"].shape == (2, 5)
        assert collated["labels"].shape == (2, 5)
        # Check padding
        assert collated["input_ids"][0, 3].item() == 0  # PAD_ID
        assert collated["input_ids"][0, 4].item() == 0
        assert collated["labels"][0, 3].item() == -100
        assert collated["labels"][0, 4].item() == -100


class TestLRScheduler:

    def test_warmup_linear_increase(self):
        base_lr = 3e-4
        min_lr = 1e-5
        warmup_steps = 100
        total_steps = 1000

        lr_step_0 = get_cosine_lr(0, total_steps, warmup_steps, base_lr, min_lr)
        lr_step_50 = get_cosine_lr(50, total_steps, warmup_steps, base_lr, min_lr)
        lr_step_100 = get_cosine_lr(100, total_steps, warmup_steps, base_lr, min_lr)

        assert lr_step_0 < lr_step_50 < lr_step_100
        assert pytest.approx(lr_step_100, rel=1e-2) == base_lr

    def test_cosine_decay_and_min_lr(self):
        base_lr = 3e-4
        min_lr = 1e-5
        warmup_steps = 100
        total_steps = 1000

        lr_step_500 = get_cosine_lr(500, total_steps, warmup_steps, base_lr, min_lr)
        lr_step_1000 = get_cosine_lr(1000, total_steps, warmup_steps, base_lr, min_lr)
        lr_step_1500 = get_cosine_lr(1500, total_steps, warmup_steps, base_lr, min_lr)

        assert lr_step_500 < base_lr
        assert pytest.approx(lr_step_1000, rel=1e-3) == min_lr
        assert pytest.approx(lr_step_1500, rel=1e-3) == min_lr


class TestTrainerSmokeRuns:

    def test_trainer_smoke_cpu(self, tmp_path: Path):
        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("BPE tokenizer artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)

        # Create tiny datasets
        mock_data = [
            {"text": "what is my attendance", "intent": "GET_ATTENDANCE", "output": "GET_ATTENDANCE|NONE|NONE|NONE|NONE"},
            {"text": "what is my DE attendance", "intent": "GET_SUBJECT_ATTENDANCE", "output": "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE"},
            {"text": "how many classes to reach 75% in ACD", "intent": "CLASSES_REQUIRED_FOR_TARGET", "output": "CLASSES_REQUIRED_FOR_TARGET|ACD|75|NONE|NONE"},
            {"text": "how many classes can I miss in OS", "intent": "MAX_CLASSES_CAN_MISS", "output": "MAX_CLASSES_CAN_MISS|OS|80|NONE|NONE"},
        ]
        train_file = tmp_path / "smoke_train.jsonl"
        val_file = tmp_path / "smoke_val.jsonl"
        test_file = tmp_path / "smoke_test.jsonl"

        for fpath in (train_file, val_file, test_file):
            with fpath.open("w", encoding="utf-8") as f:
                for d in mock_data:
                    f.write(json.dumps(d) + "\n")

        # Tiny model config for fast CPU smoke test
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            context_length=64,
        )
        model = AttendanceSLM(model_config)

        train_config = TrainingConfig(
            train_path=str(train_file),
            val_path=str(val_file),
            test_path=str(test_file),
            tokenizer_path=str(bpe_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            epochs=2,
            batch_size=2,
            device="cpu",
            mixed_precision=False,
            log_interval=1,
            warmup_steps=2,
        )

        trainer = Trainer(config=train_config, model=model, tokenizer=tokenizer)
        results = trainer.train()

        assert "total_training_time_sec" in results
        assert Path(results["best_checkpoint"]).exists()
        assert Path(results["final_checkpoint"]).exists()
        assert len(results["history"]["epoch"]) == 2

    def test_trainer_smoke_cuda_if_available(self, tmp_path: Path):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("BPE tokenizer artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)

        mock_data = [
            {"text": "what is my attendance", "intent": "GET_ATTENDANCE", "output": "GET_ATTENDANCE|NONE|NONE|NONE|NONE"},
            {"text": "what is my DE attendance", "intent": "GET_SUBJECT_ATTENDANCE", "output": "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE"},
        ]
        train_file = tmp_path / "cuda_train.jsonl"
        val_file = tmp_path / "cuda_val.jsonl"
        test_file = tmp_path / "cuda_test.jsonl"

        for fpath in (train_file, val_file, test_file):
            with fpath.open("w", encoding="utf-8") as f:
                for d in mock_data:
                    f.write(json.dumps(d) + "\n")

        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            context_length=64,
        )
        model = AttendanceSLM(model_config)

        train_config = TrainingConfig(
            train_path=str(train_file),
            val_path=str(val_file),
            test_path=str(test_file),
            tokenizer_path=str(bpe_path),
            checkpoint_dir=str(tmp_path / "checkpoints_cuda"),
            epochs=2,
            batch_size=2,
            device="cuda",
            mixed_precision=True,
            log_interval=1,
            warmup_steps=2,
        )

        trainer = Trainer(config=train_config, model=model, tokenizer=tokenizer)
        results = trainer.train()
        assert Path(results["best_checkpoint"]).exists()
        assert trainer.use_amp is True

    def test_trainer_resume(self, tmp_path: Path):
        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("BPE tokenizer artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)
        mock_data = [
            {"text": "what is my attendance", "intent": "GET_ATTENDANCE", "output": "GET_ATTENDANCE|NONE|NONE|NONE|NONE"},
            {"text": "help", "intent": "HELP", "output": "HELP|NONE|NONE|NONE|NONE"},
        ]
        train_file = tmp_path / "res_train.jsonl"
        val_file = tmp_path / "res_val.jsonl"
        test_file = tmp_path / "res_test.jsonl"

        for fpath in (train_file, val_file, test_file):
            with fpath.open("w", encoding="utf-8") as f:
                for d in mock_data:
                    f.write(json.dumps(d) + "\n")

        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            context_length=64,
        )
        model = AttendanceSLM(model_config)

        train_config = TrainingConfig(
            train_path=str(train_file),
            val_path=str(val_file),
            test_path=str(test_file),
            tokenizer_path=str(bpe_path),
            checkpoint_dir=str(tmp_path / "checkpoints_resume"),
            epochs=2,
            batch_size=2,
            device="cpu",
            mixed_precision=False,
            log_interval=1,
            warmup_steps=2,
        )

        trainer1 = Trainer(config=train_config, model=model, tokenizer=tokenizer)
        trainer1.train()

        # Resume into a second trainer instance
        model2 = AttendanceSLM(model_config)
        trainer2 = Trainer(config=train_config, model=model2, tokenizer=tokenizer)
        trainer2.resume_from(trainer1.checkpoint_manager.best_checkpoint_path)

        assert trainer2.current_epoch == 2
        assert trainer2.global_step > 0
