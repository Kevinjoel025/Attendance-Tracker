"""
tests/test_checkpoint.py
─────────────────────────
Unit tests for SLM checkpointing and state persistence.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)


class TestCheckpointing:

    def test_save_and_load_checkpoint(self, tmp_path: Path):
        config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128, context_length=64)
        model1 = AttendanceSLM(config)
        optimizer1 = AdamW(model1.parameters(), lr=1e-3)

        ckpt_file = tmp_path / "test_ckpt.pt"
        saved_path = save_checkpoint(
            filepath=ckpt_file,
            model=model1,
            optimizer=optimizer1,
            epoch=5,
            global_step=120,
            val_loss=1.234,
            train_loss=1.567,
            config=config,
            metrics={"test_metric": 42},
        )
        assert saved_path.exists()

        # Load into fresh model and optimizer
        model2 = AttendanceSLM(config)
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)

        metadata = load_checkpoint(
            filepath=ckpt_file,
            model=model2,
            optimizer=optimizer2,
            device="cpu",
        )

        assert metadata["epoch"] == 5
        assert metadata["global_step"] == 120
        assert metadata["val_loss"] == 1.234
        assert metadata["train_loss"] == 1.567
        assert metadata["metrics"]["test_metric"] == 42

        # Verify weights match
        model1.eval()
        model2.eval()
        test_input = torch.randint(0, 100, (2, 8), dtype=torch.long)
        with torch.no_grad():
            out1 = model1(test_input)
            out2 = model2(test_input)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_checkpoint_manager_best_and_latest(self, tmp_path: Path):
        config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128, context_length=64)
        model = AttendanceSLM(config)
        optimizer = AdamW(model.parameters(), lr=1e-3)

        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Epoch 1: val_loss = 2.0 (New best)
        is_best = manager.step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=1,
            global_step=10,
            val_loss=2.0,
            train_loss=2.5,
            config=config,
        )
        assert is_best is True
        assert manager.best_checkpoint_path.exists()
        assert manager.latest_checkpoint_path.exists()
        assert manager.best_val_loss == 2.0

        # Epoch 2: val_loss = 2.5 (Worse -> not best)
        is_best = manager.step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=2,
            global_step=20,
            val_loss=2.5,
            train_loss=2.2,
            config=config,
        )
        assert is_best is False
        assert manager.best_val_loss == 2.0  # Unchanged

        # Epoch 3: val_loss = 1.5 (New best)
        is_best = manager.step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=3,
            global_step=30,
            val_loss=1.5,
            train_loss=1.8,
            config=config,
        )
        assert is_best is True
        assert manager.best_val_loss == 1.5

        # Final save
        final_path = manager.save_final(
            model=model,
            optimizer=optimizer,
            epoch=3,
            global_step=30,
            val_loss=1.5,
            train_loss=1.8,
            config=config,
        )
        assert final_path.exists()
