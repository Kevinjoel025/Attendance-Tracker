"""
slm/training/checkpoint.py
──────────────────────────
Checkpoint management for the College Attendance Tracker SLM.

Handles:
- Complete state persistence (model, optimizer, scheduler, AMP scaler, metadata)
- Tracking and saving best checkpoint based on validation loss
- Restoring training state for resumption
- Safe serialization without credentials or sensitive data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.slm.model.config import ModelConfig

logger = get_logger(__name__)


def save_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    val_loss: float = float("inf"),
    train_loss: float = float("inf"),
    config: Optional[ModelConfig] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save complete training state to disk.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "config": config.to_dict() if config is not None else None,
        "metrics": metrics or {},
    }

    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path} (epoch {epoch}, val_loss {val_loss:.4f})")
    return path


def load_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load saved checkpoint and restore states of model and training components.

    Returns
    -------
    dict containing checkpoint metadata (epoch, step, val_loss, train_loss, config, metrics).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load with weights_only=False to support optimizer and scheduler states
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 1. Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 2. Restore optimizer state if provided
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Restore scheduler state if provided
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 4. Restore AMP scaler state if provided
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 0)}, "
        f"val_loss {checkpoint.get('val_loss', 0.0):.4f})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "train_loss": checkpoint.get("train_loss", float("inf")),
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics", {}),
    }


class CheckpointManager:
    """
    Manages saving of best, latest, and final checkpoints during training.
    """

    def __init__(self, checkpoint_dir: Path | str = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_checkpoint_path = self.checkpoint_dir / "best.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest.pt"
        self.final_checkpoint_path = self.checkpoint_dir / "final.pt"

    def step(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        epoch: int,
        global_step: int,
        val_loss: float,
        train_loss: float,
        config: Optional[ModelConfig] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Record an epoch's metrics. Always saves `latest.pt`.
        If validation loss improves, also updates `best.pt`.

        Returns
        -------
        bool: True if this epoch achieved a new best validation loss.
        """
        # 1. Always update latest.pt
        save_checkpoint(
            filepath=self.latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            train_loss=train_loss,
            config=config,
            metrics=metrics,
        )

        # 2. Check if this is the best validation loss
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            save_checkpoint(
                filepath=self.best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                val_loss=val_loss,
                train_loss=train_loss,
                config=config,
                metrics=metrics,
            )
            logger.info(f"New best validation loss: {val_loss:.4f} -> saved {self.best_checkpoint_path}")

        return is_best

    def save_final(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        val_loss: float = float("inf"),
        train_loss: float = float("inf"),
        config: Optional[ModelConfig] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save the final checkpoint at the end of training."""
        return save_checkpoint(
            filepath=self.final_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            train_loss=train_loss,
            config=config,
            metrics=metrics,
        )
