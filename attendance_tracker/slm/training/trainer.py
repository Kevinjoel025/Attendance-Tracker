"""
slm/training/trainer.py
───────────────────────
Complete from-scratch training pipeline for AttendanceSLM.

Features:
- Deterministic PyTorch Dataset/DataLoader pipeline
- Causal prompt-completion next-token prediction with prompt loss masking
- AdamW optimizer with cosine learning rate decay and linear warmup
- Automatic Mixed Precision (AMP FP16) on CUDA with GradScaler
- Dynamic per-batch padding with ignore_index=-100
- Throughput tracking (examples/sec, tokens/sec, latency/batch)
- Checkpointing (best.pt, latest.pt, final.pt) and resume capability
- Post-training test set evaluation
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from attendance_tracker.slm.training.checkpoint import CheckpointManager, load_checkpoint
from attendance_tracker.slm.training.evaluate import EvaluationReport, evaluate_dataset

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Hyperparameters and filepaths for SLM training."""

    train_path: str = "data/nlu/train.jsonl"
    val_path: str = "data/nlu/val.jsonl"
    test_path: str = "data/nlu/test.jsonl"
    tokenizer_path: str = "data/tokenizer/bpe.json"
    checkpoint_dir: str = "checkpoints"

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500

    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    mask_prompt: bool = True
    log_interval: int = 20
    sep_token_id: Optional[int] = None
    intent_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Dataset & Collation ───────────────────────────────────────────────────────

class NLUDataset(Dataset):
    """
    PyTorch Dataset loading natural language query -> structured output pairs.
    Supports causal prompt-completion training with field-aware loss weighting.
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        tokenizer: BPETokenizer,
        context_length: int = 256,
        mask_prompt: bool = True,
        sep_token_id: Optional[int] = None,
        intent_loss_weight: float = 1.0,
        slot_loss_weight: float = 1.0,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.mask_prompt = mask_prompt
        self.sep_token_id = sep_token_id
        self.intent_loss_weight = intent_loss_weight
        self.slot_loss_weight = slot_loss_weight

        self.records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        logger.info(f"Loaded {len(self.records)} records from {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        text = item["text"]
        output = item["output"]

        # Prompt with BOS (and optional explicit SEP token)
        prompt_ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        if self.sep_token_id is not None:
            prompt_ids = prompt_ids + [self.sep_token_id]

        # Target with EOS
        target_ids = self.tokenizer.encode(output, add_bos=False, add_eos=True)

        full_ids = prompt_ids + target_ids
        if len(full_ids) > self.context_length:
            raise ValueError(
                f"Sequence length {len(full_ids)} exceeds context length {self.context_length} "
                f"for text: '{text}'"
            )

        # Autoregressive target: input = full[:-1], target = full[1:]
        input_ids = full_ids[:-1]
        target_labels = full_ids[1:]

        prompt_loss_len = len(prompt_ids) - 1
        if self.mask_prompt:
            # Mask prompt positions with -100 so loss is computed ONLY on structured output
            labels = [-100] * prompt_loss_len + target_labels[prompt_loss_len:]
        else:
            labels = target_labels

        # Determine token spans for field-aware loss weighting
        # Output format: INTENT|SUBJECT|TARGET|COUNT|DATE
        parts = output.split("|")
        intent_str = parts[0]
        intent_tokens = self.tokenizer.encode(intent_str, add_bos=False, add_eos=False)
        intent_token_len = len(intent_tokens)

        # Construct per-position loss weights
        if self.mask_prompt:
            prompt_weights = [0.0] * prompt_loss_len
        else:
            prompt_weights = [1.0] * prompt_loss_len

        intent_weights = [self.intent_loss_weight] * intent_token_len
        remaining_target_len = len(target_ids) - intent_token_len
        slot_weights = [self.slot_loss_weight] * remaining_target_len

        loss_weights = prompt_weights + intent_weights + slot_weights

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "num_tokens": len(full_ids),
        }


def collate_nlu_batch(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """
    Collate and dynamically pad a batch of sequences to the longest sequence in the batch.
    Inputs are padded with PAD_ID (0).
    Labels are padded with -100 (ignored by CrossEntropyLoss).
    Loss weights are padded with 0.0.
    """
    batch_size = len(batch)
    max_len = max(len(item["input_ids"]) for item in batch)

    input_tensor = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    label_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
    weight_tensor = torch.zeros((batch_size, max_len), dtype=torch.float)
    total_tokens = sum(item["num_tokens"] for item in batch)

    for i, item in enumerate(batch):
        in_ids = item["input_ids"]
        lbl_ids = item["labels"]
        w_ids = item.get("loss_weights", [1.0 if l != -100 else 0.0 for l in lbl_ids])
        seq_len = len(in_ids)
        input_tensor[i, :seq_len] = torch.tensor(in_ids, dtype=torch.long)
        label_tensor[i, :seq_len] = torch.tensor(lbl_ids, dtype=torch.long)
        weight_tensor[i, :seq_len] = torch.tensor(w_ids, dtype=torch.float)

    return {
        "input_ids": input_tensor,
        "labels": label_tensor,
        "loss_weights": weight_tensor,
        "total_tokens": total_tokens,
    }


# ── Learning Rate Schedule ────────────────────────────────────────────────────

def get_cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Calculate learning rate with linear warmup and cosine decay.
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (float(step + 1) / float(max(1, warmup_steps)))

    if step >= total_steps:
        return min_lr

    # Cosine decay
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_coeff * (base_lr - min_lr)


# ── Trainer Class ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages end-to-end training, validation, checkpointing, and evaluation.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[AttendanceSLM] = None,
        tokenizer: Optional[BPETokenizer] = None,
    ) -> None:
        self.config = config if config is not None else TrainingConfig()

        # 1. Set reproducible random seeds
        self._set_seed(self.config.seed)

        # 2. Select Device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Using compute device: {self.device}")

        # 3. Load or initialize Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BPETokenizer.load(self.config.tokenizer_path)

        # 4. Load or initialize Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            model_config = ModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                context_length=256,
            )
            self.model = AttendanceSLM(model_config).to(self.device)

        # 5. Datasets & DataLoaders
        self.train_dataset = NLUDataset(
            jsonl_path=self.config.train_path,
            tokenizer=self.tokenizer,
            context_length=self.model.config.context_length,
            mask_prompt=self.config.mask_prompt,
            sep_token_id=self.config.sep_token_id,
            intent_loss_weight=self.config.intent_loss_weight,
            slot_loss_weight=self.config.slot_loss_weight,
        )
        self.val_dataset = NLUDataset(
            jsonl_path=self.config.val_path,
            tokenizer=self.tokenizer,
            context_length=self.model.config.context_length,
            mask_prompt=self.config.mask_prompt,
            sep_token_id=self.config.sep_token_id,
            intent_loss_weight=self.config.intent_loss_weight,
            slot_loss_weight=self.config.slot_loss_weight,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_nlu_batch,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_nlu_batch,
        )

        # 6. Loss, Optimizer, and Checkpoint Manager
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        # 7. Mixed Precision Scaler
        self.use_amp = (self.device.type == "cuda" and self.config.mixed_precision)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Enabled Automatic Mixed Precision (AMP FP16) on CUDA.")
        else:
            self.scaler = None
            logger.info("Training with standard FP32 precision.")

        # Total step tracking
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.config.epochs
        self.global_step = 0
        self.current_epoch = 0

        # Metrics tracking
        self.history: dict[str, list[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time_sec": [],
            "throughput_samples_per_sec": [],
            "throughput_tokens_per_sec": [],
        }

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        Train the model for a single epoch.

        Returns
        -------
        avg_loss : float
        samples_per_sec : float
        tokens_per_sec : float
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        total_samples = 0
        epoch_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            loss_weights = batch["loss_weights"].to(self.device)
            batch_tokens = batch["total_tokens"]
            batch_size = input_ids.shape[0]

            # Update learning rate with schedule
            lr = get_cosine_lr(
                step=self.global_step,
                total_steps=self.total_steps,
                warmup_steps=self.config.warmup_steps,
                base_lr=self.config.learning_rate,
                min_lr=self.config.min_learning_rate,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(input_ids)
                    unreduced = self.criterion(
                        logits.view(-1, self.model.config.vocab_size), labels.view(-1)
                    ).view(batch_size, -1)
                    eff_weights = loss_weights * (labels != -100).float()
                    denom = eff_weights.sum().clamp(min=1e-8)
                    loss = (unreduced * eff_weights).sum() / denom

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids)
                unreduced = self.criterion(
                    logits.view(-1, self.model.config.vocab_size), labels.view(-1)
                ).view(batch_size, -1)
                eff_weights = loss_weights * (labels != -100).float()
                denom = eff_weights.sum().clamp(min=1e-8)
                loss = (unreduced * eff_weights).sum() / denom

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            total_tokens += batch_tokens
            total_samples += batch_size
            self.global_step += 1

            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                logger.info(
                    f"Epoch [{epoch+1:>3}/{self.config.epochs}] | Batch [{batch_idx+1:>3}/{len(self.train_loader)}] "
                    f"| Loss: {loss_val:.4f} | LR: {lr:.6f} | Step: {self.global_step}"
                )

        epoch_duration = time.perf_counter() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        samples_per_sec = total_samples / epoch_duration if epoch_duration > 0 else 0.0
        tokens_per_sec = total_tokens / epoch_duration if epoch_duration > 0 else 0.0

        return avg_loss, samples_per_sec, tokens_per_sec

    def evaluate_val_loss(self) -> float:
        """
        Compute average validation loss on val.jsonl.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                loss_weights = batch["loss_weights"].to(self.device)
                batch_size = input_ids.shape[0]

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(input_ids)
                        unreduced = self.criterion(
                            logits.view(-1, self.model.config.vocab_size), labels.view(-1)
                        ).view(batch_size, -1)
                        eff_weights = loss_weights * (labels != -100).float()
                        denom = eff_weights.sum().clamp(min=1e-8)
                        loss = (unreduced * eff_weights).sum() / denom
                else:
                    logits = self.model(input_ids)
                    unreduced = self.criterion(
                        logits.view(-1, self.model.config.vocab_size), labels.view(-1)
                    ).view(batch_size, -1)
                    eff_weights = loss_weights * (labels != -100).float()
                    denom = eff_weights.sum().clamp(min=1e-8)
                    loss = (unreduced * eff_weights).sum() / denom

                total_loss += loss.item()

        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

    def train(self) -> dict[str, Any]:
        """
        Run the complete training loop for the configured number of epochs.
        """
        logger.info("=" * 60)
        logger.info("STARTING SLM TRAINING RUN")
        logger.info(f"Device:           {self.device}")
        logger.info(f"Model Parameters: {self.model.get_num_params():,}")
        logger.info(f"Training Samples: {len(self.train_dataset):,}")
        logger.info(f"Validation Samples: {len(self.val_dataset):,}")
        logger.info(f"Batch Size:       {self.config.batch_size}")
        logger.info(f"Batches/Epoch:    {self.steps_per_epoch}")
        logger.info(f"Total Epochs:     {self.config.epochs}")
        logger.info(f"Total Steps:      {self.total_steps}")
        logger.info(f"Initial LR:       {self.config.learning_rate}")
        logger.info(f"Mixed Precision:  {self.use_amp}")
        logger.info("=" * 60)

        overall_start_time = time.perf_counter()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            t0 = time.perf_counter()

            train_loss, samples_per_sec, tokens_per_sec = self.train_epoch(epoch)
            val_loss = self.evaluate_val_loss()
            epoch_time = time.perf_counter() - t0

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["learning_rate"].append(round(current_lr, 8))
            self.history["epoch_time_sec"].append(round(epoch_time, 2))
            self.history["throughput_samples_per_sec"].append(round(samples_per_sec, 2))
            self.history["throughput_tokens_per_sec"].append(round(tokens_per_sec, 2))

            # Step CheckpointManager
            is_best = self.checkpoint_manager.step(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=None,
                scaler=self.scaler,
                epoch=epoch + 1,
                global_step=self.global_step,
                val_loss=val_loss,
                train_loss=train_loss,
                config=self.model.config,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
            )

            best_marker = " [NEW BEST]" if is_best else ""
            logger.info(
                f"--> Epoch {epoch+1:>3}/{self.config.epochs} Complete | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{best_marker} | "
                f"Time: {epoch_time:.2f}s | Speed: {samples_per_sec:.1f} samples/s ({tokens_per_sec:.1f} tokens/s)"
            )

        total_training_time = time.perf_counter() - overall_start_time

        # Save final checkpoint
        final_path = self.checkpoint_manager.save_final(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            scaler=self.scaler,
            epoch=self.config.epochs,
            global_step=self.global_step,
            val_loss=self.history["val_loss"][-1] if self.history["val_loss"] else float("inf"),
            train_loss=self.history["train_loss"][-1] if self.history["train_loss"] else float("inf"),
            config=self.model.config,
            metrics={"final": True},
        )

        # Save training history JSON
        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETED in {total_training_time:.2f} seconds.")
        logger.info(f"Best checkpoint:  {self.checkpoint_manager.best_checkpoint_path}")
        logger.info(f"Final checkpoint: {final_path}")
        logger.info("=" * 60)

        # Post-training evaluation on test set using the best model checkpoint
        logger.info("Evaluating best model on held-out test set (test.jsonl)...")
        best_model = AttendanceSLM(self.model.config).to(self.device)
        load_checkpoint(self.checkpoint_manager.best_checkpoint_path, best_model, device=self.device)

        test_report = evaluate_dataset(
            model=best_model,
            tokenizer=self.tokenizer,
            dataset_path=self.config.test_path,
            device=self.device,
            sep_token_id=self.config.sep_token_id,
        )

        print("\n" + test_report.summary() + "\n")

        # Save evaluation report JSON
        eval_report_path = Path(self.config.checkpoint_dir) / "evaluation_report.json"
        with eval_report_path.open("w", encoding="utf-8") as f:
            json.dump(test_report.to_dict(), f, indent=2)

        return {
            "total_training_time_sec": total_training_time,
            "best_checkpoint": str(self.checkpoint_manager.best_checkpoint_path),
            "final_checkpoint": str(final_path),
            "history": self.history,
            "evaluation": test_report.to_dict(),
        }

    def resume_from(self, checkpoint_path: Path | str) -> None:
        """
        Resume training state from a saved checkpoint.
        """
        ckpt = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.checkpoint_manager.best_val_loss = ckpt["val_loss"]
        logger.info(
            f"Resumed training state from {checkpoint_path}: epoch {self.current_epoch}, step {self.global_step}"
        )


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
