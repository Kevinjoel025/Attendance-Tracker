"""
scripts/train_phase7_2_separator.py
───────────────────────────────────
Launcher for Phase 7.2 Controlled Training-Format Experiment.
Trains AttendanceSLM with explicit SEP token ID 1098 on GPU for 100 epochs.
Saves all artifacts to checkpoints/phase7_2_separator/ to preserve baseline checkpoints.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attendance_tracker.slm.training.trainer import Trainer, TrainingConfig


def main() -> None:
    config = TrainingConfig(
        train_path="data/nlu/train.jsonl",
        val_path="data/nlu/val.jsonl",
        test_path="data/nlu/test.jsonl",
        tokenizer_path="data/tokenizer/bpe.json",
        checkpoint_dir="checkpoints/phase7_2_separator",
        epochs=100,
        batch_size=128,
        learning_rate=3e-4,
        min_learning_rate=1e-5,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_steps=500,
        mixed_precision=True,
        seed=42,
        mask_prompt=True,
        log_interval=50,
        sep_token_id=1098,
    )
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
