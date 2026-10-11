"""
scripts/train_phase7_3_objective.py
───────────────────────────────────
Launcher for Phase 7.3 Field-Aware Loss Weighting Experiment.
Trains AttendanceSLM with intent_loss_weight=4.0 and sep_token_id=1098 for 100 epochs on GPU.
Saves all artifacts to checkpoints/phase7_3_objective/ to preserve baseline & separator checkpoints.
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
        checkpoint_dir="checkpoints/phase7_3_objective",
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
        intent_loss_weight=4.0,
        slot_loss_weight=1.0,
    )
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
