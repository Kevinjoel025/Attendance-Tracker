"""
slm/training package
────────────────────
Training loop, evaluation, and checkpointing for AttendanceSLM.
"""

from attendance_tracker.slm.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from attendance_tracker.slm.training.evaluate import (
    EvaluationReport,
    evaluate_dataset,
    generate_completion,
)
from attendance_tracker.slm.training.trainer import Trainer, TrainingConfig

__all__ = [
    "Trainer",
    "TrainingConfig",
    "EvaluationReport",
    "evaluate_dataset",
    "generate_completion",
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
]
