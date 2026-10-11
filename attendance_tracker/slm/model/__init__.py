"""
slm/model package
─────────────────
Custom Transformer Architecture for College Attendance Tracker SLM.
"""

from attendance_tracker.slm.model.attention import CausalSelfAttention
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.model.transformer_block import FeedForward, TransformerBlock

__all__ = [
    "ModelConfig",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "AttendanceSLM",
]
