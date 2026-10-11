"""
slm/model/config.py
───────────────────
Configuration dataclass for the custom Small Language Model (SLM).

Defines architectural hyperparameters for the 6-layer decoder-only
Transformer designed specifically for attendance NLU intent & slot extraction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ModelConfig:
    """
    Architectural configuration for the College Attendance Tracker SLM.

    Default parameters specify a ~5.8M parameter Pre-LN decoder-only Transformer.
    """

    vocab_size: int = 4096
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    pos_encoding: str = "learned"
    activation: str = "gelu"
    layer_norm: str = "pre"
    weight_tying: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        """Dimension of each individual attention head."""
        return self.d_model // self.n_heads

    def validate(self) -> None:
        """Validate all architectural constraints."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.activation != "gelu":
            raise ValueError(
                f"Unsupported activation: {self.activation}. Only 'gelu' is supported."
            )
        if self.layer_norm != "pre":
            raise ValueError(
                f"Unsupported layer_norm: {self.layer_norm}. Only 'pre' is supported."
            )
        if self.pos_encoding != "learned":
            raise ValueError(
                f"Unsupported pos_encoding: {self.pos_encoding}. Only 'learned' is supported."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Construct ModelConfig from dictionary, ignoring unknown keys."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ModelConfig:
        """Deserialize configuration from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
