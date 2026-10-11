"""
slm/model/transformer_block.py
──────────────────────────────
Pre-LayerNorm Transformer decoder block implemented from scratch.

Block Structure:
    x = x + Attention(LayerNorm1(x))
    x = x + MLP(LayerNorm2(x))

MLP Structure:
    Linear(d_model -> d_ff)
    GELU()
    Linear(d_ff -> d_model)
    Dropout(dropout)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from attendance_tracker.slm.model.attention import CausalSelfAttention
from attendance_tracker.slm.model.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP) with GELU activation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: Linear(256 -> 1024) -> GELU -> Linear(1024 -> 256) -> Dropout
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Decoder Block.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Pre-attention LayerNorm
        self.ln1 = nn.LayerNorm(config.d_model)
        # Causal multi-head self-attention
        self.attn = CausalSelfAttention(config)
        # Pre-MLP LayerNorm
        self.ln2 = nn.LayerNorm(config.d_model)
        # Feed-Forward network
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute Pre-LN Transformer block.

        Ordering:
            1. Residual + Attention(LayerNorm1(x))
            2. Residual + MLP(LayerNorm2(x))
        """
        # 1. Attention sub-layer with Pre-LN and residual
        x = x + self.attn(self.ln1(x))

        # 2. MLP sub-layer with Pre-LN and residual
        x = x + self.mlp(self.ln2(x))

        return x
