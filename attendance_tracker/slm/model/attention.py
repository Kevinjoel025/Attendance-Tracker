"""
slm/model/attention.py
──────────────────────
Multi-head causal self-attention implemented from scratch in pure PyTorch.

RULES:
- Zero external Transformer libraries (no torch.nn.Transformer, huggingface, etc.).
- Multi-head scaled dot-product attention with explicit Q, K, V linear projections.
- Registered buffer causal lower-triangular mask (future tokens cannot be attended).
- Clean device transfer between CPU and CUDA.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from attendance_tracker.slm.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.

    Given an input tensor X of shape [B, T, d_model]:
    1. Projects Q = XWq, K = XWk, V = XWv
    2. Reshapes into [B, n_heads, T, head_dim]
    3. Computes scaled attention scores: Q @ K^T / sqrt(head_dim)
    4. Applies causal mask ensuring position i cannot attend to positions > i
    5. Softmax -> weighted sum over V
    6. Concatenates heads and applies output projection: W_out
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length

        # Explicit linear projections for Query, Key, Value
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of shape [1, 1, context_length, context_length]
        # Registered as buffer so it moves across devices (CPU/CUDA) and is saved in state_dict
        mask = torch.tril(torch.ones(self.context_length, self.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, self.context_length, self.context_length))

    def forward(
        self,
        x: Tensor,
        return_attention_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Forward pass for causal self-attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [B, T, d_model].
        return_attention_weights : bool
            If True, returns a tuple (output, attention_weights).

        Returns
        -------
        Tensor of shape [B, T, d_model] or Tuple[Tensor, Tensor].
        """
        B, T, C = x.shape
        if T > self.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds maximum context length {self.context_length}"
            )

        # 1. Project inputs to Q, K, V -> [B, T, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into multi-head format: [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention scores: [B, n_heads, T, T]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 4. Apply causal mask (fill upper-triangular future positions with -inf)
        causal_mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 5. Softmax to obtain attention distribution
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # 6. Weighted sum over Value vectors: [B, n_heads, T, head_dim]
        out = torch.matmul(attn_weights_dropped, v)

        # 7. Concatenate heads back to [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Final output projection and residual dropout
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_attention_weights:
            return out, attn_weights
        return out
