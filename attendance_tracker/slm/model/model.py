"""
slm/model/model.py
──────────────────
Custom Decoder-Only Transformer Language Model for College Attendance Tracker.

Architecture:
- 6 Pre-LayerNorm Transformer decoder blocks
- d_model = 256, n_heads = 4, head_dim = 64, d_ff = 1024
- Context length = 256 tokens, Vocabulary = 4096 tokens
- Learned positional embeddings
- Weight tying between token embedding and output language modeling head
- Logits output: [B, T, 4096]
- ~5.85M total trainable parameters
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.transformer_block import TransformerBlock


class AttendanceSLM(nn.Module):
    """
    Decoder-only Transformer SLM for Attendance NLU intent & slot extraction.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.config.validate()

        # 1. Embeddings
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.pos_embedding = nn.Embedding(self.config.context_length, self.config.d_model)
        self.drop = nn.Dropout(self.config.dropout)

        # 2. Transformer Decoder Blocks (6 layers)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        # 3. Final LayerNorm
        self.ln_f = nn.LayerNorm(self.config.d_model)

        # 4. Language Modeling Output Head (tied to token_embedding)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # 5. Initialize all weights deterministically
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with standard Gaussian N(0, initializer_range)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def validate_inputs(self, input_ids: Tensor) -> None:
        """
        Validate input tensor before forward pass.

        Ensures:
        - 2D tensor of shape [B, T]
        - Long / Integer dtype
        - 1 <= T <= context_length
        - 0 <= token_id < vocab_size
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids)}")

        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be a 2D tensor [batch_size, seq_len], got shape {input_ids.shape}"
            )

        if input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError(
                f"input_ids must contain integer token IDs, got dtype {input_ids.dtype}"
            )

        B, T = input_ids.shape
        if T == 0:
            raise ValueError("Sequence length must be > 0 (empty input provided)")

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({self.config.context_length})"
            )

        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())

        if min_id < 0 or max_id >= self.config.vocab_size:
            raise ValueError(
                f"Token ID out of bounds. All IDs must be in [0, {self.config.vocab_size - 1}], "
                f"got min ID {min_id} and max ID {max_id}"
            )

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass producing next-token vocabulary logits.

        Parameters
        ----------
        input_ids : Tensor
            Integer tensor of shape [B, T] where 1 <= T <= 256 and IDs in [0, 4095].

        Returns
        -------
        logits : Tensor
            Logits of shape [B, T, vocab_size] (e.g. [B, T, 4096]).
        """
        # Validate inputs strictly
        self.validate_inputs(input_ids)

        B, T = input_ids.shape
        device = input_ids.device

        # Create position indices [0, 1, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=device)

        # 1. Embeddings: Token + Positional
        tok_emb = self.token_embedding(input_ids)       # [B, T, d_model]
        pos_emb = self.pos_embedding(positions)         # [T, d_model]
        x = self.drop(tok_emb + pos_emb)                # [B, T, d_model]

        # 2. Pass through 6 Pre-LN Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm
        x = self.ln_f(x)                                # [B, T, d_model]

        # 4. Output projection to vocabulary logits
        logits = self.lm_head(x)                        # [B, T, vocab_size]

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the total number of unique trainable parameters.
        Correctly accounts for tied weights (lm_head.weight == token_embedding.weight).
        """
        unique_params = set(self.parameters())
        if non_embedding:
            unique_params.discard(self.token_embedding.weight)
            unique_params.discard(self.pos_embedding.weight)
        return sum(p.numel() for p in unique_params if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, Any]:
        """
        Return a detailed dictionary breakdown of parameter counts by component.
        """
        tok_emb_params = self.token_embedding.weight.numel()
        pos_emb_params = self.pos_embedding.weight.numel()
        final_ln_params = sum(p.numel() for p in self.ln_f.parameters())

        block_breakdowns = []
        total_blocks_params = 0
        for i, b in enumerate(self.blocks):
            ln1_p = sum(p.numel() for p in b.ln1.parameters())
            attn_p = sum(p.numel() for p in b.attn.parameters())
            ln2_p = sum(p.numel() for p in b.ln2.parameters())
            mlp_p = sum(p.numel() for p in b.mlp.parameters())
            b_total = ln1_p + attn_p + ln2_p + mlp_p
            total_blocks_params += b_total
            block_breakdowns.append({
                "block_index": i,
                "ln1": ln1_p,
                "attention": attn_p,
                "ln2": ln2_p,
                "mlp": mlp_p,
                "total": b_total,
            })

        tied_saved = tok_emb_params if self.config.weight_tying else 0
        total_params = self.get_num_params(non_embedding=False)

        return {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "head_dim": self.config.head_dim,
            "d_ff": self.config.d_ff,
            "context_length": self.config.context_length,
            "token_embedding": tok_emb_params,
            "position_embedding": pos_emb_params,
            "total_embeddings": tok_emb_params + pos_emb_params,
            "transformer_blocks": total_blocks_params,
            "per_block": block_breakdowns[0] if block_breakdowns else {},
            "final_layernorm": final_ln_params,
            "lm_head_tied": self.config.weight_tying,
            "lm_head_weight_saved": tied_saved,
            "total_unique_parameters": total_params,
            "total_trainable_parameters": total_params,
        }
