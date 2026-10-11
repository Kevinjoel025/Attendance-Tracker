"""
tests/test_model.py
───────────────────
Comprehensive test suite for Phase 6 Custom Decoder-Only Transformer SLM.

Tests cover:
1. Model configuration and validation
2. Causal self-attention mechanics and tensor shapes
3. Strict causal masking behavior (future token isolation)
4. Pre-LN Transformer blocks and GELU MLP
5. Full AttendanceSLM forward pass and output logits shape [B, T, 4096]
6. Parameter count verification and weight tying identity
7. Input validation (bounds, length, dtypes)
8. Gradient flow across all layers
9. Phase 5 BPE tokenizer compatibility
10. State dict save / load persistence
11. Device transfers (CPU and CUDA)
12. Lightweight forward pass benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from attendance_tracker.slm.model.attention import CausalSelfAttention
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.model.transformer_block import FeedForward, TransformerBlock
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelConfig:
    """Test suite for ModelConfig hyperparameter defaults and validation rules."""

    def test_default_config_matches_specifications(self):
        config = ModelConfig()
        assert config.vocab_size == 4096
        assert config.context_length == 256
        assert config.d_model == 256
        assert config.n_layers == 6
        assert config.n_heads == 4
        assert config.d_ff == 1024
        assert config.head_dim == 64
        assert config.dropout == 0.1
        assert config.activation == "gelu"
        assert config.layer_norm == "pre"
        assert config.pos_encoding == "learned"
        assert config.weight_tying is True

    def test_invalid_d_model_n_heads_divisibility_raises(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ModelConfig(d_model=255, n_heads=4)

    def test_negative_or_zero_dimensions_raise(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ModelConfig(vocab_size=0)

        with pytest.raises(ValueError, match="context_length must be positive"):
            ModelConfig(context_length=-10)

        with pytest.raises(ValueError, match="d_model must be positive"):
            ModelConfig(d_model=0)

        with pytest.raises(ValueError, match="n_layers must be positive"):
            ModelConfig(n_layers=0)

        with pytest.raises(ValueError, match="n_heads must be positive"):
            ModelConfig(n_heads=0)

        with pytest.raises(ValueError, match="d_ff must be positive"):
            ModelConfig(d_ff=-512)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)

        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=-0.1)

    def test_unsupported_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            ModelConfig(activation="relu")

    def test_unsupported_layer_norm_raises(self):
        with pytest.raises(ValueError, match="layer_norm"):
            ModelConfig(layer_norm="post")

    def test_unsupported_pos_encoding_raises(self):
        with pytest.raises(ValueError, match="pos_encoding"):
            ModelConfig(pos_encoding="sinusoidal")

    def test_config_serialization_roundtrip(self):
        config = ModelConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["d_model"] == 256

        json_str = config.to_json()
        loaded = ModelConfig.from_json(json_str)
        assert loaded == config


# ─────────────────────────────────────────────────────────────────────────────
# 2. Causal Self-Attention Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalSelfAttention:
    """Test suite for CausalSelfAttention mechanism."""

    def test_attention_shapes(self):
        config = ModelConfig()
        attn = CausalSelfAttention(config)

        # Batch=2, Seq=16, d_model=256
        x = torch.randn(2, 16, 256)
        out = attn(x)
        assert out.shape == (2, 16, 256)

    def test_attention_shapes_with_weights(self):
        config = ModelConfig()
        attn = CausalSelfAttention(config)

        x = torch.randn(2, 8, 256)
        out, weights = attn(x, return_attention_weights=True)

        assert out.shape == (2, 8, 256)
        # Weights shape: [B, n_heads, T, T] -> [2, 4, 8, 8]
        assert weights.shape == (2, 4, 8, 8)

    @pytest.mark.parametrize("seq_len", [1, 2, 16, 128, 256])
    def test_variable_sequence_lengths(self, seq_len: int):
        config = ModelConfig()
        attn = CausalSelfAttention(config)

        x = torch.randn(1, seq_len, 256)
        out = attn(x)
        assert out.shape == (1, seq_len, 256)

    def test_sequence_length_exceeding_max_raises(self):
        config = ModelConfig(context_length=256)
        attn = CausalSelfAttention(config)

        x = torch.randn(1, 257, 256)
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            attn(x)

    def test_causal_mask_is_registered_buffer(self):
        config = ModelConfig(context_length=256)
        attn = CausalSelfAttention(config)

        assert "mask" in dict(attn.named_buffers())
        assert attn.mask.shape == (1, 1, 256, 256)
        # Check lower triangular structure
        assert attn.mask[0, 0, 0, 0].item() is True
        assert attn.mask[0, 0, 0, 1].item() is False
        assert attn.mask[0, 0, 1, 0].item() is True
        assert attn.mask[0, 0, 1, 1].item() is True

    def test_attention_device_movement(self):
        config = ModelConfig()
        attn = CausalSelfAttention(config)
        attn.to("cpu")
        assert attn.mask.device.type == "cpu"

        if torch.cuda.is_available():
            attn.to("cuda")
            assert attn.mask.device.type == "cuda"
            x = torch.randn(2, 16, 256, device="cuda")
            out = attn(x)
            assert out.device.type == "cuda"
            assert out.shape == (2, 16, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Strict Causal Masking Behavior Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalMaskBehavior:
    """Verify that future tokens have ZERO influence on earlier token representations."""

    def test_future_tokens_do_not_affect_past_attention_outputs(self):
        torch.manual_seed(42)
        config = ModelConfig(dropout=0.0)  # zero dropout for deterministic check
        attn = CausalSelfAttention(config)
        attn.eval()

        # Construct x1 and x2 of shape [1, 10, 256]
        # x1 and x2 share identical prefixes for positions 0..4,
        # but have completely different tokens for positions 5..9
        prefix = torch.randn(1, 5, 256)
        suffix1 = torch.randn(1, 5, 256)
        suffix2 = torch.randn(1, 5, 256) + 100.0  # strongly different

        x1 = torch.cat([prefix, suffix1], dim=1)
        x2 = torch.cat([prefix, suffix2], dim=1)

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Positions 0..4 MUST be strictly identical between out1 and out2
        diff_prefix = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
        assert diff_prefix < 1e-6, f"Causality violated! Prefix difference: {diff_prefix}"

        # Positions 5..9 MUST be different
        diff_suffix = (out1[:, 5:, :] - out2[:, 5:, :]).abs().max().item()
        assert diff_suffix > 1e-2, "Suffix outputs were unexpectedly identical"

    def test_upper_triangle_attention_weights_are_strictly_zero(self):
        torch.manual_seed(42)
        config = ModelConfig(dropout=0.0)
        attn = CausalSelfAttention(config)
        attn.eval()

        x = torch.randn(1, 16, 256)
        with torch.no_grad():
            _, weights = attn(x, return_attention_weights=True)

        # weights shape: [1, 4, 16, 16]
        # For every query position i and key position j where j > i, weight must be 0.0
        for i in range(16):
            for j in range(i + 1, 16):
                assert (weights[:, :, i, j] == 0.0).all().item(), f"Non-zero weight at ({i}, {j})"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Block Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerBlock:
    """Test suite for Pre-LN Transformer decoder block."""

    def test_block_shape_preservation(self):
        config = ModelConfig()
        block = TransformerBlock(config)

        x = torch.randn(2, 32, 256)
        out = block(x)
        assert out.shape == (2, 32, 256)

    def test_mlp_gelu_activation(self):
        config = ModelConfig()
        mlp = FeedForward(config)
        assert isinstance(mlp.act, nn.GELU)

        x = torch.randn(2, 10, 256)
        out = mlp(x)
        assert out.shape == (2, 10, 256)

    def test_block_pre_ln_architecture(self):
        config = ModelConfig()
        block = TransformerBlock(config)

        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.attn, CausalSelfAttention)
        assert isinstance(block.ln2, nn.LayerNorm)
        assert isinstance(block.mlp, FeedForward)

    def test_block_residual_connections(self):
        torch.manual_seed(42)
        config = ModelConfig(dropout=0.0)
        block = TransformerBlock(config)
        block.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            out = block(x)

        # Output should be different from input due to attention + MLP additions
        assert not torch.allclose(out, x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full AttendanceSLM Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAttendanceSLM:
    """Test suite for the full AttendanceSLM architecture."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [1, 8, 32, 128, 256])
    def test_forward_pass_logits_shape(self, batch_size: int, seq_len: int):
        config = ModelConfig()
        model = AttendanceSLM(config)
        model.eval()

        input_ids = torch.randint(0, 4096, (batch_size, seq_len), dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, 4096)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_cpu_forward_pass(self):
        model = AttendanceSLM()
        model.to("cpu")
        input_ids = torch.randint(0, 4096, (2, 16), dtype=torch.long, device="cpu")
        logits = model(input_ids)
        assert logits.device.type == "cpu"
        assert logits.shape == (2, 16, 4096)

    def test_cuda_forward_pass_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")

        model = AttendanceSLM()
        model.to("cuda")
        input_ids = torch.randint(0, 4096, (2, 16), dtype=torch.long, device="cuda")
        logits = model(input_ids)
        assert logits.device.type == "cuda"
        assert logits.shape == (2, 16, 4096)

    def test_weight_tying_identity(self):
        model = AttendanceSLM()

        # 1. Verify object reference identity
        assert model.lm_head.weight is model.token_embedding.weight, (
            "Weight tying failed: lm_head.weight and token_embedding.weight are different objects"
        )

        # 2. Verify underlying memory storage pointer
        assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr(), (
            "Weight tying failed: storage data_ptr does not match"
        )

        # 3. Verify modifying token_embedding immediately reflects in lm_head
        orig_val = model.token_embedding.weight[0, 0].item()
        model.token_embedding.weight.data[0, 0] += 5.0
        assert model.lm_head.weight[0, 0].item() == pytest.approx(orig_val + 5.0, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """Test that model rejects malformed inputs without silent clamping/truncation."""

    def test_empty_sequence_raises(self):
        model = AttendanceSLM()
        empty_ids = torch.empty((1, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="Sequence length must be > 0"):
            model(empty_ids)

    def test_sequence_exceeding_context_length_raises(self):
        model = AttendanceSLM()
        long_ids = torch.randint(0, 4096, (1, 257), dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds maximum context length"):
            model(long_ids)

    def test_negative_token_id_raises(self):
        model = AttendanceSLM()
        invalid_ids = torch.tensor([[10, 20, -1, 40]], dtype=torch.long)
        with pytest.raises(ValueError, match="Token ID out of bounds"):
            model(invalid_ids)

    def test_out_of_vocab_token_id_raises(self):
        model = AttendanceSLM()
        invalid_ids = torch.tensor([[10, 20, 4096, 40]], dtype=torch.long)
        with pytest.raises(ValueError, match="Token ID out of bounds"):
            model(invalid_ids)

        invalid_high = torch.tensor([[10, 20, 10000]], dtype=torch.long)
        with pytest.raises(ValueError, match="Token ID out of bounds"):
            model(invalid_high)

    def test_float_dtype_raises(self):
        model = AttendanceSLM()
        float_ids = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(TypeError, match="must contain integer token IDs"):
            model(float_ids)

    def test_non_2d_tensor_raises(self):
        model = AttendanceSLM()
        one_d = torch.tensor([1, 2, 3], dtype=torch.long)
        with pytest.raises(ValueError, match="must be a 2D tensor"):
            model(one_d)

        three_d = torch.zeros((1, 2, 3), dtype=torch.long)
        with pytest.raises(ValueError, match="must be a 2D tensor"):
            model(three_d)

    def test_non_tensor_input_raises(self):
        model = AttendanceSLM()
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            model([[1, 2, 3]])  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gradient Flow & Backward Pass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """Verify that gradients flow to all architectural parameters."""

    def test_all_parameters_receive_gradients(self):
        torch.manual_seed(42)
        model = AttendanceSLM()
        model.train()

        input_ids = torch.randint(0, 4096, (2, 16), dtype=torch.long)
        logits = model(input_ids)

        # Simple cross-entropy proxy loss
        loss = logits.sum()
        loss.backward()

        # Check token and positional embeddings
        assert model.token_embedding.weight.grad is not None
        assert not torch.isnan(model.token_embedding.weight.grad).any()
        assert model.pos_embedding.weight.grad is not None
        assert not torch.isnan(model.pos_embedding.weight.grad).any()

        # Check every transformer block's parameters
        for i, block in enumerate(model.blocks):
            # Attention LN
            assert block.ln1.weight.grad is not None, f"Block {i} ln1.weight has no grad"
            assert block.ln1.bias.grad is not None, f"Block {i} ln1.bias has no grad"

            # Attention Q, K, V, Out
            assert block.attn.q_proj.weight.grad is not None, f"Block {i} q_proj.weight has no grad"
            assert block.attn.k_proj.weight.grad is not None, f"Block {i} k_proj.weight has no grad"
            assert block.attn.v_proj.weight.grad is not None, f"Block {i} v_proj.weight has no grad"
            assert block.attn.out_proj.weight.grad is not None, f"Block {i} out_proj.weight has no grad"

            # MLP LN
            assert block.ln2.weight.grad is not None, f"Block {i} ln2.weight has no grad"
            assert block.ln2.bias.grad is not None, f"Block {i} ln2.bias has no grad"

            # MLP FC1, FC2
            assert block.mlp.fc1.weight.grad is not None, f"Block {i} fc1.weight has no grad"
            assert block.mlp.fc2.weight.grad is not None, f"Block {i} fc2.weight has no grad"

        # Final LN
        assert model.ln_f.weight.grad is not None
        assert model.ln_f.bias.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Parameter Count Verification Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterCount:
    """Verify total parameter count and structural breakdown."""

    def test_total_and_breakdown_parameter_count(self):
        model = AttendanceSLM()
        breakdown = model.get_parameter_breakdown()

        # Detailed mathematical verification:
        # Token Embedding: 4096 * 256 = 1,048,576
        # Position Embedding: 256 * 256 = 65,536
        # Total Embeddings: 1,114,112
        assert breakdown["token_embedding"] == 1_048_576
        assert breakdown["position_embedding"] == 65_536
        assert breakdown["total_embeddings"] == 1_114_112

        # Per Block:
        # ln1: 256 + 256 = 512
        # attn (q, k, v, out): 4 * (256 * 256 + 256) = 4 * 65,792 = 263,168
        # ln2: 256 + 256 = 512
        # mlp (fc1 + fc2): (256 * 1024 + 1024) + (1024 * 256 + 256) = 263,168 + 262,400 = 525,568
        # Per block total = 512 + 263,168 + 512 + 525,568 = 789,760
        assert breakdown["per_block"]["total"] == 789_760

        # 6 Blocks = 6 * 789,760 = 4,738,560
        assert breakdown["transformer_blocks"] == 4_738_560

        # Final LN = 256 + 256 = 512
        assert breakdown["final_layernorm"] == 512

        # Tied LM Head savings = 1,048,576
        assert breakdown["lm_head_tied"] is True
        assert breakdown["lm_head_weight_saved"] == 1_048_576

        # Total unique parameters: 1,114,112 + 4,738,560 + 512 = 5,853,184
        expected_total = 5_853_184
        assert breakdown["total_unique_parameters"] == expected_total
        assert breakdown["total_trainable_parameters"] == expected_total

        # Target range check (~5.8M)
        assert 5_700_000 <= expected_total <= 6_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 9. Phase 5 Tokenizer Compatibility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizerCompatibility:
    """Verify end-to-end compatibility with Phase 5 BPE tokenizer artifact."""

    def test_real_queries_tokenize_and_pass_forward(self):
        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("data/tokenizer/bpe.json artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)
        assert tokenizer.vocab_size == 4096

        model = AttendanceSLM()
        model.eval()

        test_queries = [
            "What is my attendance?",
            "what is my DE attendance",
            "How much attendance do I have in operating systems?",
            "how many classes do I need to reach 75% in IoTAP LAB?",
            "can I skip 3 classes in Gender Sensitization?",
            "did today's attendance get marked for Storytelling?",
            "if I attend 5 more classes of ACD what will my attendance be?",
        ]

        for query in test_queries:
            token_ids = tokenizer.encode(query, add_bos=True, add_eos=True)
            assert len(token_ids) > 0
            assert all(0 <= tid < 4096 for tid in token_ids)

            # Convert to batch of shape [1, T]
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_tensor)

            assert logits.shape == (1, len(token_ids), 4096)
            assert not torch.isnan(logits).any()


# ─────────────────────────────────────────────────────────────────────────────
# 10. State Dict Save / Load Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStateDictPersistence:
    """Test state_dict saving and loading."""

    def test_save_and_load_state_dict(self, tmp_path: Path):
        torch.manual_seed(123)
        model1 = AttendanceSLM()
        model1.eval()

        save_file = tmp_path / "model_state.pt"
        torch.save(model1.state_dict(), save_file)

        # Create fresh model with same config
        model2 = AttendanceSLM()
        model2.load_state_dict(torch.load(save_file, weights_only=True))
        model2.eval()

        # Weight tying should still be preserved
        assert model2.lm_head.weight is model2.token_embedding.weight

        # Outputs should be identical
        test_input = torch.randint(0, 4096, (2, 16), dtype=torch.long)
        with torch.no_grad():
            out1 = model1(test_input)
            out2 = model2(test_input)

        assert torch.allclose(out1, out2, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Forward Benchmark Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardBenchmark:
    """Lightweight forward pass latency check (no training)."""

    def test_cpu_forward_timing(self):
        model = AttendanceSLM()
        model.eval()
        input_ids = torch.randint(0, 4096, (8, 128), dtype=torch.long)

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        # Measure 5 forward passes
        times = []
        with torch.no_grad():
            for _ in range(5):
                start = time.perf_counter()
                _ = model(input_ids)
                times.append(time.perf_counter() - start)

        avg_time_ms = (sum(times) / len(times)) * 1000.0
        assert avg_time_ms > 0

    def test_cuda_forward_timing_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = AttendanceSLM().to("cuda")
        model.eval()
        input_ids = torch.randint(0, 4096, (8, 128), dtype=torch.long, device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
            torch.cuda.synchronize()

        # Measure 10 forward passes
        times = []
        with torch.no_grad():
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_ids)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time_ms = (sum(times) / len(times)) * 1000.0
        assert avg_time_ms > 0
