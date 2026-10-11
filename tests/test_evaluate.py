"""
tests/test_evaluate.py
───────────────────────
Unit tests for evaluation metrics, greedy generation, and pipe format parsing.
"""

from pathlib import Path
import json
import pytest
import torch

from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BPETokenizer
from attendance_tracker.slm.training.evaluate import (
    EvaluationReport,
    evaluate_dataset,
    generate_batch_completions,
    generate_completion,
    parse_pipe_output,
)


class TestEvaluationUtilities:

    def test_parse_pipe_output_complete(self):
        pipe_str = "CLASSES_REQUIRED_FOR_TARGET|ACD|75|NONE|NONE"
        parsed = parse_pipe_output(pipe_str)
        assert parsed["intent"] == "CLASSES_REQUIRED_FOR_TARGET"
        assert parsed["subject"] == "ACD"
        assert parsed["target"] == "75"
        assert parsed["count"] is None
        assert parsed["date"] is None

    def test_parse_pipe_output_projection(self):
        pipe_str = "ATTENDANCE_PROJECTION|WP|NONE|4|NONE"
        parsed = parse_pipe_output(pipe_str)
        assert parsed["intent"] == "ATTENDANCE_PROJECTION"
        assert parsed["subject"] == "WP"
        assert parsed["target"] is None
        assert parsed["count"] == "4"
        assert parsed["date"] is None

    def test_parse_pipe_output_all_none(self):
        pipe_str = "GET_ATTENDANCE|NONE|NONE|NONE|NONE"
        parsed = parse_pipe_output(pipe_str)
        assert parsed["intent"] == "GET_ATTENDANCE"
        assert parsed["subject"] is None
        assert parsed["target"] is None
        assert parsed["count"] is None
        assert parsed["date"] is None

    def test_parse_pipe_output_fallback(self):
        parsed = parse_pipe_output("")
        assert parsed["intent"] == "UNKNOWN"

    def test_evaluation_report_summary(self):
        report = EvaluationReport(
            total_examples=100,
            correct_intents=95,
            intent_accuracy=95.0,
            exact_matches=90,
            exact_match_accuracy=90.0,
            per_intent_metrics={
                "GET_ATTENDANCE": {"total": 50, "correct": 50, "accuracy": 100.0},
                "HELP": {"total": 50, "correct": 45, "accuracy": 90.0},
            },
            target_achieved=True,
        )
        summary = report.summary()
        assert "95.00%" in summary
        assert "GET_ATTENDANCE" in summary
        assert "YES" in summary
        assert report.to_dict()["target_achieved"] is True


class TestGenerationAndEvaluation:

    def test_greedy_generation_loop(self):
        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2, d_ff=64, context_length=64)
        model = AttendanceSLM(config)
        tokenizer = BPETokenizer(target_vocab_size=50)
        tokenizer.train(["what is attendance"])

        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt_text="what is attendance",
            max_new_tokens=5,
            device="cpu",
        )
        assert isinstance(completion, str)

    def test_batch_greedy_generation(self):
        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2, d_ff=64, context_length=64)
        model = AttendanceSLM(config)
        tokenizer = BPETokenizer(target_vocab_size=50)
        tokenizer.train(["what is attendance", "show percentage"])

        prompts = ["what is attendance", "show percentage"]
        completions = generate_batch_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=5,
            device="cpu",
        )
        assert len(completions) == 2
        assert isinstance(completions[0], str)
        assert isinstance(completions[1], str)

    def test_evaluate_dataset_mock(self, tmp_path: Path):
        bpe_path = Path("data/tokenizer/bpe.json")
        if not bpe_path.exists():
            pytest.skip("BPE tokenizer artifact not found")

        tokenizer = BPETokenizer.load(bpe_path)
        config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=2, d_ff=128, context_length=64)
        model = AttendanceSLM(config)

        # Create small mock test split
        mock_records = [
            {"text": "what is my attendance", "intent": "GET_ATTENDANCE", "output": "GET_ATTENDANCE|NONE|NONE|NONE|NONE"},
            {"text": "help me", "intent": "HELP", "output": "HELP|NONE|NONE|NONE|NONE"},
        ]
        test_file = tmp_path / "mock_test.jsonl"
        with test_file.open("w", encoding="utf-8") as f:
            for r in mock_records:
                f.write(json.dumps(r) + "\n")

        report = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_path=test_file,
            device="cpu",
            batch_size=2,
        )
        assert report.total_examples == 2
        assert isinstance(report.intent_accuracy, float)
