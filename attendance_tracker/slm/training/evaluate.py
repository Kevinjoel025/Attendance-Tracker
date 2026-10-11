"""
slm/training/evaluate.py
────────────────────────
Evaluation pipeline for AttendanceSLM on held-out test data.

Calculates:
- Overall intent classification accuracy (target: >= 80%)
- Per-intent accuracy breakdown
- Exact match accuracy on complete structured output: INTENT|SUBJECT|TARGET|COUNT|DATE
- Detailed error reporting on failure cases
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, PAD_ID, BPETokenizer

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Evaluation metrics summary on a dataset split."""

    total_examples: int
    correct_intents: int
    intent_accuracy: float
    exact_matches: int
    exact_match_accuracy: float
    per_intent_metrics: dict[str, dict[str, Any]]
    target_achieved: bool  # True if intent_accuracy >= 80.0%
    sample_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "NLU EVALUATION REPORT",
            "=" * 60,
            f"Total Test Examples:     {self.total_examples}",
            f"Correct Intents:         {self.correct_intents} / {self.total_examples}",
            f"Overall Intent Accuracy: {self.intent_accuracy:.2f}%",
            f"Exact Output Matches:    {self.exact_matches} / {self.total_examples} ({self.exact_match_accuracy:.2f}%)",
            f"Target (>= 80.0%) Met:   {'YES' if self.target_achieved else 'NO'}",
            "-" * 60,
            "PER-INTENT ACCURACY BREAKDOWN:",
            "-" * 60,
        ]
        for intent, metrics in sorted(self.per_intent_metrics.items()):
            lines.append(
                f"  {intent:<30} : {metrics['correct']:>4} / {metrics['total']:>4} "
                f"({metrics['accuracy']:>6.2f}%)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def parse_pipe_output(output_str: str) -> dict[str, Optional[str]]:
    """
    Parse 5-position pipe-delimited output string:
        INTENT|SUBJECT|TARGET|COUNT|DATE
    """
    cleaned = output_str.strip()
    parts = cleaned.split("|")
    intent = parts[0].strip() if len(parts) > 0 and parts[0].strip() else "UNKNOWN"
    subject = parts[1].strip() if len(parts) > 1 and parts[1].strip() != "NONE" else None
    target = parts[2].strip() if len(parts) > 2 and parts[2].strip() != "NONE" else None
    count = parts[3].strip() if len(parts) > 3 and parts[3].strip() != "NONE" else None
    date_val = parts[4].strip() if len(parts) > 4 and parts[4].strip() != "NONE" else None

    return {
        "intent": intent,
        "subject": subject,
        "target": target,
        "count": count,
        "date": date_val,
    }


def generate_completion(
    model: nn.Module,
    tokenizer: BPETokenizer,
    prompt_text: str,
    max_new_tokens: int = 40,
    device: str | torch.device = "cpu",
    sep_token_id: Optional[int] = None,
) -> str:
    """
    Autoregressively generate completion using greedy search.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    if sep_token_id is not None:
        prompt_ids.append(sep_token_id)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.shape[1] >= model.config.context_length:
                break

            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :]
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

            if next_token_id == EOS_ID:
                break

            generated_ids.append(next_token_id)
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_batch_completions(
    model: nn.Module,
    tokenizer: BPETokenizer,
    prompts: list[str],
    max_new_tokens: int = 40,
    device: str | torch.device = "cpu",
) -> list[str]:
    """
    Generate completions for a batch of prompt strings using greedy decoding.
    """
    model.eval()
    if not prompts:
        return []

    # Encode all prompts with BOS
    encoded_prompts = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
    batch_size = len(prompts)

    # Pad prompts to uniform length for batch processing
    max_prompt_len = max(len(ids) for ids in encoded_prompts)
    padded_inputs = torch.full((batch_size, max_prompt_len), PAD_ID, dtype=torch.long, device=device)

    # Track initial prompt lengths
    prompt_lens = [len(ids) for ids in encoded_prompts]
    for i, ids in enumerate(encoded_prompts):
        padded_inputs[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    # Track completed sequences and generated tokens
    finished = [False] * batch_size
    generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]

    current_inputs = padded_inputs

    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(finished) or current_inputs.shape[1] >= model.config.context_length:
                break

            logits = model(current_inputs)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)  # [B]

            next_col = next_tokens.unsqueeze(1)
            current_inputs = torch.cat([current_inputs, next_col], dim=1)

            for i in range(batch_size):
                if not finished[i]:
                    token_id = int(next_tokens[i].item())
                    if token_id == EOS_ID:
                        finished[i] = True
                    else:
                        generated_tokens[i].append(token_id)

    # Decode completions
    results = []
    for tokens in generated_tokens:
        decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        results.append(decoded)

    return results


def evaluate_dataset(
    model: nn.Module,
    tokenizer: BPETokenizer,
    dataset_path: Path | str = "data/nlu/test.jsonl",
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    sep_token_id: Optional[int] = None,
) -> EvaluationReport:
    """
    Run full evaluation on a held-out test split.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found at {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples is not None:
        records = records[:max_samples]

    total_examples = len(records)
    logger.info(f"Evaluating {total_examples} records from {path} on device {device}...")

    # Group metrics per intent
    intent_counts: dict[str, dict[str, int]] = {}
    for r in records:
        expected_intent = r["intent"]
        if expected_intent not in intent_counts:
            intent_counts[expected_intent] = {"total": 0, "correct": 0}
        intent_counts[expected_intent]["total"] += 1

    correct_intents = 0
    exact_matches = 0
    sample_errors: list[dict[str, Any]] = []

    for start_idx in range(0, total_examples, batch_size):
        batch_records = records[start_idx : start_idx + batch_size]

        for record in batch_records:
            pred_str = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_text=record["text"],
                max_new_tokens=40,
                device=device,
                sep_token_id=sep_token_id,
            )

            expected_intent = record["intent"]
            expected_output = record["output"].strip()

            parsed = parse_pipe_output(pred_str)
            predicted_intent = parsed["intent"]

            is_intent_correct = (predicted_intent == expected_intent)
            is_exact_match = (pred_str == expected_output)

            if is_intent_correct:
                correct_intents += 1
                intent_counts[expected_intent]["correct"] += 1

            if is_exact_match:
                exact_matches += 1

            if not is_intent_correct and len(sample_errors) < 20:
                sample_errors.append({
                    "text": record["text"],
                    "expected_output": expected_output,
                    "predicted_output": pred_str,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                })

    intent_accuracy = (correct_intents / total_examples) * 100.0 if total_examples > 0 else 0.0
    exact_match_accuracy = (exact_matches / total_examples) * 100.0 if total_examples > 0 else 0.0

    per_intent_metrics: dict[str, dict[str, Any]] = {}
    for intent, counts in intent_counts.items():
        tot = counts["total"]
        cor = counts["correct"]
        acc = (cor / tot) * 100.0 if tot > 0 else 0.0
        per_intent_metrics[intent] = {
            "total": tot,
            "correct": cor,
            "accuracy": round(acc, 2),
        }

    report = EvaluationReport(
        total_examples=total_examples,
        correct_intents=correct_intents,
        intent_accuracy=round(intent_accuracy, 2),
        exact_matches=exact_matches,
        exact_match_accuracy=round(exact_match_accuracy, 2),
        per_intent_metrics=per_intent_metrics,
        target_achieved=(intent_accuracy >= 80.0),
        sample_errors=sample_errors,
    )

    logger.info(f"Evaluation complete: Intent Accuracy = {intent_accuracy:.2f}% (Target >= 80%: {report.target_achieved})")
    return report
