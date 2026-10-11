"""
scripts/evaluate_acceptance.py
──────────────────────────────
Phase 11 Acceptance Evaluation and Diagnostic Suite.

Evaluates the College Attendance Tracker across:
A. Raw SLM Quality (direct autoregressive output parsed without deterministic overrides)
B. Final Application Quality (end-to-end ToolRouter, subject resolver, threshold safeguard)

Generates exact metrics, per-intent/per-subject breakdowns, and failure classifications.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import torch
import torch.nn.functional as F

from attendance_tracker.attendance.repository import CANONICAL_SUBJECTS, get_repository
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.subject_resolver import get_subject_resolver
from attendance_tracker.core.thresholds import COLLEGE_THRESHOLD, PERSONAL_TARGET, resolve_threshold
from attendance_tracker.core.tool_router import ToolRouter
from attendance_tracker.slm.inference.nlu import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SEP_TOKEN_ID,
    DEFAULT_TOKENIZER_PATH,
    INTENT_TOKEN_MAP,
    NLUInferenceEngine,
    get_nlu_engine,
    normalize_query_for_slm,
)
from attendance_tracker.slm.inference.parser import ParsedNLUOutput, parse_nlu_output
from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BOS_ID, EOS_ID, BPETokenizer

logger = get_logger(__name__)


@dataclass
class FailureRecord:
    query: str
    expected_intent: str
    predicted_intent: str
    expected_subject: Optional[str]
    predicted_subject: Optional[str]
    expected_target: Optional[float]
    predicted_target: Optional[float]
    expected_count: Optional[int]
    predicted_count: Optional[int]
    failure_stage: str  # "RAW_SLM" or "PIPELINE"
    failure_type: str   # "A: SLM Data Gap", "B: Tokenizer", etc.
    severity: str       # "HIGH", "MEDIUM", "LOW"
    detail: str


def classify_failure(
    query: str,
    expected_intent: str,
    pred_intent: str,
    expected_subj: Optional[str],
    pred_subj: Optional[str],
    expected_tgt: Optional[float],
    pred_tgt: Optional[float],
    expected_cnt: Optional[int],
    pred_cnt: Optional[int],
    stage: str,
    raw_output: str,
    is_malformed: bool,
) -> Tuple[str, str, str]:
    """
    Classify failure into root cause category, severity, and explanation.
    """
    if is_malformed:
        return "C: Inference / Generation Artifact", "HIGH", f"Malformed SLM output: {raw_output}"

    # Intent Mismatch
    if expected_intent != pred_intent:
        if expected_intent == "UNKNOWN":
            return "A: SLM Training / Data Gap", "MEDIUM", f"OOD query misclassified as {pred_intent}"
        if expected_intent == "ATTENDANCE_PROJECTION" and pred_intent == "CLASSES_REQUIRED_FOR_TARGET":
            if stage == "RAW_SLM":
                return "A: SLM Training / Data Gap", "HIGH", "Raw SLM confuses projection with classes required"
            else:
                return "E: Deterministic Override Failure", "HIGH", "Projection override pattern missed query"
        if stage == "PIPELINE":
            return "E: Deterministic Override Failure", "HIGH", f"Pipeline intent routing error: {expected_intent} != {pred_intent}"
        return "A: SLM Training / Data Gap", "HIGH", f"Raw SLM predicted {pred_intent} instead of {expected_intent}"

    # Subject Mismatch
    if expected_subj != pred_subj:
        if stage == "PIPELINE":
            return "F: Subject Resolver Failure", "HIGH", f"Resolver could not map subject: expected {expected_subj}, got {pred_subj}"
        return "A: SLM Training / Data Gap", "MEDIUM", f"Raw SLM extracted subject {pred_subj} instead of {expected_subj}"

    # Target Mismatch
    if expected_tgt != pred_tgt:
        if stage == "PIPELINE":
            return "D: Parser / Threshold Safeguard Bug", "MEDIUM", f"Resolved threshold {pred_tgt} != expected {expected_tgt}"
        return "A: SLM Training / Data Gap", "LOW", f"Raw SLM predicted target {pred_tgt} != expected {expected_tgt}"

    # Count Mismatch
    if expected_cnt != pred_cnt:
        if stage == "PIPELINE":
            return "D: Parser / Regex Count Bug", "MEDIUM", f"Extracted count {pred_cnt} != expected {expected_cnt}"
        return "A: SLM Training / Data Gap", "LOW", f"Raw SLM predicted count {pred_cnt} != expected {expected_cnt}"

    return "G: Ambiguous Query", "LOW", "Unknown divergence"


def run_raw_slm_inference(
    engine: NLUInferenceEngine,
    query: str,
    max_new_tokens: int = 40,
) -> Tuple[str, ParsedNLUOutput]:
    """
    Run raw autoregressive generation from the frozen model without deterministic overrides.
    """
    cleaned_query = normalize_query_for_slm(query)
    prompt_ids = engine.tokenizer.encode(cleaned_query, add_bos=True, add_eos=False)
    prompt_ids.append(engine.sep_token_id)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=engine.device)

    generated_ids: list[int] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if input_tensor.shape[1] >= engine.model_config.context_length:
                break
            logits = engine.model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            if next_token_id == EOS_ID:
                break
            generated_ids.append(next_token_id)
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=engine.device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    raw_output = engine.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed = parse_nlu_output(raw_output)
    return raw_output, parsed


def evaluate_acceptance(
    data_path: str | Path = "tests/data/phase11_acceptance.json",
) -> Dict[str, Any]:
    """
    Perform complete Phase 11 acceptance evaluation.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Acceptance data file not found at {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    nlu_engine = get_nlu_engine()
    repo = get_repository()
    resolver = get_subject_resolver(repository=repo)
    router = ToolRouter(repository=repo, subject_resolver=resolver, nlu_engine=nlu_engine)

    total = len(examples)
    print(f"Loaded {total} acceptance test cases from {data_file}.\n")

    # ── Tracking Metrics: Raw SLM ──────────────────────────────────────────────
    raw_intent_correct = 0
    raw_subject_correct = 0
    raw_subject_total = 0
    raw_target_correct = 0
    raw_target_total = 0
    raw_count_correct = 0
    raw_count_total = 0
    raw_exact_match = 0
    raw_malformed = 0
    raw_unknown_correct = 0
    raw_unknown_total = 0

    per_intent_raw: Dict[str, Dict[str, int]] = {}
    per_subj_raw: Dict[str, Dict[str, int]] = {}

    # ── Tracking Metrics: Pipeline ─────────────────────────────────────────────
    pipe_intent_correct = 0
    pipe_subject_correct = 0
    pipe_subject_total = 0
    pipe_target_correct = 0
    pipe_target_total = 0
    pipe_count_correct = 0
    pipe_count_total = 0
    pipe_exact_match = 0
    pipe_success_count = 0
    pipe_unknown_correct = 0
    pipe_unknown_total = 0

    per_intent_pipe: Dict[str, Dict[str, int]] = {}
    per_subj_pipe: Dict[str, Dict[str, int]] = {}

    failures: List[FailureRecord] = []

    for ex in examples:
        query = ex["query"]
        exp_intent = ex["expected_intent"]
        exp_subj = ex.get("expected_subject")
        exp_tgt = ex.get("expected_target")
        exp_cnt = ex.get("expected_count")

        # Initialize intent counters
        if exp_intent not in per_intent_raw:
            per_intent_raw[exp_intent] = {"total": 0, "correct": 0}
            per_intent_pipe[exp_intent] = {"total": 0, "correct": 0}
        per_intent_raw[exp_intent]["total"] += 1
        per_intent_pipe[exp_intent]["total"] += 1

        if exp_subj:
            if exp_subj not in per_subj_raw:
                per_subj_raw[exp_subj] = {"total": 0, "correct": 0}
                per_subj_pipe[exp_subj] = {"total": 0, "correct": 0}
            per_subj_raw[exp_subj]["total"] += 1
            per_subj_pipe[exp_subj]["total"] += 1

        # ── 1. Evaluate Raw SLM ────────────────────────────────────────────────
        raw_str, raw_parsed = run_raw_slm_inference(nlu_engine, query)
        raw_is_malformed = not raw_parsed.is_valid

        if raw_is_malformed:
            raw_malformed += 1

        raw_intent_ok = (raw_parsed.intent == exp_intent)
        if raw_intent_ok:
            raw_intent_correct += 1
            per_intent_raw[exp_intent]["correct"] += 1

        if exp_intent == "UNKNOWN":
            raw_unknown_total += 1
            if raw_parsed.intent == "UNKNOWN":
                raw_unknown_correct += 1

        # Subject check (normalize comparison: raw model might output 'OS', 'ACD', etc.)
        raw_subj_ok = True
        if exp_subj is not None:
            raw_subject_total += 1
            raw_subj_val = raw_parsed.subject.upper() if raw_parsed.subject else None
            raw_subj_ok = (raw_subj_val == exp_subj.upper())
            if raw_subj_ok:
                raw_subject_correct += 1
                per_subj_raw[exp_subj]["correct"] += 1
        elif raw_parsed.subject is not None:
            raw_subj_ok = False

        # Target check
        raw_tgt_ok = True
        if exp_tgt is not None:
            raw_target_total += 1
            raw_tgt_ok = (raw_parsed.target == exp_tgt)
            if raw_tgt_ok:
                raw_target_correct += 1
        elif raw_parsed.target is not None:
            raw_tgt_ok = False

        # Count check
        raw_cnt_ok = True
        if exp_cnt is not None:
            raw_count_total += 1
            raw_cnt_ok = (raw_parsed.count == exp_cnt)
            if raw_cnt_ok:
                raw_count_correct += 1
        elif raw_parsed.count is not None:
            raw_cnt_ok = False

        raw_exact = raw_intent_ok and raw_subj_ok and raw_tgt_ok and raw_cnt_ok and not raw_is_malformed
        if raw_exact:
            raw_exact_match += 1
        else:
            ftype, sev, detail = classify_failure(
                query=query,
                expected_intent=exp_intent,
                pred_intent=raw_parsed.intent,
                expected_subj=exp_subj,
                pred_subj=raw_parsed.subject,
                expected_tgt=exp_tgt,
                pred_tgt=raw_parsed.target,
                expected_cnt=exp_cnt,
                pred_cnt=raw_parsed.count,
                stage="RAW_SLM",
                raw_output=raw_str,
                is_malformed=raw_is_malformed,
            )
            failures.append(
                FailureRecord(
                    query=query,
                    expected_intent=exp_intent,
                    predicted_intent=raw_parsed.intent,
                    expected_subject=exp_subj,
                    predicted_subject=raw_parsed.subject,
                    expected_target=exp_tgt,
                    predicted_target=raw_parsed.target,
                    expected_count=exp_cnt,
                    predicted_count=raw_parsed.count,
                    failure_stage="RAW_SLM",
                    failure_type=ftype,
                    severity=sev,
                    detail=detail,
                )
            )

        # ── 2. Evaluate Final Pipeline ─────────────────────────────────────────
        pipe_resp = router.process(query)
        pipe_nlu = pipe_resp.nlu
        pipe_tool = pipe_resp.tool

        # Pipeline intent
        pipe_intent = pipe_nlu.intent
        pipe_intent_ok = (pipe_intent == exp_intent)
        if pipe_intent_ok:
            pipe_intent_correct += 1
            per_intent_pipe[exp_intent]["correct"] += 1

        if exp_intent == "UNKNOWN":
            pipe_unknown_total += 1
            if pipe_intent == "UNKNOWN":
                pipe_unknown_correct += 1

        # Pipeline subject (resolved)
        pipe_subj_ok = True
        pipe_resolved_subj = None
        if pipe_tool.subject:
            pipe_resolved_subj = pipe_tool.subject.short_name
        elif pipe_nlu.subject:
            # Fallback to resolver
            subj_obj = resolver.resolve(pipe_nlu.subject)
            pipe_resolved_subj = subj_obj.short_name if subj_obj else pipe_nlu.subject

        if exp_subj is not None:
            pipe_subject_total += 1
            pipe_subj_ok = (pipe_resolved_subj is not None and pipe_resolved_subj.upper() == exp_subj.upper())
            if pipe_subj_ok:
                pipe_subject_correct += 1
                per_subj_pipe[exp_subj]["correct"] += 1
        elif pipe_resolved_subj is not None and exp_intent not in ["GET_ATTENDANCE", "HELP", "UNKNOWN"]:
            pipe_subj_ok = False

        # Pipeline target / threshold
        pipe_tgt_ok = True
        pipe_resolved_tgt = None
        if exp_intent in ["CLASSES_REQUIRED_FOR_TARGET", "MAX_CLASSES_CAN_MISS"]:
            pipe_resolved_tgt = resolve_threshold(
                query=query,
                extracted_target=pipe_nlu.target,
                intent=pipe_intent,
                default_threshold=COLLEGE_THRESHOLD,
            )
        else:
            pipe_resolved_tgt = pipe_nlu.target

        if exp_tgt is not None:
            pipe_target_total += 1
            pipe_tgt_ok = (pipe_resolved_tgt == exp_tgt)
            if pipe_tgt_ok:
                pipe_target_correct += 1
        elif pipe_resolved_tgt is not None:
            pipe_tgt_ok = False

        # Pipeline count
        pipe_cnt_ok = True
        pipe_cnt_val = pipe_nlu.count
        if exp_cnt is not None:
            pipe_count_total += 1
            pipe_cnt_ok = (pipe_cnt_val == exp_cnt)
            if pipe_cnt_ok:
                pipe_count_correct += 1
        elif pipe_cnt_val is not None:
            pipe_cnt_ok = False

        # Pipeline success: tool executed without unexpected error (or handled missing subject / unknown / help gracefully)
        pipe_exec_ok = pipe_tool.success or (exp_intent in ["UNKNOWN", "HELP"] and bool(pipe_resp.response_text)) or (exp_subj is None and pipe_tool.error_type == "MISSING_SUBJECT" and bool(pipe_resp.response_text))
        if pipe_exec_ok:
            pipe_success_count += 1


        pipe_exact = pipe_intent_ok and pipe_subj_ok and pipe_tgt_ok and pipe_cnt_ok and pipe_exec_ok
        if pipe_exact:
            pipe_exact_match += 1
        else:
            ftype, sev, detail = classify_failure(
                query=query,
                expected_intent=exp_intent,
                pred_intent=pipe_intent,
                expected_subj=exp_subj,
                pred_subj=pipe_resolved_subj,
                expected_tgt=exp_tgt,
                pred_tgt=pipe_resolved_tgt,
                expected_cnt=exp_cnt,
                pred_cnt=pipe_cnt_val,
                stage="PIPELINE",
                raw_output=pipe_nlu.raw_output,
                is_malformed=False,
            )
            failures.append(
                FailureRecord(
                    query=query,
                    expected_intent=exp_intent,
                    predicted_intent=pipe_intent,
                    expected_subject=exp_subj,
                    predicted_subject=pipe_resolved_subj,
                    expected_target=exp_tgt,
                    predicted_target=pipe_resolved_tgt,
                    expected_count=exp_cnt,
                    predicted_count=pipe_cnt_val,
                    failure_stage="PIPELINE",
                    failure_type=ftype,
                    severity=sev,
                    detail=detail,
                )
            )

    # ── Report Summary Construction ───────────────────────────────────────────
    summary = {
        "total_queries": total,
        "raw_slm": {
            "intent_accuracy": round(raw_intent_correct / total * 100, 2),
            "intent_correct": raw_intent_correct,
            "subject_accuracy": round(raw_subject_correct / raw_subject_total * 100, 2) if raw_subject_total else 100.0,
            "subject_correct": raw_subject_correct,
            "subject_total": raw_subject_total,
            "target_accuracy": round(raw_target_correct / raw_target_total * 100, 2) if raw_target_total else 100.0,
            "target_correct": raw_target_correct,
            "target_total": raw_target_total,
            "count_accuracy": round(raw_count_correct / raw_count_total * 100, 2) if raw_count_total else 100.0,
            "count_correct": raw_count_correct,
            "count_total": raw_count_total,
            "exact_structured_match": round(raw_exact_match / total * 100, 2),
            "exact_structured_match_count": raw_exact_match,
            "malformed_rate": round(raw_malformed / total * 100, 2),
            "malformed_count": raw_malformed,
            "unknown_detection_accuracy": round(raw_unknown_correct / raw_unknown_total * 100, 2) if raw_unknown_total else 100.0,
            "unknown_correct": raw_unknown_correct,
            "unknown_total": raw_unknown_total,
            "per_intent": {
                intent: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] else 100.0,
                }
                for intent, stats in per_intent_raw.items()
            },
            "per_subject": {
                subj: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] else 100.0,
                }
                for subj, stats in per_subj_raw.items()
            },
        },
        "pipeline": {
            "intent_accuracy": round(pipe_intent_correct / total * 100, 2),
            "intent_correct": pipe_intent_correct,
            "subject_accuracy": round(pipe_subject_correct / pipe_subject_total * 100, 2) if pipe_subject_total else 100.0,
            "subject_correct": pipe_subject_correct,
            "subject_total": pipe_subject_total,
            "target_accuracy": round(pipe_target_correct / pipe_target_total * 100, 2) if pipe_target_total else 100.0,
            "target_correct": pipe_target_correct,
            "target_total": pipe_target_total,
            "count_accuracy": round(pipe_count_correct / pipe_count_total * 100, 2) if pipe_count_total else 100.0,
            "count_correct": pipe_count_correct,
            "count_total": pipe_count_total,
            "exact_pipeline_match": round(pipe_exact_match / total * 100, 2),
            "exact_pipeline_match_count": pipe_exact_match,
            "success_rate": round(pipe_success_count / total * 100, 2),
            "success_count": pipe_success_count,
            "unknown_detection_accuracy": round(pipe_unknown_correct / pipe_unknown_total * 100, 2) if pipe_unknown_total else 100.0,
            "unknown_correct": pipe_unknown_correct,
            "unknown_total": pipe_unknown_total,
            "per_intent": {
                intent: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] else 100.0,
                }
                for intent, stats in per_intent_pipe.items()
            },
            "per_subject": {
                subj: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] else 100.0,
                }
                for subj, stats in per_subj_pipe.items()
            },
        },
        "failures": [asdict(f) for f in failures],
    }

    report_path = Path("tests/data/acceptance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    print("PHASE 11 ACCEPTANCE EVALUATION REPORT")
    print("=" * 64)
    print(f"Total Test Queries: {total}\n")

    print("--- A. RAW SLM QUALITY (Without Application Layer Overrides) ---")
    print(f"Intent Accuracy:            {summary['raw_slm']['intent_accuracy']}% ({raw_intent_correct}/{total})")
    print(f"Subject Accuracy:           {summary['raw_slm']['subject_accuracy']}% ({raw_subject_correct}/{raw_subject_total})")
    print(f"Target Accuracy:            {summary['raw_slm']['target_accuracy']}% ({raw_target_correct}/{raw_target_total})")
    print(f"Count Accuracy:             {summary['raw_slm']['count_accuracy']}% ({raw_count_correct}/{raw_count_total})")
    print(f"Exact Structured Match:     {summary['raw_slm']['exact_structured_match']}% ({raw_exact_match}/{total})")
    print(f"Malformed Output Rate:      {summary['raw_slm']['malformed_rate']}% ({raw_malformed}/{total})")
    print(f"Unknown Detection Accuracy: {summary['raw_slm']['unknown_detection_accuracy']}% ({raw_unknown_correct}/{raw_unknown_total})\n")

    print("Per-Intent Raw Accuracy:")
    for intent, stats in summary["raw_slm"]["per_intent"].items():
        print(f"  - {intent:30s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\nPer-Subject Raw Accuracy:")
    for subj, stats in summary["raw_slm"]["per_subject"].items():
        print(f"  - {subj:15s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\n--- B. FINAL APPLICATION QUALITY (End-to-End Pipeline) ---------")

    print(f"Pipeline Intent Accuracy:   {summary['pipeline']['intent_accuracy']}% ({pipe_intent_correct}/{total})")
    print(f"Pipeline Subject Accuracy:  {summary['pipeline']['subject_accuracy']}% ({pipe_subject_correct}/{pipe_subject_total})")
    print(f"Pipeline Target Accuracy:   {summary['pipeline']['target_accuracy']}% ({pipe_target_correct}/{pipe_target_total})")
    print(f"Pipeline Count Accuracy:    {summary['pipeline']['count_accuracy']}% ({pipe_count_correct}/{pipe_count_total})")
    print(f"Pipeline Exact Match:       {summary['pipeline']['exact_pipeline_match']}% ({pipe_exact_match}/{total})")
    print(f"Pipeline Execution Success: {summary['pipeline']['success_rate']}% ({pipe_success_count}/{total})\n")

    print("Per-Intent Pipeline Accuracy:")
    for intent, stats in summary["pipeline"]["per_intent"].items():
        print(f"  - {intent:30s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    print("\nPer-Subject Pipeline Resolution Accuracy:")
    for subj, stats in summary["pipeline"]["per_subject"].items():
        print(f"  - {subj:15s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")

    pipe_failures = [f for f in failures if f.failure_stage == "PIPELINE"]
    print(f"\nPipeline Failures: {len(pipe_failures)}")
    for f in pipe_failures:
        print(f"  [FAIL] {f.query} -> Expected: {f.expected_intent}|{f.expected_subject}|{f.expected_target}|{f.expected_count} | Got: {f.predicted_intent}|{f.predicted_subject}|{f.predicted_target}|{f.predicted_count} | Type: {f.failure_type} | Detail: {f.detail}")

    return summary


if __name__ == "__main__":
    evaluate_acceptance()
