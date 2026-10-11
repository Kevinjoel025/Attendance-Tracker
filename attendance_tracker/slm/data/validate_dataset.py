"""
slm/data/validate_dataset.py
────────────────────────────
Dataset validation and quality audit suite for synthetic NLU training data.

Performs:
1. JSONL syntax and structure validation
2. Canonical intent and slot format checks
3. Strict slot combination consistency rules
4. Duplicate text detection
5. Cross-split template family and text leakage checks
6. Comprehensive intent and entity distribution reporting
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.slm.data.generator import CANONICAL_INTENTS, SUBJECT_CANONICAL

logger = get_logger(__name__)


# ── Report Dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    total_examples: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    intent_distribution: dict[str, int] = field(default_factory=dict)
    subject_distribution: dict[str, int] = field(default_factory=dict)
    slot_counts: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    exact_leakage_count: int = 0
    normalized_leakage_count: int = 0
    template_family_leakage_count: int = 0
    invalid_record_count: int = 0
    errors: list[str] = field(default_factory=list)
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "NLU DATASET VALIDATION REPORT",
            "=" * 60,
            f"Status:                      {'PASSED (VALID)' if self.is_valid else 'FAILED (INVALID)'}",
            f"Total Examples:              {self.total_examples}",
            f"Split Counts:                {self.split_counts}",
            f"Duplicate Count:             {self.duplicate_count}",
            f"Exact Leakage Count:         {self.exact_leakage_count}",
            f"Normalized Leakage Count:    {self.normalized_leakage_count}",
            f"Template Leakage Count:      {self.template_family_leakage_count}",
            f"Invalid Records:             {self.invalid_record_count}",
            "-" * 60,
            "INTENT DISTRIBUTION:",
        ]
        for intent, count in sorted(self.intent_distribution.items()):
            pct = (count / self.total_examples * 100) if self.total_examples > 0 else 0
            lines.append(f"  - {intent:<30}: {count:>5} ({pct:>5.1f}%)")

        lines.append("-" * 60)
        lines.append("SUBJECT DISTRIBUTION:")
        for subj, count in sorted(self.subject_distribution.items()):
            pct = (count / self.total_examples * 100) if self.total_examples > 0 else 0
            lines.append(f"  - {subj:<30}: {count:>5} ({pct:>5.1f}%)")

        lines.append("-" * 60)
        lines.append("SLOT PRESENCE:")
        for slot, count in sorted(self.slot_counts.items()):
            pct = (count / self.total_examples * 100) if self.total_examples > 0 else 0
            lines.append(f"  - {slot:<30}: {count:>5} ({pct:>5.1f}%)")

        if self.errors:
            lines.append("-" * 60)
            lines.append("ERRORS (First 10):")
            for err in self.errors[:10]:
                lines.append(f"  [ERROR] {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Validator Class ───────────────────────────────────────────────────────────

class NLUDataValidator:
    """
    Validates synthetic NLU dataset records and audits cross-split leakage.
    """

    def _normalize_text(self, text: str) -> str:
        """Lowercase and remove excess whitespace/punctuation for fuzzy duplicate check."""
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t)

    def validate_record(self, record: dict[str, Any], record_idx: int) -> list[str]:
        """
        Validate a single dataset record. Returns a list of error messages.
        """
        errors = []

        # 1. Required fields
        required_fields = ["text", "intent", "subject", "target", "count", "date", "output"]
        for f in required_fields:
            if f not in record:
                errors.append(f"Record {record_idx}: missing required field '{f}'")

        if errors:
            return errors

        text = str(record["text"]).strip()
        intent = str(record["intent"]).strip()
        subject = record["subject"]
        target = record["target"]
        count = record["count"]
        date_val = record["date"]
        output = str(record["output"]).strip()

        # 2. Text checks
        if not text:
            errors.append(f"Record {record_idx}: empty text")

        # 3. Intent checks
        if intent not in CANONICAL_INTENTS:
            errors.append(f"Record {record_idx}: invalid intent '{intent}'")

        # 4. Subject checks
        if subject is not None and subject not in SUBJECT_CANONICAL and subject != "NONE":
            errors.append(f"Record {record_idx}: invalid subject '{subject}'")

        # 5. Output format checks: INTENT|SUBJECT|TARGET|COUNT|DATE
        parts = output.split("|")
        if len(parts) != 5:
            errors.append(f"Record {record_idx}: output format must have 5 pipe-separated fields, got '{output}'")
        else:
            out_intent, out_subj, out_target, out_count, out_date = parts
            if out_intent != intent:
                errors.append(f"Record {record_idx}: output intent '{out_intent}' != record intent '{intent}'")
            expected_subj = subject if subject else "NONE"
            if out_subj != expected_subj:
                errors.append(f"Record {record_idx}: output subject '{out_subj}' != expected '{expected_subj}'")
            expected_target = str(target) if target is not None else "NONE"
            if out_target != expected_target:
                errors.append(f"Record {record_idx}: output target '{out_target}' != expected '{expected_target}'")
            expected_count = str(count) if count is not None else "NONE"
            if out_count != expected_count:
                errors.append(f"Record {record_idx}: output count '{out_count}' != expected '{expected_count}'")
            expected_date = str(date_val) if date_val is not None else "NONE"
            if out_date != expected_date:
                errors.append(f"Record {record_idx}: output date '{out_date}' != expected '{expected_date}'")

        # 6. Intent-specific slot rules
        if intent == "GET_ATTENDANCE":
            if subject is not None:
                errors.append(f"Record {record_idx}: GET_ATTENDANCE should not have a subject slot ('{subject}')")
        elif intent == "GET_SUBJECT_ATTENDANCE":
            if subject is None or subject == "NONE":
                errors.append(f"Record {record_idx}: GET_SUBJECT_ATTENDANCE must have a valid subject slot")
        elif intent == "ATTENDANCE_PROJECTION":
            if count is None or count == "NONE":
                errors.append(f"Record {record_idx}: ATTENDANCE_PROJECTION must have a count slot")
        elif intent in ("HELP", "UNKNOWN"):
            if subject is not None or target is not None or count is not None or date_val is not None:
                errors.append(f"Record {record_idx}: {intent} must have all slots as None")

        return errors

    def validate_splits(
        self,
        splits: dict[str, list[dict[str, Any]]],
    ) -> ValidationReport:
        """
        Validate full multi-split dataset and perform cross-split leakage checks.
        """
        report = ValidationReport()

        seen_exact_by_split: dict[str, set[str]] = {}
        seen_norm_by_split: dict[str, set[str]] = {}
        families_by_split: dict[str, set[str]] = {}

        global_exact_texts: dict[str, str] = {}
        global_norm_texts: dict[str, str] = {}

        total_records = 0

        for split_name, records in splits.items():
            report.split_counts[split_name] = len(records)
            seen_exact_by_split[split_name] = set()
            seen_norm_by_split[split_name] = set()
            families_by_split[split_name] = set()

            for idx, rec in enumerate(records):
                total_records += 1
                rec_errors = self.validate_record(rec, idx + 1)
                if rec_errors:
                    report.invalid_record_count += 1
                    report.errors.extend(rec_errors)

                text = rec.get("text", "")
                norm_text = self._normalize_text(text)
                intent = rec.get("intent", "UNKNOWN")
                subj = rec.get("subject")
                target = rec.get("target")
                count = rec.get("count")
                date_val = rec.get("date")
                family = rec.get("template_family")

                # Distribution tracking
                report.intent_distribution[intent] = report.intent_distribution.get(intent, 0) + 1
                if subj and subj != "NONE":
                    report.subject_distribution[subj] = report.subject_distribution.get(subj, 0) + 1

                if target and target != "NONE":
                    report.slot_counts["target_slot"] = report.slot_counts.get("target_slot", 0) + 1
                if count and count != "NONE":
                    report.slot_counts["count_slot"] = report.slot_counts.get("count_slot", 0) + 1
                if date_val and date_val != "NONE":
                    report.slot_counts["date_slot"] = report.slot_counts.get("date_slot", 0) + 1
                if not any([subj, target, count, date_val]):
                    report.slot_counts["no_slots"] = report.slot_counts.get("no_slots", 0) + 1

                if family:
                    families_by_split[split_name].add(family)

                # Within-split duplicate detection
                if text in seen_exact_by_split[split_name]:
                    report.duplicate_count += 1
                    report.errors.append(f"Duplicate text within {split_name}: '{text}'")
                seen_exact_by_split[split_name].add(text)
                seen_norm_by_split[split_name].add(norm_text)

                # Cross-split duplicate / leakage check
                if text in global_exact_texts and global_exact_texts[text] != split_name:
                    report.exact_leakage_count += 1
                    report.errors.append(
                        f"Exact cross-split leakage: '{text}' in both {global_exact_texts[text]} and {split_name}"
                    )
                else:
                    global_exact_texts[text] = split_name

                if norm_text in global_norm_texts and global_norm_texts[norm_text] != split_name:
                    report.normalized_leakage_count += 1
                    report.errors.append(
                        f"Normalized cross-split leakage: '{text}' in both {global_norm_texts[norm_text]} and {split_name}"
                    )
                else:
                    global_norm_texts[norm_text] = split_name

        # Template family leakage check across splits
        split_names = list(splits.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = families_by_split[s1].intersection(families_by_split[s2])
                if overlap:
                    report.template_family_leakage_count += len(overlap)
                    report.errors.append(f"Template family leakage between {s1} and {s2}: {overlap}")

        report.total_examples = total_records
        report.is_valid = (
            report.invalid_record_count == 0
            and report.exact_leakage_count == 0
            and report.normalized_leakage_count == 0
            and report.template_family_leakage_count == 0
            and report.duplicate_count == 0
        )

        return report

    def validate_jsonl_files(self, data_dir: Path | str) -> ValidationReport:
        """
        Load and validate train.jsonl, val.jsonl, and test.jsonl from directory.
        """
        dir_path = Path(data_dir)
        splits: dict[str, list[dict[str, Any]]] = {}

        for split_name in ["train", "val", "test"]:
            filepath = dir_path / f"{split_name}.jsonl"
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required dataset file: {filepath}")

            records = []
            with filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        records.append(json.loads(line_str))
            splits[split_name] = records

        return self.validate_splits(splits)


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic NLU training datasets.")
    parser.add_argument("--data-dir", type=str, default="data/nlu", help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    args = parser.parse_args()

    validator = NLUDataValidator()
    report = validator.validate_jsonl_files(args.data_dir)
    print(report.summary())

    if not report.is_valid:
        exit(1)


if __name__ == "__main__":
    main()
