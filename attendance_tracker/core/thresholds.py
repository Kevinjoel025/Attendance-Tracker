"""
core/thresholds.py
──────────────────
Authoritative attendance threshold and target resolution service.

Rules:
1. Explicit numeric percentage in user query takes highest precedence (e.g. "80%", "at 85").
2. Personal target language ("my target", "my goal", "personal target") resolves to personal_target (70.0%).
3. College requirement language ("college requirement", "college threshold", "minimum attendance") resolves to college_threshold (75.0%).
4. If no explicit percentage and no target language are present in the query, falls back to the operation default:
   - MAX_CLASSES_CAN_MISS: college_threshold (75.0%)
   - CLASSES_REQUIRED_FOR_TARGET: college_threshold (75.0%)
"""

from __future__ import annotations

import re
from typing import Optional

from attendance_tracker.core.config import config

COLLEGE_THRESHOLD: float = config.attendance.college_threshold  # 75.0
PERSONAL_TARGET: float = config.attendance.personal_target      # 70.0
DEFAULT_THRESHOLD: float = config.attendance.default_threshold  # 75.0

# Regex patterns for explicit percentages in query text
_PERCENT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent\b)', re.IGNORECASE)
_EXPLICIT_NUMBER_PATTERN = re.compile(
    r'\b(?:at|reach|to|above|maintain|keep|stay\s+at|target\s+of|threshold\s+of|for)\s+(\d+(?:\.\d+)?)\b',
    re.IGNORECASE,
)


# Regex patterns for personal target language
_PERSONAL_TARGET_PATTERN = re.compile(
    r'\b(?:my\s+target|my\s+goal|my\s+personal\s+target|personal\s+target|personal\s+goal|stay\s+above\s+my\s+target|stay\s+at\s+my\s+target|reach\s+my\s+target|reach\s+my\s+goal|hit\s+my\s+target|for\s+my\s+target|at\s+my\s+target|to\s+my\s+target)\b',
    re.IGNORECASE,
)

# Regex patterns for college requirement language
_COLLEGE_REQUIREMENT_PATTERN = re.compile(
    r'\b(?:college\s+requirement|college\s+threshold|required\s+attendance|minimum\s+attendance|required\s+percentage|college\s+minimum|college\s+standard|official\s+threshold|for\s+the\s+college|for\s+college\s+requirement|meet\s+the\s+college|meet\s+college|for\s+the\s+college\s+requirement|college)\b',
    re.IGNORECASE,
)


def extract_explicit_percentage(query: str) -> Optional[float]:
    """
    Extract an explicit percentage number from query text if present.
    """
    if not query:
        return None

    # 1. Look for explicit "%" notation e.g. "80%", "75 %"
    m_pct = _PERCENT_PATTERN.search(query)
    if m_pct:
        try:
            val = float(m_pct.group(1))
            if 0.0 <= val <= 100.0:
                return val
        except ValueError:
            pass

    # 2. Look for "reach 85", "at 80", "target of 80", etc.
    m_num = _EXPLICIT_NUMBER_PATTERN.search(query)
    if m_num:
        try:
            val = float(m_num.group(1))
            if 0.0 <= val <= 100.0:
                return val
        except ValueError:
            pass

    return None


def resolve_threshold(
    query: str = "",
    extracted_target: Optional[float] = None,
    intent: Optional[str] = None,
    default_threshold: Optional[float] = None,
) -> float:
    """
    Resolve the authoritative threshold or target percentage for an operation.

    Parameters
    ----------
    query : str
        The raw natural language query text from the user.
    extracted_target : Optional[float]
        The target slot extracted by the NLU model (if any).
    intent : Optional[str]
        The intent (e.g. "MAX_CLASSES_CAN_MISS", "CLASSES_REQUIRED_FOR_TARGET").
    default_threshold : Optional[float]
        Custom default threshold (e.g. subject-specific threshold), defaults to COLLEGE_THRESHOLD (75.0).

    Returns
    -------
    float
        The resolved target/threshold percentage (e.g. 75.0, 70.0, 80.0).
    """
    q_clean = query.strip() if query else ""

    # Rule A: Explicit percentage in user query
    explicit_val = extract_explicit_percentage(q_clean)
    if explicit_val is not None:
        return explicit_val

    # If extracted_target was provided, check if its numeric value was actually written in the query
    if extracted_target is not None:
        tgt_str = str(int(extracted_target)) if extracted_target.is_integer() else str(extracted_target)
        if re.search(r'\b' + re.escape(tgt_str) + r'\b', q_clean):
            return extracted_target

    # Rule B: Personal target language
    if q_clean and _PERSONAL_TARGET_PATTERN.search(q_clean):
        return PERSONAL_TARGET

    # Rule C: College requirement language
    if q_clean and _COLLEGE_REQUIREMENT_PATTERN.search(q_clean):
        return COLLEGE_THRESHOLD

    # Rule D: Default for operation
    fallback = default_threshold if default_threshold is not None else COLLEGE_THRESHOLD
    return fallback
