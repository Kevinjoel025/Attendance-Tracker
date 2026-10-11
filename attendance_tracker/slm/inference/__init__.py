"""
slm/inference/__init__.py
─────────────────────────
Inference and parsing package for the College Attendance Tracker SLM.
"""

from attendance_tracker.slm.inference.nlu import (
    NLUInferenceEngine,
    NLUResult,
    get_nlu_engine,
)
from attendance_tracker.slm.inference.parser import (
    VALID_INTENTS,
    ParsedNLUOutput,
    parse_nlu_output,
)

__all__ = [
    "NLUInferenceEngine",
    "NLUResult",
    "get_nlu_engine",
    "ParsedNLUOutput",
    "parse_nlu_output",
    "VALID_INTENTS",
]
