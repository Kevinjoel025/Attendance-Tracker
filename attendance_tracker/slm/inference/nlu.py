"""
slm/inference/nlu.py
────────────────────
Runtime NLU inference engine for the College Attendance Tracker SLM.

Features:
- Loads the frozen Phase 7.6 decoder-only transformer checkpoint.
- Tokenizes user query with scratch BPE tokenizer using [BOS] + QUERY + [SEP].
- Performs autoregressive greedy generation stopping at [EOS] or max token limit.
- Computes exact intent confidence and top-k intent distribution from the first token logits.
- Parses generated completion into structured NLUResult.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from attendance_tracker.core.config import config as app_config
from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.core.thresholds import extract_explicit_percentage
from attendance_tracker.slm.inference.parser import ParsedNLUOutput, parse_nlu_output

from attendance_tracker.slm.model.config import ModelConfig
from attendance_tracker.slm.model.model import AttendanceSLM
from attendance_tracker.slm.tokenizer.bpe_tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    BPETokenizer,
)

logger = get_logger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/phase7_6_boundary/best.pt")
DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/bpe.json")
DEFAULT_SEP_TOKEN_ID = 1098  # <unused_0>

INTENT_TOKEN_MAP = {
    "GET_ATTENDANCE": 223,
    "GET_SUBJECT_ATTENDANCE": 226,
    "ATTENDANCE_PROJECTION": 190,
    "CLASSES_REQUIRED_FOR_TARGET": 219,
    "MAX_CLASSES_CAN_MISS": 231,
    "CHECK_ATTENDANCE_UPDATE": 205,
    "HELP": 227,
    "UNKNOWN": 233,
}


@dataclass(frozen=True)
class NLUResult:
    """Complete structured output from NLU inference."""

    raw_output: str
    intent: str
    subject: Optional[str]
    target: Optional[float]
    count: Optional[int]
    date: Optional[str]
    is_valid: bool
    malformed_reason: Optional[str] = None
    confidence: float = 0.0
    top_intents: list[tuple[str, float]] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "intent": self.intent,
            "subject": self.subject,
            "target": self.target,
            "count": self.count,
            "date": self.date,
            "is_valid": self.is_valid,
            "malformed_reason": self.malformed_reason,
            "confidence": round(self.confidence, 4),
            "top_intents": [(name, round(prob, 4)) for name, prob in self.top_intents],
            "inference_time_ms": round(self.inference_time_ms, 2),
        }


_CANONICAL_ACRONYMS = {
    "os": "OS",
    "ops": "OS",
    "acd": "ACD",
    "de": "DE",
    "wp": "WP",
    "gs": "GS",
    "oe": "OE-1",
    "oe1": "OE-1",
    "oe-1": "OE-1",
    "iotap": "IoTAP",
    "iot": "IoT",
    "lab": "LAB",
    "gender": "GS",
    "storytelling": "OE-1",
    "compiler": "ACD",
}


def normalize_query_for_slm(query: str) -> str:
    """Normalize whitespace, duplicate punctuation, and word casing for SLM tokenizer."""
    cleaned = re.sub(r"\s+", " ", query.strip())
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    cleaned = re.sub(r"!{2,}", "!", cleaned)

    if not cleaned:
        return cleaned

    if cleaned.isupper():
        cleaned = cleaned.lower()

    tokens = cleaned.split(" ")
    normalized_tokens = []
    for i, tok in enumerate(tokens):
        m = re.match(r"^([a-zA-Z0-9_\-]+)([\?\!\.\,]*)$", tok)
        if m:
            word, punct = m.group(1), m.group(2)
            word_lower = word.lower()
            if word_lower in _CANONICAL_ACRONYMS:
                normalized_tokens.append(_CANONICAL_ACRONYMS[word_lower] + punct)
            elif i > 0 and word.istitle() and not word.isupper():
                normalized_tokens.append(word.lower() + punct)
            else:
                normalized_tokens.append(tok)
        else:
            normalized_tokens.append(tok)

    return " ".join(normalized_tokens)


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a": 1,
    "an": 1,
    "single": 1,
}

_PROJECTION_INTENT_PATTERNS = [
    re.compile(r'\bwhat\s+happens\s+if\s+i\s+(?:attend|go\s+to|miss|skip|bunk)\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+(?:will|would)\s+(?:my\s+)?attendance\s+(?:be|become|look\s+like)\b', re.IGNORECASE),
    re.compile(r'\bhow\s+(?:will|would)\s+(?:my\s+)?attendance\s+(?:change|suffer|improve|be\s+affected|increase|decrease)\b', re.IGNORECASE),
    re.compile(r'\bif\s+i\s+(?:attend|go\s+to|miss|skip|bunk)\b', re.IGNORECASE),
    re.compile(r'\b(?:the\s+)?next\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)?\s*class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\bafter\s+(?:attending|going\s+to|missing|skipping|bunking)\b', re.IGNORECASE),
    re.compile(r'\b(?:attending|attend|going\s+to|go\s+to|missing|miss|skipping|skip|bunking|bunk)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:more\s+)?class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\babsent\s+(?:myself\s+)?(?:from)?\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)?\s*class\b', re.IGNORECASE),
    re.compile(r'\bproject(?:ed|ion)?\s+attendance\b', re.IGNORECASE),
]

_COUNT_PATTERNS = [
    re.compile(r'\b(?:the\s+)?next\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)\s+class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+more\s+class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\b(?:attend(?:ing)?|go(?:ing)?\s+to|miss(?:ing)?|skip(?:ping)?|bunk(?:ing)?|absent\s+(?:myself\s+)?(?:from)?|from|for|to)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)\s+class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\bafter\s+(?:attending|going\s+to|missing|skipping|bunking)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)\s+class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[a-zA-Z\-]+\s+)?class(?:es)?\b', re.IGNORECASE),
    re.compile(r'\b(?:the\s+)?next\s+class\b', re.IGNORECASE),
]

_KNOWN_SUBJECT_ALIASES = [
    ("storytelling for effective communication", "OE-1"),
    ("storytelling communication", "OE-1"),
    ("storytelling comms", "OE-1"),
    ("storytelling", "OE-1"),
    ("gender sensitization", "GS"),
    ("gender", "GS"),
    ("data engineering laboratory", "DE LAB"),
    ("data engineering lab", "DE LAB"),
    ("data engineering", "DE"),
    ("data eng lab", "DE LAB"),
    ("data eng", "DE"),
    ("automata and compiler design", "ACD"),
    ("compiler design", "ACD"),
    ("compiler", "ACD"),
    ("automata", "ACD"),
    ("iot architecture and protocols laboratory", "IoTAP LAB"),
    ("iot architecture and protocols lab", "IoTAP LAB"),
    ("iot architecture and protocols", "IoTAP"),
    ("iot architecture lab", "IoTAP LAB"),
    ("iot architecture", "IoTAP"),
    ("iotap lab", "IoTAP LAB"),
    ("iot lab", "IoTAP LAB"),
    ("iotap", "IoTAP"),
    ("iot", "IoTAP"),
    ("operating systems", "OS"),
    ("operating system", "OS"),
    ("ops", "OS"),
    ("web programming", "WP"),
    ("web prog", "WP"),
    ("web", "WP"),
    ("de lab", "DE LAB"),
    ("acd", "ACD"),
    ("oe-1", "OE-1"),
    ("oe1", "OE-1"),
    ("oe", "OE-1"),
    ("os", "OS"),
    ("de", "DE"),
    ("wp", "WP"),
    ("gs", "GS"),
]


def is_projection_language(query: str) -> bool:
    """Check if query contains explicit attendance projection phrasing."""
    if not query:
        return False
    if re.search(r'\bhow\s+many\s+(?:more\s+)?classes\s+(?:must|do|should|need|required|to\s+reach)\b', query, re.IGNORECASE):
        return False
    return any(p.search(query) for p in _PROJECTION_INTENT_PATTERNS)


def extract_projection_count(query: str) -> Optional[int]:
    """Extract integer count from projection phrase in query."""
    if not query:
        return None
    for p in _COUNT_PATTERNS:
        m = p.search(query)
        if m:
            if m.groups() and m.group(1):
                raw = m.group(1).lower()
                if raw.isdigit():
                    return int(raw)
                if raw in _NUMBER_WORDS:
                    return _NUMBER_WORDS[raw]
            else:
                return 1
    return None


def extract_subject_from_text(query: str) -> Optional[str]:
    """Detect subject code from query text."""
    if not query:
        return None
    q_lower = " " + query.lower() + " "
    q_clean = re.sub(r'[^a-zA-Z0-9\-]', ' ', q_lower)
    q_spaced = " " + " ".join(q_clean.split()) + " "

    for alias, code in _KNOWN_SUBJECT_ALIASES:
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, q_spaced, re.IGNORECASE):
            return code
    return None



class NLUInferenceEngine:
    """
    Inference service wrapping AttendanceSLM and BPETokenizer.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        tokenizer_path: Optional[Path | str] = None,
        device: Optional[str | torch.device] = None,
        sep_token_id: int = DEFAULT_SEP_TOKEN_ID,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.tokenizer_path = Path(tokenizer_path or DEFAULT_TOKENIZER_PATH)
        self.sep_token_id = sep_token_id

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load Tokenizer
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
        self.tokenizer = BPETokenizer.load(self.tokenizer_path)

        # Load Model
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SLM checkpoint not found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg_dict = checkpoint.get("config", {})
        self.model_config = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()
        self.model = AttendanceSLM(self.model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Cache intent token IDs for fast confidence calculation
        self._intent_token_ids = {}
        for intent_name, token_id in INTENT_TOKEN_MAP.items():
            self._intent_token_ids[intent_name] = token_id

        logger.info(
            f"NLUInferenceEngine initialized successfully from {self.checkpoint_path} "
            f"on device {self.device}."
        )

    @torch.inference_mode()
    def predict(
        self,
        query: str,
        max_new_tokens: int = 40,
    ) -> NLUResult:
        """
        Run NLU inference on a user query string.

        Parameters
        ----------
        query : str
            Natural language user request.
        max_new_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        NLUResult
            Structured extraction result with intent, slots, and confidence.
        """
        start_time = time.perf_counter()

        # Encode input prompt: [BOS] + query_tokens + [SEP]
        cleaned_query = normalize_query_for_slm(query)

        prompt_ids = self.tokenizer.encode(cleaned_query, add_bos=True, add_eos=False)
        prompt_ids.append(self.sep_token_id)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_ids: list[int] = []
        confidence = 0.0
        top_intents: list[tuple[str, float]] = []

        # Step 1: First token generation (intent token) & confidence calculation
        logits = self.model(input_tensor)
        first_step_logits = logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(first_step_logits, dim=-1)

        first_token_id = int(torch.argmax(first_step_logits, dim=-1).item())
        confidence = float(probs[first_token_id].item())

        # Compute intent probabilities
        intent_probs: list[tuple[str, float]] = []
        for intent_name, tid in self._intent_token_ids.items():
            if tid < probs.shape[0]:
                p = float(probs[tid].item())
                intent_probs.append((intent_name, p))
        intent_probs.sort(key=lambda x: x[1], reverse=True)
        top_intents = intent_probs

        if first_token_id != EOS_ID:
            generated_ids.append(first_token_id)
            next_tensor = torch.tensor([[first_token_id]], dtype=torch.long, device=self.device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

            # Step 2: Autoregressive decoding for remainder of sequence
            for _ in range(max_new_tokens - 1):
                if input_tensor.shape[1] >= self.model_config.context_length:
                    break

                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())

                if next_token_id == EOS_ID:
                    break

                generated_ids.append(next_token_id)
                next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse structured output
        parsed: ParsedNLUOutput = parse_nlu_output(raw_output)

        intent = parsed.intent
        subject = parsed.subject
        target = parsed.target
        count = parsed.count

        # Reconcile explicit query patterns deterministically at inference boundary
        text_subject = extract_subject_from_text(query)
        q_lower = query.lower().strip()
        q_clean_words = re.sub(r'[^\w\s\-]', ' ', q_lower).strip()
        q_norm_phrase = " ".join(q_clean_words.split())

        # 1. Projection language
        if is_projection_language(query):
            intent = "ATTENDANCE_PROJECTION"
            proj_count = extract_projection_count(query)
            if proj_count is not None:
                count = proj_count
            elif count is None:
                count = 1
            target = None
            subject = text_subject

        # 2. Check attendance update language
        elif re.search(r'\b(?:did\s+(?:my\s+|the\s+)?attendance\s+(?:change|update|updates)|has\s+(?:my\s+|new\s+)?attendance\s+been\s+(?:updated|posted)|check\s+(?:if\s+)?attendance\s+(?:was\s+)?updated|check\s+attendance\s+updates?|was\s+attendance\s+updated|any\s+new\s+attendance\s+updates?|did\s+the\s+teacher\s+mark\s+attendance|attendance\s+updates?|attendance\s+changes?)\b', query, re.IGNORECASE):
            intent = "CHECK_ATTENDANCE_UPDATE"
            subject = text_subject
            target = None
            count = None


        # 3. Bunk margin / Max classes can miss language
        elif re.search(r'\b(?:bunk\s+margin|can\s+i\s+bunk|how\s+many\s+(?:can\s+i\s+bunk|classes\s+can\s+i\s+(?:miss|skip|bunk|forgo)))\b', query, re.IGNORECASE):
            intent = "MAX_CLASSES_CAN_MISS"
            if text_subject:
                subject = text_subject
            explicit_pct = extract_explicit_percentage(query)
            if explicit_pct is not None:
                target = explicit_pct

        # 4. Classes required language
        elif re.search(r'\bhow\s+many\s+(?:more\s+)?classes\s+(?:must|do|should|need|needed|require|required|to\s+reach|for)\b', query, re.IGNORECASE):
            intent = "CLASSES_REQUIRED_FOR_TARGET"
            if text_subject:
                subject = text_subject
            explicit_pct = extract_explicit_percentage(query)
            if explicit_pct is not None:
                target = explicit_pct

        # 5. Direct Subject Query / Subject Attendance Phrasing
        elif text_subject and (
            q_norm_phrase in [a.lower() for a, _ in _KNOWN_SUBJECT_ALIASES]
            or (re.search(r'\b(?:attendance|percent(?:age)?|status|how\s+much\s+(?:have\s+i\s+)?attended)\b', query, re.IGNORECASE) and not re.search(r'\bhow\s+many\b', query, re.IGNORECASE))
        ):
            intent = "GET_SUBJECT_ATTENDANCE"
            subject = text_subject
            target = None
            count = None

        # 6. Overall attendance query without subject
        elif text_subject is None and re.search(r'\b(?:what\s+is\s+my\s+attendance|overall\s+attendance|total\s+attendance|attendance\s+summary|how\s+is\s+my\s+attendance\s+overall)\b', query, re.IGNORECASE):
            intent = "GET_ATTENDANCE"
            subject = None
            target = None
            count = None

        # 7. Fallback subject cleaning for intents that do not take subjects unless present in query text
        else:
            if intent in ["CHECK_ATTENDANCE_UPDATE", "GET_ATTENDANCE", "HELP", "UNKNOWN"]:
                if text_subject is None:
                    subject = None
            elif text_subject is not None:
                subject = text_subject

        if intent in ["CLASSES_REQUIRED_FOR_TARGET", "MAX_CLASSES_CAN_MISS"]:
            explicit_pct = extract_explicit_percentage(query)
            if explicit_pct is not None:
                target = explicit_pct



        return NLUResult(
            raw_output=raw_output,
            intent=intent,
            subject=subject,
            target=target,
            count=count,
            date=parsed.date,
            is_valid=parsed.is_valid,
            malformed_reason=parsed.malformed_reason,
            confidence=confidence,
            top_intents=top_intents,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(
        self,
        queries: list[str],
        max_new_tokens: int = 40,
    ) -> list[NLUResult]:
        """Run NLU inference on a list of user query strings."""
        return [self.predict(q, max_new_tokens=max_new_tokens) for q in queries]


# ── Singleton Accessor ────────────────────────────────────────────────────────

_default_engine: Optional[NLUInferenceEngine] = None


def get_nlu_engine(
    checkpoint_path: Optional[Path | str] = None,
    tokenizer_path: Optional[Path | str] = None,
    device: Optional[str | torch.device] = None,
) -> NLUInferenceEngine:
    """Return or initialize the singleton NLUInferenceEngine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = NLUInferenceEngine(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
    return _default_engine
