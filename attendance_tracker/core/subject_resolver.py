"""
core/subject_resolver.py
────────────────────────
Deterministic and fuzzy subject resolution service.

Resolves freeform subject names and abbreviations to canonical Subject records.

Resolution Hierarchy:
1. Exact canonical subject code match (case-insensitive).
2. Exact short name match (case-insensitive).
3. Exact full name match (case-insensitive).
4. Authoritative alias match (case-insensitive, normalized).
5. Token / Word subset match.
6. High-confidence fuzzy string match.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

from attendance_tracker.attendance.repository import (
    CANONICAL_ALIASES,
    CANONICAL_SUBJECTS,
    AttendanceRepository,
    Subject,
    get_repository,
)
from attendance_tracker.core.logging_setup import get_logger

logger = get_logger(__name__)


def _normalize_string(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


class SubjectResolver:
    """
    Service for resolving subject query strings to canonical Subject instances.
    """

    def __init__(self, repository: Optional[AttendanceRepository] = None) -> None:
        self.repo = repository or get_repository()
        self._canonical_subjects: list[Subject] = []
        self._code_map: dict[str, Subject] = {}
        self._short_name_map: dict[str, Subject] = {}
        self._full_name_map: dict[str, Subject] = {}
        self._alias_map: dict[str, Subject] = {}
        self._fuzzy_targets: dict[str, Subject] = {}
        self._load_subjects()

    def _load_subjects(self) -> None:
        """Load and cache canonical subjects and aliases."""
        subjects = self.repo.list_subjects()
        if not subjects:
            subjects = [
                Subject(
                    code=s["code"],
                    full_name=s["full_name"],
                    short_name=s["short_name"],
                    threshold=float(s.get("threshold", 75.0)),
                )
                for s in CANONICAL_SUBJECTS
            ]

        self._canonical_subjects = subjects
        self._code_map = {s.code.lower(): s for s in subjects}
        self._short_name_map = {s.short_name.lower(): s for s in subjects}
        self._full_name_map = {s.full_name.lower(): s for s in subjects}

        # Build alias map from database or fallback CANONICAL_ALIASES
        self._alias_map = {}
        for s in subjects:
            aliases = CANONICAL_ALIASES.get(s.code, [])
            for a in aliases:
                self._alias_map[a.lower().strip()] = s
                self._alias_map[_normalize_string(a)] = s

            # Also register short name and full name variants
            self._alias_map[s.short_name.lower()] = s
            self._alias_map[_normalize_string(s.short_name)] = s
            self._alias_map[s.full_name.lower()] = s
            self._alias_map[_normalize_string(s.full_name)] = s

        _RUNTIME_EXTENDED_ALIASES = {
            "22PC1IT202": ["ops"],
            "22OE1EN301": ["storytelling comms", "storytelling communication"],
            "22PC2AM202": ["data eng lab"],
        }
        for code, extra_aliases in _RUNTIME_EXTENDED_ALIASES.items():
            if code.lower() in self._code_map:
                subj = self._code_map[code.lower()]
                for a in extra_aliases:
                    self._alias_map[a.lower().strip()] = subj
                    self._alias_map[_normalize_string(a)] = subj


        # Build search targets for fuzzy matching
        self._fuzzy_targets = {}
        for alias_key, subj in self._alias_map.items():
            if alias_key:
                self._fuzzy_targets[alias_key] = subj

    def resolve(self, query: Optional[str]) -> Optional[Subject]:
        """
        Resolve a subject query string to a canonical Subject.

        Parameters
        ----------
        query : Optional[str]
            Subject code, short name, full name, or alias.

        Returns
        -------
        Optional[Subject]
            Resolved Subject instance, or None if no match found.
        """
        if not query:
            return None

        q_clean = query.strip()
        if not q_clean or q_clean.upper() == "NONE":
            return None

        q_lower = q_clean.lower()
        q_norm = _normalize_string(q_clean)

        # 1. Exact canonical subject code match
        if q_lower in self._code_map:
            return self._code_map[q_lower]

        # 2. Exact short name match
        if q_lower in self._short_name_map:
            return self._short_name_map[q_lower]

        # 3. Exact full name match
        if q_lower in self._full_name_map:
            return self._full_name_map[q_lower]

        # 4. Authoritative alias match (direct and normalized)
        if q_lower in self._alias_map:
            return self._alias_map[q_lower]
        if q_norm in self._alias_map:
            return self._alias_map[q_norm]

        # 5. Check if repository database has dynamic alias
        repo_subj = self.repo.resolve_subject(q_clean)
        if repo_subj is not None:
            return repo_subj

        # 6. Word / Token subset matching (e.g. "automata design" matching "automata and compiler design")
        q_words = set(q_norm.split())
        if q_words:
            # Check for specific strong keywords
            for key, subj in self._alias_map.items():
                key_words = set(key.split())
                # If all words of query are in key, or key words are subset of query words
                if q_words and key_words and (q_words.issubset(key_words) or key_words.issubset(q_words)):
                    if len(q_words & key_words) >= 1 and any(len(w) >= 3 for w in (q_words & key_words)):
                        return subj

        # 7. High-confidence fuzzy match against known aliases and names
        candidates = list(self._fuzzy_targets.keys())
        matches = difflib.get_close_matches(q_lower, candidates, n=1, cutoff=0.60)
        if matches:
            matched_key = matches[0]
            logger.debug(f"Fuzzy resolved '{query}' to '{matched_key}'")
            return self._fuzzy_targets[matched_key]

        if q_norm:
            matches_norm = difflib.get_close_matches(q_norm, candidates, n=1, cutoff=0.60)
            if matches_norm:
                matched_key = matches_norm[0]
                logger.debug(f"Fuzzy resolved normalized '{q_norm}' to '{matched_key}'")
                return self._fuzzy_targets[matched_key]

        return None

    def get_canonical_subjects(self) -> list[Subject]:
        """Return list of all canonical subjects."""
        return list(self._canonical_subjects)

    def get_all_aliases(self) -> dict[str, list[str]]:
        """Return dict of canonical subject code to aliases."""
        return {s.code: CANONICAL_ALIASES.get(s.code, []) for s in self._canonical_subjects}


# ── Singleton Accessor ────────────────────────────────────────────────────────

_default_resolver: Optional[SubjectResolver] = None


def get_subject_resolver(repository: Optional[AttendanceRepository] = None) -> SubjectResolver:
    """Return or initialize the singleton SubjectResolver."""
    global _default_resolver
    if _default_resolver is None or repository is not None:
        _default_resolver = SubjectResolver(repository=repository)
    return _default_resolver
