"""
slm/tokenizer/char_tokenizer.py
───────────────────────────────
Deterministic character-level tokenizer prototype.

RULES:
- Deterministic vocabulary construction with stable IDs.
- Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
- Lossless round-trip for all known characters.
- Unknown character fallback to <UNK>.
- JSON serialization / deserialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class CharTokenizer:
    """
    Deterministic character-level tokenizer.
    """

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
    ) -> None:
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

        if vocab is not None:
            self._token_to_id: dict[str, int] = dict(vocab)
            self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}
        else:
            self._token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
            self._id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

    @property
    def vocab_size(self) -> int:
        """Total size of the vocabulary including special tokens."""
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        return dict(self._id_to_token)

    def train(self, texts: list[str]) -> None:
        """
        Build vocabulary deterministically from a list of strings.
        Characters are sorted alphabetically to ensure stable ID assignment.
        """
        unique_chars: set[str] = set()
        for text in texts:
            unique_chars.update(text)

        # Reset vocabulary with special tokens
        self._token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

        # Sort characters for deterministic IDs
        sorted_chars = sorted(unique_chars)
        next_id = len(SPECIAL_TOKENS)

        for ch in sorted_chars:
            if ch not in self._token_to_id:
                self._token_to_id[ch] = next_id
                next_id += 1

        self._id_to_token = {i: t for t, i in self._token_to_id.items()}

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode a string into a list of token IDs.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for ch in text:
            ids.append(self._token_to_id.get(ch, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a list of token IDs back into a string.
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        chars: list[str] = []

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token_str = self._id_to_token.get(tid, self.unk_token)
            # If special token, include its representation unless skipped
            chars.append(token_str)

        return "".join(chars)

    def save(self, filepath: Path | str) -> None:
        """Serialize tokenizer state to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "type": "char",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                self.pad_token: self.pad_id,
                self.unk_token: self.unk_id,
                self.bos_token: self.bos_id,
                self.eos_token: self.eos_id,
            },
            "vocab": self._token_to_id,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> CharTokenizer:
        """Load tokenizer state from a JSON file."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = data["vocab"]
        return cls(vocab=vocab)
