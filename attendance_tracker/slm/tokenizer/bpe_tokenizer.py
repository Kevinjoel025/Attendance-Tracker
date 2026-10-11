"""
slm/tokenizer/bpe_tokenizer.py
──────────────────────────────
Custom Byte Pair Encoding (BPE) tokenizer implemented entirely from scratch.

RULES:
- Zero external dependencies (no HuggingFace, tiktoken, tokenizers, sentencepiece).
- Target vocabulary size: 4096 total (including 4 special tokens).
- Deterministic training with explicit tie-breaking rule.
- Lossless round-trip encoding/decoding.
- Special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
- JSON serialization and deserialization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# GPT-2 style pre-tokenization pattern (split into words, contractions, punctuation, whitespace)
SPLIT_REGEX = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""",
    re.IGNORECASE,
)

SPACE_MARKER = "Ġ"


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer built from scratch for the College Attendance SLM.
    """

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
        merges: Optional[list[Tuple[str, str]]] = None,
        target_vocab_size: int = 4096,
    ) -> None:
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

        self.target_vocab_size = target_vocab_size

        if vocab is not None and merges is not None:
            self._token_to_id: dict[str, int] = dict(vocab)
            self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}
            self._merges: list[Tuple[str, str]] = list(merges)
            self._bpe_ranks: dict[Tuple[str, str], int] = {
                (p0, p1): i for i, (p0, p1) in enumerate(self._merges)
            }
        else:
            self._token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
            self._id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
            self._merges = []
            self._bpe_ranks = {}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        return dict(self._id_to_token)

    @property
    def merges(self) -> list[Tuple[str, str]]:
        return list(self._merges)

    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Split raw text into pre-token chunks and mark spaces with Ġ.
        """
        chunks = SPLIT_REGEX.findall(text)
        formatted_chunks = []
        for chunk in chunks:
            # Replace space with space marker
            formatted = chunk.replace(" ", SPACE_MARKER)
            if formatted:
                formatted_chunks.append(formatted)
        return formatted_chunks

    def train(
        self,
        texts: list[str],
        target_vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> None:
        """
        Train BPE merge rules deterministically from a list of strings.

        Parameters
        ----------
        texts : list[str]
            Training corpus queries.
        target_vocab_size : int
            Desired total vocabulary size (default: 4096).
        min_frequency : int
            Minimum frequency for a pair to be considered for merging (default: 2).
        """
        self.target_vocab_size = target_vocab_size

        # 1. Initialize special tokens
        self._token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        next_id = len(SPECIAL_TOKENS)

        # 2. Count initial word frequencies from training corpus
        word_freqs: dict[Tuple[str, ...], int] = {}
        initial_chars: set[str] = {SPACE_MARKER}

        for text in texts:
            chunks = self._pre_tokenize(text)
            for chunk in chunks:
                symbols = tuple(chunk)
                word_freqs[symbols] = word_freqs.get(symbols, 0) + 1
                initial_chars.update(chunk)

        # 3. Add base single-character symbols to vocabulary in sorted order
        for ch in sorted(initial_chars):
            if ch not in self._token_to_id:
                self._token_to_id[ch] = next_id
                next_id += 1

        self._merges = []
        self._bpe_ranks = {}

        # 4. Iteratively learn merges
        num_merges_to_learn = target_vocab_size - len(self._token_to_id)

        for merge_idx in range(num_merges_to_learn):
            # Count adjacent pairs across all words
            pair_counts: dict[Tuple[str, str], int] = {}
            for symbols, freq in word_freqs.items():
                if len(symbols) < 2:
                    continue
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            # Find best pair deterministically:
            # Sort by: 1. highest frequency (-count), 2. lexicographical tie-break (pair[0], pair[1])
            best_pair = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[0]

            if pair_counts[best_pair] < min_frequency:
                # Stop if no frequent pairs remain
                break

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self._token_to_id:
                self._token_to_id[merged_token] = next_id
                next_id += 1

            self._merges.append(best_pair)
            self._bpe_ranks[best_pair] = merge_idx

            # Update word frequencies with merged pair
            new_word_freqs: dict[Tuple[str, ...], int] = {}
            for symbols, freq in word_freqs.items():
                if len(symbols) < 2:
                    new_word_freqs[symbols] = freq
                    continue

                new_symbols: list[str] = []
                i = 0
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and symbols[i] == best_pair[0]
                        and symbols[i + 1] == best_pair[1]
                    ):
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_word_freqs[tuple(new_symbols)] = freq

            word_freqs = new_word_freqs

            if len(self._token_to_id) >= target_vocab_size:
                break

        # 5. If target_vocab_size was not reached (e.g. all frequent pairs merged),
        # pad the remaining vocabulary slots with deterministic reserved tokens <unused_k>
        # so the model embedding table matches target_vocab_size (4096) exactly.
        unused_idx = 0
        while len(self._token_to_id) < target_vocab_size:
            unused_token = f"<unused_{unused_idx}>"
            if unused_token not in self._token_to_id:
                self._token_to_id[unused_token] = next_id
                next_id += 1
            unused_idx += 1

        self._id_to_token = {i: t for t, i in self._token_to_id.items()}

    def _bpe_encode_chunk(self, chunk: str) -> list[str]:
        """
        Apply learned BPE merge rules to a single pre-tokenized chunk.
        """
        if not chunk:
            return []

        symbols = list(chunk)
        if len(symbols) <= 1:
            return symbols

        while len(symbols) > 1:
            # Find all possible pairs in current symbol sequence
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find the pair with the lowest merge rank (earliest learned)
            candidate_pairs = [p for p in pairs if p in self._bpe_ranks]
            if not candidate_pairs:
                break

            best_pair = min(candidate_pairs, key=lambda p: self._bpe_ranks[p])

            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_symbols.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode a string into a list of BPE token IDs.
        """
        token_ids: list[int] = []

        if add_bos:
            token_ids.append(self.bos_id)

        chunks = self._pre_tokenize(text)
        for chunk in chunks:
            subwords = self._bpe_encode_chunk(chunk)
            for sw in subwords:
                token_ids.append(self._token_to_id.get(sw, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a list of BPE token IDs back into raw text.
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        tokens: list[str] = []

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token_str = self._id_to_token.get(tid, self.unk_token)
            tokens.append(token_str)

        full_text = "".join(tokens)
        # Restore original spaces
        return full_text.replace(SPACE_MARKER, " ")

    def save(self, filepath: Path | str) -> None:
        """Serialize BPE vocabulary and merge rules to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                self.pad_token: self.pad_id,
                self.unk_token: self.unk_id,
                self.bos_token: self.bos_id,
                self.eos_token: self.eos_id,
            },
            "vocab": self._token_to_id,
            "merges": self._merges,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> BPETokenizer:
        """Load BPE tokenizer from JSON file."""
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = data["vocab"]
        merges = [tuple(m) for m in data["merges"]]
        target_size = data.get("target_vocab_size", len(vocab))

        return cls(vocab=vocab, merges=merges, target_vocab_size=target_size)
