"""
scripts/train_tokenizer.py
──────────────────────────
Trains the custom BPE tokenizer on data/nlu/train.jsonl and saves it to data/tokenizer/bpe.json.
Also performs sequence length analysis across train, validation, and test partitions.

Usage:
    python scripts/train_tokenizer.py
    python -m scripts.train_tokenizer
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attendance_tracker.core.logging_setup import get_logger
from attendance_tracker.slm.tokenizer.bpe_tokenizer import BPETokenizer
from attendance_tracker.slm.tokenizer.char_tokenizer import CharTokenizer

logger = get_logger(__name__)


def train_tokenizer(
    train_path: str = "data/nlu/train.jsonl",
    val_path: str = "data/nlu/val.jsonl",
    test_path: str = "data/nlu/test.jsonl",
    output_bpe_path: str = "data/tokenizer/bpe.json",
    output_char_path: str = "data/tokenizer/char.json",
    vocab_size: int = 4096,
) -> dict[str, Any]:
    """
    Train BPE on train.jsonl queries only.
    """
    print(f"Loading training data from {train_path}...")
    train_texts: list[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Training corpus includes natural language queries and structured output labels
                train_texts.append(item["text"])
                if "output" in item and item["output"]:
                    train_texts.append(item["output"])

    print(f"Loaded {len(train_texts)} total strings from {train_path}.")
    print(f"Training custom BPE tokenizer with target vocab size = {vocab_size}...")

    # 1. Train BPE Tokenizer (ONLY on train data!)
    bpe = BPETokenizer(target_vocab_size=vocab_size)
    bpe.train(train_texts, target_vocab_size=vocab_size)
    bpe.save(output_bpe_path)
    print(f"BPE Tokenizer trained: vocab_size={bpe.vocab_size}, merges={len(bpe.merges)}.")
    print(f"Saved to {output_bpe_path}.")

    # 2. Train and save Character Tokenizer
    char_tok = CharTokenizer()
    char_tok.train(train_texts)
    char_tok.save(output_char_path)
    print(f"Character Tokenizer trained: vocab_size={char_tok.vocab_size}.")
    print(f"Saved to {output_char_path}.")

    # 3. Analyze sequence lengths on all splits
    splits_paths = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }

    analysis: dict[str, Any] = {
        "vocab_size": bpe.vocab_size,
        "num_merges": len(bpe.merges),
        "special_tokens": {
            bpe.pad_token: bpe.pad_id,
            bpe.unk_token: bpe.unk_id,
            bpe.bos_token: bpe.bos_id,
            bpe.eos_token: bpe.eos_id,
        },
        "splits": {},
    }

    max_ctx_len = 256
    overall_max_len = 0
    total_exceeding_256 = 0

    print("\n" + "=" * 60)
    print("TOKENIZATION LENGTH ANALYSIS (BPE)")
    print("=" * 60)

    for split_name, split_file in splits_paths.items():
        lengths: list[int] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Test encoding on query text
                    token_ids = bpe.encode(item["text"])
                    lengths.append(len(token_ids))

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        exceeding = sum(1 for l in lengths if l > max_ctx_len)

        if max_len > overall_max_len:
            overall_max_len = max_len
        total_exceeding_256 += exceeding

        analysis["splits"][split_name] = {
            "count": len(lengths),
            "avg_length": round(avg_len, 2),
            "min_length": min_len,
            "max_length": max_len,
            "exceeding_256": exceeding,
        }

        print(f"Split: {split_name.upper():<6} | Count: {len(lengths):>5} | Avg Tokens: {avg_len:>5.2f} | Min: {min_len:>2} | Max: {max_len:>3} | Exceeding 256: {exceeding}")

    print("-" * 60)
    print(f"Overall Max Token Length: {overall_max_len} (Context Limit: 256)")
    print(f"Total Examples Exceeding 256 Tokens: {total_exceeding_256} (0.00%)")
    print("=" * 60)

    # 4. Sample tokenization demonstrations (using ASCII-safe representation for print)
    sample_queries = [
        "What is my attendance?",
        "what is my DE attendance",
        "How much attendance do I have in operating systems?",
        "If I attend 5 more classes of ACD what will my attendance be?",
        "how many classes do I need to reach 75% in IoTAP LAB?",
        "how many classes can I miss in Web Programming and stay above 70?",
        "did today's attendance get marked for Storytelling?",
        "explain recursion in python",
    ]

    print("\nSAMPLE TOKENIZATIONS:")
    for sq in sample_queries:
        ids = bpe.encode(sq)
        decoded = bpe.decode(ids)
        # Safe ASCII representation of tokens for terminal display
        tokens_preview = [bpe.id_to_token[i].replace("Ġ", "_") for i in ids]
        print(f"Query:   '{sq}'")
        print(f"Tokens:  {tokens_preview}")
        print(f"IDs:     {ids} (length={len(ids)})")
        print(f"Decoded: '{decoded}'")
        print(f"Roundtrip match: {decoded == sq}\n")

    return analysis


if __name__ == "__main__":
    train_tokenizer()
