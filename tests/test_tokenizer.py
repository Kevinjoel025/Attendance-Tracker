"""
tests/test_tokenizer.py
───────────────────────
Comprehensive unit and integration tests for Character Tokenizer and custom BPE Tokenizer.

Covers:
- Character Tokenizer: vocabulary, encode, decode, roundtrip, special tokens, serialization
- BPE Tokenizer: deterministic training, tie-breaking, exact 4096 vocab size, merge learning
- BPE Tokenizer: lossless roundtrip encoding/decoding, unknown character handling, save/load
- Real Tokenizer verification against the 30,000 example dataset partitions
"""

from pathlib import Path
import pytest

from attendance_tracker.slm.tokenizer.bpe_tokenizer import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
    BPETokenizer,
)
from attendance_tracker.slm.tokenizer.char_tokenizer import CharTokenizer


class TestCharTokenizer:

    def test_special_tokens_ids(self):
        tok = CharTokenizer()
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert tok.token_to_id[PAD_TOKEN] == 0
        assert tok.token_to_id[UNK_TOKEN] == 1
        assert tok.token_to_id[BOS_TOKEN] == 2
        assert tok.token_to_id[EOS_TOKEN] == 3

    def test_deterministic_char_vocab_training(self):
        tok = CharTokenizer()
        corpus = ["abc", "cba", "hello world!"]
        tok.train(corpus)

        # Characters should be sorted deterministically
        expected_chars = sorted(set("abc" + "cba" + "hello world!"))
        for ch in expected_chars:
            assert ch in tok.token_to_id
            assert tok.token_to_id[ch] >= 4

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        corpus = ["What is my attendance in DE? 75%"]
        tok.train(corpus)

        text = "What is my attendance in DE? 75%"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_bos_eos_flags(self):
        tok = CharTokenizer()
        tok.train(["test"])

        ids = tok.encode("test", add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID
        assert tok.decode(ids, skip_special_tokens=True) == "test"

    def test_unknown_character_fallback(self):
        tok = CharTokenizer()
        tok.train(["abc"])

        ids = tok.encode("abz")  # 'z' is unknown
        assert ids == [tok.token_to_id["a"], tok.token_to_id["b"], UNK_ID]

    def test_empty_string(self):
        tok = CharTokenizer()
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_save_and_load(self, tmp_path: Path):
        tok = CharTokenizer()
        tok.train(["hello world 123"])

        save_file = tmp_path / "char_tok.json"
        tok.save(save_file)

        loaded_tok = CharTokenizer.load(save_file)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.encode("hello") == tok.encode("hello")


class TestBPETokenizerUnit:

    def test_bpe_training_on_mini_corpus(self):
        corpus = [
            "low low low low low",
            "lower lower",
            "newest newest newest newest newest newest",
            "widest widest widest",
        ]
        bpe = BPETokenizer(target_vocab_size=30)
        bpe.train(corpus, target_vocab_size=30)

        assert bpe.vocab_size == 30
        assert len(bpe.merges) > 0
        assert bpe.pad_id == 0
        assert bpe.unk_id == 1
        assert bpe.bos_id == 2
        assert bpe.eos_id == 3

    def test_bpe_training_determinism(self):
        corpus = [
            "what is my attendance in DE",
            "show attendance for operating systems",
            "if i attend 5 classes what is my percentage",
            "how many classes to reach 75%",
        ]
        bpe1 = BPETokenizer(target_vocab_size=100)
        bpe1.train(corpus, target_vocab_size=100)

        bpe2 = BPETokenizer(target_vocab_size=100)
        bpe2.train(corpus, target_vocab_size=100)

        assert bpe1.vocab_size == bpe2.vocab_size
        assert bpe1.merges == bpe2.merges
        assert bpe1.token_to_id == bpe2.token_to_id

        # Verify exact same encoding
        test_text = "what is attendance in DE"
        assert bpe1.encode(test_text) == bpe2.encode(test_text)

    def test_bpe_lossless_roundtrip(self):
        corpus = [
            "What is my DE attendance?",
            "How many classes can I skip in OS to stay above 70%?",
            "Did today's attendance get marked for Storytelling?",
            "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE",
        ]
        bpe = BPETokenizer(target_vocab_size=100)
        bpe.train(corpus, target_vocab_size=100)

        for text in corpus:
            ids = bpe.encode(text)
            decoded = bpe.decode(ids)
            assert decoded == text, f"Roundtrip failed for: '{text}' -> got '{decoded}'"

    def test_bpe_bos_eos_addition(self):
        bpe = BPETokenizer(target_vocab_size=50)
        bpe.train(["attendance query"])

        ids = bpe.encode("attendance query", add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID

        decoded_clean = bpe.decode(ids, skip_special_tokens=True)
        assert decoded_clean == "attendance query"

    def test_bpe_save_and_load(self, tmp_path: Path):
        corpus = ["what is my DE attendance", "check OS attendance 75%"]
        bpe = BPETokenizer(target_vocab_size=60)
        bpe.train(corpus, target_vocab_size=60)

        save_path = tmp_path / "bpe.json"
        bpe.save(save_path)

        loaded_bpe = BPETokenizer.load(save_path)
        assert loaded_bpe.vocab_size == bpe.vocab_size
        assert loaded_bpe.merges == bpe.merges

        query = "what is my DE attendance"
        assert loaded_bpe.encode(query) == bpe.encode(query)
        assert loaded_bpe.decode(loaded_bpe.encode(query)) == query


class TestTrainedBPEArtifacts:

    def test_real_trained_bpe_file_exists_and_loads(self):
        bpe_file = Path("data/tokenizer/bpe.json")
        if not bpe_file.exists():
            pytest.skip("BPE tokenizer artifact data/tokenizer/bpe.json not yet generated")

        bpe = BPETokenizer.load(bpe_file)
        assert bpe.vocab_size == 4096, f"Expected exactly 4096 vocab size, got {bpe.vocab_size}"
        assert bpe.pad_id == 0
        assert bpe.unk_id == 1
        assert bpe.bos_id == 2
        assert bpe.eos_id == 3

    def test_real_trained_bpe_encodes_all_subjects_and_aliases(self):
        bpe_file = Path("data/tokenizer/bpe.json")
        if not bpe_file.exists():
            pytest.skip("data/tokenizer/bpe.json not found")

        bpe = BPETokenizer.load(bpe_file)

        test_cases = [
            "what is my DE attendance?",
            "how much attendance in operating systems?",
            "tell me my attendance in Automata and Compiler Design",
            "how many classes do I need to reach 75% in IoTAP LAB?",
            "can I skip 3 classes in Gender Sensitization?",
            "did today's attendance get marked for Storytelling?",
            "what is my WP percentage?",
            "DE LAB attendance status",
        ]

        for query in test_cases:
            ids = bpe.encode(query)
            assert len(ids) > 0
            assert all(0 <= tid < 4096 for tid in ids)
            decoded = bpe.decode(ids)
            assert decoded == query

    def test_real_trained_bpe_encodes_pipe_output_format(self):
        bpe_file = Path("data/tokenizer/bpe.json")
        if not bpe_file.exists():
            pytest.skip("data/tokenizer/bpe.json not found")

        bpe = BPETokenizer.load(bpe_file)

        pipe_labels = [
            "GET_ATTENDANCE|NONE|NONE|NONE|NONE",
            "GET_SUBJECT_ATTENDANCE|DE|NONE|NONE|NONE",
            "CLASSES_REQUIRED_FOR_TARGET|ACD|75|NONE|NONE",
            "MAX_CLASSES_CAN_MISS|OS|70|NONE|NONE",
            "ATTENDANCE_PROJECTION|DE|NONE|5|NONE",
            "CHECK_ATTENDANCE_UPDATE|DE|NONE|NONE|TODAY",
            "HELP|NONE|NONE|NONE|NONE",
            "UNKNOWN|NONE|NONE|NONE|NONE",
        ]

        for label in pipe_labels:
            ids = bpe.encode(label)
            assert all(0 <= tid < 4096 for tid in ids)
            decoded = bpe.decode(ids)
            assert decoded == label
