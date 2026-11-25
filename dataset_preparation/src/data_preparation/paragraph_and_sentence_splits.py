#!/usr/bin/env python3
"""
Process author corpus: first paragraph split (save), then sentence split from paragraphs.

1. Paragraph: split text by 1+ newlines → clean → save to paragraph_splits/text/ and
   paragraph_splits/json_format/ with train/val/test (no seed, document order).
2. Sentence: read paragraph train/val/test.txt → sentence-tokenize each paragraph →
   clean → shuffle with seed → save to sentence_splits (train/val/test). Paragraph level: no shuffle.
"""

import json
import random
import re
import sys
from pathlib import Path

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Split on 1+ newlines (paragraph/excerpt boundary)
PARAGRAPH_DELIMITER = re.compile(r"\n\s*\n+")


def clean_text(text: str) -> str:
    """
    Clean text:
    1. Replace curly quotes, em-dash, ellipsis, etc. with ASCII.
    2. Remove '_', '*', '[', ']'.
    3. Collapse multiple spaces to one, strip.
    """
    replacements = {
        "\n": " ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "…": "...",
    }
    if not text:
        return ""
    pattern = re.compile("|".join(re.escape(k) for k in replacements.keys()))
    text = pattern.sub(lambda m: replacements[m.group(0)], text.strip())
    text = re.sub(r"[_*\[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _collect_txt_files(author_dir: Path) -> list[Path]:
    """All .txt files in author_dir (no recursion)."""
    author_dir = Path(author_dir)
    if not author_dir.is_dir():
        return []
    return sorted(author_dir.glob("*.txt"))


SPLIT_NAMES = ("train", "val", "test")
MIN_VAL_TEST_CHARS = 50  # drop items with len < this in train, val, and test


def _read_paragraph_split_file(paragraph_text_dir: Path, split_name: str) -> list[str]:
    """Read paragraphs from paragraph_text_dir/<split_name>.txt (one paragraph per line)."""
    fp = paragraph_text_dir / f"{split_name}.txt"
    if not fp.is_file():
        return []
    try:
        return [s for line in fp.read_text(encoding="utf-8", errors="replace").splitlines() if (s := line.strip())]
    except Exception:
        return []


def _ensure_nltk_punkt():
    if not NLTK_AVAILABLE:
        raise RuntimeError("nltk is required for sentence mode. Install with: pip install nltk")
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(f"Could not download NLTK resource: {resource}")


def process_author_dir_paragraph(
    author_dir: str | Path,
    output_text_dir: str | Path,
    output_json_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_excerpt_chars: int = 20,
) -> None:
    """
    Process at paragraph/excerpt level: split by 1+ newlines, clean each,
    then split into train/val/test in document order (no shuffle/seed).
    Writes to output_text_dir (train/val/test.txt) and output_json_dir (train/val/test.json).
    """
    author_dir = Path(author_dir)
    output_text_dir = Path(output_text_dir)
    output_json_dir = Path(output_json_dir)
    output_text_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    all_excerpts: list[str] = []
    for fp in _collect_txt_files(author_dir):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        parts = PARAGRAPH_DELIMITER.split(raw)
        for p in parts:
            s = p.strip()
            if len(s) >= min_excerpt_chars:
                cleaned = clean_text(s)
                if cleaned:
                    all_excerpts.append(cleaned)

    if not all_excerpts:
        return

    # No shuffle: keep document order, split by ratio
    n = len(all_excerpts)
    t1 = int(n * train_ratio)
    t2 = t1 + int(n * val_ratio)
    train_list = all_excerpts[:t1]
    val_list = all_excerpts[t1:t2]
    test_list = all_excerpts[t2:]

    for split_name, lines in [("train", train_list), ("val", val_list), ("test", test_list)]:
        if not lines:
            continue
        lines = [line for line in lines if len(line) >= MIN_VAL_TEST_CHARS]
        if not lines:
            continue
        json_data = [{"text": line} for line in lines]
        output_json_dir.joinpath(f"{split_name}.json").write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output_text_dir.joinpath(f"{split_name}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def _is_valid_sentence(sentence: str) -> bool:
    if not sentence or not sentence.strip():
        return False
    s = sentence.strip()
    if re.fullmatch(r"\.*", s):
        return False
    if re.fullmatch(r"[\d\s.,%-]+", s):
        return False
    return True


def process_author_dir_sentence(
    paragraph_text_dir: str | Path,
    output_text_dir: str | Path,
    output_json_dir: str | Path,
    *,
    seed: int = 42,
) -> None:
    """
    Read paragraph splits train/val/test.txt from paragraph_text_dir (one para per line).
    For each split: sentence-tokenize, clean, filter invalid, shuffle with seed, then save.
    """
    _ensure_nltk_punkt()
    paragraph_text_dir = Path(paragraph_text_dir)
    output_text_dir = Path(output_text_dir)
    output_json_dir = Path(output_json_dir)
    output_text_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    for i, split_name in enumerate(SPLIT_NAMES):
        paragraphs = _read_paragraph_split_file(paragraph_text_dir, split_name)
        if not paragraphs:
            continue

        sentences: list[str] = []
        for para in paragraphs:
            for s in nltk.sent_tokenize(para):
                cleaned = clean_text(s)
                if _is_valid_sentence(cleaned):
                    sentences.append(cleaned)

        sentences = [s for s in sentences if len(s) >= MIN_VAL_TEST_CHARS]
        if not sentences:
            continue

        random.seed(seed + i)
        random.shuffle(sentences)

        output_text_dir.joinpath(f"{split_name}.txt").write_text(
            "\n".join(sentences), encoding="utf-8"
        )
        output_json_dir.joinpath(f"{split_name}.json").write_text(
            json.dumps([{"text": line} for line in sentences], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def process_corpus(
    corpus_dir: Path,
    output_base: Path,
    mode: str,
    *,
    seed: int = 42,
) -> None:
    """
    Paragraph mode: corpus_dir = raw author dirs (book_*.txt).
    Writes to output_base/text/<author>/ and output_base/json_format/<author>/ (train/val/test).

    Sentence mode: corpus_dir = paragraph_splits base (contains text/ and json_format/).
    Reads from corpus_dir/text/<author>/ (train/val/test.txt), writes
    output_base/text/<author>/ and output_base/json_format/<author>/ (train/val/test.txt and .json).
    """
    corpus_dir = Path(corpus_dir)
    output_base = Path(output_base)
    text_base = output_base / "text"
    json_base = output_base / "json_format"

    if mode == "paragraph":
        for author_dir in sorted(corpus_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            name = author_dir.name
            process_author_dir_paragraph(author_dir, text_base / name, json_base / name)
    else:
        # Sentence: input is paragraph_splits base; author list from corpus_dir/text/
        paragraph_text_base = corpus_dir / "text"
        if not paragraph_text_base.is_dir():
            raise FileNotFoundError(
                f"For sentence mode, corpus_dir must be paragraph_splits base with text/ subdir: {corpus_dir}"
            )
        for author_dir in sorted(paragraph_text_base.iterdir()):
            if not author_dir.is_dir():
                continue
            name = author_dir.name
            process_author_dir_sentence(author_dir, text_base / name, json_base / name, seed=seed)


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "Usage: paragraph_and_sentence_splits.py <mode> <input_dir> <output_text_dir> <output_json_dir> [seed]",
            file=sys.stderr,
        )
        print("  paragraph: input_dir = raw author dir (book_*.txt); output = train/val/test (no seed)", file=sys.stderr)
        print("  sentence:  input_dir = paragraph_splits/text/<author>; output = train/val/test; [seed] for shuffle (default 42)", file=sys.stderr)
        print("  Or:   paragraph_and_sentence_splits.py --corpus <corpus_dir> <output_base> <mode> [seed]", file=sys.stderr)
        print("  paragraph: corpus_dir = raw corpus; sentence: corpus_dir = paragraph_splits base; seed only for sentence", file=sys.stderr)
        return 1

    if sys.argv[1] == "--corpus":
        if len(sys.argv) < 5:
            print("--corpus requires: <corpus_dir> <output_base> <mode> [seed]", file=sys.stderr)
            return 1
        corpus_dir = Path(sys.argv[2])
        output_base = Path(sys.argv[3])
        mode = sys.argv[4].lower()
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
        if mode not in ("paragraph", "sentence"):
            print("mode must be 'paragraph' or 'sentence'", file=sys.stderr)
            return 1
        process_corpus(corpus_dir, output_base, mode, seed=seed)
        return 0

    mode = sys.argv[1].lower()
    input_dir = sys.argv[2]
    output_text_dir = sys.argv[3]
    output_json_dir = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    if mode == "paragraph":
        process_author_dir_paragraph(input_dir, output_text_dir, output_json_dir)
    elif mode == "sentence":
        process_author_dir_sentence(input_dir, output_text_dir, output_json_dir, seed=seed)
    else:
        print("mode must be 'paragraph' or 'sentence'", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
