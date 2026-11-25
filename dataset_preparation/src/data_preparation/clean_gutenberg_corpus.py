#!/usr/bin/env python3
"""
Clean Project Gutenberg corpus: remove headers (title, author, TOC, etc.)
and footers (PG legal text), keeping only the author's writing.
Output is written to gutenberg_corpus_cleaned with the same directory structure.
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Footer patterns: strip everything from these lines onward
# ---------------------------------------------------------------------------
FOOTER_PATTERNS = [
    re.compile(r"^\*{3}\s*END OF (THIS )?PROJECT GUTENBERG", re.IGNORECASE),
    re.compile(r"^End of (the )?Project Gutenberg", re.IGNORECASE),
    re.compile(r"^End of Project Gutenberg", re.IGNORECASE),
]


def find_footer_start(lines: list[str]) -> int | None:
    """Return index of first line that starts the PG footer, or None."""
    for i, line in enumerate(lines):
        for pat in FOOTER_PATTERNS:
            if pat.search(line.strip()):
                return i
    return None


# ---------------------------------------------------------------------------
# Header / content start detection
# ---------------------------------------------------------------------------

# TOC markers
TOC_MARKERS = ("contents", "table of contents", "table of content")


def looks_like_prose(line: str, min_len: int = 50) -> bool:
    """True if line looks like narrative prose (length and has lowercase)."""
    s = line.strip()
    if len(s) < min_len:
        return False
    if not any(c.islower() for c in s):
        return False
    # Skip lines that are clearly metadata
    if re.match(r"^(Produced by|Prepared by|\[Note|\[Etext|Release Date)", s, re.IGNORECASE):
        return False
    if re.match(r"^By\s+[A-Z][a-z].*$", s) and len(s) < 80:  # "By Author Name"
        return False
    return True


def is_section_header_line(line: str) -> bool:
    """True if line looks like a chapter/section header (short, often caps)."""
    s = line.strip()
    if not s:
        return False
    # Standalone Roman numeral or "Chapter N" or "CHAPTER X"
    if re.match(r"^\s*[IVXLCDM]+\s*\.?\s*$", s, re.IGNORECASE):
        return True
    if re.match(r"^(CHAPTER|Chapter|BOOK|PART)\s+[IVXLCDM0-9]+", s, re.IGNORECASE):
        return True
    base = s.rstrip(".").strip().upper()
    if base in ("PRELUDE", "PROLOGUE", "INTRODUCTION", "FOREWORD", "FINALE"):
        return True
    if re.match(r"^(BOOK|PART)\s+[IVXLCDM]+\.?\s*[-–—]", s, re.IGNORECASE):
        return True
    return False


def find_content_start(lines: list[str]) -> int:
    """
    Find the line index where the main body of the book starts.
    Returns 0 if no clear start is found (keep all and rely on footer strip).
    """
    n = len(lines)
    if n == 0:
        return 0

    # Search in first 250 lines for TOC / section start
    search_limit = min(250, n)

    toc_line_idx: int | None = None
    for i in range(search_limit):
        lower = lines[i].strip().lower()
        if lower in TOC_MARKERS or "table of contents" in lower:
            toc_line_idx = i
            break

    if toc_line_idx is not None:
        # After TOC: find first section header that is *immediately* followed by
        # prose (only blank lines in between), so we don't pick a TOC entry.
        i = toc_line_idx + 1
        while i < search_limit:
            line = lines[i]
            if is_section_header_line(line):
                j = i + 1
                while j < min(i + 15, n) and not lines[j].strip():
                    j += 1
                if j < n and lines[j].strip():
                    next_line = lines[j].strip()
                    # Next non-blank must be prose, not another section header
                    if looks_like_prose(next_line, min_len=40) and not is_section_header_line(next_line):
                        return i
            i += 1

    # No TOC: find first prose or section header after typical front matter
    # Skip "Produced by", "[Note", title (often all caps), "By Author"
    i = 0
    while i < search_limit:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # Skip known front matter
        if re.match(r"^(Produced by|Prepared by|\[Note|\[Etext|Release Date)", s, re.IGNORECASE):
            i += 1
            continue
        if re.match(r"^By\s+[A-Z]", s) and len(s) < 70:
            i += 1
            continue
        # First section header (e.g. "I", "CHAPTER I") or first long prose
        if is_section_header_line(s):
            return i
        if looks_like_prose(line, min_len=50):
            return i
        i += 1

    return 0


def clean_text(content: str) -> str:
    """
    Extract only the author's content: remove header and PG footer.
    """
    lines = content.splitlines()
    if not lines:
        return ""

    start = find_content_start(lines)
    end_idx = find_footer_start(lines)
    if end_idx is None:
        end_idx = len(lines)

    # Don't include start if it's past the footer
    if start >= end_idx:
        start = 0

    selected = lines[start:end_idx]
    # Normalize: join with newlines, strip trailing/leading blank lines
    text = "\n".join(selected)
    return text.strip()


def process_corpus(
    input_root: Path,
    output_root: Path,
    *,
    copy_metadata: bool = True,
    verbose: bool = True,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """
    Process all author dirs under input_root; write cleaned .txt to output_root.
    Only processes .txt files; copies metadata.json if copy_metadata.
    Returns (files_processed, files_failed, list of (path, error) for failures).
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed: list[tuple[Path, str]] = []

    for author_dir in sorted(input_root.iterdir()):
        if not author_dir.is_dir():
            continue
        out_author = output_root / author_dir.name
        out_author.mkdir(parents=True, exist_ok=True)

        if copy_metadata:
            meta = author_dir / "metadata.json"
            if meta.exists():
                import shutil
                shutil.copy2(meta, out_author / "metadata.json")

        for f in sorted(author_dir.iterdir()):
            if f.suffix.lower() != ".txt":
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
                cleaned = clean_text(raw)
                out_path = out_author / f.name
                out_path.write_text(cleaned, encoding="utf-8")
                processed += 1
                if verbose and processed % 100 == 0:
                    print(f"  Processed {processed} files ...", file=sys.stderr)
            except Exception as e:
                failed.append((f, str(e)))

    return processed, len(failed), failed


def main() -> int:
    # Project root is two levels up from src/data_collection/
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    input_dir = data_dir / "gutenberg_corpus"
    output_dir = data_dir / "gutenberg_corpus_cleaned"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("Cleaning corpus (removing headers and PG footers)...")

    processed, num_failed, failures = process_corpus(
        input_dir,
        output_dir,
        copy_metadata=True,
        verbose=True,
    )

    print(f"Done. Processed: {processed}, Failed: {num_failed}")
    if failures:
        print("Failures:", file=sys.stderr)
        for path, err in failures[:20]:
            print(f"  {path}: {err}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
