#!/usr/bin/env python3
"""Download the recorded Gutenberg book IDs rather than a changing top-author list."""
import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'gutenberg_manifest.json')
    parser.add_argument('--output', type=Path, default=root / 'data/gutenberg_corpus')
    args = parser.parse_args()
    from src.data_collection.download_gotemberg import GutenbergAuthorDownloader

    class RecordedCorpusDownloader(GutenbergAuthorDownloader):
        def get_top_authors(self, n=100):
            return records[:n]

        def get_author_books(self, author_url, author_name):
            return next(record['books'] for record in records if record['url'] == author_url)

    records = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    RecordedCorpusDownloader(args.output).download_all(n_authors=len(records))


if __name__ == '__main__':
    main()
