#!/usr/bin/env python3
"""Format aligned neutral rewrites and original author sentences for LoRA SFT."""
import argparse
import json
from pathlib import Path


def build_pairs(neutral_path, author_path):
    neutral = neutral_path.read_text().splitlines()
    original = author_path.read_text().splitlines()
    if len(neutral) != len(original):
        raise ValueError('Neutral and author files must contain the same number of aligned lines.')
    if any(not a.strip() or not b.strip() for a,b in zip(neutral, original)):
        raise ValueError('Empty lines would break sentence alignment; resolve them before formatting.')
    return [{'instruction':'Paraphrase', 'input':a.strip(), 'output':b.strip()}
            for a,b in zip(neutral, original)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neutral', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build_pairs(args.neutral, args.original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(f'Wrote {len(rows)} aligned training pairs to {args.output}')


if __name__ == '__main__':
    main()
