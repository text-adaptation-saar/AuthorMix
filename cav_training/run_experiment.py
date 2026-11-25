#!/usr/bin/env python3
"""Run the paper's CAV exp04 or regenerate its content-controlled triplets."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_command(config):
    trainer = HERE / 'train_style_embeddings_cav.py'
    options = set()
    for node in ast.walk(ast.parse(trainer.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            options.update(a.value[2:] for a in node.args if isinstance(a, ast.Constant)
                           and isinstance(a.value, str) and a.value.startswith('--'))
    unknown = set(config) - options
    if unknown:
        raise ValueError(f'Unknown training options: {sorted(unknown)}')
    command = [sys.executable, str(trainer)]
    for key, value in config.items():
        if value is None or value is False:
            continue
        command.append('--' + key)
        if value is not True:
            command.append(str(value))
    return command


def validate_triplets(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    for split, expected in [('train', 100000), ('test', 10000)]:
        data = json.loads((folder / f'{split}_triplets.json').read_text())
        if data['count'] != expected or len(data['triplets']) != expected:
            raise ValueError(f'{split}: expected {expected} saved triplets')
        if metadata[f'{split}_size'] != expected:
            raise ValueError(f'{split}: metadata count disagrees')
        if not all(set(row) == {'anchor', 'positive', 'negative'} and
                   all(isinstance(v, str) and v.strip() for v in row.values())
                   for row in data['triplets']):
            raise ValueError(f'{split}: malformed triplet')
    if not metadata['use_cav'] or metadata['num_authors_train'] != 50:
        raise ValueError('Expected the 50-author CAV triplets')


def validate_corpus(folder):
    expected = {a['directory'] for a in json.loads((HERE / 'authors.json').read_text())}
    found = {p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.')}
    if found != expected:
        raise ValueError(f'Author mismatch. Missing: {sorted(expected-found)}; extra: {sorted(found-expected)}')
    for author in expected:
        for split in ('train', 'val'):
            if not (folder / author / f'{split}.txt').is_file():
                raise ValueError(f'Missing {author}/{split}.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['train', 'prepare'], default='train')
    parser.add_argument('--data-dir', type=Path, help='50-author paragraph corpus; required for prepare')
    parser.add_argument('--triplets-dir', type=Path, default=HERE / 'data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads((HERE / ('exp04.json' if args.phase == 'train' else 'prepare_triplets.json')).read_text())
    if args.phase == 'train':
        folder = args.triplets_dir.expanduser().resolve()
        validate_triplets(folder)
        config['load_triplets_from'] = str(folder)
    else:
        if args.data_dir is None:
            parser.error('--data-dir is required for triplet preparation')
        folder = args.data_dir.expanduser().resolve()
        validate_corpus(folder)
        config['data_dir'] = str(folder)
    if args.output_dir:
        config['output_dir'] = str(args.output_dir.expanduser().resolve())
    config['device'] = args.device
    command = build_command(config)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    out = Path(config['output_dir'])
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new output directory to preserve existing experiment files.')
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


if __name__ == '__main__':
    main()
