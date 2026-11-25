#!/usr/bin/env python3
"""Train the high-resource adapters with a separate, full LLaMA-Factory checkout."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CODES = ('MT', 'VW', 'VL', 'CPG', 'GO', 'JA', 'NH', 'OW', 'PGW', 'SR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llamafactory-root', type=Path, required=True)
    parser.add_argument('--authors', nargs='+', choices=CODES, default=list(CODES))
    parser.add_argument('--base-model', default='models/Llama-3.1-8B-Instruct')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    factory = args.llamafactory_root.expanduser().resolve()
    entry = factory / 'src/train.py'
    if not entry.is_file():
        parser.error('Use a full LLaMA-Factory checkout containing src/train.py.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(factory / 'src')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Use the recorded experiment settings, including this compatibility override.
    env['DISABLE_VERSION_CHECK'] = '1'
    for code in args.authors:
        source = ROOT / f'PTA/10_author_models_ft/configs/{code}.yaml'
        config = source.read_text()
        config = re.sub(r'^model_name_or_path:.*$', lambda _: 'model_name_or_path: ' + json.dumps(args.base_model), config, flags=re.M)
        path = ROOT / f'outputs/adapter_training_configs/{code}.yaml'
        command = [sys.executable, str(entry), str(path)]
        print(f'{code}: {shlex.join(command)}', flush=True)
        if args.dry_run:
            print(config)
        else:
            if (ROOT / 'adapters' / code).exists():
                parser.error(f'adapters/{code} already exists; choose a clean output before retraining.')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config)
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
