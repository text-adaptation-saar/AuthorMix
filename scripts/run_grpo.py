#!/usr/bin/env python3
"""Train AuthorMix using a JSON configuration."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "layerwise": ROOT / "PTA/19_RL_on_10src_15tgt_authors/weight_optimization_per_tgt_author/run_rl_per_pair.py",
    "adapterwise": ROOT / "PTA/19_RL_on_10src_15tgt_authors/adapterwise_weight_optimization_per_tgt_author_layerwiseFalse/run_rl_per_pair.py",
}


def build_command(config):
    variant = config.get("variant", "layerwise")
    action = config.get("action", "train")
    if variant not in VARIANTS or action != "train":
        raise ValueError("Unknown variant or action")
    entry = VARIANTS[variant]
    if not entry.is_file():
        raise ValueError(f"{action} is not available for {variant}")
    # Read argparse declarations without importing ML libraries or loading models.
    options = {}
    for node in ast.walk(ast.parse(entry.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            kwargs = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    options[arg.value[2:]] = kwargs
    arguments = config.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    unknown = set(arguments) - set(options)
    if unknown:
        raise ValueError(f"Unknown arguments: {sorted(unknown)}")
    required = {key for key, kw in options.items() if kw.get("required")}
    missing = required - {key for key, value in arguments.items() if value is not None}
    if missing:
        raise ValueError(f"Missing required arguments: {sorted(missing)}")
    command = [sys.executable, str(entry)]
    for key, value in arguments.items():
        if value is None:
            continue
        if options[key].get("action") in ("store_true", "store_false"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true (include flag) or false (omit flag)")
            if value:
                command.append("--" + key)
        else:
            command.extend(["--" + key, str(int(value)) if isinstance(value, bool) else str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print command; load no models")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        command = build_command(config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Working directory: {ROOT}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env["LLAMA_FACTORY_SRC"] = str(ROOT / "src")
    env["DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
