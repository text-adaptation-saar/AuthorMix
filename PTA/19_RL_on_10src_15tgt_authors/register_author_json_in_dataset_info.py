"""
Register author LLaMA-Factory JSON files into data/dataset_info.json.

Scans data/dataset_author_eval/sents/llama_factory_json/ for all author
JSON files and adds (or updates) entries in data/dataset_info.json.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LLAMA_FACTORY_JSON_DIR = PROJECT_ROOT / "data/dataset_author_eval/sents/llama_factory_json"
DATASET_INFO_PATH = PROJECT_ROOT / "data/dataset_info.json"


def main():
    if not LLAMA_FACTORY_JSON_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {LLAMA_FACTORY_JSON_DIR}")

    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    added = 0
    updated = 0

    for author_dir in sorted(LLAMA_FACTORY_JSON_DIR.iterdir()):
        if not author_dir.is_dir():
            continue
        author_name = author_dir.name
        safe_author = author_name.replace(".", "_").replace(" ", "_")

        for json_file in sorted(author_dir.glob("*.json")):
            split = json_file.stem  # "train" or "val"
            ds_key = f"author_eval_{safe_author}_{split}"
            rel_path = str(json_file.relative_to(PROJECT_ROOT / "data"))

            if ds_key in dataset_info:
                old_path = dataset_info[ds_key].get("file_name", "")
                if old_path == rel_path:
                    print(f"  SKIP (unchanged): {ds_key} -> {rel_path}")
                    continue
                print(f"  UPDATE: {ds_key}: {old_path} -> {rel_path}")
                dataset_info[ds_key]["file_name"] = rel_path
                updated += 1
            else:
                print(f"  ADD:    {ds_key} -> {rel_path}")
                dataset_info[ds_key] = {"file_name": rel_path}
                added += 1

    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone! Added {added}, updated {updated} entries in {DATASET_INFO_PATH}")
    print(f"\nRegistered dataset keys:")
    for author_dir in sorted(LLAMA_FACTORY_JSON_DIR.iterdir()):
        if not author_dir.is_dir():
            continue
        safe_author = author_dir.name.replace(".", "_").replace(" ", "_")
        for split in ("train", "val"):
            print(f"  author_eval_{safe_author}_{split}")


if __name__ == "__main__":
    main()
