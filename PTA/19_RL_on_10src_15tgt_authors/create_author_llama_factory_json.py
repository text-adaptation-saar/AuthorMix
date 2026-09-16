"""
Create LLaMA-Factory compatible JSON files for each source author.

Reads source author names from test_src10_tgt15_with_neutral.json,
loads their text from data/dataset_author_eval/sents/text/{author}/{split}.txt,
and creates JSON files in the same format as the wiki paraphrase data:
    [{"instruction": "Paraphrase", "input": "<text>", "output": ""}, ...]

Output is saved to data/dataset_author_eval/llama_factory_json/{author}/{split}.json
and entries are appended to data/dataset_info.json.
"""
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_DATA_PATH = PROJECT_ROOT / "data/dataset_author_eval/sents/json/test_src10_tgt15_with_neutral.json"
TEXT_BASE_DIR = PROJECT_ROOT / "data/dataset_author_eval/sents/text"
OUTPUT_BASE_DIR = PROJECT_ROOT / "data/dataset_author_eval/sents/llama_factory_json"
DATASET_INFO_PATH = PROJECT_ROOT / "data/dataset_info.json"

SPLITS = ["train", "val"]


def load_text_file(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def texts_to_llama_factory_json(texts: list[str]) -> list[dict]:
    return [
        {"instruction": "Paraphrase", "input": text, "output": ""}
        for text in texts
    ]


def main():
    # 1. Get source author names
    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)
    source_authors = list(test_data["source_authors"].keys())
    print(f"Found {len(source_authors)} source authors: {source_authors}")

    # 2. Load dataset_info.json
    with open(DATASET_INFO_PATH, "r") as f:
        dataset_info = json.load(f)

    new_entries = {}

    for author in source_authors:
        author_text_dir = TEXT_BASE_DIR / author
        if not author_text_dir.is_dir():
            # Try comma variant (e.g. Dickens,_Charles)
            alt = author.replace("_", ",_", 1)
            author_text_dir = TEXT_BASE_DIR / alt
            if not author_text_dir.is_dir():
                print(f"  WARNING: No text directory found for {author}, skipping")
                continue

        author_out_dir = OUTPUT_BASE_DIR / author
        author_out_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            txt_file = author_text_dir / f"{split}.txt"
            if not txt_file.exists():
                print(f"  WARNING: {txt_file} not found, skipping")
                continue

            texts = load_text_file(txt_file)
            records = texts_to_llama_factory_json(texts)

            out_file = author_out_dir / f"{split}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

            print(f"  {author}/{split}.json: {len(records)} samples")

            # Build dataset_info key
            safe_author = author.replace(".", "_").replace(" ", "_")
            ds_key = f"author_eval_{safe_author}_{split}"
            rel_path = str(out_file.relative_to(PROJECT_ROOT / "data"))
            new_entries[ds_key] = {"file_name": rel_path}

    # 3. Merge new entries into dataset_info.json
    added = 0
    for key, value in new_entries.items():
        if key not in dataset_info:
            dataset_info[key] = value
            added += 1
        else:
            print(f"  dataset_info key '{key}' already exists, skipping")

    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone! Created JSON files under {OUTPUT_BASE_DIR}")
    print(f"Added {added} new entries to {DATASET_INFO_PATH}")
    print(f"\nDataset keys for use in run_rl_per_pair.py --train_dataset_name / --val_dataset_name:")
    for key in sorted(new_entries.keys()):
        print(f"  {key}")


if __name__ == "__main__":
    main()
