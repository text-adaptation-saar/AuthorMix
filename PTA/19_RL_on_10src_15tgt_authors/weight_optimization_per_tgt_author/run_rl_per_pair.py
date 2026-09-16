"""Default per-target GRPO training; stops after saving training weights and metadata."""
import os
import sys
import json
import gc
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import set_seed
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from llamafactory.x_my_pta_scripts.weight_optimize_hf.layerwise_weight_optimizer_with_RL_astrapop import train_layerwise_weights_with_grpo_astrapop
from llamafactory.x_my_pta_scripts.evaluation_patel.evaluation_metrics import _get_doc_embedding
DEFAULT_AUTHOR_TRAIN_DATA_DIR = PROJECT_ROOT / "data/dataset_author_eval/sents/text"
DEFAULT_STYLE_MODEL_PATH = "AIDA-UPM/star"


def generate_with_rl_model(
    model,
    tokenizer,
    input_texts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 128,
) -> List[str]:
    """Optional single-pass generation with an already loaded GRPO mixing model.

    Not called by training. Uses the original experiment's Llama prompt and
    decoding settings; adapt the prompt if your adapters use another format.
    Pass a tokenizer configured for left padding and with a pad token set.
    Leaves the model in eval mode. No scoring or repeated-seed evaluation.

    Example with an existing trained model and its tokenizer:
        outputs = generate_with_rl_model(model, tokenizer, ["Text to rewrite."])
    """
    device = str(next(model.parameters()).device)
    model.eval()
    outputs = []

    bos_token = ""
    prompts = []
    for text in input_texts:
        content = "\n\nOnly output the paraphrased version. Paraphrase\n" + text
        prompt = (
            f"{bos_token}<|start_header_id|>user<|end_header_id|>"
            f"{content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt)


    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                max_length=256,
                truncation=True,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            input_length = encoded["input_ids"].shape[1]

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated[:, input_length:], skip_special_tokens=True
            )
            outputs.extend([t.strip() for t in decoded])

    return outputs

def to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 else float(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj

def load_json(path) -> dict:
    with open(str(path), "r") as f:
        return json.load(f)

def load_text_file(path) -> List[str]:
    with open(str(path), "r") as f:
        return [line.strip() for line in f if line.strip()]

def source_author_to_dataset_key(author_name: str, split: str = "train") -> str:
    """Convert source author name to the dataset registry key.

    E.g.  'Baum_L._Frank' + 'train' -> 'author_eval_Baum_L__Frank_train'
    """
    safe = author_name.replace(".", "_").replace(" ", "_")
    return f"author_eval_{safe}_{split}"

def get_author_to_adapter_abbreviation() -> Dict[str, str]:
    return {
        "Mark_Twain": "MT",
        "Virginia_Woolf": "VW",
        "Vernon_Lee": "VL",
        "Charlotte_Perkins_Gilman": "CPG",
        "George_Orwell": "GO",
        "Jane_Austen": "JA",
        "Nathaniel_Hawthorne": "NH",
        "Oscar_Wilde": "OW",
        "P._G._Wodehouse": "PGW",
        "Samuel_Richardson": "SR",
    }

def author_name_to_train_data_dir(author_name: str, author_train_data_dir: Path) -> Path:
    author_dir = author_train_data_dir / author_name
    if author_dir.is_dir():
        return author_dir
    if "_" in author_name:
        fs_name = author_name.replace("_", ",_", 1)
        author_dir = author_train_data_dir / fs_name
        if author_dir.is_dir():
            return author_dir
    return author_train_data_dir / author_name

def get_adapter_paths(adapter_pattern_path: str, author_names: List[str]) -> List[str]:
    adapter_paths = []
    abbreviation_mapping = get_author_to_adapter_abbreviation()

    for author in author_names:
        adapter_abbr = abbreviation_mapping.get(author, None)
        if adapter_abbr is None:
            parts = author.replace("_", " ").replace(".", "").split()
            if len(parts) >= 2:
                adapter_abbr = (
                    "".join([p[0].upper() for p in parts])
                    if len(parts) > 2
                    else parts[0][0].upper() + parts[1][0].upper()
                )
            else:
                adapter_abbr = author[:3].upper().replace("_", "").replace(".", "")
        print(f"  Looking for adapter: {author} -> abbreviation: {adapter_abbr}")
        adapter_path_str = adapter_pattern_path.replace("{AUTHOR}", adapter_abbr).replace("*", adapter_abbr)
        adapter_path = Path(adapter_path_str)
        print(f"    Checking: {adapter_path} (exists: {adapter_path.exists()})")
        if adapter_path.exists() and adapter_path.is_dir():
            adapter_paths.append(str(adapter_path))
            print(f"  Found adapter for {author}: {adapter_path}")
        else:
            print(f"  Warning: Adapter not found for {author} at {adapter_path}")
    return adapter_paths

def find_related_authors_by_style_similarity(
    target_author: str,
    target_author_texts: List[str],
    style_model: SentenceTransformer,
    author_train_data_dir: Path,
    num_related: int = 3,
    train_samples_per_author: int = 1000,
) -> tuple:
    """Return (top_author_names, top_similarity_scores) sorted by descending similarity."""
    print(f"\nFinding related authors for {target_author} using style embeddings...")
    target_emb = _get_doc_embedding(target_author_texts, style_model)
    adapter_abbr_mapping = get_author_to_adapter_abbreviation()
    authors_with_adapters = set(adapter_abbr_mapping.keys())
    authors_with_adapters.discard(target_author)
    print(f"  Checking {len(authors_with_adapters)} candidate authors...")

    author_similarities = []
    for author_name in tqdm(authors_with_adapters, desc="  Computing similarities"):
        author_dir = author_name_to_train_data_dir(author_name, author_train_data_dir)
        train_file = author_dir / "train.txt"
        if not train_file.exists():
            continue
        train_texts = load_text_file(train_file)[:train_samples_per_author]
        if not train_texts:
            continue
        author_emb = _get_doc_embedding(train_texts, style_model)
        similarity = float((target_emb @ author_emb.T).item())
        author_similarities.append((author_name, similarity, len(train_texts)))

    author_similarities.sort(key=lambda x: x[1], reverse=True)
    top_related = [a for a, s, n in author_similarities[:num_related]]
    top_scores = [s for a, s, n in author_similarities[:num_related]]
    print(f"\n  Top {num_related} related authors:")
    for i, (a, s, n) in enumerate(author_similarities[:num_related], 1):
        print(f"    {i}. {a}: {s:.4f} (from {n} train samples)")
    return top_related, top_scores

def run_grpo_for_target(
    target_author: str,
    source_authors: List[str],
    test_data: dict,
    style_model: SentenceTransformer,
    base_model_path: str,
    adapter_pattern_path: str,
    author_train_data_dir: Path,
    num_adapters: int,
    val_sample_size: int,
    train_sample_size: int,
    rewrite_task_adapter: Optional[str] = None,
    learning_rate: float = 1e-3,
    max_steps: int = 500,
    num_generations: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.95,
    beta: float = 0.0,
    rl_batch_size: int = 4,
    max_length: int = 256,
    max_new_tokens: int = 256,
    logging_steps: int = 10,
    rl_style_model_path: str = DEFAULT_STYLE_MODEL_PATH,
    toward_exp: float = 0.5,
    mis_exp: float = 0.5,
    train_dataset_name: str = "wiki_train_for_paraNMT_model",
    val_dataset_name: str = "wiki_val_for_paraNMT_model",
    template: str = "llama3",
    weight_init: str = "zeros",
    seed: int = 42,
    grpo_output_dir: str = ".",
    resume_from_checkpoint: bool = True,
    save_checkpoint_without_model: bool = True,
):
    """Run one GRPO training for a target author using combined train/val from all source authors.

    Returns (final_weights, trainer, weight_history, results_base_dict).
    results_base_dict contains optimized_weights, adapter_paths, related_authors,
    rl_training_summary, rewrite_task_adapter, rl_config (training metadata).
    """
    print(f"\n{'='*80}")
    print(f"GRPO (per-target): {target_author} | sources: {len(source_authors)}")
    print(f"{'='*80}")

    if target_author not in test_data.get("target_authors", {}):
        raise ValueError(f"Target author {target_author} not found in author reference data")
    target_author_texts = test_data["target_authors"][target_author]

    # Combined source texts for reward (doc-level source embedding)
    source_texts_for_reward = []
    for src in source_authors:
        source_texts_for_reward.extend(test_data["source_authors"].get(src, []))
    if not source_texts_for_reward:
        raise ValueError("No source author texts for reward")

    # Find related authors and adapter paths (same as per-pair)
    top_related, top_style_scores = find_related_authors_by_style_similarity(
        target_author=target_author,
        target_author_texts=target_author_texts,
        style_model=style_model,
        author_train_data_dir=author_train_data_dir,
        num_related=num_adapters,
        train_samples_per_author=1000,
    )
    top_related = top_related[:num_adapters]
    top_style_scores = top_style_scores[:num_adapters]
    print(f"Selected top {num_adapters} related authors: {top_related}")

    adapter_paths = get_adapter_paths(adapter_pattern_path, top_related)
    rewrite_given = (
        rewrite_task_adapter is not None
        and str(rewrite_task_adapter).strip().strip('"').strip("'").strip()
    )
    if rewrite_given:
        rewrite_path = Path(str(rewrite_task_adapter).strip().strip('"').strip("'")).resolve()
        if not rewrite_path.exists() or not rewrite_path.is_dir():
            raise FileNotFoundError(f"rewrite_task_adapter not found: {rewrite_task_adapter}")
        adapter_paths = [str(rewrite_path)] + adapter_paths
        print(f"Prepended rewrite_task_adapter (total adapters: {len(adapter_paths)})")
    if not adapter_paths:
        raise ValueError("No adapters found.")

    adapter_names = []
    for ra in top_related:
        abbr = get_author_to_adapter_abbreviation().get(ra, ra[:3].upper())
        adapter_names.append(f"adapter-{abbr}")
    if rewrite_given:
        adapter_names = ["rewrite-adapter"] + adapter_names

    custom_init_weights = None
    if weight_init == "style_similarity":
        custom_init_weights = list(top_style_scores)
        if rewrite_given:
            mean_score = float(np.mean(custom_init_weights)) if custom_init_weights else 0.5
            custom_init_weights = [mean_score] + custom_init_weights
        print(f"  style_similarity init weights: {[f'{w:.4f}' for w in custom_init_weights]}")

    print(f"\nStarting GRPO training (max_steps={max_steps}, train_ds={train_dataset_name[:60]}...)")
    final_weights, trainer, weight_history = train_layerwise_weights_with_grpo_astrapop(
        tokenizer_path=base_model_path,
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        adapter_names=adapter_names,
        target_author_texts=target_author_texts,
        source_author_texts=source_texts_for_reward,
        output_dir=grpo_output_dir,
        batch_size=rl_batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        style_model_name=rl_style_model_path,
        toward_exp=toward_exp,
        mis_exp=mis_exp,
        num_generations=num_generations,
        beta=beta,
        logging_steps=logging_steps,
        temperature=temperature,
        top_p=top_p,
        max_steps=max_steps,
        train_dataset_name_as_llamafactory=train_dataset_name,
        val_dataset_name_as_llamafactory=val_dataset_name,
        template=template,
        max_samples_train=train_sample_size,
        max_samples_eval=val_sample_size,
        weight_init=weight_init,
        custom_init_weights=custom_init_weights,
        seed=seed,
        resume_from_checkpoint=resume_from_checkpoint,
        save_checkpoint_without_model=save_checkpoint_without_model,
        load_best_model_at_end=False, #just run for 300 steps and get final weights
    )

    results_base = {
        "optimized_weights": to_json_serializable(final_weights),
        "adapter_paths": adapter_paths,
        "related_authors": top_related,
        "rewrite_task_adapter": (
            str(rewrite_task_adapter).strip().strip('"').strip("'") if rewrite_given else None
        ),
        "rl_config": {
            "method": "GRPO",
            "reward_type": "astrapop_weighted_joint",
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "num_generations": num_generations,
            "temperature": temperature,
            "top_p": top_p,
            "beta": beta,
            "rl_batch_size": rl_batch_size,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "logging_steps": logging_steps,
            "rl_style_model_path": rl_style_model_path,
            "toward_exp": toward_exp,
            "mis_exp": mis_exp,
            "train_dataset_name": train_dataset_name,
            "val_dataset_name": val_dataset_name,
            "template": template,
            "weight_init": weight_init,
            "seed": seed,
            "num_adapters": len(adapter_paths),
                    },
    }
    if weight_history:
        rewards_over_time = [
            h.get("reward") for h in weight_history if h.get("reward") is not None
        ]
        results_base["rl_training_summary"] = {
            "total_steps": weight_history[-1].get("step", 0) if weight_history else 0,
            "final_reward": rewards_over_time[-1] if rewards_over_time else None,
            "best_reward": max(rewards_over_time) if rewards_over_time else None,
            "num_logged_steps": len(weight_history),
        }
    else:
        results_base["rl_training_summary"] = {}

    return final_weights, trainer, weight_history, results_base

def main():
    parser = argparse.ArgumentParser(description="Default per-target GRPO training")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_pattern_path", type=str, required=True,
                            help="Adapter path pattern with {AUTHOR} placeholder")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_directory_path", type=str, required=True)
    parser.add_argument("--author_train_data_dir", type=str, default=str(DEFAULT_AUTHOR_TRAIN_DATA_DIR))
    parser.add_argument("--style_model_path", type=str, default=DEFAULT_STYLE_MODEL_PATH,
                            help="Style model for adapter selection (STAR)")
    parser.add_argument("--target_author", type=str, default=None,
                            help="If set, iterate only source authors for this target")
    parser.add_argument("--num_adapters", type=int, required=True)
    parser.add_argument("--val_sample_size", type=int, default=50)
    parser.add_argument("--train_sample_size", type=int, default=100)
    parser.add_argument("--rewrite_task_adapter", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--rl_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--rl_style_model_path", type=str,
                            default=DEFAULT_STYLE_MODEL_PATH,
                            help="Style model used by ASTRAPOP reward (default: STAR)")
    parser.add_argument("--toward_exp", type=float, default=0.5,
                            help="Exponent for toward score in ASTRAPOP reward (default: 0.5)")
    parser.add_argument("--mis_exp", type=float, default=0.5,
                            help="Exponent for MIS score in ASTRAPOP reward (default: 0.5)")
    parser.add_argument("--train_dataset_name", type=str, default="wiki_train_for_paraNMT_model",
                            help="Registered dataset name for GRPO training (ignored when --use_author_specific_dataset)")
    parser.add_argument("--val_dataset_name", type=str, default="wiki_val_for_paraNMT_model",
                            help="Registered dataset name for GRPO validation (ignored when --use_author_specific_dataset)")
    parser.add_argument("--use_author_specific_dataset", type=int, default=0, choices=[0, 1],
                            help="1 = use per-source-author datasets (author_eval_<AUTHOR>_train/val) "
                                 "instead of the generic train/val dataset. Default: 0")
    parser.add_argument("--template", type=str, default="llama3", choices=["llama3"])
    parser.add_argument("--weight_init", type=str, default="zeros",
                            choices=["zeros", "ones", "equal",
                                     "randint_0_1", "randint_-2_2", "uniform_-2_2",
                                     "random_0_1", "random_-1_1", "random_0.5_1.3",
                                     "random_0_1_adapter_level", "random_-1_1_adapter_level",
                                     "random_0.5_1.3_adapter_level",
                                     "style_similarity"])
    parser.add_argument("--seed", type=int, default=42,
                            help="Random seed for RL training (default: 42)")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True,
                            help="Resume GRPO from latest checkpoint in output dir if present (default: True).")
    parser.add_argument("--no_resume_from_checkpoint", action="store_false", dest="resume_from_checkpoint",
                            help="Disable resume; start training from scratch.")
    parser.add_argument("--save_checkpoint_without_model", action="store_true", default=True,
                            help="Save only trainer/optimizer/scheduler state and optimized_weights.json, not model weights (default: True).")
    parser.add_argument("--no_save_checkpoint_without_model", action="store_false", dest="save_checkpoint_without_model",
                            help="Save full checkpoints including model.safetensors.")

    args = parser.parse_args()
    set_seed(args.seed)
    test_data = load_json(Path(args.test_data_path))
    source_authors = list(test_data.get("source_authors", {}))
    target_authors = list(test_data.get("target_authors", {}))
    if not source_authors or not target_authors:
        parser.error("Author reference JSON must contain nonempty source_authors and target_authors")
    if args.target_author and args.target_author not in target_authors:
        parser.error("Requested target author is absent from the author reference JSON")
    style_model = SentenceTransformer(args.style_model_path)
    for tgt_author in ([args.target_author] if args.target_author else target_authors):
        tgt_output_dir = Path(args.output_directory_path) / f"tgt_author_{tgt_author}"
        tgt_output_dir.mkdir(parents=True, exist_ok=True)
        train_dataset_name = args.train_dataset_name
        val_dataset_name = args.val_dataset_name
        if args.use_author_specific_dataset:
            train_dataset_name = ",".join(source_author_to_dataset_key(src, "train") for src in source_authors)
            val_dataset_name = ",".join(source_author_to_dataset_key(src, "val") for src in source_authors)
        grpo_output_dir = str(tgt_output_dir / "grpo_training")
        final_weights, trainer, weight_history, training_result = run_grpo_for_target(
                    target_author=tgt_author,
                    source_authors=source_authors,
                    test_data=test_data,
                    style_model=style_model,
                    base_model_path=args.base_model_path,
                    adapter_pattern_path=args.adapter_pattern_path,
                    author_train_data_dir=Path(args.author_train_data_dir),
                    num_adapters=args.num_adapters,
                    val_sample_size=args.val_sample_size,
                    train_sample_size=args.train_sample_size,
                    rewrite_task_adapter=args.rewrite_task_adapter,
                    learning_rate=args.learning_rate,
                    max_steps=args.max_steps,
                    num_generations=args.num_generations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    beta=args.beta,
                    rl_batch_size=args.rl_batch_size,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    logging_steps=args.logging_steps,
                    rl_style_model_path=args.rl_style_model_path,
                    toward_exp=args.toward_exp,
                    mis_exp=args.mis_exp,
                    train_dataset_name=train_dataset_name,
                    val_dataset_name=val_dataset_name,
                    template=args.template,
                    weight_init=args.weight_init,
                    seed=args.seed,
                    grpo_output_dir=grpo_output_dir,
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    save_checkpoint_without_model=args.save_checkpoint_without_model,
                )
        (tgt_output_dir / "training_results.json").write_text(
            json.dumps(to_json_serializable(training_result), indent=2) + "\n"
        )
        (tgt_output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        del trainer, final_weights, weight_history
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
