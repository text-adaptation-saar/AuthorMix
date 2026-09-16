"""
train_weights_with_grpo_astrapop.py - Optimize adapter weights using GRPO
with ASTRAPOP weighted joint score as the reward function.

Compared to layerwise_weight_optimizer_with_RL.py (harmonic-mean reward),
this version uses:
  reward_i = (toward_i ** toward_exp) * (mis_i ** mis_exp)
where toward and MIS are computed per-sample (not doc-level).
"""

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_callback import TrainerState
from datasets import load_dataset
from typing import List, Optional, Union
import json
import os
import matplotlib.pyplot as plt

from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_data_loader.custom_data_prep import (
    prepare_dataset,
    prepare_grpo_dataset,
    prepare_grpo_dataset_llama_facotry_compatible,
)
from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_lm.learnable_layerwise_weight_module_with_layer_wrapper_layerwiseFalse import (
    AdapterwiseLearnableWeightModule,
)
from llamafactory.x_my_pta_scripts.weight_optimize_hf.reward_compter.custom_loss_computer_astrapop import (
    ASTRAPOPRewardComputer,
    TRLAstrapopRewardWrapper,
)

from transformers import TrainerCallback
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

import matplotlib.pyplot as plt
import numpy as np


def visualize_layer_weights(model, output_path="layer_weights.png"):
    """Visualize adapter-wise weights as a bar chart (one weight per adapter)."""
    weights = model.get_all_weights().detach().cpu().numpy()
    if weights.ndim == 1:
        plt.figure(figsize=(max(6, len(model.adapter_names) * 0.8), 5))
        x = range(len(model.adapter_names))
        plt.bar(x, weights, color='steelblue', edgecolor='black')
        plt.xticks(x, model.adapter_names, rotation=45, ha='right')
        plt.xlabel('Adapter')
        plt.ylabel('Weight')
        plt.title('Adapter-wise Weights')
        plt.tight_layout()
        plt.savefig(output_path)
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, aspect='auto', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Adapter')
        plt.ylabel('Layer')
        plt.title('Per-Layer Adapter Weights')
        plt.xticks(range(len(model.adapter_names)), model.adapter_names)
        plt.yticks(range(0, model.num_layers, 4))
        plt.tight_layout()
        plt.savefig(output_path)
    print(f"Saved weight visualization to {output_path}")


def _to_json_serializable(obj):
    """Convert numpy/torch scalars to native Python for JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to checkpoint dir with highest step number, or None."""
    out = os.path.join(output_dir)
    if not os.path.isdir(out):
        return None
    checkpoints = []
    for name in os.listdir(out):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, os.path.join(out, name)))
            except (IndexError, ValueError):
                continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _step_from_checkpoint_path(checkpoint_path: str) -> Optional[int]:
    """Extract step number from path like '.../checkpoint-4600' or 'checkpoint-4600'. Returns None if not parseable."""
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.split("-")[1])
    except (IndexError, ValueError):
        return None


def _load_optimized_weights_into_model(model, checkpoint_dir: str) -> bool:
    """Load weights from checkpoint's optimized_weights.json into model.weight_logits. Supports adapter-wise (1D) and per-layer (2D). Returns True if loaded."""
    path = os.path.join(checkpoint_dir, "optimized_weights.json")
    if not os.path.isfile(path) or not hasattr(model, "weight_logits"):
        return False
    with open(path, "r") as f:
        data = json.load(f)
    adapter_names = list(model.adapter_names)
    w = model.weight_logits
    with torch.no_grad():
        if w.ndim == 1:
            # Adapter-wise: load from adapter_weights or per_layer_weights["layer_0"]
            num_adapters = w.shape[0]
            arr = np.zeros((num_adapters,), dtype=np.float32)
            adapter_weights = data.get("adapter_weights")
            if isinstance(adapter_weights, dict):
                for adapter_idx, name in enumerate(adapter_names):
                    if name in adapter_weights:
                        arr[adapter_idx] = float(adapter_weights[name])
            elif isinstance(adapter_weights, (list, tuple)) and len(adapter_weights) == num_adapters:
                for i in range(num_adapters):
                    arr[i] = float(adapter_weights[i])
            else:
                per_layer = data.get("per_layer_weights") or {}
                row = per_layer.get("layer_0") or per_layer.get("0")
                if row:
                    for adapter_idx, name in enumerate(adapter_names):
                        if name in row:
                            arr[adapter_idx] = float(row[name])
                else:
                    return False
            model.weight_logits.copy_(torch.tensor(arr, dtype=w.dtype, device=w.device))
        else:
            # Per-layer (2D)
            per_layer = data.get("per_layer_weights")
            if not per_layer:
                return False
            num_layers, num_adapters = w.shape[0], w.shape[1]
            arr = np.zeros((num_layers, num_adapters), dtype=np.float32)
            for layer_idx in range(num_layers):
                key = f"layer_{layer_idx}"
                if key not in per_layer:
                    continue
                row = per_layer[key]
                for adapter_idx, name in enumerate(adapter_names):
                    if name in row:
                        arr[layer_idx, adapter_idx] = float(row[name])
            model.weight_logits.copy_(torch.tensor(arr, dtype=w.dtype, device=w.device))
    return True


def _checkpoint_has_model_weights(ckpt_dir: str) -> bool:
    """Return True if checkpoint dir contains model weight files (safetensors or pytorch)."""
    return (
        os.path.isfile(os.path.join(ckpt_dir, SAFE_WEIGHTS_NAME))
        or os.path.isfile(os.path.join(ckpt_dir, WEIGHTS_NAME))
        or os.path.isfile(os.path.join(ckpt_dir, SAFE_WEIGHTS_INDEX_NAME))
        or os.path.isfile(os.path.join(ckpt_dir, WEIGHTS_INDEX_NAME))
    )


def _remove_model_weight_files_from_checkpoint(ckpt_dir: str) -> None:
    """Remove model weight files from a checkpoint dir; keep optimizer, scheduler, trainer_state, optimized_weights.json."""
    removed = []
    for name in (
        SAFE_WEIGHTS_NAME,
        WEIGHTS_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
    ):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(name)
            except OSError as e:
                print(f"  [checkpoint] Warning: could not remove {path}: {e}")
    # Sharded safetensors: model.safetensors.index.json references model-00001-of-00003.safetensors etc.
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("model-") and (f.endswith(".safetensors") or f.endswith(".bin")):
                path = os.path.join(ckpt_dir, f)
                try:
                    os.remove(path)
                    removed.append(f)
                except OSError as e:
                    print(f"  [checkpoint] Warning: could not remove {path}: {e}")
    if removed:
        print(f"  [checkpoint] Removed model weight files (save_checkpoint_without_model): {removed}")


class WeightMonitorCallback(TrainerCallback):
    """Monitor weight evolution during GRPO training and save weights at checkpoints. Supports adapter-wise (1D) and per-layer (2D) models."""

    def __init__(self, save_checkpoint_without_model: bool = False):
        self.weight_history = []
        self.save_checkpoint_without_model = save_checkpoint_without_model

    @staticmethod
    def _weights_to_dict(model):
        """Extract current weights as a JSON-serialisable dict. Supports adapter-wise (1D) and per-layer (2D)."""
        weights = model.get_all_weights().detach().cpu().numpy()
        adapter_names = list(model.adapter_names)
        if weights.ndim == 1:
            # Adapter-wise: one value per adapter (num_layers needed for merge_layerwise_weights)
            adapter_weights = {name: float(weights[adapter_idx]) for adapter_idx, name in enumerate(adapter_names)}
            summary = {name: {"value": float(weights[i]), "mean": float(weights[i]), "std": 0.0, "min": float(weights[i]), "max": float(weights[i])} for i, name in enumerate(adapter_names)}
            return {
                "adapter_names": adapter_names,
                "num_adapters": int(weights.shape[0]),
                "num_layers": int(getattr(model, "num_layers", 0)),
                "adapter_weights": adapter_weights,
                "adapter_summary": summary,
            }
        # Per-layer (2D)
        per_layer = {}
        for layer_idx in range(weights.shape[0]):
            per_layer[f"layer_{layer_idx}"] = {
                name: float(weights[layer_idx, adapter_idx])
                for adapter_idx, name in enumerate(adapter_names)
            }
        summary = {}
        for adapter_idx, name in enumerate(adapter_names):
            col = weights[:, adapter_idx]
            summary[name] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
            }
        return {
            "adapter_names": adapter_names,
            "num_layers": int(weights.shape[0]),
            "num_adapters": int(weights.shape[1]),
            "per_layer_weights": per_layer,
            "adapter_summary": summary,
        }

    @staticmethod
    def _get_latest_eval_metrics(state):
        """Return the most recent eval metrics from state.log_history (for current checkpoint)."""
        if not state.log_history:
            return None
        for i in range(len(state.log_history) - 1, -1, -1):
            entry = state.log_history[i]
            if "eval_loss" in entry or "eval_reward" in entry:
                return _to_json_serializable(entry)
        return None

    def on_save(self, args, state, control, model=None, **kwargs):
        """Save optimised weights JSON alongside every checkpoint and append to eval history."""
        if model is None or not hasattr(model, "get_all_weights"):
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        payload = self._weights_to_dict(model)
        payload["step"] = state.global_step
        payload["epoch"] = state.epoch
        out_path = os.path.join(ckpt_dir, "optimized_weights.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [checkpoint] Saved weights to {out_path}")

        # Append to single eval checkpoint history JSON
        history_path = os.path.join(args.output_dir, "eval_checkpoint_history.json")
        eval_metrics = self._get_latest_eval_metrics(state)
        entry = {
            "step": state.global_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "eval_metrics": eval_metrics,
            "weights": self._weights_to_dict(model),
        }
        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    data = json.load(f)
                checkpoints = data.get("checkpoints", [])
            else:
                checkpoints = []
            checkpoints.append(entry)
            with open(history_path, "w") as f:
                json.dump({"checkpoints": checkpoints}, f, indent=2)
            print(f"  [checkpoint] Appended eval + weights to {history_path}")
        except Exception as e:
            print(f"  [checkpoint] Failed to update eval history: {e}")

        if self.save_checkpoint_without_model:
            _remove_model_weight_files_from_checkpoint(ckpt_dir)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if model is not None and hasattr(model, 'get_all_weights'):
            weights = model.get_all_weights().detach().cpu().numpy()
            if weights.ndim == 1:
                adapter_stats = {
                    name: {"mean": float(weights[i]), "std": 0.0, "min": float(weights[i]), "max": float(weights[i])}
                    for i, name in enumerate(model.adapter_names)
                }
            else:
                adapter_stats = {}
                for adapter_idx, adapter_name in enumerate(model.adapter_names):
                    col = weights[:, adapter_idx]
                    adapter_stats[adapter_name] = {
                        "mean": float(col.mean()),
                        "std": float(col.std()),
                        "min": float(col.min()),
                        "max": float(col.max()),
                    }

            step_info = {
                "step": state.global_step,
                "epoch": state.epoch,
                "weights_shape": weights.shape,
                "adapter_stats": adapter_stats,
                "reward": logs.get("reward", None) if logs else None,
                "kl": logs.get("kl", None) if logs else None,
            }
            self.weight_history.append(step_info)

            if logs:
                print(f"\n[Step {state.global_step}]")
                label = "Adapter weights:" if weights.ndim == 1 else f"Per-Adapter Statistics (across {weights.shape[0]} layers):"
                print(f"  {label}")
                for adapter_name, stats in adapter_stats.items():
                    print(f"    {adapter_name}: mean={stats['mean']:.4f}, "
                          f"std={stats['std']:.4f}, "
                          f"min={stats['min']:.4f}, "
                          f"max={stats['max']:.4f}")
                if "reward" in logs:
                    print(f"  Mean Reward: {logs['reward']:.4f}")
                if "kl" in logs:
                    print(f"  KL Divergence: {logs['kl']:.4f}")


# Checkpoint file names used by HuggingFace Trainer (must match trainer.py)
_TRAINER_STATE_NAME = "trainer_state.json"
_OPTIMIZER_NAME = "optimizer.pt"
_SCHEDULER_NAME = "scheduler.pt"


class GRPOTrainerResumableWithoutModel(GRPOTrainer):
    """
    GRPOTrainer that can resume from checkpoints that have no model weight files.
    When the checkpoint dir has no model.safetensors/pytorch_model.bin, only
    optimizer, scheduler, and trainer_state are loaded; the model is assumed
    to be already correct (e.g. from optimized_weights.json loaded before train()).
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, **kwargs):
        if _checkpoint_has_model_weights(resume_from_checkpoint):
            return super()._load_from_checkpoint(resume_from_checkpoint, **kwargs)
        # Load only optimizer, scheduler, trainer_state; skip model.
        if self.args.should_save and self.is_world_process_zero():
            print(f"  [resume] Checkpoint has no model weights; loading optimizer/scheduler/state only.")
        state_path = os.path.join(resume_from_checkpoint, _TRAINER_STATE_NAME)
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                state_dict = json.load(f)
            from_dict = getattr(TrainerState, "from_dict", None)
            if callable(from_dict):
                self.state = from_dict(state_dict)
            else:
                for k, v in state_dict.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
        opt_path = os.path.join(resume_from_checkpoint, _OPTIMIZER_NAME)
        if os.path.isfile(opt_path) and self.optimizer is not None:
            opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
            if isinstance(opt_state, dict):
                self.optimizer.load_state_dict(opt_state)
            elif hasattr(opt_state, "state_dict"):
                self.optimizer.load_state_dict(opt_state.state_dict())
            else:
                self.optimizer.load_state_dict(opt_state)
        sched_path = os.path.join(resume_from_checkpoint, _SCHEDULER_NAME)
        if os.path.isfile(sched_path) and self.lr_scheduler is not None:
            sched_state = torch.load(sched_path, map_location="cpu", weights_only=False)
            self.lr_scheduler.load_state_dict(sched_state)
        return None


def train_adapterwise_weights_with_grpo_astrapop(
    base_model_path: str,
    adapter_paths: List[str],
    adapter_names: List[str],
    target_author_texts: List[str],
    source_author_texts: Optional[List[str]] = None,
    train_dataset=None,
    eval_dataset=None,
    output_dir: str = "./grpo_weight_optimization_astrapop",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    max_length: int = 256,
    max_new_tokens: int = 256,
    style_model_name: str = "/scratch/common_models/AIDA-UPM/star",
    toward_exp: float = 0.5,
    mis_exp: float = 0.5,
    num_generations: int = 4,
    beta: float = 0.0,
    logging_steps: int = 10,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_steps: int = 3000,
    tokenizer_path: str = None,
    train_dataset_name_as_llamafactory=None,
    val_dataset_name_as_llamafactory=None,
    template="llama3",
    max_samples_train: int = 100,
    max_samples_eval: int = 1000,
    weight_init: str = "zeros",
    custom_init_weights: Optional[List[float]] = None,
    seed: int = 42,
    resume_from_checkpoint: Optional[Union[bool, str]] = None,
    save_checkpoint_without_model: bool = False,
):
    """
    Optimize adapter merge weights using GRPO with ASTRAPOP weighted joint-score reward.

    Per-sample reward:  reward_i = (toward_i ** toward_exp) * (mis_i ** mis_exp)

    This optimizes ONLY the merge weights (not adapter or base-model parameters).

    Args:
        base_model_path: Base model path
        adapter_paths: List of adapter paths to merge
        adapter_names: Names for adapters
        target_author_texts: Style reference texts for the target author
        source_author_texts: Optional source author texts (for doc-level source embedding)
        output_dir: Output directory
        style_model_name: SentenceTransformer model for style embeddings (STAR, etc.)
        toward_exp: Exponent for toward score (higher = more style-dominant)
        mis_exp: Exponent for MIS score (higher = more content-preservation)
        num_generations: Number of samples per prompt (GRPO)
        beta: KL divergence coefficient
        resume_from_checkpoint: If None or True or "auto", resume from latest checkpoint in output_dir if present.
            If a path string, resume from that checkpoint. If False or "", do not resume.
        save_checkpoint_without_model: If True, after each checkpoint save we remove model weight files
            (model.safetensors, pytorch_model.bin, etc.) from the checkpoint dir, keeping only trainer state,
            optimizer, scheduler, and optimized_weights.json. Resuming still works: we pre-load
            optimized_weights.json into the model and use a trainer that loads only optimizer/scheduler/state
            when the checkpoint has no model file. Final save to output_dir/final_model is also skipped
            when True (weights are in output_dir/optimized_weights.json).
        (remaining args same as the harmonic variant)

    Resume behaviour and verification:
        - We pre-load optimized_weights.json into model.weight_logits; Trainer.train(resume_from_checkpoint=...)
          then restores full trainer state from the checkpoint (model state_dict, optimizer, scheduler, RNG, global_step).
        - After training, a "RESUME VERIFICATION" block is printed: resumed step (from path), final global_step.
          If resume worked, final_step should be > resumed_step (or equal if no new steps ran). Check logs for this block.

    Does resume work like default ML (true resume)?
        Yes. When resume_from_checkpoint is passed, the Hugging Face Trainer (and TRL GRPOTrainer inheriting it):
        - Loads model state_dict from the checkpoint (overwriting current model, including weight_logits).
        - Loads optimizer state (e.g. Adam momentum buffers) from optimizer.pt.
        - Loads scheduler state from scheduler.pt so LR schedule continues from the resumed step.
        - Restores trainer.state.global_step so the training loop continues from that step (e.g. 4601 after 4600).
        - Restores RNG states for reproducibility.
        So training continues from where it stopped: same weights, optimizer state, step count, and LR schedule.
        Data: the dataloader is rebuilt each run; the loop runs from global_step to max_steps, so only remaining
        steps are executed. Shuffling uses the restored RNG/epoch so behaviour is consistent.
    """

    print("\n" + "="*80)
    print("GRPO WEIGHT OPTIMIZATION (ASTRAPOP REWARD)")
    print("="*80)
    print(f"Base model: {base_model_path}")
    print(f"Adapters: {len(adapter_paths)}")
    for i, (path, name) in enumerate(zip(adapter_paths, adapter_names), 1):
        print(f"  {i}. {name}: {path}")
    print(f"Train samples: {len(train_dataset) if train_dataset is not None else train_dataset_name_as_llamafactory}")
    print(f"Eval samples: {len(eval_dataset) if eval_dataset is not None else val_dataset_name_as_llamafactory}")
    print(f"Style references: {len(target_author_texts)}")
    print(f"Reward: ASTRAPOP  toward^{toward_exp} * mis^{mis_exp}")
    print(f"Style model: {style_model_name}")
    print(f"Optimization target: Adapter merge weights only!")
    print(f"Seed: {seed}")
    print("="*80 + "\n")

    # 0. Set seed before any random initialization (weight logits, etc.)
    from transformers import set_seed
    set_seed(seed)

    # Resolve resume_from_checkpoint: None/"auto"/True = latest in output_dir; False/"" = no resume; else path
    # When resuming, Trainer.train(resume_from_checkpoint=...) restores: model state, optimizer, scheduler, RNG, global_step.
    # We also pre-load optimized_weights.json into model.weight_logits so weights are correct before the trainer loads the checkpoint.
    resume_path = None
    resumed_step = None  # step number from checkpoint path, for verification logging
    if resume_from_checkpoint not in (None, False, ""):
        if resume_from_checkpoint in (True, "auto"):
            resume_path = _find_latest_checkpoint(output_dir)
            if resume_path:
                resumed_step = _step_from_checkpoint_path(resume_path)
                print(f"Resume: auto-detected latest checkpoint: {resume_path} (step {resumed_step})")
        elif isinstance(resume_from_checkpoint, str) and os.path.isdir(resume_from_checkpoint):
            resume_path = resume_from_checkpoint
            resumed_step = _step_from_checkpoint_path(resume_path)
            print(f"Resume: using checkpoint: {resume_path} (step {resumed_step})")
    if resume_path and not os.path.isdir(resume_path):
        resume_path = None
        resumed_step = None
        print("Resume: checkpoint path not found, starting from scratch.")

    # 1. Create model with learnable adapter-wise weights (one weight per adapter)
    print("\nInitializing model with learnable adapter-wise weights...")
    model = AdapterwiseLearnableWeightModule(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        adapter_names=adapter_names,
        weight_init=weight_init,
        custom_init_weights=custom_init_weights,
    )

    if resume_path:
        if _load_optimized_weights_into_model(model, resume_path):
            print(f"Loaded optimized weights from {resume_path}/optimized_weights.json")
        else:
            print(f"Could not load optimized_weights.json from {resume_path}; trainer will load from checkpoint.")

    print("\nInitial adapter-wise weights:")
    model.print_weights(verbose=False)

    # Verify only weights are trainable
    print("\n" + "="*80)
    print("TRAINABLE PARAMETERS CHECK")
    print("="*80)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
            print(f"TRAINABLE: {name} - shape: {param.shape}")

    if len(trainable_params) == 1 and trainable_params[0][0] == 'weight_logits':
        print("\nCorrect! Only weight_logits are trainable")
    else:
        print("\nWarning: More than just weights are trainable!")
    print("="*80 + "\n")

    # 2. Create ASTRAPOP reward function
    print("Initializing ASTRAPOP reward function...")
    reward_computer = ASTRAPOPRewardComputer(
        target_author_texts=target_author_texts,
        source_author_texts=source_author_texts,
        style_model_name=style_model_name,
        toward_exp=toward_exp,
        mis_exp=mis_exp,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    reward_fn = TRLAstrapopRewardWrapper(reward_computer)
    print("ASTRAPOP reward function ready\n")

    # 3. Prepare datasets
    print("Preparing datasets for GRPO...")
    train_dataset_grpo, tokenizer = prepare_grpo_dataset_llama_facotry_compatible(
        model_name_or_path=tokenizer_path if tokenizer_path is not None else base_model_path,
        adapter_name_or_path=adapter_paths,
        dataset=train_dataset_name_as_llamafactory,
        template=template,
        cutoff_len=max_length,
        max_samples=max_samples_train,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    eval_dataset_grpo, tokenizer_val = prepare_grpo_dataset_llama_facotry_compatible(
        model_name_or_path=tokenizer_path if tokenizer_path is not None else base_model_path,
        adapter_name_or_path=adapter_paths,
        dataset=val_dataset_name_as_llamafactory,
        template=template,
        cutoff_len=max_length,
        max_samples=max_samples_eval,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    print(f"Prepared {len(train_dataset_grpo)} train queries")
    print(f"Prepared {len(eval_dataset_grpo)} eval queries")

    if len(train_dataset_grpo) > 0:
        sample = train_dataset_grpo[0]
        print(f"\nSample query:\n{sample}...\n")

    print("Configuring GRPO trainer...")

    explicit_config = {
        "output_dir": output_dir,
        "logging_steps": logging_steps,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "num_generations": num_generations,
        "gradient_accumulation_steps": 8,
        "report_to": "none",
        "beta": beta,
        "per_device_eval_batch_size": num_generations,
        "max_prompt_length": max_length,
        "max_completion_length": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "save_steps": logging_steps * 10,
        "eval_steps": logging_steps * 10,
        "fp16": True,
        "remove_unused_columns": True,
        "log_completions": True,
        "gradient_checkpointing": False,
        "seed": seed,
    }
        #    "metric_for_best_model": "eval_reward",  # GRPO logs eval reward; higher is better
        # "greater_is_better": True,
        # "save_total_limit": 3,  

    print("\n" + "=" * 80)
    print("GRPO TRAINER CONFIGURATION (Explicitly Set Values)")
    print("=" * 80)
    print(json.dumps(explicit_config, indent=2, default=str))
    print("=" * 80 + "\n")

    grpo_config = GRPOConfig(**explicit_config)

    print("GRPO config ready")
    print("\n" + "=" * 80)
    print("GRPO TRAINER CONFIGURATION")
    print("=" * 80)
    print(json.dumps(grpo_config.to_dict(), indent=2, default=str))
    print("=" * 80 + "\n")

    # 4. Create GRPO Trainer
    print("\nInitializing GRPO trainer...")
    if save_checkpoint_without_model:
        print("  Using GRPOTrainerResumableWithoutModel (checkpoints will not store model weights).")

    weight_monitor = WeightMonitorCallback(save_checkpoint_without_model=save_checkpoint_without_model)

    trainer_cls = GRPOTrainerResumableWithoutModel if save_checkpoint_without_model else GRPOTrainer
    trainer = trainer_cls(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset_grpo,
        eval_dataset=eval_dataset_grpo,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[weight_monitor],
    )

    print("Trainer ready\n")

    # 5. Train
    print("="*80)
    print("STARTING GRPO WEIGHT OPTIMIZATION (ASTRAPOP REWARD)")
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
    print("="*80 + "\n")

    train_result = trainer.train(resume_from_checkpoint=resume_path)

    # Resume verification: Trainer restores model, optimizer, scheduler, RNG, and global_step from checkpoint.
    if resume_path:
        final_step = getattr(trainer.state, "global_step", None)
        print("\n" + "="*80)
        print("RESUME VERIFICATION")
        print("="*80)
        print(f"  Resumed from checkpoint: {resume_path}")
        print(f"  Resumed step (from path): {resumed_step}")
        print(f"  Final global_step (trainer.state): {final_step}")
        if resumed_step is not None and final_step is not None:
            if final_step > resumed_step:
                print(f"  -> OK: training continued from step {resumed_step} to {final_step}")
            elif final_step == resumed_step:
                print(f"  -> Step unchanged (may have hit max_steps or no new steps run)")
            else:
                print(f"  -> WARNING: final_step < resumed_step (unexpected)")
        print("  (Trainer restores: model state_dict, optimizer, scheduler, RNG, global_step)")
        print("="*80 + "\n")

    # 6. Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    print("\nFinal adapter-wise weights:")
    model.print_weights(verbose=False)

    visualize_layer_weights(model, os.path.join(output_dir, "adapter_weights_final.png"))

    print("\nFinal optimized weights:")
    final_weights = model.print_weights()

    if weight_monitor.weight_history:
        print("\n--- Weight Evolution During GRPO ---")
        print(f"{'Step':<8} {'Epoch':<8} {'Reward':<10} {'KL':<10}")
        print("-" * 60)

        history = weight_monitor.weight_history
        indices = []
        indices.extend(range(min(3, len(history))))
        if len(history) > 6:
            indices.append(len(history) // 2)
        if len(history) > 3:
            indices.extend(range(max(3, len(history) - 3), len(history)))
        indices = sorted(set(indices))

        for i in indices:
            entry = history[i]
            reward_str = f"{entry['reward']:.4f}" if entry['reward'] is not None else "N/A"
            kl_str = f"{entry['kl']:.4f}" if entry['kl'] is not None else "N/A"
            epoch_str = f"{entry['epoch']:.1f}" if entry['epoch'] is not None else "N/A"

            print(f"{entry['step']:<8} {epoch_str:<8} {reward_str:<10} {kl_str:<10}")

            if 'adapter_stats' in entry:
                for adapter_name, stats in entry['adapter_stats'].items():
                    print(f"  {adapter_name}: mean={stats['mean']:.4f}, "
                          f"std={stats['std']:.4f}, "
                          f"min={stats['min']:.4f}, "
                          f"max={stats['max']:.4f}")

            if i < len(indices) - 1 and indices[i + 1] - i > 1:
                print("...")

        print("-" * 60)

    # 7. Save results
    os.makedirs(output_dir, exist_ok=True)

    weight_detail = WeightMonitorCallback._weights_to_dict(model)
    results = {
        'optimized_weights': final_weights,
        'adapter_summary': weight_detail['adapter_summary'],
        'adapter_names': adapter_names,
        'adapter_paths': adapter_paths,
    }
    if 'adapter_weights' in weight_detail:
        results['adapter_weights'] = weight_detail['adapter_weights']
    if 'per_layer_weights' in weight_detail:
        results['per_layer_weights'] = weight_detail['per_layer_weights']
    results.update({
        'training_config': {
            'method': 'GRPO',
            'reward_type': 'astrapop_weighted_joint',
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_generations': num_generations,
            'kl_coef': beta,
            'style_model': style_model_name,
            'toward_exp': toward_exp,
            'mis_exp': mis_exp,
        },
        'weight_history': weight_monitor.weight_history,
    })

    with open(f"{output_dir}/optimized_weights.json", 'w') as f:
        json.dump(results, f, indent=2)

    if not save_checkpoint_without_model:
        trainer.save_model(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")
        print(f"\nResults saved to {output_dir}/optimized_weights.json")
        print(f"Model saved to {output_dir}/final_model")
    else:
        print(f"\nResults saved to {output_dir}/optimized_weights.json")
        print("(Skipped saving model weights to final_model; use optimized_weights.json for weights.)")
    print("="*80 + "\n")

    # 8. Plots
    log_history = trainer.state.log_history

    reward_steps, rewards = [], []
    loss_steps, losses = [], []
    kl_steps, kls = [], []

    for e in log_history:
        s = e.get("step")
        if s is None:
            continue
        if "reward" in e:
            reward_steps.append(s)
            rewards.append(e["reward"])
        if "loss" in e:
            loss_steps.append(s)
            losses.append(e["loss"])
        if "kl" in e:
            kl_steps.append(s)
            kls.append(e["kl"])

    if rewards:
        plt.figure(figsize=(10, 4))
        plt.plot(reward_steps, rewards, label="Rewards")
        plt.xlabel("Training Steps")
        plt.ylabel("Reward")
        plt.title("ASTRAPOP Reward Over Steps")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "reward_over_steps.png"))
    else:
        print("No reward logs found to plot.")

    if losses:
        plt.figure(figsize=(10, 4))
        plt.plot(loss_steps, losses, label="Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Steps")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_over_steps.png"))
    else:
        print("No loss logs found to plot.")

    if kls:
        plt.figure(figsize=(10, 4))
        plt.plot(kl_steps, kls, label="KL")
        plt.xlabel("Training Steps")
        plt.ylabel("KL")
        plt.title("KL Over Steps")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "kl_over_steps.png"))
    else:
        print("No KL logs found to plot.")

    return final_weights, trainer, weight_monitor.weight_history


if __name__ == "__main__":
    print("main — use train_layerwise_weights_with_grpo_astrapop()")
