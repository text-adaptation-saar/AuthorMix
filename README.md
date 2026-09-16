# AuthorMix

## Paper workflow

AuthorMix adapts a language model to a target author's style by learning to mix
existing author LoRA adapters:

1. **Train author adapters.** In our experiments, we used LLaMA-Factory to train
   one LoRA adapter for each high-resource author, using the same base Llama model
   and training setup.
2. **Select adapters for a target author.** AuthorMix uses style similarity to
   select relevant adapters from this pool, using the target author's reference
   texts.
3. **Learn the mixing weights.** GRPO optimizes either layerwise weights or one
   weight per adapter. The base model and author adapters remain frozen. The
   reward combines similarity to the target author's style with preservation of
   the input's meaning.

We implemented AuthorMix in `src/llamafactory/x_my_pta_scripts/` and integrated it
with LLaMA-Factory to reuse its dataset loading, formatting, and model utilities.
This repository contains the mixing-weight training stage; the pretrained author
adapters are supplied separately.

You can train the author LoRA adapters with your own scripts and data format;
LLaMA-Factory's training pipeline and dataset format are not required. Keep your
existing prompt and data formatting, and adapt AuthorMix's GRPO data loader to
use that same format when training the mixing weights. The adapters must share
the same base model and use compatible LoRA configurations and target modules.
The current adapter loader expects PEFT-compatible checkpoints.

## Dependencies

Use a LLaMA-Factory-compatible environment with PyTorch, Transformers, PEFT,
TRL (for GRPO), Sentence Transformers, and `mutual-implication-score` (MIS).
The LLaMA-Factory modules used by AuthorMix are included in this repository.
Training requires an NVIDIA GPU; supply the base model, author adapters, and
datasets separately.

## Data

Provide an author reference JSON containing:

```json
{
  "source_authors": {"Source_Author": ["Source reference text."]},
  "target_authors": {"Target_Author": ["Target style reference text."]}
}
```

The `test_data_path` configuration field points to this file. Despite the historical
name, these references are used in training for adapter selection and rewards.
Keep held-out test texts separate.

The preparation scripts expect:

```text
data/dataset_author_eval/sents/json/test_src10_tgt15_with_neutral.json
data/dataset_author_eval/sents/text/<Source_Author>/train.txt
data/dataset_author_eval/sents/text/<Source_Author>/val.txt
data/dataset_author_eval/sents/text/<Adapter_Author>/train.txt
```

Each text file contains one example per line. Create `data/dataset_info.json` with
`{}` if no registry exists, then run:

```bash
python PTA/19_RL_on_10src_15tgt_authors/create_author_llama_factory_json.py
```

This creates and registers source-author training/validation datasets. The separate
`register_author_json_in_dataset_info.py` script can register existing files.

## Train

The two GRPO weight-mixing implementations are:

- **Adapterwise GRPO weight mixing:**
  [`PTA/19_RL_on_10src_15tgt_authors/adapterwise_weight_optimization_per_tgt_author_layerwiseFalse/`](PTA/19_RL_on_10src_15tgt_authors/adapterwise_weight_optimization_per_tgt_author_layerwiseFalse/)
  learns one mixing weight per adapter, shared across layers.
- **Layerwise GRPO weight mixing:**
  [`PTA/19_RL_on_10src_15tgt_authors/weight_optimization_per_tgt_author/`](PTA/19_RL_on_10src_15tgt_authors/weight_optimization_per_tgt_author/)
  learns separate adapter mixing weights for each layer.

Each directory contains its `run_rl_per_pair.py` training entry point and an
optional `generate_with_rl_model()` sample for generating text with an already
loaded mixing model. The sample is not called during training and does not run
metric scoring or repeated-seed evaluation.

Place your Llama base model and author adapters locally, then edit an example config:

```bash
cp configs/layerwise.example.json configs/layerwise.local.json
python scripts/run_grpo.py configs/layerwise.local.json --dry-run
python scripts/run_grpo.py configs/layerwise.local.json
```

Use `configs/adapterwise.example.json` for adapterwise training. Paths in configs
resolve from the repository root. `adapter_pattern_path` substitutes `{AUTHOR}`
with MT, VW, VL, CPG, GO, JA, NH, OW, PGW, or SR. Set `target_author` in
`arguments` to train one target; otherwise all targets are processed.

The launcher sets `PYTHONPATH`, `LLAMA_FACTORY_SRC`, and `DISABLE_VERSION_CHECK=1`
for the GRPO environment. Both examples use Table 11 hyperparameters: learning
rate 0.02, 300 steps, top-p 0.95, temperature 1.0, and zero-initialized mixing
weights. They use 16 generations and beta 0, consistent with the saved training
configuration. The layerwise example selects four adapters and the adapterwise example selects
three, matching their settings in the main results table. Full GPU reproduction
has not been rerun with this repository.

Training saves learned weights and checkpoints in
`results/<method>/tgt_author_<name>/grpo_training/`, plus `training_results.json`
and `training_config.json` in the target directory. GRPO retains its in-training
validation and checkpoint resumption. No post-training evaluation is launched.

## Structure

```text
PTA/19_RL_on_10src_15tgt_authors/
  weight_optimization_per_tgt_author/  # Layerwise GRPO weight mixing
    run_rl_per_pair.py
  adapterwise_weight_optimization_per_tgt_author_layerwiseFalse/  # Adapterwise GRPO weight mixing
    run_rl_per_pair.py
  create_author_llama_factory_json.py
  register_author_json_in_dataset_info.py
src/llamafactory/         # LLaMA-Factory framework with custom AuthorMix modules
  x_my_pta_scripts/       # AuthorMix: GRPO, weighted LoRA, reward, and data helpers
  data/                  # LLaMA-Factory: dataset loading and formatting
  model/                 # LLaMA-Factory: model and tokenizer support
  hparams/               # LLaMA-Factory: configuration handling
  extras/                # LLaMA-Factory: shared utilities
scripts/                 # Config launcher
configs/                 # Layerwise and adapterwise examples
```

The `data/`, `model/`, `hparams/`, and `extras/` directories under
`src/llamafactory/` come from the LLaMA-Factory codebase and retain its module
paths. AuthorMix training and mixing helpers are in `x_my_pta_scripts/`.
`evaluation_metrics.py` contains only the embedding and MIS helpers needed for
GRPO training.

## Citation

**AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing**  
Sarubi Thillainathan, Ji-Ung Lee, Michael Sullivan, and Alexander Koller.  
Accepted to **[EMNLP 2026](https://2026.emnlp.org/)**, Budapest, Hungary.

[Paper on arXiv](https://arxiv.org/abs/2603.23069)

If you use AuthorMix in your research, please cite:

```bibtex
@inproceedings{thillainathan2026authormix,
  title = {{AuthorMix}: Modular Authorship Style Transfer via Layer-wise Adapter Mixing},
  author = {Thillainathan, Sarubi and Lee, Ji-Ung and Sullivan, Michael and Koller, Alexander},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  address = {Budapest, Hungary},
  eprint = {2603.23069},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2603.23069},
  note = {Accepted; publication forthcoming}
}
```

The citation will be updated with the official ACL Anthology entry when available.
