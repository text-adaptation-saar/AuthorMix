# AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing

Sarubi Thillainathan, Ji-Ung Lee, Michael Sullivan, and Alexander Koller.  
Published at **EMNLP 2026**, Budapest, Hungary.

[Published paper (ACL Anthology)](https://aclanthology.org/2026.emnlp-main.2215/) ·
[arXiv preprint](https://arxiv.org/abs/2603.23069)

## Paper workflow

AuthorMix adapts a language model to a target author's style by learning to mix
existing author LoRA adapters:

1. **Train author adapters.** In our experiments, we used LLaMA-Factory to train
   one LoRA adapter for each of the ten high-resource authors from
   [StyleTunedLM](https://github.com/cauchy221/StyleTunedLM), using the same base
   Llama model and training setup.
2. **Select adapters for a target author.** AuthorMix uses style similarity to
   select relevant adapters from this pool, using the target author's reference
   texts.
3. **Learn the mixing weights.** GRPO optimizes either layerwise weights or one
   weight per adapter. The base model and author adapters remain frozen. The
   reward combines similarity to the target author's style with preservation of
   the input's meaning.

We implemented AuthorMix in `src/llamafactory/x_my_pta_scripts/` and integrated it
with LLaMA-Factory to reuse its dataset loading, formatting, and model utilities.
This repository includes the evaluation dataset, high-resource adapter training
configs, GRPO mixing code, and single-pass evaluation. Download the ten trained
author adapters from [Hugging Face](https://huggingface.co/sarubi/AuthorMix)
or train them locally. Obtain the Llama base model separately.

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
Training requires an NVIDIA GPU and base-model weights. Adapter SFT uses a
separate full LLaMA-Factory checkout; the included framework subset supports
AuthorMix GRPO training.

## Data

The released test set contains **10 source authors × 10 target authors** (100
pairs), with 16 texts per author. The configs use
`data/dataset_author_eval/sents/json/test_src10_tgt10.json`.
`data/authors.json` lists the high-resource, source, and target authors.

The 10 × 10 test set is the only bundled dataset. The repository also includes
the author lists and the LLaMA-Factory dataset registry, which describes where
to place locally prepared files. The test JSON supplies original source
sentences for evaluation and target references for adapter selection and GRPO
rewards.

See [Dataset preparation](dataset_preparation/README.md) for the corpus sources
and instructions for preparing GRPO prompts, adapter selection reference texts,
and optional SFT data. The StyleTunedLM corpus supplies the high-resource
authors. The evaluation authors come from the additional Project Gutenberg
corpus.

## Train high-resource adapters

The ten configs in `PTA/10_author_models_ft/configs/` use the saved experiment
settings: LoRA rank 8, alpha 16, dropout 0.05, all target modules, learning rate
0.0001, 5,000 steps, 200 warmup steps, sequence length 256, batch size 8 with
four gradient accumulation steps, and bf16. The best validation-loss checkpoint
is selected for each adapter.

You can skip this step if you download the trained adapters from Hugging Face.
To retrain them, [prepare the local SFT data](dataset_preparation/README.md#prepare-adapter-training-pairs),
then run from this repository root with a full LLaMA-Factory checkout and its SFT
environment:

```bash
python PTA/10_author_models_ft/train_adapters.py \
  --llamafactory-root /path/to/LLaMA-Factory \
  --base-model models/Llama-3.1-8B-Instruct --authors MT --dry-run
```

Remove `--dry-run` to train. Omit `--authors MT` to train all ten authors in
sequence. The resulting adapters are saved under `adapters/<code>/`, matching
the GRPO configs. The launcher uses the external checkout's training code and
the locally prepared data.

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

Place your Llama base model and author adapters locally and
[prepare the GRPO training data](dataset_preparation/README.md#prepare-grpo-training-data).
Then edit an example config:

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

## Evaluate

Run one generation pass per source–target pair using saved GRPO weights:

```bash
python evaluation/evaluate.py \
  --variant layerwise --training-dir results/layerwise \
  --base-model models/Llama-3.1-8B-Instruct \
  --output-dir results/evaluation_layerwise --cola
```

Use `--variant adapterwise --training-dir results/adapterwise` for adapterwise
mixing. `--target-author` selects one target and `--dry-run` checks the test data
and saved training metadata without loading models. The metadata's adapter paths
must still resolve from the repository root. Use a fresh output directory for
each evaluation.

This evaluates the final weights in `training_results.json`, using the original
`generate_with_rl_model()` helper, seed 42 and up to 256 new tokens. It does not
retry, select, or duplicate generations, and has no repeated-seed loop. A single
pass is not an exact reproduction of the paper's multi-run aggregate.

Each pair JSON records the generated texts and metric components; `summary.json`
contains means across pairs:

| Metric | Implementation |
| --- | --- |
| Toward | Original document-level STAR style similarity, scaled relative to the source–target baseline |
| MIS | Mean Mutual Implication Score for outputs versus source texts |
| Joint | Original `joint_without_away_scaled_mis`: square root of scaled Toward × scaled MIS |
| CoLA (`--cola`) | `textattack/roberta-base-CoLA`, with the original long-text chunking |
| CAV Toward / SBERT / Joint (`--cav-model PATH`) | Original CAV angular Toward, all-mpnet-base-v2 semantic similarity, and their geometric mean |

The original STAR/MIS functions are in `evaluation/metrics.py`; fluency functions
are in `evaluation/cola.py`. To use the paper's independent CAV encoder, supply
the **CAV exp04 gutenberg50-2k-chunk** SentenceTransformer checkpoint. That locally
trained checkpoint is not bundled. Its [CAV training folder](cav_training/README.md)
contains the original 50-author list, triplet preparation guidance, and
experiment scripts.

## Structure

```text
data/                    # 10 × 10 evaluation dataset and metadata
dataset_preparation/     # Gutenberg collection, cleaning, splitting, and SFT pairing
evaluation/              # Single-pass generation and paper metrics
cav_training/            # 50-author CAV encoder training and data preparation
PTA/10_author_models_ft/  # High-resource adapter training launcher and configs
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

If you use AuthorMix in your research, please cite:

```bibtex
@inproceedings{thillainathan-etal-2026-authormix,
  title = {{A}uthor{M}ix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing},
  author = {Thillainathan, Sarubi and Lee, Ji-Ung and Sullivan, Michael and Koller, Alexander},
  editor = {Lee, Hung-Yi and Dev, Sunipa and Li, Junyi Jessy and Elliott, Desmond},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  month = oct,
  year = {2026},
  address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2026.emnlp-main.2215/},
  doi = {10.18653/2026.emnlp-main.2215},
  pages = {50382--50416},
  isbn = {979-8-89176-459-0}
}
```
