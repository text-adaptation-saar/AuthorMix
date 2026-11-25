# CAV style encoder

This folder provides the paper's **CAV exp04: 50 authors, 2k chunks per author,
cosine triplet loss** experiment. It trains the independent style encoder used
for the CAV evaluation columns; it is separate from LoRA adapter training and
GRPO mixing.

## Data and author pool

The 50 authors comprise **10 high-resource authors from
[StyleTunedLM](https://github.com/cauchy221/StyleTunedLM)** and **40 additional
Project Gutenberg authors**. The latter include all 10 source and 10 target
authors used in AuthorMix evaluation, plus 20 other authors. These are 50 author
identities; `n_clusters=50` is a separate setting for content-topic clustering.

Prepare the triplets locally using the steps below. The reference run used
100,000 training triplets and 10,000 validation triplets. The preparation step
writes `train_triplets.json`, `test_triplets.json`, and `metadata.json`.
The `test_triplets.json` file is generated from the authors' validation splits
and used for model selection.

Each triplet has an anchor, a same-author positive, and a different-author
negative. Content clustering guides the training triplets; validation triplets
use the original standard sampling. The preparation run sampled 2,000 training
chunks per author (100,000 total) and up to 500 validation chunks per author
(24,951 total; Emily Brontë had 451).

## Train the encoder

Use the CAV training environment, separate from the newer GRPO environment.
The saved encoder records Sentence Transformers **4.1.0**, Transformers **4.48.3**,
and PyTorch **2.5.1+cu124**. The script also uses NumPy, scikit-learn, and tqdm.
After preparing the triplets, run from the repository root:

```bash
python cav_training/run_experiment.py \
  --triplets-dir cav_training/outputs/regenerated_triplets/datasets --dry-run
python cav_training/run_experiment.py \
  --triplets-dir cav_training/outputs/regenerated_triplets/datasets
```

`exp04.json` uses the saved model settings: RoBERTa-base with mean pooling,
256-token model input, batch size 32, 20 epochs, learning rate 2e-5, cosine
triplet margin 0.5, warmup ratio 0.1, and seed 42. The model is validated every
epoch and the best checkpoint is saved under
`cav_training/outputs/exp04/weights/style_model`.

The training config retains `no_cav: true` from the recorded run. With
`load_triplets_from` set, this flag does **not** generate new triplets or disable
CAV: training loads the precomputed CAV triplets whose metadata has `use_cav: true`.

Use the resulting encoder in AuthorMix evaluation:

```bash
python evaluation/evaluate.py \
  --variant layerwise --training-dir results/layerwise \
  --output-dir results/evaluation_layerwise_cav \
  --cav-model cav_training/outputs/exp04/weights/style_model --cola
```

## Prepare triplets

Follow [dataset preparation](../dataset_preparation/README.md) to get both
corpora. Combine the StyleTunedLM paragraph splits with the prepared Gutenberg
paragraph splits under one directory. Use exactly the 50 directory names in
`authors.json`, each with a `train.txt` and `val.txt`:

```bash
python cav_training/run_experiment.py --phase prepare \
  --data-dir /path/to/50_author_paragraph_corpus
```

`prepare_triplets.json` specifies the
original content-clustering recipe: all-MiniLM-L6-v2, 50 KMeans clusters,
2,000 training chunks and at most 500 validation chunks per author, seed 42.
The chunk builder groups sentences using a **character-length** budget of 256;
the encoder separately truncates to 256 **tokens**. Regenerated files are saved
in `cav_training/outputs/regenerated_triplets/datasets/`.

The training launcher expects 100,000 training and 10,000 validation triplets.
Use enough source text to reach those counts. Fresh preprocessing, filesystem
order, and library versions can change which triplets are sampled.

## Exact 50-author list

`authors.json` retains the on-disk names and per-author chunk counts. “Source” and
“target” below refer to AuthorMix generation evaluation; all 50 authors participate
in CAV training.

| Author | Corpus source | AuthorMix role |
| --- | --- | --- |
| Louisa May Alcott | Project Gutenberg | cav only |
| L. Frank Baum | Project Gutenberg | source |
| Charlotte Brontë | Project Gutenberg | source |
| Emily Brontë | Project Gutenberg | cav only |
| Lewis Carroll | Project Gutenberg | source |
| Robert W. Chambers | Project Gutenberg | target |
| Charlotte Perkins Gilman | StyleTunedLM | high resource |
| G. K. Chesterton | Project Gutenberg | cav only |
| Agatha Christie | Project Gutenberg | target |
| Joseph Conrad | Project Gutenberg | target |
| Charles Dickens | Project Gutenberg | source |
| Arthur Conan Doyle | Project Gutenberg | target |
| George Eliot | Project Gutenberg | cav only |
| F. Scott Fitzgerald | Project Gutenberg | target |
| E. M. Forster | Project Gutenberg | target |
| Elizabeth Cleghorn Gaskell | Project Gutenberg | cav only |
| George Orwell | StyleTunedLM | high resource |
| Thomas Hardy | Project Gutenberg | source |
| G. A. Henty | Project Gutenberg | source |
| Washington Irving | Project Gutenberg | cav only |
| Henry James | Project Gutenberg | cav only |
| Jane Austen | StyleTunedLM | high resource |
| James Joyce | Project Gutenberg | cav only |
| Rudyard Kipling | Project Gutenberg | target |
| Jack London | Project Gutenberg | target |
| Mark Twain | StyleTunedLM | high resource |
| Herman Melville | Project Gutenberg | cav only |
| George Meredith | Project Gutenberg | cav only |
| L. M. Montgomery | Project Gutenberg | target |
| Nathaniel Hawthorne | StyleTunedLM | high resource |
| Oscar Wilde | StyleTunedLM | high resource |
| P. G. Wodehouse | StyleTunedLM | high resource |
| Edgar Allan Poe | Project Gutenberg | cav only |
| John Ruskin | Project Gutenberg | source |
| Samuel Richardson | StyleTunedLM | high resource |
| Walter Scott | Project Gutenberg | source |
| William Shakespeare | Project Gutenberg | cav only |
| Mary Wollstonecraft Shelley | Project Gutenberg | cav only |
| Robert Louis Stevenson | Project Gutenberg | cav only |
| Bram Stoker | Project Gutenberg | cav only |
| Jonathan Swift | Project Gutenberg | cav only |
| Henry David Thoreau | Project Gutenberg | cav only |
| Anthony Trollope | Project Gutenberg | cav only |
| Vernon Lee | StyleTunedLM | high resource |
| Virginia Woolf | StyleTunedLM | high resource |
| H. G. Wells | Project Gutenberg | cav only |
| Edith Wharton | Project Gutenberg | target |
| Henry B. Wheatley | Project Gutenberg | source |
| David Widger | Project Gutenberg | source |
| Charlotte M. Yonge | Project Gutenberg | cav only |
