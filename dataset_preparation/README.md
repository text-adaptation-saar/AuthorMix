# Dataset preparation

## Corpus sources

- **Ten high-resource authors:** the author corpus released by
  [StyleTunedLM](https://github.com/cauchy221/StyleTunedLM): Charlotte Perkins
  Gilman, George Orwell, Jane Austen, Mark Twain, Nathaniel Hawthorne, Oscar Wilde,
  P. G. Wodehouse, Samuel Richardson, Vernon Lee, and Virginia Woolf. We prepared
  neutral-input/original-output pairs from this corpus for adapter training.
- **Forty additional authors:** collected separately from Project Gutenberg using
  the download and preparation scripts here. `gutenberg_manifest.json` records
  their author IDs, book IDs, titles, and source URLs. Twenty of these authors
  form the 10-source × 10-target evaluation set.
- **CAV training:** combines those 10 + 40 authors. See the
  [exact 50-author list and training recipe](../cav_training/README.md).

StyleTunedLM provides its author corpus through the `gs://author-style/` bucket.
Its README documents downloading it with Google Cloud Storage tools, for example:

```bash
gsutil -m cp -r 'gs://author-style/*' dataset_preparation/data/styletunedlm/
```

The Gutenberg download command below fetches only the additional 40-author pool;
it does not replace the StyleTunedLM source of the high-resource adapter data.
Book texts retain their source rights; the repository's code license does not
assign a new license to the text.

## Use the released evaluation dataset

The [10 × 10 evaluation JSON](../data/dataset_author_eval/sents/json/test_src10_tgt10.json)
contains 16 texts per source and target author. Its target references also
define the desired style for adapter selection and GRPO rewards.
[`authors.json`](../data/authors.json) lists the author groups, and
[`dataset_info.json`](../data/dataset_info.json) maps the training dataset
names to local files.

## Download and prepare the corpus

Run these commands from the AuthorMix repository root. The downloader uses
`requests`, `beautifulsoup4`, `lxml`, and `tqdm`; sentence splitting uses `nltk`
and its `punkt` / `punkt_tab` resources.

```bash
python dataset_preparation/download_corpus.py
python dataset_preparation/src/data_preparation/clean_gutenberg_corpus.py
python dataset_preparation/src/data_preparation/paragraph_and_sentence_splits.py --corpus \
  dataset_preparation/data/gutenberg_corpus_cleaned \
  dataset_preparation/data/paragraph_splits paragraph
python dataset_preparation/src/data_preparation/paragraph_and_sentence_splits.py --corpus \
  dataset_preparation/data/paragraph_splits \
  dataset_preparation/data/sentence_splits sentence 42
```

`download_corpus.py` uses the recorded book list, rather than today's changing
popular-author ranking. It resumes downloads using a progress file. The original
ranking-based downloader is also retained under `src/data_collection/`.

The cleaner removes Gutenberg headers/footers and introductory material using
heuristics. The splitter creates paragraph train/validation/test partitions
(80/10/10) in document order, then tokenizes and shuffles sentences within each
partition. These are not book-disjoint splits. Generated corpus files stay in
Git-ignored `dataset_preparation/data/`.

Fresh downloads and tokenizer versions can change the resulting splits. Keep
the released evaluation JSON as the reference test set.

## Prepare GRPO training data

1. For each of the ten source authors in `data/authors.json`, copy the
   Gutenberg sentence `train.txt` and `val.txt` files from
   `dataset_preparation/data/sentence_splits/text/` to
   `data/dataset_author_eval/sents/text/<author>/`. Use the source keys from
   `data/authors.json` as directory names.
2. Add a `train.txt` of original StyleTunedLM sentences for each high-resource
   author under the same text directory. Use the author name with spaces
   replaced by underscores, for example `Jane_Austen` or `P._G._Wodehouse`.
   The selector uses up to the first 1,000 lines per author.
3. Generate the LLaMA-Factory JSON files for the source authors:

   ```bash
   python PTA/19_RL_on_10src_15tgt_authors/create_author_llama_factory_json.py
   ```

The generated train and validation JSON files go to
`data/dataset_author_eval/sents/llama_factory_json/`. The GRPO trainer resolves
them through `data/dataset_info.json`.

## Prepare adapter training pairs

For new data, first produce a neutral rewrite for each original sentence, keeping
line order and split membership unchanged. Format each split with a command
like this one for Mark Twain:

```bash
python dataset_preparation/build_adapter_pairs.py \
  --neutral path/to/neutral_train.txt \
  --original path/to/author_train.txt \
  --output data/adapter_training/MT/train.json
```

Repeat for `val.json` and `test.json` and use the adapter codes from
`data/authors.json`. The script writes `instruction`, `input`, and `output`
fields with the `Paraphrase` instruction and rejects mismatched line counts.
Neutral rewriting is a separate model generation step. The
[trained adapters](https://huggingface.co/sarubi/AuthorMix) are available if
you want to skip SFT.
