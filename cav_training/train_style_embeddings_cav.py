"""
CAV-Style Training Script for Style Embeddings
================================================
This script trains a style embedding model using the Contrastive Authorship
Verification (CAV) approach from Wegmann et al. (2022).

Key insight: To learn STYLE (not CONTENT), we need:
- Positive pairs: Same author, DIFFERENT topic
- Negative pairs: Different author, SAME topic

Three modes available:
1. --use_cav: Use content clustering to create pseudo-topics
2. --use_paraphrased_negatives: Use pre-paraphrased texts as same-content negatives (BEST)
3. --no_cav: Standard triplets without topic control

Data structure expected:
    data_dir/
        Author1/
            train.txt
            test.txt
            val.txt
            paraphrased/           # Optional, for --use_paraphrased_negatives
                neutral_train.txt
                neutral_test.txt
                neutral_val.txt
        Author2/
            ...
"""

import argparse
import random
import os
import torch
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    losses,
    models,
    InputExample
)
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import re
import json
from datetime import datetime

import pickle
import json


def save_triplets(triplets, output_path, format='pickle'):
    """
    Save triplets to disk.

    Args:
        triplets: List of (anchor, positive, negative) tuples
        output_path: Path to save file (without extension)
        format: 'pickle', 'json', or 'both'
    """
    if format in ['pickle', 'both']:
        pickle_path = f"{output_path}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(triplets, f)
        print(f"  Saved pickle: {pickle_path}")

    if format in ['json', 'both']:
        json_path = f"{output_path}.json"
        # Convert to serializable format
        triplets_dict = {
            'triplets': [
                {'anchor': a, 'positive': p, 'negative': n}
                for a, p, n in triplets
            ],
            'count': len(triplets)
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(triplets_dict, f, indent=2, ensure_ascii=False)
        print(f"  Saved JSON: {json_path}")


def load_triplets(input_path, format='pickle'):
    """
    Load triplets from disk.

    Args:
        input_path: Path to load file (with or without extension)
        format: 'pickle' or 'json'

    Returns:
        List of (anchor, positive, negative) tuples
    """
    if format == 'pickle':
        if not input_path.endswith('.pkl'):
            input_path = f"{input_path}.pkl"
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    elif format == 'json':
        if not input_path.endswith('.json'):
            input_path = f"{input_path}.json"
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(t['anchor'], t['positive'], t['negative'])
                for t in data['triplets']]


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def split_into_chunks(text, max_length=512, min_length=100):
    """
    Split text into chunks at sentence boundaries.
    This preserves natural language flow better than arbitrary splitting.
    """
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(current) + len(sent) + 1 <= max_length:
            current += " " + sent if current else sent
        else:
            if len(current) >= min_length:
                chunks.append(current.strip())
            current = sent

    # Don't forget the last chunk
    if len(current) >= min_length:
        chunks.append(current.strip())

    return chunks


def load_author_data(data_dir, split="train", max_seq_length=512, sentence_wise_no_chunks=False,
                     skip_dirs={'paraNMT_src', 'paraNMT_tgt', 'WikiLarge_src', 'WikiLarge_tgt', '.DS_Store', '__pycache__'},
                     data_size_limit_per_author=None, min_chars=None, random_sample=True):
    """
    Load data from your directory structure.

    Args:
        data_dir: Path to text_format_pt_as_text_excerpts/
        split: "train", "test", or "val"
        max_seq_length: Maximum chunk length

    Returns:
        author_texts: {author_name: [list of text chunks]}
    """
    author_texts = {}

    # Skip non-author directories
    # skip_dirs = {'paraNMT', 'WikiLarge', '.DS_Store', '__pycache__'}

    for author in os.listdir(data_dir):
        if author in skip_dirs or author.startswith('.'):
            continue

        author_path = os.path.join(data_dir, author)
        if not os.path.isdir(author_path):
            continue

        filepath = os.path.join(author_path, f"{split}.txt")

        if os.path.exists(filepath):
            with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
                # text = f.read()
                if sentence_wise_no_chunks:
                    # Each line is a separate text
                    text = f.readlines()
                    chunks = [line.strip() for line in text if line.strip()]
                    # below is used to create PTA/eval_metric_models/style_embeddings_cav/all_possible_runs/shared_triplets
                    # even though shuffed dataset is give, first it did f.read() and then split it.
                    # chunks = [line.strip() for line in text.splitlines() if line.strip()]
                else:
                    text = f.read()
                    chunks = split_into_chunks(text, max_seq_length)

            if chunks:
                if min_chars is not None:
                    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chars]
                if not chunks:
                    continue
                if data_size_limit_per_author is not None:
                    if random_sample and len(chunks) > data_size_limit_per_author:
                        # ✅ Random sample instead of first N
                        chunks = random.sample(chunks, data_size_limit_per_author)
                    else:
                        # Take first N (sequential)
                        chunks = chunks[:data_size_limit_per_author]
                author_texts[author] = chunks
                print(f"  {author}: {len(chunks)} chunks")
        else:
            print(f"  {author}: WARNING - {split}.txt not found")

    return author_texts


def load_author_data_with_paraphrased(data_dir, split="train", max_seq_length=512,
                                       sentence_wise_no_chunks=False,
                                       skip_dirs={'paraNMT_src', 'paraNMT_tgt', 'WikiLarge_src', 'WikiLarge_tgt', '.DS_Store', '__pycache__'},
                                       data_size_limit_per_author=None,
                                       paraphrased_path="paraphrased",
                                       paraphrased_prefix="neutral_"):
    """
    Load data with paraphrased (neutral) versions for CAV training.

    Expected structure:
        data_dir/
            Author1/
                train.txt              <- Original texts (line-wise)
                paraphrased/
                    neutral_train.txt  <- Paraphrased versions (line-aligned)

    Args:
        data_dir: Path to data directory
        split: "train", "test", or "val"
        max_seq_length: Maximum chunk length (only used if not sentence_wise)
        sentence_wise_no_chunks: If True, treat each line as a separate text
        paraphrased_path: Subdirectory containing paraphrased files
        paraphrased_prefix: Prefix for paraphrased files (e.g., "neutral_")

    Returns:
        author_texts: {author_name: [list of original text chunks]}
        author_paraphrased: {author_name: [list of paraphrased text chunks]}
    """
    author_texts = {}
    author_paraphrased = {}

    print(f"\nLoading data with paraphrased versions...")
    print(f"  Looking for: {paraphrased_path}/AUTHOR NAME/{paraphrased_prefix}{split}.txt")

    for author in os.listdir(data_dir):
        if author in skip_dirs or author.startswith('.'):
            continue

        author_path = os.path.join(data_dir, author)
        if not os.path.isdir(author_path):
            continue

        # Paths
        original_path = os.path.join(author_path, f"{split}.txt")
        # paraphrased_path = os.path.join(author_path, paraphrased_subdir, f"{paraphrased_prefix}{split}.txt")
        paraphrased_path_full = os.path.join(paraphrased_path, author, f"{paraphrased_prefix}{split}.txt")

        # Check both files exist
        if not os.path.exists(original_path):
            print(f"  {author}: WARNING - {split}.txt not found, skipping")
            continue

        if not os.path.exists(paraphrased_path_full):
            print(f"  {author}: WARNING - {paraphrased_path_full} not found, skipping")
            continue

        # Load original texts (line-wise for alignment with paraphrased)
        with open(original_path, "r", encoding='utf-8', errors='ignore') as f:
            original_lines = f.readlines()
            original_chunks = [line.strip() for line in original_lines if line.strip()]

        # Load paraphrased texts (always line-wise, must align with original)
        with open(paraphrased_path_full, "r", encoding='utf-8', errors='ignore') as f:
            paraphrased_lines = f.readlines()
            paraphrased_chunks = [line.strip() for line in paraphrased_lines if line.strip()]

        # Check alignment
        if len(original_chunks) != len(paraphrased_chunks):
            print(f"  {author}: WARNING - length mismatch!")
            print(f"    Original: {len(original_chunks)}, Paraphrased: {len(paraphrased_chunks)}")
            # Use minimum length to maintain alignment
            min_len = min(len(original_chunks), len(paraphrased_chunks))
            original_chunks = original_chunks[:min_len]
            paraphrased_chunks = paraphrased_chunks[:min_len]
            print(f"    Using first {min_len} aligned pairs")

        # Apply size limit if specified
        if data_size_limit_per_author is not None:
            original_chunks = original_chunks[:data_size_limit_per_author]
            paraphrased_chunks = paraphrased_chunks[:data_size_limit_per_author]

        if original_chunks and paraphrased_chunks:
            author_texts[author] = original_chunks
            author_paraphrased[author] = paraphrased_chunks
            print(f"  {author}: {len(original_chunks)} original + {len(paraphrased_chunks)} paraphrased (aligned)")

    print(f"\nLoaded {len(author_texts)} authors with paraphrased data")

    return author_texts, author_paraphrased


def generate_cav_triplets_with_paraphrased(author_texts, author_paraphrased, size=100000):
    """
    Generate CAV triplets using paraphrased versions as same-content negatives.

    This is the CLEANEST CAV approach because:
    - Anchor: Original text from Author A (has author's STYLE + CONTENT)
    - Positive: Different original text from Author A (same STYLE, different CONTENT)
    - Negative: Paraphrased version of anchor (same CONTENT, neutral/no STYLE)

    The model learns to distinguish:
    - Author's style vs neutral style
    - On the EXACT SAME content (perfect topic control!)

    This is better than clustering because:
    - Clustering only approximates "same topic"
    - Paraphrasing gives EXACT same content

    Args:
        author_texts: {author: [original texts]} - line i corresponds to line i in paraphrased
        author_paraphrased: {author: [paraphrased texts]} - aligned with author_texts
        size: Maximum number of triplets to generate

    Returns:
        List of (anchor, positive, negative) tuples
    """
    print("\nGenerating CAV triplets with paraphrased negatives...")
    print("  Anchor:   Original text (Author's style + content)")
    print("  Positive: Different original text (same style, diff content)")
    print("  Negative: Paraphrased anchor (same content, neutral style)")

    triplets = []
    authors = list(author_texts.keys())

    stats = {
        'total': 0,
        'skipped_no_positive': 0
    }

    for anchor_author in tqdm(authors, desc="Generating paraphrased-CAV triplets"):
        original_texts = author_texts[anchor_author]
        paraphrased_texts = author_paraphrased[anchor_author]

        for i, anchor in enumerate(original_texts):
            # NEGATIVE: Paraphrased version of the SAME text
            # This has the EXACT same content but neutral/no style
            negative = paraphrased_texts[i]

            # POSITIVE: Different text from same author
            # This has the SAME style but DIFFERENT content
            pos_candidates_idx = [j for j in range(len(original_texts)) if j != i]

            if not pos_candidates_idx:
                stats['skipped_no_positive'] += 1
                continue

            pos_idx = random.choice(pos_candidates_idx)
            positive = original_texts[pos_idx]

            triplets.append((anchor, positive, negative))
            stats['total'] += 1

    random.shuffle(triplets)

    print(f"\n  Statistics:")
    print(f"    Total triplets generated: {stats['total']}")
    print(f"    Skipped (no positive available): {stats['skipped_no_positive']}")
    print(f"    All triplets use paraphrased same-content negatives (100% true CAV)")

    return triplets[:size]


def generate_cav_triplets_with_paraphrased_cross_author(author_texts, author_paraphrased, size=100000):
    """
    Alternative CAV triplet generation using paraphrased texts from OTHER authors.

    This version uses:
    - Anchor: Original text from Author A
    - Positive: Different original text from Author A (same style, diff content)
    - Negative: Paraphrased text from Author B that has similar content (approximated)

    Note: This requires finding content-similar texts across authors, which adds complexity.
    For most cases, use generate_cav_triplets_with_paraphrased() instead.

    Args:
        author_texts: {author: [original texts]}
        author_paraphrased: {author: [paraphrased texts]}
        size: Maximum number of triplets

    Returns:
        List of (anchor, positive, negative) tuples
    """
    print("\nGenerating cross-author CAV triplets with paraphrased negatives...")

    # For cross-author, we need to find similar content across authors
    # We'll use a simple approach: cluster all paraphrased texts by content

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    content_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Collect all paraphrased texts with metadata
    all_paraphrased = []
    paraphrased_metadata = []  # (author, index)

    for author, texts in author_paraphrased.items():
        for i, text in enumerate(texts):
            all_paraphrased.append(text)
            paraphrased_metadata.append((author, i))

    print(f"  Encoding {len(all_paraphrased)} paraphrased texts...")
    embeddings = content_model.encode(all_paraphrased, show_progress_bar=True, batch_size=64)

    # Cluster by content
    n_clusters = min(100, len(all_paraphrased) // 10)
    print(f"  Clustering into {n_clusters} content groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Build cluster -> [(author, idx, paraphrased_text)]
    cluster_texts = defaultdict(list)
    for i, (author, idx) in enumerate(paraphrased_metadata):
        cluster_id = labels[i]
        cluster_texts[cluster_id].append((author, idx, all_paraphrased[i]))

    # Find usable clusters (2+ authors)
    usable_clusters = {
        cid: texts for cid, texts in cluster_texts.items()
        if len(set(author for author, _, _ in texts)) >= 2
    }
    print(f"  Clusters with 2+ authors: {len(usable_clusters)}")

    # Generate triplets
    triplets = []
    authors = list(author_texts.keys())

    for anchor_author in tqdm(authors, desc="Cross-author CAV"):
        original_texts = author_texts[anchor_author]
        paraphrased_texts = author_paraphrased[anchor_author]

        for i, anchor in enumerate(original_texts):
            # Find which cluster this text's paraphrase belongs to
            para_text = paraphrased_texts[i]
            para_idx = all_paraphrased.index(para_text)
            cluster_id = labels[para_idx]

            if cluster_id not in usable_clusters:
                continue

            # Find paraphrased text from DIFFERENT author in same cluster
            other_author_paraphrased = [
                (a, idx, text) for a, idx, text in usable_clusters[cluster_id]
                if a != anchor_author
            ]

            if not other_author_paraphrased:
                continue

            # NEGATIVE: Paraphrased text from different author, similar content
            neg_author, neg_idx, negative = random.choice(other_author_paraphrased)

            # POSITIVE: Different text from same author
            pos_candidates = [t for j, t in enumerate(original_texts) if j != i]
            if not pos_candidates:
                continue
            positive = random.choice(pos_candidates)

            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    print(f"\n  Generated {len(triplets)} cross-author CAV triplets")

    return triplets[:size]


def create_pseudo_topics(author_texts, n_clusters=50, content_model_name='all-MiniLM-L6-v2'):
    """
    Cluster all texts by CONTENT to create pseudo-topics.

    This allows us to find same-topic texts from different authors,
    which is essential for CAV training.

    Returns:
        cluster_author_texts: {cluster_id: {author: [texts]}}
        text_to_cluster: {text: cluster_id}
        text_to_author: {text: author}
    """
    print(f"\nCreating pseudo-topics using {content_model_name}...")

    # Use a general content model (NOT a style model)
    content_model = SentenceTransformer(content_model_name)

    # Flatten all texts with metadata
    all_texts = []
    text_to_author = {}

    for author, texts in author_texts.items():
        for text in texts:
            all_texts.append(text)
            text_to_author[text] = author

    print(f"  Total texts to cluster: {len(all_texts)}")

    # Encode all texts
    print("  Encoding texts...")
    embeddings = content_model.encode(
        all_texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )

    # Cluster by content
    print(f"  Clustering into {n_clusters} pseudo-topics...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Build indices
    text_to_cluster = {text: int(label) for text, label in zip(all_texts, labels)}

    # cluster -> author -> [texts]
    cluster_author_texts = defaultdict(lambda: defaultdict(list))
    for text, label in zip(all_texts, labels):
        author = text_to_author[text]
        cluster_author_texts[int(label)][author].append(text)

    # Count usable clusters (2+ authors)
    usable = sum(1 for c in cluster_author_texts.values() if len(c) >= 2)
    print(f"  Clusters with 2+ authors (usable for CAV): {usable}/{n_clusters}")

    # Show cluster distribution
    print("\n  Cluster distribution by author:")
    for cluster_id in sorted(cluster_author_texts.keys())[:10]:
        authors_in_cluster = list(cluster_author_texts[cluster_id].keys())
        if len(authors_in_cluster) >= 2:
            counts = {a: len(texts) for a, texts in cluster_author_texts[cluster_id].items()}
            print(f"    Cluster {cluster_id}: {counts}")

    if usable < n_clusters:
        print(f"  ... and {n_clusters - 10} more clusters")

    return dict(cluster_author_texts), text_to_cluster, text_to_author


def generate_cav_triplets(
        author_texts,
        cluster_author_texts,
        text_to_cluster,
        size=100000
):
    """
    Generate triplets following CAV methodology:

    - Anchor: text from Author A, Cluster X
    - Positive: text from Author A, Cluster Y (DIFFERENT cluster/topic)
    - Negative: text from Author B, Cluster X (SAME cluster/topic!)

    This forces the model to learn STYLE, not CONTENT.
    """
    print("\nGenerating CAV triplets (same-topic negatives via clustering)...")

    triplets = []
    authors = list(author_texts.keys())

    # Build author -> cluster -> texts index
    author_cluster_texts = defaultdict(lambda: defaultdict(list))
    for cluster_id, author_dict in cluster_author_texts.items():
        for author, texts in author_dict.items():
            author_cluster_texts[author][cluster_id] = texts

    # Find usable clusters (2+ authors)
    usable_clusters = {
        cid: dict(authors)
        for cid, authors in cluster_author_texts.items()
        if len(authors) >= 2
    }

    cav_count = 0
    fallback_count = 0

    for anchor_author in tqdm(authors, desc="Generating triplets"):
        anchor_all_texts = author_texts[anchor_author]
        anchor_clusters = author_cluster_texts[anchor_author]

        for anchor_text in anchor_all_texts:
            anchor_cluster = text_to_cluster.get(anchor_text)

            if anchor_cluster is None:
                continue

            # Try to find SAME-TOPIC NEGATIVE
            if anchor_cluster in usable_clusters:
                other_authors = [
                    a for a in usable_clusters[anchor_cluster].keys()
                    if a != anchor_author
                ]

                if other_authors:
                    # SUCCESS: We have a same-topic negative!
                    neg_author = random.choice(other_authors)
                    neg_texts = usable_clusters[anchor_cluster][neg_author]
                    negative = random.choice(neg_texts)

                    # Find DIFFERENT-TOPIC POSITIVE
                    other_clusters = [c for c in anchor_clusters.keys() if c != anchor_cluster]

                    if other_clusters:
                        pos_cluster = random.choice(other_clusters)
                        pos_texts = anchor_clusters[pos_cluster]
                        positive = random.choice(pos_texts)
                    else:
                        # Fallback: any other text from same author
                        other_texts = [t for t in anchor_all_texts if t != anchor_text]
                        if not other_texts:
                            continue
                        positive = random.choice(other_texts)

                    triplets.append((anchor_text, positive, negative))
                    cav_count += 1
                    continue

            # FALLBACK: Standard triplet (no same-topic guarantee)
            other_texts = [t for t in anchor_all_texts if t != anchor_text]
            if not other_texts:
                continue
            positive = random.choice(other_texts)

            # Random negative from any other author
            other_authors = [a for a in authors if a != anchor_author]
            neg_author = random.choice(other_authors)
            negative = random.choice(author_texts[neg_author])

            triplets.append((anchor_text, positive, negative))
            fallback_count += 1

    random.shuffle(triplets)

    total = cav_count + fallback_count
    print(f"\n  Total triplets generated: {total}")
    print(f"  CAV triplets (same-topic negatives): {cav_count} ({100 * cav_count / total:.1f}%)")
    print(f"  Fallback triplets (random negatives): {fallback_count} ({100 * fallback_count / total:.1f}%)")

    return triplets[:size]


def generate_standard_triplets(author_texts, size=10000):
    """
    Generate standard triplets (for comparison or when CAV isn't possible).
    No topic control - just same author positive, different author negative.
    """
    print("\nGenerating standard triplets...")

    triplets = []
    authors = list(author_texts.keys())

    for anchor_author in tqdm(authors, desc="Standard triplets"):
        anchor_texts = author_texts[anchor_author]

        # All texts from other authors
        negative_pool = []
        for other_author in authors:
            if other_author != anchor_author:
                negative_pool.extend(author_texts[other_author])

        for anchor in anchor_texts:
            # Positive: same author, different text
            pos_candidates = [t for t in anchor_texts if t != anchor]
            if not pos_candidates:
                continue
            positive = random.choice(pos_candidates)

            # Negative: different author
            negative = random.choice(negative_pool)

            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    print(f"  Generated {len(triplets)} standard triplets")

    return triplets[:size]


def triplets_to_examples(triplets):
    """Convert triplets to SentenceTransformer InputExamples."""
    return [InputExample(texts=[a, p, n]) for a, p, n in triplets]


def main():
    parser = argparse.ArgumentParser(description="Train CAV-style embeddings")

    # Data
    parser.add_argument("--data_dir", type=str,
                        default=None,
                        # default="data/author_dataset_cleaned/text_format_pt_as_text_excerpts",
                        help="Path to author data directory")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--max_train_size", type=int, default=100000)
    parser.add_argument("--max_test_size", type=int, default=10000)

    # Clustering (for --use_cav mode)
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Number of pseudo-topic clusters")
    parser.add_argument("--content_model", type=str, default="all-MiniLM-L6-v2",
                        help="Model for content clustering")

    # Model
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Triplet loss margin")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Output
    parser.add_argument("--output_dir", type=str, default="weights/style_cav_model")
    parser.add_argument("--seed", type=int, default=42)

    # ============================================================================
    # CAV MODE SELECTION (mutually exclusive)
    # ============================================================================
    parser.add_argument("--use_cav", action="store_true", default=False,
                        help="Use CAV training with content clustering (pseudo-topics)")
    parser.add_argument("--use_paraphrased_negatives", action="store_true", default=False,
                        help="Use paraphrased texts as same-content negatives (BEST CAV approach)")
    parser.add_argument("--no_cav", action="store_true", default=False,
                        help="Disable CAV, use standard triplets only")

    # Paraphrased data options (for --use_paraphrased_negatives)
    parser.add_argument("--paraphrased_path", type=str, default="paraphrased",
                        help="Subdirectory containing paraphrased files")
    parser.add_argument("--paraphrased_prefix", type=str, default="neutral_",
                        help="Prefix for paraphrased files (e.g., 'neutral_' for neutral_train.txt)")
    parser.add_argument("--use_cross_author_paraphrased", action="store_true", default=False,
                        help="Use paraphrased texts from other authors (requires content clustering)")

    # Other options
    parser.add_argument("--triplet_distance_metric_COSINE", action="store_true",
                        help="Use COSINE distance metric for Triplet Loss")
    parser.add_argument("--sentence_wise_no_chunks", action="store_true",
                        help="Treat each line as a separate text (no chunking)")

    # Load/save options
    parser.add_argument("--load_triplets_from", type=str, default=None,
                        help="Path to directory containing pre-generated triplets")
    parser.add_argument("--skip_training", action="store_true",
                        help="Only generate and save triplets, don't train model")

    # Data size limit options
    parser.add_argument("--data_size_limit_per_author_train", type=int, default=900,
                        help="Maximum number of chunks per author for training")
    parser.add_argument("--data_size_limit_per_author_test", type=int, default=500,
                        help="Maximum number of chunks per author for testing")

    args = parser.parse_args()

    # ============================================================================
    # Validate CAV mode selection
    # ============================================================================
    cav_modes = [args.use_cav, args.use_paraphrased_negatives, args.no_cav]
    if sum(cav_modes) > 1:
        parser.error("Only one of --use_cav, --use_paraphrased_negatives, --no_cav can be specified")

    # Default to standard triplets if nothing specified
    if not any(cav_modes):
        print("No CAV mode specified, defaulting to --no_cav (standard triplets)")
        args.no_cav = True

    # Setup
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, "exp_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Determine mode string for logging
    if args.use_paraphrased_negatives:
        mode_str = "CAV with paraphrased negatives"
    elif args.use_cav:
        mode_str = "CAV with content clustering"
    else:
        mode_str = "Standard triplets (no CAV)"

    print("=" * 60)
    print("CAV-STYLE EMBEDDING TRAINING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model}")
    print(f"Training mode: {mode_str}")
    print(f"Output: {args.output_dir}")

    # ============================================================================
    # LOAD DATA
    # ============================================================================
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # ============================================================================
    # GENERATE OR LOAD TRIPLETS
    # ============================================================================
    print("\n" + "=" * 60)
    if args.load_triplets_from:
        print("LOADING PRE-GENERATED TRIPLETS")
        print("=" * 60)

        train_path = os.path.join(args.load_triplets_from, "train_triplets")
        test_path = os.path.join(args.load_triplets_from, "test_triplets")

        print(f"Loading from: {args.load_triplets_from}")

        if not os.path.exists(train_path + ".json") or not os.path.exists(test_path + ".json"):
            raise FileNotFoundError(f"JSON triplets not found at {args.load_triplets_from}")

        train_triplets = load_triplets(train_path, format='json')
        test_triplets = load_triplets(test_path, format='json')

        print(f"  Loaded {len(train_triplets)} train triplets")
        print(f"  Loaded {len(test_triplets)} test triplets")

        # Copy metadata for reference
        metadata_src = os.path.join(args.load_triplets_from, "metadata.json")
        if os.path.exists(metadata_src):
            import shutil
            dataset_dir = os.path.join(args.output_dir, "datasets")
            os.makedirs(dataset_dir, exist_ok=True)
            shutil.copy(metadata_src, os.path.join(dataset_dir, "loaded_from_metadata.json"))
            print(f"  Copied metadata from source")

    else:
        print("GENERATING TRIPLETS")
        print("=" * 60)

        if args.use_paraphrased_negatives:
            # Load with paraphrased versions
            print("\nTraining data (with paraphrased):")
            author_texts_train, author_paraphrased_train = load_author_data_with_paraphrased(
                args.data_dir, "train", args.max_sequence_length,
                sentence_wise_no_chunks=args.sentence_wise_no_chunks,
                paraphrased_path=args.paraphrased_path,
                paraphrased_prefix=args.paraphrased_prefix
            )

            print("\nValidation data (with paraphrased):")
            author_texts_test, author_paraphrased_test = load_author_data_with_paraphrased(
                args.data_dir, "val", args.max_sequence_length,
                sentence_wise_no_chunks=args.sentence_wise_no_chunks,
                paraphrased_path=args.paraphrased_path,
                paraphrased_prefix=args.paraphrased_prefix
            )
        else:
            # Load regular data
            print("\nTraining data:")
            author_texts_train = load_author_data(args.data_dir, "train", args.max_sequence_length,
                                                  args.sentence_wise_no_chunks,
                                                  data_size_limit_per_author=args.data_size_limit_per_author_train)

            print("\nValidation data:")
            # author_texts_test = load_author_data(args.data_dir, "test", args.max_sequence_length, args.sentence_wise_no_chunks)
            author_texts_test = load_author_data(args.data_dir, "val", args.max_sequence_length,
                                                 args.sentence_wise_no_chunks,
                                                 data_size_limit_per_author=args.data_size_limit_per_author_test)

        total_train = sum(len(texts) for texts in author_texts_train.values())
        total_test = sum(len(texts) for texts in author_texts_test.values())

        print(f"\nTotal: {len(author_texts_train)} authors")
        print(f"Training chunks: {total_train}")
        print(f"Validation chunks: {total_test}")

        # Reset seed for reproducibility
        seed_everything(args.seed)

        # ============================================================================
        # Generate training triplets based on mode
        # ============================================================================
        if args.use_paraphrased_negatives:
            if args.use_cross_author_paraphrased:
                # Cross-author paraphrased negatives (requires clustering)
                train_triplets = generate_cav_triplets_with_paraphrased_cross_author(
                    author_texts_train,
                    author_paraphrased_train,
                    size=args.max_train_size
                )
            else:
                # Same-author paraphrased negatives (cleaner approach)
                train_triplets = generate_cav_triplets_with_paraphrased(
                    author_texts_train,
                    author_paraphrased_train,
                    size=args.max_train_size
                )

        elif args.use_cav:
            # CAV with content clustering
            cluster_author_texts, text_to_cluster, text_to_author = create_pseudo_topics(
                author_texts_train,
                n_clusters=args.n_clusters,
                content_model_name=args.content_model
            )

            # Generate CAV triplets
            train_triplets = generate_cav_triplets(
                author_texts_train,
                cluster_author_texts,
                text_to_cluster,
                size=args.max_train_size
            )
        else:
            # Standard triplets (no CAV)
            train_triplets = generate_standard_triplets(
                author_texts_train,
                size=args.max_train_size
            )

        # ============================================================================
        # Generate test triplets
        # ============================================================================
        if args.use_paraphrased_negatives:
            # For evaluation, we can use paraphrased negatives too
            test_triplets = generate_cav_triplets_with_paraphrased(
                author_texts_test,
                author_paraphrased_test,
                size=args.max_test_size
            )
        else:
            # Standard triplets for evaluation
            test_triplets = generate_standard_triplets(
                author_texts_test,
                size=args.max_test_size
            )

        print(f"\nFinal triplet counts:")
        print(f"  Train: {len(train_triplets)}")
        print(f"  Test: {len(test_triplets)}")

        # ============================================================================
        # Save datasets
        # ============================================================================
        print("\n" + "=" * 60)
        print("SAVING DATASETS")
        print("=" * 60)

        dataset_dir = os.path.join(args.output_dir, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)

        # Save train triplets
        print("\nSaving training dataset...")
        train_path = os.path.join(dataset_dir, "train_triplets")
        save_triplets(train_triplets, train_path, format='both')

        # Save test triplets
        print("\nSaving test dataset...")
        test_path = os.path.join(dataset_dir, "test_triplets")
        save_triplets(test_triplets, test_path, format='both')

        # Save metadata
        metadata = {
            'train_size': len(train_triplets),
            'test_size': len(test_triplets),
            'num_authors_train': len(author_texts_train),
            'num_authors_test': len(author_texts_test),
            'total_train_chunks': sum(len(texts) for texts in author_texts_train.values()),
            'total_test_chunks': sum(len(texts) for texts in author_texts_test.values()),
            'mode': mode_str,
            'use_cav': args.use_cav,
            'use_paraphrased_negatives': args.use_paraphrased_negatives,
            'n_clusters': args.n_clusters if args.use_cav else None,
            'paraphrased_path': args.paraphrased_path if args.use_paraphrased_negatives else None,
            'paraphrased_prefix': args.paraphrased_prefix if args.use_paraphrased_negatives else None,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved metadata: {metadata_path}")

        print(f"\nDatasets saved to: {dataset_dir}")
        print(f"  - train_triplets.pkl ({len(train_triplets)} triplets)")
        print(f"  - train_triplets.json")
        print(f"  - test_triplets.pkl ({len(test_triplets)} triplets)")
        print(f"  - test_triplets.json")
        print(f"  - metadata.json")

    # ============================================================================
    # Early exit if only generating triplets
    # ============================================================================
    if args.skip_training:
        print("\n" + "=" * 60)
        print("SKIPPING TRAINING (--skip_training flag set)")
        print("=" * 60)
        print(f"Triplets saved to: {os.path.join(args.output_dir, 'datasets')}")
        return

    # ============================================================================
    # Convert to InputExamples and train
    # ============================================================================
    train_examples = triplets_to_examples(train_triplets)
    test_examples = triplets_to_examples(test_triplets)

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    word_embedding_model = models.Transformer(
        args.model,
        max_seq_length=args.max_sequence_length
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,  # Mean pooling works well for style
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device=args.device
    )

    print(f"Model: {args.model}")
    print(f"Embedding dimension: {word_embedding_model.get_word_embedding_dimension()}")
    print(f"Device: {args.device}")

    # DataLoader
    train_dl = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Loss
    if args.triplet_distance_metric_COSINE:
        print("Using COSINE distance metric for Triplet Loss")
        loss_fct = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=args.margin
        )
    else:
        print("Using Default distance metric for Triplet Loss: TripletDistanceMetric.EUCLIDEAN")
        loss_fct = losses.TripletLoss(
            model=model
        )

    # Evaluator
    evaluator = TripletEvaluator.from_input_examples(
        test_examples,
        name='style-eval',
        show_progress_bar=True
    )

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Triplet margin: {args.margin}")

    model.fit(
        train_objectives=[(train_dl, loss_fct)],
        evaluator=evaluator,
        evaluation_steps=len(train_dl),  # Evaluate every epoch
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        save_best_model=True,
        optimizer_params={'lr': args.learning_rate},
        show_progress_bar=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")

    # Quick evaluation
    print("\n" + "=" * 60)
    print("QUICK EVALUATION")
    print("=" * 60)

    if not args.load_triplets_from:
        evaluate_model(model, author_texts_test)


def evaluate_model(model, author_texts, n_samples=500):
    """Quick evaluation: can the model distinguish authors?"""
    from sklearn.metrics.pairwise import cosine_similarity

    authors = list(author_texts.keys())

    correct = 0
    total = 0

    for _ in range(n_samples):
        # Pick random author and two of their texts
        author = random.choice(authors)
        texts = author_texts[author]
        if len(texts) < 2:
            continue

        anchor, positive = random.sample(texts, 2)

        # Pick negative from different author
        other_author = random.choice([a for a in authors if a != author])
        if not author_texts[other_author]:
            continue
        negative = random.choice(author_texts[other_author])

        # Encode
        embs = model.encode([anchor, positive, negative])

        # Check if positive is closer than negative
        pos_sim = cosine_similarity([embs[0]], [embs[1]])[0][0]
        neg_sim = cosine_similarity([embs[0]], [embs[2]])[0][0]

        if pos_sim > neg_sim:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Triplet accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  (random baseline would be ~0.50)")

    return accuracy


if __name__ == "__main__":
    main()