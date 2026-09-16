"""Style embeddings and MIS helpers used by GRPO training."""


from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@staticmethod
def _l2norm(v, eps: float = 1e-12):
    """L2 normalize - works for both NumPy arrays and PyTorch tensors"""
    if isinstance(v, torch.Tensor):
        return v / (v.norm() + eps)
    else:
        return v / (np.linalg.norm(v) + eps)


# Global MIS model instance (lazy-loaded).
_mis_model = None


def _get_mis_model(device="cuda"):
    """
    Lazy-load MIS model instance.
    
    Args:
        device: Device to use for MIS model (default: "cuda")
    
    Returns:
        MIS model instance or None if import fails
    """
    global _mis_model
    if _mis_model is None:
        try:
            from mutual_implication_score import MIS
            _mis_model = MIS(device=device)
        except ImportError:
            print("Warning: mutual_implication_score module not available. MIS score calculation will be disabled.")
            _mis_model = False  # Use False to indicate unavailable (None means not tried yet)
    return _mis_model if _mis_model is not False else None


def mis_compute(a_texts: List[str], b_texts: List[str], device: str = "cuda", batch_size: Optional[int] = None) -> List[float]:
    """
    Compute Mutual Implication Score (MIS) between two lists of texts.
    
    Args:
        a_texts: First list of texts
        b_texts: Second list of texts (must have same length as a_texts)
        device: Device to use for MIS model (default: "cuda")
        batch_size: Optional batch size for MIS computation
    
    Returns:
        List of MIS scores (one per text pair)
    
    Raises:
        ValueError: If MIS model is not available or if input lists have different lengths
    """
    if len(a_texts) != len(b_texts):
        raise ValueError(f"a_texts and b_texts must have the same length. Got {len(a_texts)} and {len(b_texts)}")
    
    mis_model = _get_mis_model(device)
    if mis_model is None:
        raise ValueError("MIS model is not available. Please install mutual_implication_score package.")
    
    # Compute MIS scores
    # Type assertion: mis_model cannot be None or False at this point due to check above
    assert mis_model is not None and mis_model is not False, "MIS model should be available here"
    if batch_size is not None:
        scores = mis_model.compute(a_texts, b_texts, batch_size=batch_size)  # type: ignore
    else:
        scores = mis_model.compute(a_texts, b_texts)  # type: ignore
    
    # Handle identical texts (set score to 1.0)
    for idx, (score, a_text, b_text) in enumerate(zip(scores, a_texts, b_texts)):
        if a_text == b_text:
            scores[idx] = 1.0
    
    return scores


def _get_doc_embedding(
    texts: List[str],
    style_model: SentenceTransformer,
) -> torch.Tensor:
    """L2-normalized mean sentence embedding for a list of texts (doc-level), shape (1, D)."""
    emb = style_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
        device="cuda",
    )
    doc = _l2norm(emb.mean(dim=0))
    return doc.unsqueeze(0)
