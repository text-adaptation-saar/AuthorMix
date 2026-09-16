"""
ASTRAPOP-based reward function for GRPO weight optimization.

The existing joint_score_weighted() computes doc-level (aggregated) scores,
returning a single scalar.  GRPO needs per-sample rewards.

This module provides:
  - ASTRAPOPRewardComputer: computes per-sample toward, away, and MIS scores
    then combines them as  (toward_i ** toward_exp) * (mis_i ** mis_exp)
  - TRLAstrapopRewardWrapper: adapts the computer for TRL's
    reward_fn(prompts, completions) -> List[float] interface.
"""

import math
import re
from typing import List, Optional

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

from llamafactory.x_my_pta_scripts.evaluation_patel.evaluation_metrics import (
    _get_doc_embedding,
    mis_compute,
    _l2norm,
)

# ---------------------------------------------------------------------------
# Prompt / output cleaning (shared with custom_loss_computer_v2)
# ---------------------------------------------------------------------------

# PARAPHRASE_INPUT_PATTERN = re.compile(
#     r"Paraphrase\n(.*?)(?=<\|eot_id\|>)",
#     flags=re.DOTALL,
# )

PARAPHRASE_INPUT_PATTERN = re.compile(
    r"Only output the paraphrased version. Paraphrase\n(.*?)(?=<\|eot_id\|>)",
    flags=re.DOTALL,
)

OUTPUT_PREFIX_PATTERNS = [
    re.compile(r"^\s*Here is the paraphrased version:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Here is the paraphrased plot summary:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*The following paraphrase conveys the same meaning:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Here is a paraphrased version:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Paraphrase:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Paraphrased, this sentence could be:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*Here is the paraphrased sentence:\s*", flags=re.IGNORECASE),
]


def extract_paraphrase_input(text: str) -> str:
    m = PARAPHRASE_INPUT_PATTERN.search(text)
    return m.group(1).strip() if m else text.strip()


# def clean_output_text(text: str) -> str:
#     cleaned = text.strip()
#     for pattern in OUTPUT_PREFIX_PATTERNS:
#         cleaned = pattern.sub("", cleaned).strip()
#     if re.match(r"^\s*here\b", cleaned, flags=re.IGNORECASE) and ":" in cleaned:
#         cleaned = cleaned.split(":", 1)[1].strip()
#     if re.match(r"^\s*parap", cleaned, flags=re.IGNORECASE) and ":" in cleaned:
#         cleaned = cleaned.split(":", 1)[1].strip()
#     return cleaned


# ---------------------------------------------------------------------------
# Per-sample ASTRAPOP helpers
# ---------------------------------------------------------------------------

def _S(us: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    """Angular similarity (element-wise).  Input: (..., D), output: (...)."""
    cossims = cosine_similarity(us, vs)
    return 1 - torch.acos(cossims.clamp(-1, 1)) / math.pi


def _Sc(us: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    return 1 - _S(us, vs)


def _per_sample_toward(
    r_s: torch.Tensor,     # (batch, D) — source per-text embeddings
    r_t: torch.Tensor,     # (1, D)     — target doc embedding
    r_s2t: torch.Tensor,   # (batch, D) — transferred per-text embeddings
) -> torch.Tensor:
    """Per-sample toward score, shape (batch,)."""
    numerator = _S(r_s2t, r_t.expand_as(r_s2t)) - _S(r_s, r_t.expand_as(r_s))
    numerator = numerator.clamp(min=0)
    denominator = _Sc(r_s, r_t.expand_as(r_s))
    return numerator / denominator.clamp(min=1e-8)


def _per_sample_away(
    r_s: torch.Tensor,
    r_t: torch.Tensor,
    r_s2t: torch.Tensor,
) -> torch.Tensor:
    """Per-sample away score, shape (batch,)."""
    t_exp = r_t.expand_as(r_s2t)
    numerator = torch.min(
        _Sc(r_s2t, r_s),
        _Sc(t_exp, r_s),
    )
    denominator = _Sc(t_exp, r_s)
    return numerator / denominator.clamp(min=1e-8)


def _get_per_text_embeddings(
    texts: List[str],
    style_model: SentenceTransformer,
    device: str = "cuda",
) -> torch.Tensor:
    """L2-normalised per-text embeddings, shape (N, D)."""
    emb = style_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
    )
    return emb


# ---------------------------------------------------------------------------
# Reward computer
# ---------------------------------------------------------------------------

class ASTRAPOPRewardComputer:
    """
    Computes per-sample reward using the ASTRAPOP weighted joint-score formula:

        reward_i = (toward_i ** toward_exp) * (mis_i ** mis_exp)

    where toward_i and mis_i are computed per generated sample.

    Target author embedding is computed once (doc-level) and shared.
    Source text embedding is per-sample (matches each prompt).
    """

    def __init__(
        self,
        target_author_texts: List[str],
        source_author_texts: Optional[List[str]] = None,
        style_model_name: str = "/scratch/common_models/AIDA-UPM/star",
        toward_exp: float = 0.5,
        mis_exp: float = 0.5,
        device: str = "cuda",
    ):
        self.toward_exp = toward_exp
        self.mis_exp = mis_exp
        self.device = device

        print(f"Loading ASTRAPOP style model: {style_model_name}...")
        self.style_model = SentenceTransformer(style_model_name).to(device)
        self.style_model.eval()

        print(f"Computing target author doc embedding from {len(target_author_texts)} texts...")
        self.target_doc_emb = _get_doc_embedding(target_author_texts, self.style_model)  # (1, D)

        self.source_doc_emb: Optional[torch.Tensor] = None
        if source_author_texts is not None and len(source_author_texts) > 0:
            print(f"Computing source author doc embedding from {len(source_author_texts)} texts...")
            self.source_doc_emb = _get_doc_embedding(source_author_texts, self.style_model)

    def compute_rewards(
        self,
        input_texts: List[str],
        output_texts: List[str],
    ) -> np.ndarray:
        """
        Compute per-sample reward.

        Args:
            input_texts: prompts (raw or template-wrapped)
            output_texts: model completions

        Returns:
            numpy array of shape (batch,) with per-sample rewards.
        """
        cleaned_inputs = [extract_paraphrase_input(t) for t in input_texts]
        # cleaned_outputs = [clean_output_text(t) for t in output_texts]
        cleaned_outputs = output_texts

        # Per-text embeddings
        r_s = _get_per_text_embeddings(cleaned_inputs, self.style_model, self.device)   # (B, D)
        r_s2t = _get_per_text_embeddings(cleaned_outputs, self.style_model, self.device)  # (B, D)
        r_t = self.target_doc_emb  # (1, D)

        toward_scores = _per_sample_toward(r_s, r_t, r_s2t).cpu()  # (B,)
        away_scores = _per_sample_away(r_s, r_t, r_s2t).cpu()       # (B,)

        # MIS per pair (returns List[float])
        mis_scores_list = mis_compute(cleaned_outputs, cleaned_inputs, device=self.device)
        mis_scores = torch.tensor(
            [s if not math.isnan(s) else 0.0 for s in mis_scores_list],
            dtype=torch.float32,
        )

        # Clamp to [0, 1] before exponentiation
        eps = 1e-4 #1e-8 
        toward_clamped = toward_scores.clamp(min=0.0, max=1.0) + eps
        mis_clamped = mis_scores.clamp(min=0.0, max=1.0) + eps

        rewards = (toward_clamped ** self.toward_exp) * (mis_clamped ** self.mis_exp)

        print(f"  ASTRAPOP per-sample | toward mean={toward_clamped.mean():.4f} "
              f"mis mean={mis_clamped.mean():.4f} "
              f"away mean={away_scores.clamp(0,1).mean():.4f} "
              f"reward mean={rewards.mean():.4f}")

        return rewards.numpy()

    def __call__(self, input_texts: List[str], output_texts: List[str]) -> np.ndarray:
        return self.compute_rewards(input_texts, output_texts)


class TRLAstrapopRewardWrapper:
    """
    Wrapper to make ASTRAPOPRewardComputer compatible with TRL's interface.

    TRL expects:  reward_fn(prompts, completions) -> List[float]
    """

    def __init__(self, reward_computer: ASTRAPOPRewardComputer, name: str = "astrapop_reward"):
        self.reward_computer = reward_computer
        self.__name__ = name

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = self.reward_computer.compute_rewards(prompts, completions)
        return rewards.tolist()
