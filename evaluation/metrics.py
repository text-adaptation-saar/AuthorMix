"""Original STAR/MIS evaluation metrics (document-level style aggregation)."""
import math
from statistics import mean
from typing import List, Optional
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from llamafactory.x_my_pta_scripts.evaluation_patel.evaluation_metrics import (
    _get_doc_embedding, mis_compute,
)


def compute_semantic_similarity(
        input_texts: List[str],
        output_texts: List[str],
        semantic_model: SentenceTransformer = None
) -> torch.Tensor:
    semantic_model.eval()
    with torch.no_grad():
        input_emb = semantic_model.encode(
            input_texts,
            convert_to_tensor=True,
            device="cuda",
            show_progress_bar=False
        )
        output_emb = semantic_model.encode(
            output_texts,
            convert_to_tensor=True,
            device="cuda",
            show_progress_bar=False
        )

    # Compute cosine similarities in one call
    # PTA/python_scripts/eval_metrics/calculate_sts.py
    # def evaluate_on_gpu(src_lines, out_lines, overlap_weight=0.5, model_name="/local/common_models/all-MiniLM-L6-v2",
    #                     threshold=0.5)
    cosine_similarities = util.cos_sim(input_emb, output_emb).diagonal()

    # Print semantic similarity scores
    mean_sim = float(cosine_similarities.mean().item())
    print(f"\nSemantic Similarity Score:")
    print(f"  mean_cosine_similarity:  {mean_sim:.6f}")
    print(f"  per_text_scores:         {len(cosine_similarities)} texts")

    return cosine_similarities


def _sim(a, b):
    return clean_fp_error(cosine_similarity(a, b))


def _sim_raw(a, b):
    """Cosine similarity without clamping to [0,1]; for reporting raw scores (can be in [-1, 1])."""
    v = cosine_similarity(a, b, dim=1)
    return v.item() if hasattr(v, "item") else float(v)


def asim(a, b):
    return (1 - np.arccos(_sim(a, b)) / np.pi)


def clean_fp_error(v):
    ''' Helps ignore floating point error ie. 1.00001 or -0.0000001 '''
    if type(v) != float:
        v = v.item()
    v = min(max(v, 0), 1)
    if v > .99999 and v < 1.0:
        v = 1.0
    elif v < .00001 and v > 0.0:
        v = 0.0
    return v


def away_score(emb, source_emb):
    return clean_fp_error(1.0-((asim(emb, source_emb) + 1.0) / 2.0))


def towards_score(emb, target_emb):
    return clean_fp_error((asim(emb, target_emb) + 1.0) / 2.0)


def sim_score(texts, source_texts, semantic_model):
    # return clean_fp_error(mean([s for s in mis_compute(source_texts, texts) if not math.isnan(s)]))
    return clean_fp_error(mean([s for s in compute_semantic_similarity(source_texts, texts, semantic_model=semantic_model).tolist() if not math.isnan(s)]))


def mis_score(texts: List[str], reference_texts: List[str], device: str = "cuda", batch_size: Optional[int] = None) -> float:
    """
    Compute mean Mutual Implication Score (MIS) between texts and reference texts.
    Similar to sim_score but using MIS instead of semantic similarity.

    Args:
        texts: List of texts to evaluate
        reference_texts: List of reference texts (must have same length as texts)
        device: Device to use for MIS model (default: "cuda")
        batch_size: Optional batch size for MIS computation

    Returns:
        Mean MIS score (cleaned for floating point errors)
    """
    scores = mis_compute(texts, reference_texts, device=device, batch_size=batch_size)
    # Filter out NaN values and compute mean
    valid_scores = [s for s in scores if not math.isnan(s)]
    if not valid_scores:
        return 0.0
    return clean_fp_error(mean(valid_scores))


def joint_score(
    source_texts: List[str],
    target_texts: List[str],
    style_transferred_texts: List[str],
    semantic_model: SentenceTransformer,
    style_model: SentenceTransformer,
    return_components: bool = False,
):
    """
    Joint score combining style transfer (away from source, towards target) and content preservation (sim).

    Uses style_model for away/towards in style space and semantic_model for content similarity.
    """
    # Doc-level style embeddings via style_model (aligned with get_style_similarity_for_author)
    source_emb = _get_doc_embedding(source_texts, style_model)
    target_emb = _get_doc_embedding(target_texts, style_model)
    st_emb = _get_doc_embedding(style_transferred_texts, style_model)

    # Calculate the raw scores (unclamped so target/source similarity can be negative)
    raw_output_vs_target = _sim_raw(st_emb, target_emb)
    raw_output_vs_source = _sim_raw(st_emb, source_emb)
    raw_target_vs_source = _sim_raw(target_emb, source_emb)
    raw_source_vs_target = _sim_raw(source_emb, target_emb)

    # Calculate asim scores
    asim_output_vs_target = asim(st_emb, target_emb)
    asim_output_vs_source = asim(st_emb, source_emb)
    asim_target_vs_source = asim(target_emb, source_emb)
    asim_source_vs_target = asim(source_emb, target_emb)

    # Away score: how far style-transferred is from source (scaled by max possible = target vs source)
    score_away = away_score(st_emb, source_emb)
    score_away_max = away_score(target_emb, source_emb)
    score_away_scaled = clean_fp_error(min(score_away, score_away_max)/score_away_max) # Ensures we can achieve a 1.0 when reaching the away maximum

    # Towards score: how close style-transferred is to target (scaled above source→target baseline)
    score_towards = towards_score(st_emb, target_emb)
    score_towards_min = towards_score(source_emb, target_emb)
    score_towards_scaled = clean_fp_error(max(score_towards - score_towards_min, 0)/(1.0 - score_towards_min)) # Ensures we can achieve a 0.0 when scoring at or below the towards minimum

    # Sim score: content preservation (semantic_model), scaled above target→source baseline
    score_sim = sim_score(style_transferred_texts, source_texts, semantic_model)
    score_sim_min = sim_score(target_texts, source_texts, semantic_model)
    score_sim_scaled = clean_fp_error(max(score_sim - score_sim_min, 0) / (1.0 - score_sim_min)) # Ensures we can achieve a 0.0 when scoring at or below the sim minimum

    # Calculate the joint score
    score_joint = clean_fp_error((((score_away_scaled * score_towards_scaled) ** (1/2)) * score_sim_scaled) ** (1/2))
    # calculate joint score without score_away_scaled
    score_joint_without_away_scaled = clean_fp_error(score_towards_scaled * score_sim_scaled) ** (1/2)


    mis_score_mean = mis_score(style_transferred_texts, source_texts, device="cuda")
    mis_score_min = mis_score(target_texts, source_texts, device="cuda")
    mis_score_scaled = clean_fp_error(max(mis_score_mean - mis_score_min, 0) / (1.0 - mis_score_min)) # Ensures we can achieve a 0.0 when scoring at or below the sim minimum
    score_joint_mis = clean_fp_error((((score_towards_scaled * mis_score_scaled) ** (1/2)) * score_sim_scaled) ** (1/2))
    score_joint_without_away_scaled_mis = clean_fp_error(score_towards_scaled * mis_score_scaled) ** (1/2)


    # Print all scores for visibility
    print("\n" + "="*60)
    print("EVALUATION SCORES")
    print("="*60)
    print(f"Raw Scores:")
    print(f"  raw_output_vs_target:    {raw_output_vs_target:.6f}")
    print(f"  raw_output_vs_source:    {raw_output_vs_source:.6f}")
    print(f"  raw_target_vs_source:    {raw_target_vs_source:.6f}")
    print(f"  raw_source_vs_target:    {raw_source_vs_target:.6f}")
    print(f"Asim Scores:")
    print(f"  asim_output_vs_target:    {asim_output_vs_target:.6f}")
    print(f"  asim_output_vs_source:    {asim_output_vs_source:.6f}")
    print(f"  asim_target_vs_source:    {asim_target_vs_source:.6f}")
    print(f"  asim_source_vs_target:    {asim_source_vs_target:.6f}")
    print(f"Away Score:")
    print(f"  score_away:              {score_away:.6f}")
    print(f"  score_away_max:          {score_away_max:.6f}")
    print(f"  score_away_scaled:       {score_away_scaled:.6f}")
    print(f"Towards Score:")
    print(f"  score_towards:           {score_towards:.6f}")
    print(f"  score_towards_min:       {score_towards_min:.6f}")
    print(f"  score_towards_scaled:    {score_towards_scaled:.6f}")
    print(f"Sim Score:")
    print(f"  score_sim:               {score_sim:.6f}")
    print(f"  score_sim_min:           {score_sim_min:.6f}")
    print(f"  score_sim_scaled:        {score_sim_scaled:.6f}")
    print(f"Scaled Scores:")
    print(f"  score_away_scaled:       {score_away_scaled:.6f}")
    print(f"  score_towards_scaled:    {score_towards_scaled:.6f}")
    print(f"  score_sim_scaled:        {score_sim_scaled:.6f}")
    print(f"Joint Score:")
    print(f"  score_joint:             {score_joint:.6f}")
    print("="*60 + "\n")

    print(f"MIS Score:")
    print(f"  mis_score_mean:          {mis_score_mean:.6f}")
    print(f"  mis_score_min:           {mis_score_min:.6f}")
    print(f"  mis_score_scaled:        {mis_score_scaled:.6f}")
    print(f"Joint MIS Score:")
    print(f"  joint_mis:               {score_joint_mis:.6f}")
    print(f"  joint_without_away_scaled_mis: {score_joint_without_away_scaled_mis:.6f}")

    # return clean_fp_error(score_joint)


    if return_components:
        return {
            "raw_output_vs_target": raw_output_vs_target,
            "raw_output_vs_source": raw_output_vs_source,
            "raw_target_vs_source": raw_target_vs_source,
            "raw_source_vs_target": raw_source_vs_target,
            "asim_output_vs_target": asim_output_vs_target,
            "asim_output_vs_source": asim_output_vs_source,
            "asim_target_vs_source": asim_target_vs_source,
            "asim_source_vs_target": asim_source_vs_target,
            "score_away": score_away,
            "score_away_max": score_away_max,
            "score_away_scaled": score_away_scaled,
            "score_towards": score_towards,
            "score_towards_min": score_towards_min,
            "score_towards_scaled": score_towards_scaled,
            "score_semantic_similarity": score_sim,
            "score_semantic_similarity_min": score_sim_min,
            "score_semantic_similarity_scaled": score_sim_scaled,
            "score_joint": score_joint,
            "score_joint_without_away_scaled": score_joint_without_away_scaled,

            # mis scores
            "mis_score_mean": mis_score_mean,
            "mis_score_min": mis_score_min,
            "mis_score_scaled": mis_score_scaled,
            "joint_mis": score_joint_mis,
            "joint_without_away_scaled_mis": score_joint_without_away_scaled_mis,
        }
    return clean_fp_error(score_joint)


# Independent CAV angular similarity, as used in the original re-evaluation.
def S_raw(us, vs):
    cossims = cosine_similarity(us, vs)
    return cossims


def S(us, vs):
    cossims = cosine_similarity(us, vs)
    sims = 1 - torch.acos(cossims) / math.pi
    return sims


def Sc(us, vs):
    return 1 - S(us, vs)


def away(r_s, r_t, r_s2t, agg=True):
    numerator = torch.vstack((Sc(r_s2t, r_s), Sc(r_t, r_s)))
    numerator = numerator.min(dim=0).values
    demoninator = Sc(r_t, r_s)
    if not agg:
        return (numerator / demoninator)
    return float((numerator / demoninator).mean())


def toward(r_s, r_t, r_s2t, agg=True):
    numerator = S(r_s2t, r_t) - S(r_s, r_t)
    numerator[numerator < 0] = 0
    demoninator = Sc(r_s, r_t)
    if not agg:
        return (numerator / demoninator)
    return float((numerator / demoninator).mean())
