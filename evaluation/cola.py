"""CoLA scoring used by the paper's separate fluency analysis."""
import math
import sys
import torch

COLA_MODEL_NAME = "textattack/roberta-base-CoLA"
COLA_MAX_LENGTH = 512
COLA_CHUNK_CHARS = 200
COLA_LONG_THRESHOLD = 300


def cola_score(
    text: str,
    cola_tokenizer,
    cola_model,
    device: str,
) -> float:
    """Single-text CoLA acceptability score (probability of grammatically acceptable)."""
    text = (text or "").strip()
    if not text:
        return float("nan")
    inputs = cola_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=COLA_MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = cola_model(**inputs)
    # index 1 = acceptable (CoLA label 1)
    return output.logits.softmax(-1)[0][1].item()


def eval_grammar(
    text_ls: list[str],
    cola_tokenizer,
    cola_model,
    device: str,
    verbose: bool = False,
) -> list[float]:
    """CoLA grammar scores for a list of texts. Long texts are split and averaged."""
    cola_ls = []
    for i, text in enumerate(text_ls):
        text = (text or "").strip()
        if text == "":
            continue
        if len(text) > COLA_LONG_THRESHOLD:
            o = text
            if verbose:
                print(f"  CoLA: text index {i} length {len(o)} > {COLA_LONG_THRESHOLD}, splitting into chunks", file=sys.stderr)
            num_groups = max(1, math.ceil(len(o) / COLA_CHUNK_CHARS))
            cola_score_list = []
            for n in range(num_groups):
                start = n * (len(o) // num_groups)
                end = (n + 1) * (len(o) // num_groups) if n < num_groups - 1 else len(o)
                o_split = o[start:end]
                if o_split.strip():
                    cola_score_list.append(cola_score(o_split, cola_tokenizer, cola_model, device))
            if cola_score_list:
                cola_ls.append(sum(cola_score_list) / len(cola_score_list))
        else:
            cola_ls.append(cola_score(text, cola_tokenizer, cola_model, device))
    return cola_ls
