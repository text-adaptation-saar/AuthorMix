#!/usr/bin/env python3
"""Generate once per source-target pair and score the resulting rewrites."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['DISABLE_VERSION_CHECK'] = '1'
EXPERIMENT = ROOT / 'PTA/19_RL_on_10src_15tgt_authors'


def load_entry(variant):
    folder = ('weight_optimization_per_tgt_author' if variant == 'layerwise' else
              'adapterwise_weight_optimization_per_tgt_author_layerwiseFalse')
    spec = importlib.util.spec_from_file_location('authormix_training_entry', EXPERIMENT / folder / 'run_rl_per_pair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_data(data):
    if len(data['source_authors']) != 10 or len(data['target_authors']) != 10:
        raise ValueError('Use the released 10-source, 10-target test dataset.')
    for role in ('source_authors', 'target_authors'):
        for author, texts in data[role].items():
            if len(texts) != 16 or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError(f'{author}: expected 16 nonempty reference texts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['layerwise', 'adapterwise'], required=True)
    parser.add_argument('--training-dir', type=Path, required=True,
                        help='Directory containing tgt_author_*/training_results.json')
    parser.add_argument('--base-model', default='models/Llama-3.1-8B-Instruct')
    parser.add_argument('--test-data', type=Path, default=ROOT / 'data/dataset_author_eval/sents/json/test_src10_tgt10.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--style-model', default='AIDA-UPM/star')
    parser.add_argument('--semantic-model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--cav-model', help='Optional SentenceTransformer path to CAV exp04 gutenberg50-2k-chunk')
    parser.add_argument('--cola', action='store_true', help='Also compute the paper fluency metric')
    parser.add_argument('--target-author', help='Evaluate one of the released target authors')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true', help='Check data and saved metadata without loading models')
    args = parser.parse_args()
    data = json.loads(args.test_data.read_text())
    validate_data(data)
    if args.target_author and args.target_author not in data['target_authors']:
        parser.error('Unknown target author')
    targets = [args.target_author] if args.target_author else list(data['target_authors'])
    records = []
    for target in targets:
        path = args.training_dir / f'tgt_author_{target}' / 'training_results.json'
        record = json.loads(path.read_text())
        if not record.get('adapter_paths') or not record.get('optimized_weights'):
            raise ValueError(f'{path}: missing adapter paths or optimized weights')
        if record.get('rl_config', {}).get('layerwise') is not None:
            if record['rl_config']['layerwise'] != (args.variant == 'layerwise'):
                raise ValueError('Training metadata and requested variant disagree')
        records.append((target, record))
    print(f'{len(targets)} targets x 10 sources; one generation pass per pair, seed {args.seed}')
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory to avoid overwriting an evaluation.')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    from sentence_transformers import SentenceTransformer
    from metrics import joint_score, toward, sim_score, _get_doc_embedding
    from cola import COLA_MODEL_NAME, eval_grammar
    if not torch.cuda.is_available():
        parser.error('The original metric helpers require CUDA.')
    entry = load_entry(args.variant)
    if args.variant == 'layerwise':
        from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_lm.learnable_layerwise_weight_module_with_layer_wrapper import LayerwiseLearnableWeightModule as MixingModel
    else:
        from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_lm.learnable_layerwise_weight_module_with_layer_wrapper_layerwiseFalse import AdapterwiseLearnableWeightModule as MixingModel
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    style = SentenceTransformer(args.style_model, device='cuda')
    semantic = SentenceTransformer(args.semantic_model, device='cuda')
    cav = SentenceTransformer(args.cav_model, device='cuda') if args.cav_model else None
    if args.cola:
        cola_tokenizer = AutoTokenizer.from_pretrained(COLA_MODEL_NAME)
        cola_model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL_NAME).to('cuda').eval()
    rows = []
    for target, record in records:
        names = [f'adapter-{entry.get_author_to_adapter_abbreviation()[a]}' for a in record['related_authors']]
        if record.get('rewrite_task_adapter'):
            names.insert(0, 'rewrite-adapter')
        model = MixingModel(args.base_model, record['adapter_paths'], adapter_names=names)
        weights = torch.tensor(record['optimized_weights'], dtype=model.weight_logits.dtype, device=model.weight_logits.device)
        if weights.shape != model.weight_logits.shape or not torch.isfinite(weights).all():
            raise ValueError(f'{target}: invalid weights for {args.variant}')
        with torch.no_grad():
            model.weight_logits.copy_(weights)
        for source, inputs in data['source_authors'].items():
            set_seed(args.seed)
            outputs = entry.generate_with_rl_model(model, tokenizer, inputs, args.batch_size, args.max_new_tokens)
            target_texts = data['target_authors'][target]
            scores = joint_score(inputs, target_texts, outputs, semantic, style, return_components=True)
            row = {'source_author': source, 'target_author': target, 'seed': args.seed,
                   'test_source_texts': inputs, 'test_target_texts': target_texts,
                   'generated_texts': outputs, 'joint_score_patel': scores}
            if cav is not None:
                r_s = _get_doc_embedding(inputs, cav)
                r_t = _get_doc_embedding(target_texts, cav)
                r_o = _get_doc_embedding(outputs, cav)
                toward_cav = toward(r_s, r_t, r_o)
                sbert = sim_score(outputs, inputs, semantic)
                row['cav_scores'] = {'toward': toward_cav, 'sbert': sbert,
                                     'joint': (toward_cav * sbert) ** 0.5}
            if args.cola:
                values = eval_grammar(outputs, cola_tokenizer, cola_model, 'cuda')
                row['cola'] = {'scores': values, 'mean': sum(values) / len(values) if values else None}
            path = args.output_dir / f'src_{source}_to_tgt_{target}.json'
            path.write_text(json.dumps(entry.to_json_serializable(row), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
            rows.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    keys = {'toward': 'score_towards_scaled', 'mis': 'mis_score_mean', 'joint': 'joint_without_away_scaled_mis'}
    summary = {name: sum(r['joint_score_patel'][key] for r in rows)/len(rows) for name,key in keys.items()}
    summary.update(pairs=len(rows), seed=args.seed, passes_per_pair=1)
    if cav is not None:
        summary['cav'] = {key: sum(r['cav_scores'][key] for r in rows)/len(rows) for key in ('toward','sbert','joint')}
    if args.cola:
        values = [r['cola']['mean'] for r in rows if r['cola']['mean'] is not None]
        summary['cola'] = sum(values)/len(values) if values else None
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    settings = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    (args.output_dir/'evaluation_config.json').write_text(json.dumps(settings,indent=2)+'\n')


if __name__ == '__main__':
    main()
