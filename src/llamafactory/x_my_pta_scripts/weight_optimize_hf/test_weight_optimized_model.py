"""
test_weight_optimized_model.py - FIXED VERSION

Comprehensive testing script for weight-optimized adapter models.
Supports loading from optimized weights JSON or saved checkpoints.
Also supports testing base model only (no adapters).
"""

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, Dataset
from typing import List, Dict, Optional, Union
import json
import os
from tqdm import tqdm
import argparse



def load_model_with_weights(
    base_model_path: str,
    adapter_paths: Optional[List[str]] = None,
    adapter_names: Optional[List[str]] = None,
    weights_source: Union[str, List[float], None] = None,
    use_adapterwise: bool = False,
):
    """
    Load model with optimized weights or base model only.

    Args:
        base_model_path: Path to base model
        adapter_paths: List of adapter paths (None for base model only)
        adapter_names: List of adapter names (None for base model only)
        weights_source: Can be:
            - Path to optimized_weights.json
            - List of weights directly (1D for adapterwise, 2D list-of-lists for layerwise)
            - None (use default initialization or base model only)
        use_adapterwise: If True, use AdapterwiseLearnableWeightModule (one weight per adapter)
            and load from adapter_weights or 1D optimized_weights. If False, use layerwise module.

    Returns:
        Model with weights loaded (or base model if no adapters)
    """

    print("[test_weight_optimized_model.load_model_with_weights()] LOADING MODEL")
    print(f"[test_weight_optimized_model.load_model_with_weights()] Base model: {base_model_path}")
    if use_adapterwise:
        print("[test_weight_optimized_model.load_model_with_weights()] Mode: adapter-wise (one weight per adapter)")

    # Base model only (no adapters)
    if adapter_paths is None or len(adapter_paths) == 0:
        print("[test_weight_optimized_model.load_model_with_weights()] Loading BASE MODEL ONLY (no adapters)")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("[test_weight_optimized_model.load_model_with_weights()] ✓ Base model loaded")
        return model

    if len(adapter_paths) == 1 and weights_source is None:
        print("[test_weight_optimized_model.load_model_with_weights()] Loading BASE MODEL and single adapter without weights")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        for param in base_model.parameters():
            param.requires_grad = False

        print("[test_weight_optimized_model.load_model_with_weights()] ✓ Base model loaded")

        model = PeftModel.from_pretrained(
            base_model,
            adapter_paths[0],
            adapter_name=adapter_names[0],
            is_trainable=False  # Freeze adapters!
        )
        # Freeze all adapter parameters
        for param in model.parameters():
            param.requires_grad = False
        print(f"[test_weight_optimized_model.load_model_with_weights()] ✓ Loaded single adapter from: {adapter_paths[0]}")
        return model

    # Model with adapters (adapter-wise or layerwise)
    if use_adapterwise:
        from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_lm.learnable_layerwise_weight_module_with_layer_wrapper_layerwiseFalse import AdapterwiseLearnableWeightModule
        model_cls = AdapterwiseLearnableWeightModule
    else:
        from llamafactory.x_my_pta_scripts.weight_optimize_hf.custom_lm.learnable_layerwise_weight_module_with_layer_wrapper import LayerwiseLearnableWeightModule
        model_cls = LayerwiseLearnableWeightModule

    print(f"[test_weight_optimized_model.load_model_with_weights()] Adapters: {len(adapter_paths)}")
    for i, (path, name) in enumerate(zip(adapter_paths, adapter_names), 1):
        print(f"[test_weight_optimized_model.load_model_with_weights()]  {i}. {name}: {path}")

    model = model_cls(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        adapter_names=adapter_names
    )

    # Load weights if provided
    if weights_source is not None:
        if isinstance(weights_source, str):
            print(f"\n[test_weight_optimized_model.load_model_with_weights()] Loading weights from: {weights_source}")
            with open(weights_source, 'r') as f:
                data = json.load(f)

            if use_adapterwise:
                adapter_weights = data.get("adapter_weights")
                if isinstance(adapter_weights, dict):
                    optimized_weights = [float(adapter_weights.get(n, 0.0)) for n in adapter_names]
                elif isinstance(adapter_weights, (list, tuple)) and len(adapter_weights) == len(adapter_names):
                    optimized_weights = [float(x) for x in adapter_weights]
                else:
                    ow = data.get("optimized_weights")
                    if isinstance(ow, list) and len(ow) > 0 and not isinstance(ow[0], list):
                        optimized_weights = [float(x) for x in ow]
                    else:
                        optimized_weights = None
                if optimized_weights is None:
                    raise ValueError("adapter_weights or 1D optimized_weights not found in JSON for adapter-wise mode")
                print(f"[test_weight_optimized_model.load_model_with_weights()] Loaded adapter-wise weights: {len(optimized_weights)} adapters")
            else:
                optimized_weights = data["optimized_weights"]
                print(f"[test_weight_optimized_model.load_model_with_weights()] Loaded weights shape: {len(optimized_weights)} layers × {len(optimized_weights[0])} adapters")

        elif isinstance(weights_source, list):
            optimized_weights = weights_source
            print(f"\n[test_weight_optimized_model.load_model_with_weights()] Using provided weights")
            if use_adapterwise and optimized_weights and isinstance(optimized_weights[0], list):
                # Flatten if 2D passed by mistake
                optimized_weights = [float(optimized_weights[0][i]) for i in range(len(optimized_weights[0]))]
        else:
            raise ValueError(
                f"weights_source must be either:\n"
                f"  - str (path to optimized_weights.json)\n"
                f"  - list (weights directly)\n"
                f"  - None (random initialization)\n"
                f"Got type: {type(weights_source)}"
            )

        with torch.no_grad():
            weights_tensor = torch.tensor(
                optimized_weights,
                dtype=torch.float32,
                device=model.weight_logits.device
            )
            model.weight_logits.copy_(weights_tensor)

        print(f"\n[test_weight_optimized_model.load_model_with_weights()] ✓ Weights loaded successfully!")
        model.print_weights(verbose=True)
    else:
        print("\n[test_weight_optimized_model.load_model_with_weights()] ⚠ No weights provided - using random initialization")
        model.print_weights(verbose=True)

    return model


def load_test_dataset(
        dataset_path: str,
        tokenizer,
        dataset_type: str = "auto",
        max_samples: Optional[int] = None,
):
    """
    Load test dataset from various formats.

    Args:
        dataset_path: Path to dataset
        tokenizer: Tokenizer to use
        dataset_type: "auto", "pretokenized", "huggingface", or "json"
        max_samples: Limit number of samples (None = all)

    Returns:
        Loaded dataset
    """
    print("\n" + "=" * 80)
    print("LOADING TEST DATASET")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset type: {dataset_type}")

    # Auto-detect dataset type
    if dataset_type == "auto":
        if os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
            dataset_type = "pretokenized"
        elif dataset_path.endswith(".json"):
            dataset_type = "json"
        else:
            dataset_type = "huggingface"

    # Load dataset based on type
    if dataset_type == "pretokenized":
        print(f"Loading pre-tokenized dataset...")

        # Check for validation split
        val_path = os.path.join(dataset_path, "validation")
        if os.path.exists(val_path):
            dataset = load_from_disk(val_path)
        else:
            # Try test split
            test_path = os.path.join(dataset_path, "test")
            if os.path.exists(test_path):
                dataset = load_from_disk(test_path)
            else:
                # Use train split
                dataset = load_from_disk(os.path.join(dataset_path, "train"))

        # Remove extra columns
        if 'images' in dataset.column_names:
            dataset = dataset.remove_columns(['images', 'videos', 'audios'])

        print(f"✓ Loaded {len(dataset)} pre-tokenized samples")

    elif dataset_type == "json":
        print(f"Loading from JSON file...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # Convert to dataset
        dataset = Dataset.from_list(data)
        print(f"✓ Loaded {len(dataset)} samples from JSON")

    else:  # huggingface
        print(f"Loading HuggingFace dataset...")
        # Try test split first
        try:
            dataset = load_dataset(dataset_path, split='test')
        except:
            # Fall back to validation
            try:
                dataset = load_dataset(dataset_path, split='validation')
            except:
                # Fall back to train
                dataset = load_dataset(dataset_path, split='train')

        print(f"✓ Loaded {len(dataset)} samples")

    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        print(f"Limiting to first {max_samples} samples")
        dataset = dataset.select(range(max_samples))

    print(f"Final dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    print("=" * 80 + "\n")

    return dataset


def prepare_inputs_for_generation(
        dataset,
        tokenizer,
        max_length: int = 512,
):
    """
    Prepare inputs for generation from dataset.
    Matches the working code's input preparation.
    """
    inputs = []

    print("Preparing inputs for generation...")

    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        # Handle instruction/input/output format
        if 'instruction' in example and 'input' in example:
            instruction = example['instruction']
            input_text = example['input']

            # Create messages format
            messages = [
                {"role": "user", "content": f"\n\n{instruction}\n{input_text}"}
            ]

            # Apply chat template with generation prompt
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = formatted_text
            # # Remove unwanted system block
            # if isinstance(formatted_text, list):
            #     prompt = [remove_system_block(t) for t in formatted_text]
            # else:
            #     prompt = remove_system_block(formatted_text)


            inputs.append({
                'prompt': prompt,
                'instruction': instruction,
                'input': input_text,
                'ground_truth': example.get('output', None),
                'index': i
            })

        elif 'text' in example:
            # Plain text
            inputs.append({
                'prompt': example['text'],
                'text': example['text'],
                'index': i
            })

        else:
            print(f"⚠ Warning: Example {i} has unknown format, skipping")
            continue

    print(f"✓ Prepared {len(inputs)} inputs")
    return inputs


def get_llama3_stop_token_ids(tokenizer):
    """Get Llama 3 specific stop tokens"""
    stop_ids = [tokenizer.eos_token_id]

    # Add Llama 3 specific tokens
    special_tokens = [
        "<|eot_id|>",  # End of turn
        "<|end_header_id|>",  # End of header (less common)
    ]

    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            stop_ids.append(token_id)

    return stop_ids


def generate_predictions_single(
        model,
        tokenizer,
        inputs: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 0.0,  # Match your vLLM setting
        repetition_penalty: float = 1.0,
        do_sample: bool = False,  # Greedy when temperature=0
):
    """Generate with vLLM-compatible parameters"""

    model.eval()
    device = next(model.parameters()).device

    # ✅ Get proper stop tokens (like vLLM does)
    stop_token_ids = get_llama3_stop_token_ids(tokenizer)
    print(f"Using stop tokens: {stop_token_ids}")

    results = []

    with torch.no_grad():
        for item in tqdm(inputs, desc="Generating"):
            # Tokenize single prompt (no padding!)
            prompt = item['prompt']

            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)

            # Generate with proper stop tokens
            outputs = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                eos_token_id=stop_token_ids,  # ← Multiple stop tokens!
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode output (simple - just remove input length)
            # input_length = tokenized['input_ids'].shape[1]
            input_length = len(tokenized.input_ids[0])
            generated_ids = outputs[0][input_length:]  # outputs[0] since batch=1
            # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            # https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb#scrollTo=5UNOw-E0LWAs
            # Decode without special tokens (like vLLM result)
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True  # ← Clean output
            )

            result = {
                'index': item['index'],
                'prompt': prompt,
                'generated_output': generated_text.strip(),  # Strip whitespace
            }

            # Add instruction/input if available
            if 'instruction' in item:
                result['instruction'] = item['instruction']
            result['input'] = item['input']

            # Add ground truth if available
            if 'ground_truth' in item:
                result['ground_truth'] = item['ground_truth']

            results.append(result)

    print(f"\n✓ Generated {len(results)} predictions")
    return results



def generate_predictions(
        model,
        tokenizer,
        inputs: List[Dict],
        batch_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
):
    """
    Generate predictions for all inputs.
    Uses the same approach as the working code.
    """
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    print(f"Number of samples: {len(inputs)}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Do sample: {do_sample}")
    print("=" * 80 + "\n")

    model.eval()
    device = next(model.parameters()).device
    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch_items = inputs[i:i + batch_size]

            # Tokenize prompts
            prompts = [item['prompt'] for item in batch_items]

            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            # Generate
            outputs = model.generate(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                # top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode outputs
            for j, output in enumerate(outputs):
                # # Get only the generated part (remove input)
                # input_length = tokenized['input_ids'][j].ne(tokenizer.pad_token_id).sum()
                # generated_ids = output[input_length:]

                # ✅ FIX: Use full input length (including padding)
                input_length = tokenized['input_ids'].shape[1]
                generated_ids = output[input_length:]



                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                result = {
                    'index': batch_items[j]['index'],
                    'prompt': batch_items[j].get('prompt', ''),
                    'generated_output': generated_text,
                }

                # Add instruction/input if available
                if 'instruction' in batch_items[j]:
                    result['instruction'] = batch_items[j]['instruction']
                result['input'] = batch_items[j]['input']

                # Add ground truth if available
                if 'ground_truth' in batch_items[j] and batch_items[j]['ground_truth']:
                    result['ground_truth'] = batch_items[j]['ground_truth']

                results.append(result)

    print(f"\n✓ Generated {len(results)} predictions")
    return results


def save_results(
        results: List[Dict],
        output_dir: str,
        base_name: str = "test_results",
):
    """
    Save results in multiple formats.
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save full results as JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Full results saved to: {json_path}")

    # 2. Save inputs and outputs separately
    inputs_path = os.path.join(output_dir, f"inputs.txt")
    outputs_path = os.path.join(output_dir, f"outputs.txt")

    with open(inputs_path, 'w', encoding='utf-8') as f_in, \
            open(outputs_path, 'w', encoding='utf-8') as f_out:

        for i, result in enumerate(results):
            f_in.write(result['input'] + "\n")
            f_out.write(result['generated_output'].replace("\n", " ").strip() + "\n")

    print(f"✓ Inputs saved to: {inputs_path}")
    print(f"✓ Outputs saved to: {outputs_path}")

    # 3. Save summary statistics
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Results Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total samples: {len(results)}\n")

        # Compute average generation length
        avg_length = sum(len(r['generated_output'].split()) for r in results) / len(results)
        f.write(f"Average generation length: {avg_length:.1f} words\n")

        # Show first 3 examples
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Sample Predictions (First 3)\n")
        f.write(f"{'=' * 80}\n\n")

        for i, result in enumerate(results[:3]):
            f.write(f"\n--- Example {i + 1} ---\n")
            if 'instruction' in result:
                f.write(f"Instruction: {result['instruction']}\n")
                f.write(f"Input: {result['input']}\n")
            else:
                f.write(f"Input: {result['input'][:200]}...\n")
            f.write(f"\ngenerated_output: {result['generated_output']}\n")
            if 'ground_truth' in result:
                f.write(f"\nGround Truth: {result['ground_truth']}\n")
            f.write(f"{'-' * 80}\n")

    print(f"✓ Summary saved to: {summary_path}")
    print("=" * 80 + "\n")


def test_weight_optimized_model(
        base_model_path: str,
        test_dataset_path: str,
        output_dir: str,
        adapter_paths: Optional[List[str]] = None,
        adapter_names: Optional[List[str]] = None,
        weights_source: Union[str, List[float], None] = None,
        tokenizer_path: Optional[str] = None,
        dataset_type: str = "auto",
        max_samples: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,  # Match vLLM
        repetition_penalty: float = 1.0,  # Match vLLM
        max_length: int = 512,
):
    """
    Test with vLLM-compatible settings (temperature=0, proper stop tokens)
    """

    print("\n" + "=" * 80)
    print("MODEL TESTING (vLLM-compatible)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # 1. Load tokenizer
    print("Loading tokenizer...")
    tokenizer_path = tokenizer_path or base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # padding_side="left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # No need to set padding_side!
    # Single generation = no padding needed
    # tokenizer.padding_side = 'left'
    # https://github.com/tloen/alpaca-lora/issues/514
    # https://github.com/unslothai/unsloth/issues/267
    # https://github.com/huggingface/transformers/issues/26072
    print(f"✓ Tokenizer loaded")
    print(f"  Pad token: {tokenizer.pad_token}")
    print()

    # 2. Load model (with or without adapters)
    model = load_model_with_weights(
        base_model_path=base_model_path,
        adapter_paths=adapter_paths,
        adapter_names=adapter_names,
        weights_source=weights_source,
    )

    # 3. Load test dataset
    dataset = load_test_dataset(
        dataset_path=test_dataset_path,
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        max_samples=max_samples,
    )

    # 4. Prepare inputs
    inputs = prepare_inputs_for_generation(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # # 5. Generate predictions
    # results = generate_predictions(
    # 5. Generate predictions (single, no batching)
    results = generate_predictions_single(  # ← Changed function
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        # batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0),  # Sample if temp > 0
    )

    # 6. Save results
    save_results(
        results=results,
        output_dir=output_dir,
        base_name="test_results",
    )

    print("\n✅ Testing complete!")
    return {'results': results, 'num_samples': len(results)}


if __name__ == "__main__":

    # Example 1: Test BASE MODEL ONLY
    print("Example 1: Testing base model only (no adapters)")

    test_weight_optimized_model(
        base_model_path="/scratch/common_models/Llama-3.1-8B-Instruct",
        # No adapters - will use base model only
        adapter_paths=None,
        adapter_names=None,
        weights_source=None,

        test_dataset_path="data/paraNMT/sft_wiki_test/from_gr_10_to_12_subset",
        output_dir="./test_results/base_model_only",

        max_samples=10,
        batch_size=2,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,  # Greedy
    )

    print("\n" + "="*80 + "\n")

    # Example 2: Test with optimized weights
    # test_weight_optimized_model(
    #     base_model_path="/scratch/common_models/Llama-3.1-8B-Instruct",
    #     adapter_paths=[
    #         "PTA/.../author_adapter/model",
    #         "PTA/.../rewrite_adapter/model"
    #     ],
    #     adapter_names=["author-adapter", "rewrite-adapter"],
    #     weights_source="PTA/.../optimized_weights.json",
    #     test_dataset_path="data/test",
    #     output_dir="./test_results/optimized",
    # )