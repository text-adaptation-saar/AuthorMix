# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from tqdm import tqdm

# import fire
from peft import PeftModel
from transformers import Seq2SeqTrainingArguments, AutoModelForCausalLM

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.x_my_pta_scripts.weight_optimize_hf.test_weight_optimized_model import load_model_with_weights
from typing import List, Dict, Optional, Union


def hf_infer(
        model_name_or_path: str,
        adapter_name_or_path: Union[str, List[str]] = None,
        dataset: str = "alpaca_en_demo",
        dataset_dir: str = "data",
        template: str = "default",
        cutoff_len: int = 2048,
        max_samples: int = None,
        save_name: str = "generated_predictions.jsonl",
        temperature: float = 0.0,
        top_p: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.0,
        image_resolution: int = 512 * 512,
        adapter_names: Optional[List[str]] = None,
        weights_source: Union[str, List[float], None] = None,
):
    r"""
    Performs generation using HuggingFace Transformers (instead of vLLM).
    Uses LlamaFactory's data pipeline for consistency.

    Usage:
    python hf_infer_with_llamafactory.py \
        --model_name_or_path /path/to/model \
        --adapter_name_or_path /path/to/adapter \
        --dataset your_dataset \
        --template llama3 \
        --temperature 0 \
        --max_samples 100
    """

    print("[vllm_infer_script_adapted_for_using_hf.hf_infer()] HF INFERENCE (Using LlamaFactory Data Pipeline)")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Model: {model_name_or_path}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Adapter: {adapter_name_or_path}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Adapter Names: {adapter_names}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Weights source: {weights_source}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Dataset: {dataset}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Template: {template}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Cutoff length: {cutoff_len}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Temperature: {temperature}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Max samples: {max_samples}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Max new tokens: {max_new_tokens}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Repetition penalty: {repetition_penalty}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Save name: {save_name}")

    if adapter_name_or_path is None:
        adapter_paths_str = None
    else:
        adapter_paths_str = ",".join(adapter_name_or_path) if isinstance(adapter_name_or_path, list) else str(adapter_name_or_path)
    # Get directory path (without filename)
    output_folder = os.path.dirname(save_name)

    # Create folder if it doesn't exist
    if output_folder:  # Check if not empty string
        os.makedirs(output_folder, exist_ok=True)

    inputs, attention_mask, labels, model_args, prompts, template_obj, tokenizer = prepare_dataset_for_generation_llamafactory_compatible(
        adapter_paths_str, cutoff_len, dataset, dataset_dir, max_new_tokens, max_samples, model_name_or_path,
        repetition_penalty, temperature, template, top_k, top_p)

    # ========================================================================
    # 3. Load model with HuggingFace (instead of vLLM)
    # ========================================================================

    print("\n[vllm_infer_script_adapted_for_using_hf.hf_infer()] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Load base model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    #
    # # Load adapter if provided
    # if model_args.adapter_name_or_path is not None:
    #     print(f"Loading adapter from: {model_args.adapter_name_or_path[0]}")
    #     model = PeftModel.from_pretrained(
    #         model,
    #         model_args.adapter_name_or_path[0],
    #         adapter_name="loaded_adapter",
    #         is_trainable=False
    #     )
    #     print("✓ Adapter loaded")

    model = load_model_with_weights(
        base_model_path=model_args.model_name_or_path,
        adapter_paths=model_args.adapter_name_or_path,
        adapter_names=adapter_names,
        weights_source=weights_source,
    )

    model.eval()
    print("[vllm_infer_script_adapted_for_using_hf.hf_infer()] ✓ Model loaded and set to eval mode")

    # ========================================================================
    # 4. Get stop tokens (like vLLM does)
    # ========================================================================

    stop_token_ids = template_obj.get_stop_token_ids(tokenizer)
    print(f"\n[vllm_infer_script_adapted_for_using_hf.hf_infer()] Stop token IDs: {stop_token_ids}")

    # ========================================================================
    # 5. Generate predictions (single sample at a time, no batching)
    # ========================================================================

    print("[vllm_infer_script_adapted_for_using_hf.hf_infer()] GENERATING PREDICTIONS")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Mode: Single sample generation (no batching)")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Temperature: {temperature}")
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Max new tokens: {max_new_tokens}")

    preds = []

    with torch.no_grad():
        for idx, (input_ids, attention_mask_i) in enumerate(tqdm(zip(inputs, attention_mask), desc="Generating")):
            # Prepare input (single sample)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            mask_tensor = torch.tensor([attention_mask_i], dtype=torch.long).to(device)  # Assigning the mask

            # Generate
            # Match vLLM's behavior with temperature
            if temperature == 0 or temperature < 0.01:
                # Greedy decoding
                do_sample = False
                actual_temp = 1.0  # Ignored when do_sample=False
            else:
                # Sampling
                do_sample = True
                actual_temp = temperature

            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=mask_tensor,  # Pass the assigned mask here
                max_new_tokens=max_new_tokens,

                # Sampling parameters (match vLLM)
                do_sample=do_sample,
                temperature=actual_temp,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else -1,
                repetition_penalty=repetition_penalty,

                # Stop tokens (match vLLM)
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Extract only generated tokens
            input_length = len(input_ids)
            generated_ids = outputs[0][input_length:]

            # Decode (skip_special_tokens=False to match vLLM)
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            preds.append(generated_text)

    # ========================================================================
    # 6. Save results (same format as vLLM)
    # ========================================================================

    print(f"\n[vllm_infer_script_adapted_for_using_hf.hf_infer()] Saving results to: {save_name}")

    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(prompts, preds, labels):
            f.write(
                json.dumps(
                    {"prompt": text, "predict": pred, "label": label},
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] {len(prompts)} generated results have been saved at {save_name}.")

    # get the folder of save_name
    save_name_folder = os.path.dirname(save_name)
    # save outputs to separate files
    with open(save_name_folder + "/output.txt", "w", encoding="utf-8") as gen_f:
        for pred in preds:
            pred_single_line = " ".join(pred.splitlines())
            # pred_single_line = " ".join(pred.strip().splitlines())
            # pred_single_line = pred.replace("\n", " ")
            gen_f.write(pred_single_line + "\n")

    with open(save_name_folder + "/input.txt", "w", encoding="utf-8") as inp_f:
        for prompt in prompts:
            inp_f.write(extract_clean_input(prompt) + "\n")

    results = {
        'weights': model.print_weights(verbose=False) if hasattr(model, 'print_weights') else None,
        'adapter_names': adapter_names,
        'adapter_paths': adapter_name_or_path,
        'model_name_or_path': model_name_or_path,
        'dataset': dataset,
        'template': template,
        'cutoff_len': cutoff_len,
        'temperature': temperature,
        'max_samples': max_samples,
        'max_new_tokens': max_new_tokens,
        'repetition_penalty': repetition_penalty,
        'save_name': save_name,

    }

    with open(save_name_folder + "/weights.json", "w", encoding="utf-8") as f_w:
        json.dump(results, f_w, indent=2)

    # ========================================================================
    # 7. Print sample outputs
    # ========================================================================

    print("[vllm_infer_script_adapted_for_using_hf.hf_infer()] SAMPLE OUTPUTS (First 3)")

    for i in range(min(3, len(preds))):
        print(f"\n[vllm_infer_script_adapted_for_using_hf.hf_infer()] --- Sample {i + 1} ---")
        print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Prompt: {prompts[i][:150]}...")
        print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Prediction: {preds[i][:150]}...")
        print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] Label: {labels[i][:150]}...")

    print("\n[vllm_infer_script_adapted_for_using_hf.hf_infer()] ✅ Inference complete!")


def prepare_dataset_for_generation_llamafactory_compatible(adapter_paths_str, cutoff_len, dataset, dataset_dir,
                                                           max_new_tokens, max_samples, model_name_or_path,
                                                           repetition_penalty, temperature, template, top_k, top_p):
    # ========================================================================
    # 1. Load tokenizer and template (using LlamaFactory)
    # ========================================================================
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_paths_str,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    # ========================================================================
    # 2. Load dataset (using LlamaFactory)
    # ========================================================================
    print("[vllm_infer_script_adapted_for_using_hf.hf_infer()] Loading dataset...")
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    # Prepare inputs
    inputs, prompts, labels, attention_mask = [], [], [], []
    for sample in dataset_module["train_dataset"]:
        # For now, skip multimodal (can add later if needed)
        if sample["images"]:
            continue

        inputs.append(sample["input_ids"])
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        attention_mask.append(sample["attention_mask"])
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )
    print(f"[vllm_infer_script_adapted_for_using_hf.hf_infer()] ✓ Loaded {len(inputs)} samples")
    return inputs, attention_mask, labels, model_args, prompts, template_obj, tokenizer


import json
import re

def extract_clean_input(raw_input: str) -> str:
    """
    Extract the actual user content after 'Paraphrase'
    and before the <|eot_id|> token.
    """
    match = re.search(r"Paraphrase\s*(.*?)(?=<\|eot_id\|>)", raw_input, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""



if __name__ == "__main__":
    # fire.Fire(hf_infer)
    # create a default call to hf_infer as a standalone script
    BASE_MODEL = "/scratch/common_models/Llama-3.1-8B-Instruct"
    # # ADAPTER_1 = "PTA/6/author_sft_lora/author_sft_lora_EXP_10_sent_shuffled_2025_10_08_Instruct-MT/model"
    # ADAPTER_2 = "PTA/6/rewrite_sft_lora/rewrite_sft_lora_EXP_24_paraNMT_2025_10_08_Instruct/model"
    output_dir = "PTA/8_weight_optimize_with_hf_trainer/1_harmonic_mean_based_sft/weight_optimization_custom"
    output_dir_new = f"{output_dir}_0.0_0.0_truout_2026_wikitest"
    # weights_source = [[0.0, 0.0]] * 32,  # 32 layers

    adapter_paths = [
        "PTA/10_author_models_ft/1_random_input_vs_author_text_sft_with_instruction_lora/1_random_input_vs_author_text_sft_with_instruction_lora_EXP_31_sent_shuffled_2025_12_25_Instruct-MT/model",
        "PTA/6/rewrite_sft_lora/rewrite_sft_lora_EXP_24_paraNMT_2025_10_08_Instruct/model"
    ]
    adapter_paths_str = ",".join(adapter_paths)
    adapter_names = ["author-adapter", "rewrite-adapter"]

    # Manual weights: Equal weights [0.5, 0.5] for all layers
    # weights_source = [[2.0, 1.0]] * 32  # 32 layers
    weights_source="PTA/11_weight_training_sft/exp_1-author_sft+rewrite_sft-random_init_weights_uncontraintined/optimized_weights.json"

    # hf_infer(
    #     model_name_or_path=BASE_MODEL,
    #     adapter_name_or_path=adapter_paths_str,
    #     dataset="wiki_test_for_paraNMT_model",
    #     template="llama3",
    #     cutoff_len=256,
    #     max_samples=100,
    #     save_name=f"{output_dir_new}/generated_predictions.jsonl",
    #     temperature=0.0,
    #     max_new_tokens=256,
    #     repetition_penalty=1.0,
    #     weights_source=weights_source,
    #     adapter_names=adapter_names,
    # )
    #
    # hf_infer(
    #     # Model configuration
    #     model_name_or_path="/scratch/common_models/Llama-3.1-8B-Instruct",
    #     adapter_name_or_path=[
    #         "PTA/10_author_models_ft/1_random_input_vs_author_text_sft_with_instruction_lora/1_random_input_vs_author_text_sft_with_instruction_lora_EXP_31_sent_shuffled_2025_12_25_Instruct-NH/model",
    #         "PTA/10_author_models_ft/1_random_input_vs_author_text_sft_with_instruction_lora/1_random_input_vs_author_text_sft_with_instruction_lora_EXP_31_sent_shuffled_2025_12_25_Instruct-VW/model",
    #         "PTA/6/rewrite_sft_lora/rewrite_sft_lora_EXP_24_paraNMT_2025_10_08_Instruct/model"
    #     ],
    #     adapter_names=["author-adapter", "rewrite-adapter"],
    #
    #     # Weights from training
    #     weights_source="PTA/11_weight_training_sft/exp_2-two_author_sft+rewrite_sft-random_init_weights_uncontraintined/optimized_weights.json",
    #
    #     # Test dataset (HuggingFace format)
    #     dataset="wiki_test_for_paraNMT_model",
    #     template="llama3",
    #     cutoff_len=256,
    #     # Output
    #     save_name="PTA/13_evaluation_on_testing_style_paraphrase/test_results/mark_twain_optimized_exp_2-two_author_sft+rewrite_sft-random_init_weights_uncontraintined/generated_predictions.jsonl",
    #
    #     # Generation settings
    #     max_samples=100,  # Test on first 100 samples
    #     max_new_tokens=256,
    #     temperature=0.0,
    #     repetition_penalty=1.0,
    # )
    #
    # print("\n" + "=" * 80 + "\n")
    #
    # print("\n" + "=" * 80 + "\n")


    hf_infer(
        # Model configuration
        model_name_or_path="/scratch/common_models/Llama-3.1-8B-Instruct",
        adapter_name_or_path=[
            "PTA/10_author_models_ft/2_paraphrased_to_remove_original_style_input_vs_author_text_sft_with_instruction_lora/2_paraphrased_to_remove_original_style_input_vs_author_text_sft_with_instruction_lora_EXP_32_sent_shuffled_2025_12_26_Instruct-NH/model",
             "PTA/10_author_models_ft/2_paraphrased_to_remove_original_style_input_vs_author_text_sft_with_instruction_lora/2_paraphrased_to_remove_original_style_input_vs_author_text_sft_with_instruction_lora_EXP_32_sent_shuffled_2025_12_26_Instruct-VW/model"
        ],
        adapter_names=["author-adapter", "rewrite-adapter"],

        # Weights from training
        weights_source="PTA/11_weight_training_sft/exp_4-two_author_sft_from_synthetic_parallel-random_init_weights_uncontraintined/optimized_weights.json",

        # Test dataset (HuggingFace format)
        dataset="wiki_test_for_paraNMT_model",
        template="llama3",
        cutoff_len=256,
        # Output
        save_name="PTA/13_evaluation_on_testing_style_paraphrase/test_results/mark_twain_optimized_exp_4-two_author_sft_from_synthetic_parallel-random_init_weights_uncontraintined/generated_predictions.jsonl",

        # Generation settings
        max_samples=100,  # Test on first 100 samples
        max_new_tokens=256,
        temperature=0.0,
        repetition_penalty=1.0,
    )

    print("\n" + "=" * 80 + "\n")