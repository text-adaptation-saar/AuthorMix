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

"""LLaMA-Factory dataset preparation used by GRPO training."""

from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


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
