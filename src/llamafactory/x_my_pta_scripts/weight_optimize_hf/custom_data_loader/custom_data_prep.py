import re
from typing import List, Dict, Optional, Union
from datasets import Dataset

from llamafactory.x_my_pta_scripts.weight_optimize_hf.inference.vllm_infer_script_adapted_for_using_hf import \
    prepare_dataset_for_generation_llamafactory_compatible

SYSTEM_BLOCK_PATTERN = re.compile(
    r"<\|start_header_id\|\>system<\|end_header_id\|\>.*?<\|eot_id\|\>",
    flags=re.DOTALL
)

def remove_system_block(text):
    """Remove system message block inserted by chat template."""
    return re.sub(SYSTEM_BLOCK_PATTERN, "", text).strip()

def prepare_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize dataset for training.
    Converts instruction/input/output format to messages, applies chat template, then tokenizes.
    """

    def convert_to_messages_and_tokenize(examples):
        """Convert to messages format, apply chat template, then tokenize"""

        if 'instruction' in examples:
            # Instruction/input/output format - convert to messages
            num_examples = len(examples['instruction'])
            messages_all=[]

            print(f"\n{'='*80}")
            print(f"DATASET PROCESSING - Processing {num_examples} examples")
            print(f"{'='*80}")

            for i in range(num_examples):
                instruction = examples['instruction'][i]
                output = examples['output'][i]
                input_text = examples['input'][i]

                # Show first example in detail
                if i == 0:
                    print(f"\n--- Example 1: Raw Data ---")
                    print(f"Input: {instruction[:100]} {input_text[:100]}..." if len(input_text) > 100 else f"Input: {input_text}")
                    print(f"Output: {output[:100]}..." if len(output) > 100 else f"Output: {output}")

                # Create messages
                messages = [
                    {"role": "user", "content": f"\n\n{instruction}\n{input_text}"},
                    {"role": "assistant", "content": output}
                ]
                messages_all.append(messages)

                # Show first example in messages format
                if i == 0:
                    print(f"\n--- Example 1: Messages Format ---")
                    for msg in messages:
                        content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                        print(f"  {msg['role']}: {content_preview}")

            # Apply chat template (without tokenization)
            print(f"\n--- Applying Chat Template ---")
            formated_text = tokenizer.apply_chat_template(messages_all, tokenize=False)

            # --- Remove unwanted system block ---
            if isinstance(formated_text, list):
                formated_text = [remove_system_block(t) for t in formated_text]
            else:
                formated_text = remove_system_block(formated_text)


            # Show first example after template
            if isinstance(formated_text, list):
                print(f"\n--- Example 1: After Chat Template ---")
                print(formated_text[0][:500])
                print("..." if len(formated_text[0]) > 500 else "")
            else:
                print(f"\n--- After Chat Template (single string) ---")
                print(formated_text[:500])
                print("..." if len(formated_text) > 500 else "")

        else:
            raise ValueError(
                "Dataset must have 'messages', 'instruction'+'output', or 'text' field"
            )

        # Now tokenize all texts
        print(f"\n--- Tokenizing ---")
        tokenized = tokenizer(
            formated_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True
        )

        # Show tokenization results for first example
        if isinstance(tokenized['input_ids'], list) and len(tokenized['input_ids']) > 0:
            print(f"\n--- Example 1: Tokenization Results ---")
            print(f"Input IDs length: {len(tokenized['input_ids'][0])}")
            print(f"First 20 tokens: {tokenized['input_ids'][0][:20]}")
            print(f"Decoded (first 200 chars): {tokenizer.decode(tokenized['input_ids'][0][:50])}...")
            print(f"Attention mask length: {len(tokenized['attention_mask'][0])}")

        print(f"\n{'='*80}")
        print(f"DATASET PROCESSING COMPLETE")
        print(f"{'='*80}\n")

        return tokenized

    # Process the dataset
    tokenized_dataset = dataset.map(
        convert_to_messages_and_tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Converting to messages and tokenizing",
        load_from_cache_file=False
    )

    return tokenized_dataset


def prepare_grpo_dataset(dataset, tokenizer, max_length: int = 256):
    """
    Prepare dataset for GRPO.
    GRPO expects: {"query": prompt_text}
    """

    def format_examples(examples):
        # queries = []

        # for i in range(len(examples['instruction'])):
        instruction = examples['instruction']
        input_text = examples['input']

        # Create prompt
        messages = [
            {"role": "user", "content": f"\n\n{instruction}\n{input_text}"},
        ]

        query = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Keep as text for GRPO
            add_generation_prompt=True
        )

        # --- Remove unwanted system block ---
        if isinstance(query, list):
            formated_text = [remove_system_block(t) for t in query]
        else:
            formated_text = remove_system_block(query)


        return {
            "prompt": formated_text,
        }

        # # Print first example only (outside the loop)
        # if len(queries) > 0:
        #     print(f"\n--- Sample Query (first example) ---")
        #     print(f"Raw text (first 300 chars):\n{queries[0][:300]}...")
        #
        #     # Show tokenized version
        #     tokenized = tokenizer(queries[0], add_special_tokens=False)
        #     print(f"\nFirst 20 token IDs: {tokenized['input_ids'][:20]}")
        #     print(f"Decoded (first 20 tokens): {tokenizer.decode(tokenized['input_ids'][:20])}")
        #     print(f"Total tokens in this query: {len(tokenized['input_ids'])}")
        #     print("-" * 60 + "\n")

        # return {"query": queries}
        # return queries

    dataset = dataset.map(
        format_examples,
        # batched=True,
        # remove_columns=dataset.column_names
    )

    return dataset

def prepare_grpo_dataset_llama_facotry_compatible(model_name_or_path: str,
        adapter_name_or_path: Union[str, List[str]] = None,
        dataset: str = "alpaca_en_demo",
        dataset_dir: str = "data",
        template: str = "default",
        cutoff_len: int = 2048,
        max_samples: int = None,
        temperature: float = 0.0,
        top_p: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.0,
):
    if adapter_name_or_path is None:
        adapter_paths_str = None
    else:
        adapter_paths_str = ",".join(adapter_name_or_path) if isinstance(adapter_name_or_path, list) else str(adapter_name_or_path)

    inputs, attention_mask, labels, model_args, prompts, template_obj, tokenizer = \
        prepare_dataset_for_generation_llamafactory_compatible(adapter_paths_str, cutoff_len, dataset, dataset_dir,
                                                           max_new_tokens, max_samples, model_name_or_path,
                                                           repetition_penalty, temperature, template, top_k, top_p)

    # GRPO expects this:
    grpo_dataset = Dataset.from_dict({
        "prompt": prompts  # List of strings (already formatted with template)
    })

    return grpo_dataset, tokenizer

    # NOT this:
    # grpo_dataset = Dataset.from_dict({
    #     "prompt": prompts,
    #     "input_ids": inputs  # ❌ Not needed
    # })
