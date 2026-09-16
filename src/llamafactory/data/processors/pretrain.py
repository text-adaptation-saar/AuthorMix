# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...hparams import DataArguments


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    print(f"data_args.template: {data_args.template}, data_args.packing: {data_args.packing}")
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    # print(f"Input examples: {examples}")
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
    print(f"Text examples: {text_examples}")
    print(f"Length of Input text examples (before pre-process): {len(text_examples)}")

    if not data_args.packing:
        if getattr(tokenizer, "add_bos_token", False):
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        result = tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)
    else:
        # print("hit here")
        # print("Text Examples:")
        # print(text_examples)

        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        # print("Tokenized Examples:")
        # print(tokenized_examples)

        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        # print("Concatenated Examples:")
        # print(concatenated_examples)

        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        print(f"Total Length: {total_length}")

        block_size = data_args.cutoff_len
        print(f"Block Size: {block_size}")

        total_length = (total_length // block_size) * block_size
        print(f"Adjusted Total Length: {total_length}")

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # print("Result:")
        # print(result)

        if getattr(tokenizer, "add_bos_token", False):
            print("Adding BOS Token...")
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id
            # print("Result after adding BOS Token:")
            # print(result)
    # print("Preprocessed output:")
    # print(result)
    print(f"Length of preprocessed output: {len(result['input_ids'])}")

    return result


def print_pretrain_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
