#  Copyright (C) 2025 lukerm of www.zl-labs.tech
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
"""
Example usage:
$ python3 -u src/run_prompt.py --prompt "Give me a knock-knock joke" --n-reps 25
"""
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--prompt", type=str, required=True)
    arg_parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    arg_parser.add_argument("--n-reps", type=int, default=1)
    args = arg_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto")#, low_cpu_mem_usage=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generations = []
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    for n in range(args.n_reps):
        print(f'{n} / {args.n_reps} reps')
        stream = model.generate(
            input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            max_length=2000, do_sample=True, temperature=0.6, top_k=50, return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

        gen = ""
        for token_id in stream.sequences[0][len(input_ids[0]):]:
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            print(token, end="", flush=True)  # Print without newlines, flush the buffer
            gen += token
        print("\n")
        generations.append(gen)
