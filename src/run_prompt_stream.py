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
"""
Example usage:
$ python3 -u src/run_prompt_stream.py --prompt "Do any two people in London have the same number of hairs on their heads?"
"""
import argparse
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--prompt", type=str, required=True)
    arg_parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    arg_parser.add_argument("--max-tokens", type=int, default=1000)
    args = arg_parser.parse_args()

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Run generation in a separate thread
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    thread = threading.Thread(target=model.generate, kwargs={
        "input_ids": input_ids,
        "max_new_tokens": args.max_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    })
    thread.start()

    # Print tokens as they arrive
    for token in streamer:
        print(token, end="", flush=True)

    thread.join()
    print("\n\nEnd of generation")
