#!/bin/bash
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

# NOTE: it is recommended to use the AMI described in ami_setup.sh as the base image before running this script
sudo apt install -y tree

# Fast model download
# TODO: ensure you configure the model you want
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type model $MODEL_NAME

# Check the download worked
tree -h $HOME/.cache/huggingface/hub/
