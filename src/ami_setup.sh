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


# Note: you need about 30GB volume
# Use a g5.[2]xlarge (~$1 per hour)

# <<< AMI START: Ubuntu 24.04 Server >>>

sudo apt update
sudo apt -y upgrade

# following: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202
# nvidia drivers
sudo apt autoremove nvidia* --purge
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt install -y nvidia-driver-550  # note: exact number (550) can change (check recommended version from `ubuntu-drivers devices`)
# Note: 550.xx is compatible with CUDA 12.4
sudo reboot
nvidia-smi

sudo snap install aws-cli --classic

# python + dependencies
sudo apt install -y python3 python3-pip
python3 -m pip config set global.break-system-packages true  # required for system-wide installs (pip 23.0+)
pip install torch torchvision transformers
pip install accelerate hf_transfer
pip install gpustat

# <<< AMI END: CUDA pytorch GPU training + inference >>>
