#!/bin/bash
echo "Checking for CUDA and installing."

# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 17.04 installer works with 17.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda-9-0 -y
fi

# Enable persistence mode
sudo nvidia-smi -pm 1
