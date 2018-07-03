#!/bin/bash

# install CUDA Toolkit v9.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))
cd $HOME
CUDA_REPO_PKG="cuda-repo-ubuntu1604_9.0.176-1_amd64.deb"
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-9-0
sudo nvidia-smi -pm 1

# install cuDNN v7.1.4
# The tar file has to be transferred from local machine
tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# set environment variables
echo '' >> .bashrc
echo '# Set environment variables for CUDA and cuDNN' >> .bashrc
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> .bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> .bashrc
echo '' >> .bashrc
echo "Reboot the shell by typing 'source ~/.bashrc'"
