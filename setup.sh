#!/bin/bashsh
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt-get install libglm-dev -y

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge colmap -y
conda install -c conda-forge imagemagick -y
conda install matplotlib -y
conda install pycuda -y
conda install pytorch3d -c pytorch3d -y

yes | pip install opt_einsum
yes | pip install einops
yes | pip install scikit-image
yes | pip install opencv-python
yes | pip install open3d
yes | pip install numba
yes | pip install plyfile
yes | pip install roma
yes | pip install wandb
yes | pip install kornia
yes | pip install lpips
yes | pip install jaxtyping
yes | pip install einops
yes | pip install torchmetrics
yes | pip install kmeans1d
yes | pip install huggingface-hub[torch]>=0.22
yes | pip install diffusers transformers accelerate scipy safetensors

yes | pip install submodules/diff-gaussian-rasterization-confidence
yes | pip install submodules/simple-knn
