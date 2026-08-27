conda create -n foundation_stereo python=3.11 pip
conda activate foundation_stereo
pip install torch==2.8 torchvision==0.23 --index-url https://download.pytorch.org/whl/cu129

conda install -c nvidia cuda-nvcc=12.9 -y
pip install scikit-image omegaconf opencv-contrib-python imgaug ninja timm albumentations jupyterlab scipy joblib scikit-learn ruamel.yaml trimesh pyyaml imageio open3d transformations einops gdown nodejs
pip install flash-attn --no-build-isolation
# pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu129
pip install xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu129