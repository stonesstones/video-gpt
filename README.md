# VideoGPT: Video Generation using VQ-VAE and Transformers

[[Paper]](https://arxiv.org/abs/2104.10157)[[Website]](https://wilson1yan.github.io/videogpt/index.html)[[Colab]](https://colab.research.google.com/github/wilson1yan/VideoGPT/blob/master/notebooks/Using_VideoGPT.ipynb)

VideoGPT uses VQ-VAE that learns downsampled discrete latent representations of a raw video by employing 3D convolutions and axial self-attention. A simple GPT-like architecture is then used to autoregressively model the discrete latents using spatio-temporal position encodings.

## Installation

```bash
# Install dependencies
uv sync
uv pip install pytorch-lightning==2.4.0
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install nuscenes-devkit

# Generate dataset paths if needed
uv run get_video_path.py
```

## Training

```bash
qsub train.sh
```

## Usage