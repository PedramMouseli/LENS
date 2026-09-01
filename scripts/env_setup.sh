#!/usr/bin/env bash
set -e

ENV_DIR="msk_pain_env"

python3.11 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio

pip install \
    "numpy==1.26.4" \
    scipy \
    pandas \
    scikit-learn \
    h5py \
    matplotlib \
    tqdm \
    tensorboard \
    einops \
    fla \
    "networkx>=2.8"

echo "Done. Activate with:  source ${ENV_DIR}/bin/activate"