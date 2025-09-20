#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# Kairos environment setup script
# -------------------------------------------------------
ENV_NAME="kairos"
PY_VERSION="3.9"
MINIFORGE_DIR="/home/pworth/miniforge3"

# Helper: run conda in this shell
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

# 1. Create env if not exists
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env '${ENV_NAME}' already exists."
else
    echo "[1/5] Creating env ${ENV_NAME} with Python ${PY_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi

# 2. Activate env
echo "[2/5] Activating environment..."
conda activate "${ENV_NAME}"

# 3. Install base conda packages
echo "[3/5] Installing psycopg2 and tqdm..."
conda install -y psycopg2 tqdm

# 4. Install pinned pip packages
echo "[4/5] Installing pinned pip packages..."
pip install scikit-learn==1.2.0 \
            networkx==2.8.7 \
            xxhash==3.2.0 \
            graphviz==0.20.1

# 5. Install PyTorch GPU + PyG stack
echo "[5/5] Installing PyTorch 1.13.1 (CUDA 11.7) and PyTorch Geometric..."
conda install -y -c pytorch -c nvidia pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7

pip install torch_geometric==2.0.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Summary
cat <<EOF

✅ Kairos environment setup complete!

Environment:
  - Name: ${ENV_NAME}
  - Python: ${PY_VERSION}
  - Installed:
      * psycopg2, tqdm (conda)
      * scikit-learn==1.2.0, networkx==2.8.7, xxhash==3.2.0, graphviz==0.20.1 (pip)
      * PyTorch 1.13.1 + CUDA 11.7, TorchVision 0.14.1, Torchaudio 0.13.1 (conda)
      * torch_geometric==2.0.0, pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv (pip)

Next steps:
  conda activate ${ENV_NAME}
  python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"
EOF
