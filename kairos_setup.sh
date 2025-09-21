#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# Kairos environment setup script
# -------------------------------------------------------
# 1. Install base conda packages
echo "[1/3] Installing psycopg2 and tqdm..."
conda install -y psycopg2 tqdm

# 2. Install pinned pip packages
echo "[2/3] Installing pinned pip packages..."
pip install scikit-learn networkx xxhash graphviz

# 3. Install PyTorch GPU + PyG stack
echo "[3/3] Installing PyTorch 1.13.1 (CUDA 11.7) and PyTorch Geometric..."
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda

pip install torch_geometric
# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

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
