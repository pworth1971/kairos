#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# Script: setup.sh
# Purpose: Install Java 1.8, Maven, Python 3.9 env, and PostgreSQL 18
# -----------------------------------------------------------------------------

# Variables
ENV_NAME="python39"
PY_VER="3.9"
GIT_USER_NAME="Peter Worth"
GIT_USER_EMAIL="peterworthjr@gmail.com"

# Step 1: Update apt and install essentials
echo "[*] Updating apt and installing prerequisites..."
sudo apt update
sudo apt install -y git wget bzip2 curl gnupg lsb-release software-properties-common

# Step 2: Install Java 1.8 and Maven
echo "[*] Installing Java 1.8 (OpenJDK 8) and Maven..."
sudo apt install -y openjdk-8-jdk maven

# Verify installation
echo "[*] Java version:"
java -version || true
echo "[*] Maven version:"
mvn -version || true

# Step 3: Configure git global settings
echo "[*] Configuring git global user..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# Step 4: Create Python 3.9 environment
echo "[*] Creating conda environment '$ENV_NAME' with Python $PY_VER..."
conda create -y -n "$ENV_NAME" python="$PY_VER"

# Step 5: Source conda.sh so conda activate works in this shell
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "[*] Sourcing conda.sh..."
    source /root/miniconda3/etc/profile.d/conda.sh
else
    echo "[!] Could not find /root/miniconda3/etc/profile.d/conda.sh"
    echo "    Make sure Miniconda/Conda is installed in /root/miniconda3"
fi

# Step 6: Activate the new environment
echo "[*] Activating conda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Step 7: Install base conda packages
echo "[*] Updating conda packages and installing psycopg2, tqdm, pytz..."
conda update --all -y
conda install -y psycopg2 tqdm pytz

# Step 8: Install pinned pip packages
echo "[*] Installing pinned pip packages..."
pip install scikit-learn networkx xxhash graphviz gdown

# Step 9: Install PyTorch GPU + PyG stack
echo "[*] Installing PyTorch (CUDA) and PyTorch Geometric..."
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda

pip install torch_geometric
# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Summary
cat <<EOF

✅ Kairos environment setup complete!

------------------------------------------------------------
✅ Git configured: $(git config --global user.name) <$(git config --global user.email)>
✅ Conda environment '$ENV_NAME' (Python $PY_VER) created and activated
✅ Java 1.8 (OpenJDK 8) and Maven installed
------------------------------------------------------------

To activate the conda environment again later, run:
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_NAME

EOF
