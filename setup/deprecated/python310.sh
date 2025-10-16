#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Script: setup_python310_user.sh
# Purpose: Securely install Python 3.10 + Conda environment under a non-root user
# -----------------------------------------------------------------------------

# === Configurable variables ===
ENV_NAME="python310"
PY_VER="3.10"
LINUX_USER="pworthjr"
GIT_USER_NAME="Peter Worth"
GIT_USER_EMAIL="peterworthjr@gmail.com"
CONDA_HOME="/home/${LINUX_USER}/miniconda3"
CONDA_SH="${CONDA_HOME}/etc/profile.d/conda.sh"

# -----------------------------------------------------------------------------
# Step 1: Create user (if not exists)
# -----------------------------------------------------------------------------
echo "[*] Creating user '$LINUX_USER' (if not exists)..."
if ! id "$LINUX_USER" >/dev/null 2>&1; then
    sudo adduser --disabled-password --gecos "" "$LINUX_USER"
fi

# -----------------------------------------------------------------------------
# Step 2: Install Miniconda (user-local)
# -----------------------------------------------------------------------------
echo "[*] Installing Miniconda under /home/$LINUX_USER..."
sudo -u "$LINUX_USER" bash -c "
    cd /home/$LINUX_USER
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    bash miniconda.sh -b -p $CONDA_HOME
    rm miniconda.sh
    echo 'export PATH=\"$CONDA_HOME/bin:\$PATH\"' >> /home/$LINUX_USER/.bashrc
    echo 'source \"$CONDA_SH\"' >> /home/$LINUX_USER/.bashrc
"

# -----------------------------------------------------------------------------
# Step 3: Configure Git (as user)
# -----------------------------------------------------------------------------
echo "[*] Configuring git global settings..."
sudo -u "$LINUX_USER" git config --global user.name "$GIT_USER_NAME"
sudo -u "$LINUX_USER" git config --global user.email "$GIT_USER_EMAIL"

# -----------------------------------------------------------------------------
# Step 4: Create Conda environment and install packages
# -----------------------------------------------------------------------------
echo "[*] Creating conda env '$ENV_NAME' with Python $PY_VER..."
sudo -u "$LINUX_USER" bash -c "
    source \"$CONDA_SH\"
    conda create -y -n \"$ENV_NAME\" python=\"$PY_VER\"
    conda activate \"$ENV_NAME\"

    echo '[*] Installing conda packages...'
    conda update -y --all
    conda install -y psycopg2 tqdm pytz scikit-learn

    echo '[*] Installing pip packages...'
    pip install networkx xxhash graphviz gdown torch_geometric pytz psycopg2-binary xxhash python-louvain

    echo '[*] Installing PyTorch with CUDA 12.8...'
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"

# -----------------------------------------------------------------------------
# Step 5: Install system-level Graphviz
# -----------------------------------------------------------------------------
echo "[*] Installing system-level Graphviz..."
sudo apt update && sudo apt install -y graphviz

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------
cat <<EOF

✅ Kairos Python 3.10 environment installed under user '$LINUX_USER'

Git user: $GIT_USER_NAME <$GIT_USER_EMAIL>
Conda environment: $ENV_NAME (Python $PY_VER)
Miniconda path: $CONDA_HOME

To activate this environment as $LINUX_USER:
    sudo -u $LINUX_USER -i
    source "$CONDA_SH"
    conda activate $ENV_NAME

EOF
