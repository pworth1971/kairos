#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# Script: setup_env.sh
# Purpose: Install Python 3.10 environment (Conda), PostgreSQL, and libraries
# Compatible with macOS (Homebrew) and Ubuntu (APT)
# -----------------------------------------------------------------------------

# Variables
ENV_NAME="python310"
PY_VER="3.10"
GIT_USER_NAME="Peter Worth"
GIT_USER_EMAIL="peterworthjr@gmail.com"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

echo "[*] Detected OS: $OS"

# -----------------------------------------------------------------------------
# Step 1: Install prerequisites
# -----------------------------------------------------------------------------
if [[ "$OS" == "darwin" ]]; then
    echo "[*] macOS detected — using Homebrew..."
    # Install Homebrew if not installed
    if ! command -v brew &> /dev/null; then
        echo "[*] Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    echo "[*] Installing system dependencies..."
    brew install git wget curl graphviz postgresql@16

else
    echo "[*] Ubuntu detected — using apt..."
    sudo apt update -y
    sudo apt install -y git wget bzip2 curl gnupg lsb-release software-properties-common graphviz
fi

# -----------------------------------------------------------------------------
# Step 2: Configure Git
# -----------------------------------------------------------------------------
echo "[*] Configuring Git..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# -----------------------------------------------------------------------------
# Step 3: Install or Update Conda
# -----------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[*] Installing Miniconda..."
    wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[*] Conda already installed."
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# -----------------------------------------------------------------------------
# Step 4: Create and activate environment
# -----------------------------------------------------------------------------
echo "[*] Creating conda environment '$ENV_NAME'..."
conda create -y -n "$ENV_NAME" python="$PY_VER"
conda activate "$ENV_NAME"

# -----------------------------------------------------------------------------
# Step 5: Install Python packages
# -----------------------------------------------------------------------------
echo "[*] Installing Python packages..."
conda update -y --all
conda install -y psycopg2 tqdm pytz scikit-learn
pip install networkx xxhash graphviz gdown torch_geometric pytz psycopg2-binary xxhash python-louvain flask

# -----------------------------------------------------------------------------
# Step 6: Install PyTorch
# -----------------------------------------------------------------------------
if [[ "$OS" == "darwin" ]]; then
    echo "[*] Installing PyTorch (CPU or MPS for Apple Silicon)..."
    pip install torch torchvision torchaudio
else
    echo "[*] Installing PyTorch CUDA 12.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

# -----------------------------------------------------------------------------
# Step 7: Setup PostgreSQL (if installed)
# -----------------------------------------------------------------------------
if command -v psql &> /dev/null; then
    echo "[*] PostgreSQL detected: $(psql --version)"
else
    echo "[!] PostgreSQL not detected — skipping DB setup."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat <<EOF

✅ Kairos environment setup complete!

------------------------------------------------------------
Git:        $(git config --global user.name) <$(git config --global user.email)>
Python:     $(python --version)
Conda Env:  $ENV_NAME
Location:   $(which python)
------------------------------------------------------------

To activate your environment:
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate $ENV_NAME
------------------------------------------------------------
EOF
