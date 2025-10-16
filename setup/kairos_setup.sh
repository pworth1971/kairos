#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Kairos Unified Setup Script (ROOT-ONLY)
# - Installs Miniconda to /root/miniconda3
# - Creates conda env python310 (Python 3.10) and installs deps
# - Installs PostgreSQL 16 and configures tc_cadet_dataset_db
# - No extra users, no logs, minimal stdout
# ------------------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

# === Config ===
ENV_NAME="python310"
PY_VER="3.10"
GIT_USER_NAME="Peter Worth"
GIT_USER_EMAIL="peterworthjr@gmail.com"

# Miniconda install location (root)
CONDA_HOME="/root/miniconda3"
CONDA_BIN="${CONDA_HOME}/bin/conda"
CONDA_SH="${CONDA_HOME}/etc/profile.d/conda.sh"
DOWNLOAD_PATH="/root/miniconda.sh"

# ------------------------------------------------------------------------------
# Step 0: Locale & base packages
# ------------------------------------------------------------------------------
echo "[*] Locale & base packages..."
apt-get update -qq
apt-get install -y -qq locales curl wget gnupg lsb-release ca-certificates graphviz
sed -i 's/^[#[:space:]]*en_US\.UTF-8[[:space:]]\+UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
locale-gen en_US.UTF-8 >/dev/null
update-locale LANG=en_US.UTF-8 LC_CTYPE=en_US.UTF-8
echo "[+] Locale configured."

# ------------------------------------------------------------------------------
# Step 1: Miniconda (root) + env
# ------------------------------------------------------------------------------
echo "[*] Installing Miniconda (root) and setting up conda env..."

# Clean up incomplete installs
if [ -d "$CONDA_HOME" ] && [ ! -f "$CONDA_SH" ]; then
  echo "[!] Incomplete Miniconda detected at $CONDA_HOME — removing..."
  rm -rf "$CONDA_HOME"
fi

# Install Miniconda once
if [ ! -d "$CONDA_HOME" ]; then
  echo "-> Installing Miniconda to $CONDA_HOME..."
  wget -q -O "$DOWNLOAD_PATH" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$DOWNLOAD_PATH" -b -p "$CONDA_HOME" >/dev/null
  rm -f "$DOWNLOAD_PATH"
  echo "[+] Miniconda installed."
else
  echo "[=] Miniconda already present."
fi

# Validate
if [ ! -f "$CONDA_SH" ]; then
  echo "❌ conda.sh missing — installation failed."
  exit 1
fi

# Ensure root's bashrc initializes conda
if ! grep -q "conda.sh" /root/.bashrc; then
  echo "export PATH=\"$CONDA_HOME/bin:\$PATH\"" >> /root/.bashrc
  echo "source \"$CONDA_SH\"" >> /root/.bashrc
fi

# Accept Conda ToS (non-interactive; required on fresh installs)
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# Create env if missing
if ! "$CONDA_BIN" info --envs | grep -q "^${ENV_NAME}\s"; then
  echo "-> Creating conda env: $ENV_NAME (Python $PY_VER)"
  "$CONDA_BIN" create -y -n "$ENV_NAME" python="$PY_VER" >/dev/null
else
  echo "[=] Conda env '$ENV_NAME' already exists."
fi

# Install Python deps (quiet)
echo "-> Installing Python dependencies..."
"$CONDA_BIN" run -n "$ENV_NAME" conda install -y -q psycopg2 tqdm pytz scikit-learn >/dev/null
"$CONDA_BIN" run -n "$ENV_NAME" pip install -q networkx xxhash graphviz gdown torch_geometric pytz psycopg2-binary xxhash python-louvain
# PyTorch CUDA 12.8 wheels (works on CPU too; will just use CPU if no GPU)
"$CONDA_BIN" run -n "$ENV_NAME" pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[+] Conda env '$ENV_NAME' ready."

# ------------------------------------------------------------------------------
# Step 2: Git (global for root)
# ------------------------------------------------------------------------------
echo "[*] Configuring Git (global for root)..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
echo "[+] Git configured."



# ------------------------------------------------------------------------------
# Step 4: Verification
# ------------------------------------------------------------------------------
echo "[*] Verifying..."

# Quick conda check (non-interactive)
"$CONDA_BIN" info --envs | grep -q "^${ENV_NAME}\s" || { echo "❌ Conda env missing"; exit 1; }

cat <<EOF

------------------------------------------------------------
✅ Kairos Dev Environment Installed (root)

Conda:
  Home:         $CONDA_HOME
  Env:          $ENV_NAME (Python $PY_VER)

To use conda (root):
  source "$CONDA_SH"
  conda activate $ENV_NAME
------------------------------------------------------------
EOF
