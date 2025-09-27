#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# Script: setup.sh
# Purpose: Install Python 3.10 env and PostgreSQL 18
#
# -----------------------------------------------------------------------------

# Variables
ENV_NAME="python310"
PY_VER="3.10"
GIT_USER_NAME="Peter Worth"
GIT_USER_EMAIL="peterworthjr@gmail.com"

# Step 1: Update apt and install essentials
echo "[*] Updating apt and installing prerequisites..."
sudo apt update
sudo apt install -y git wget bzip2 curl gnupg lsb-release software-properties-common

# Step 2: Configure git global settings
echo "[*] Configuring git global user..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# Step 3: Create Python 3.10 environment
echo "[*] Creating conda environment '$ENV_NAME' with Python $PY_VER..."
conda create -y -n "$ENV_NAME" python="$PY_VER"
