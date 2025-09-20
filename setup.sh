#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Ubuntu setup: non-root dev user + Conda (Python 3.12) + PyTorch + Git identity
# ----------------------------------------

REQUIRED_OS="ubuntu"
DEV_USER="pworth"
DEV_GROUP="$DEV_USER"
DEV_SHELL="/bin/bash"
MINIFORGE_DIR="/home/${DEV_USER}/miniforge3"
ENV_NAME="py312"
PY_VERSION="3.12"

# Git identity (override with env vars if you wish)
GIT_USER_NAME="${GIT_USER_NAME:-Peter Worth}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-peterworthjr@gmail.com}"

# ---- helpers ---------------------------------------------------------------

is_ubuntu() {
  [[ "${OSTYPE:-}" == "linux-gnu"* ]] || return 1
  if command -v lsb_release >/dev/null 2>&1; then
    [[ "$(lsb_release -is 2>/dev/null | tr '[:upper:]' '[:lower:]')" == "ubuntu" ]]
  else
    [[ -f /etc/os-release ]] && grep -qi '^id=ubuntu' /etc/os-release
  fi
}

need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "This script must be run as root (use: sudo bash setup.sh)." >&2
    exit 1
  fi
}

user_exists() {
  id -u "$1" >/dev/null 2>&1
}

su_user() {
  # Run a command as the DEV_USER with login shell, preserving HOME
  sudo -Hiu "$DEV_USER" bash -lc "$*"
}

arch_triplet() {
  local machine
  machine="$(uname -m)"
  case "$machine" in
    x86_64)   echo "x86_64" ;;
    aarch64)  echo "aarch64" ;;
    arm64)    echo "aarch64" ;;
    *)        echo "x86_64" ;;
  esac
}

has_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1
}

# ---- preflight -------------------------------------------------------------

need_root

if ! is_ubuntu; then
  echo "This script targets Ubuntu. Detected non-Ubuntu Linux or unsupported OS." >&2
  exit 1
fi

echo "[1/7] Updating apt and installing base dev tools..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  sudo ca-certificates curl wget git build-essential pkg-config \
  libssl-dev openssl bzip2

echo "[2/7] Creating non-root dev user '${DEV_USER}' (idempotent)..."
if ! user_exists "$DEV_USER"; then
  adduser --disabled-password --gecos "" "$DEV_USER"
  usermod -aG sudo "$DEV_USER"
  chsh -s "$DEV_SHELL" "$DEV_USER"
else
  echo "User '${DEV_USER}' already exists. Ensuring sudo and shell..."
  usermod -aG sudo "$DEV_USER" || true
  chsh -s "$DEV_SHELL" "$DEV_USER" || true
fi
chown -R "${DEV_USER}:${DEV_GROUP}" "/home/${DEV_USER}"

echo "[3/7] Installing Miniforge (Conda) for '${DEV_USER}'..."
ARCH="$(arch_triplet)"
INSTALLER="Miniforge3-Linux-${ARCH}.sh"
INSTALL_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"

if [[ -d "$MINIFORGE_DIR" ]]; then
  echo "Miniforge already present at ${MINIFORGE_DIR} — skipping install."
else
  su_user "cd ~ && wget -q ${INSTALL_URL} -O ${INSTALLER}"
  su_user "bash ${INSTALLER} -b -p ${MINIFORGE_DIR}"
  su_user "rm -f ${INSTALLER}"
  su_user "${MINIFORGE_DIR}/bin/conda init bash"
fi

PROFILE_FILE="/home/${DEV_USER}/.bashrc"
if ! su_user "grep -q 'conda initialize' '${PROFILE_FILE}'"; then
  su_user "echo '# conda initialized by setup.sh (if not already)' >> '${PROFILE_FILE}'"
  su_user "${MINIFORGE_DIR}/bin/conda init bash"
fi

echo "[4/7] Creating Python ${PY_VERSION} environment '${ENV_NAME}' (idempotent)..."
if su_user "${MINIFORGE_DIR}/bin/conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}'"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  su_user "${MINIFORGE_DIR}/bin/conda create -y -n '${ENV_NAME}' python='${PY_VERSION}'"
fi

echo "[5/7] Configuring Conda channels and installing packages..."
su_user "${MINIFORGE_DIR}/bin/conda config --add channels conda-forge || true"
su_user "${MINIFORGE_DIR}/bin/conda config --add channels pytorch || true"
su_user "${MINIFORGE_DIR}/bin/conda config --set channel_priority strict || true"

if has_nvidia; then
  echo "NVIDIA GPU detected — installing CUDA-enabled PyTorch..."
  su_user "source '${MINIFORGE_DIR}/etc/profile.d/conda.sh' && \
           conda activate '${ENV_NAME}' && \
           conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 && \
           conda install -y torchtext transformers safetensors=0.4.5"
else
  echo "No NVIDIA GPU detected — installing CPU-only PyTorch..."
  su_user "source '${MINIFORGE_DIR}/etc/profile.d/conda.sh' && \
           conda activate '${ENV_NAME}' && \
           conda install -y -c pytorch pytorch torchvision torchaudio cpuonly && \
           conda install -y torchtext transformers safetensors=0.4.5"
fi

echo "[6/7] Setting Git identity for ${DEV_USER} (idempotent)..."
# Only set if not already configured
if ! su_user "git config --global user.name >/dev/null 2>&1 && git config --global user.email >/dev/null 2>&1"; then
  su_user "git config --global user.name \"${GIT_USER_NAME}\""
  su_user "git config --global user.email \"${GIT_USER_EMAIL}\""
else
  echo "Global Git user.name and user.email already set for ${DEV_USER}."
fi

echo "[7/7] Adding convenience aliases to ${DEV_USER}'s .bashrc..."
ALIAS_BLOCK=$(cat <<'EOF'
# ---- Python dev shortcuts (added by setup.sh) ----
alias ca='conda activate'
alias cde='conda deactivate'
alias workon='conda activate py312'
EOF
)
if ! su_user "grep -q 'Python dev shortcuts (added by setup.sh)' '${PROFILE_FILE}'"; then
  su_user "printf '%s\n' \"${ALIAS_BLOCK}\" >> '${PROFILE_FILE}'"
fi

cat <<EOF

✅ All set!

User:
  - Created/updated user: ${DEV_USER} (member of 'sudo')

Conda:
  - Installed: ${MINIFORGE_DIR}
  - Env: ${ENV_NAME} (Python ${PY_VERSION})

PyTorch:
  - Installed for $(has_nvidia && echo 'CUDA (12.1)' || echo 'CPU-only')

Git (global for ${DEV_USER}):
  - user.name = ${GIT_USER_NAME}
  - user.email = ${GIT_USER_EMAIL}

Next steps:
  sudo -iu ${DEV_USER}
  conda activate ${ENV_NAME}
  git config --list --show-origin | egrep 'user.name|user.email'
  python -V
  python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"

Override Git identity on install by exporting:
  GIT_USER_NAME="Your Name" GIT_USER_EMAIL="you@domain.tld" sudo -E bash setup.sh

EOF
