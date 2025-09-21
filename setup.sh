#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Ubuntu setup for Kairos:
# - Create user
# - Use existing Conda if available for that user; otherwise install Miniforge
# - Create env py310, install PyTorch/PyG + deps
# - Grant user access to /home/kairos
# - Switch into the user with env active
# ----------------------------------------

DEV_USER="pworth"
DEV_GROUP="$DEV_USER"
DEV_SHELL="/bin/bash"
ENV_NAME="py310"
PY_VERSION="3.10"
MINIFORGE_DIR="/home/${DEV_USER}/miniforge3"   # only used if we need to install conda
KAIROS_DIR="/home/kairos"

GIT_USER_NAME="${GIT_USER_NAME:-Peter Worth}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-peterworthjr@gmail.com}"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
}

user_exists() { id -u "$1" >/dev/null 2>&1; }

arch_triplet() {
  case "$(uname -m)" in
    x86_64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) echo "x86_64" ;;
  esac
}

has_nvidia() { command -v nvidia-smi >/dev/null 2>&1; }

RUNUSER() { runuser -l "$DEV_USER" -c "$*"; }

require_root

# --- Fix sudo hostname warning (ensure hostname resolvable) ---
HOST="$(hostname)"
if ! grep -qE "(^|[[:space:]])${HOST}([[:space:]]|$)" /etc/hosts; then
  echo "127.0.1.1 ${HOST}" >> /etc/hosts
fi

echo "[1/10] apt update + base tools + locales..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  sudo ca-certificates curl wget git build-essential pkg-config \
  libssl-dev openssl bzip2 unzip htop \
  graphviz libgraphviz-dev \
  postgresql postgresql-contrib libpq-dev \
  locales

# Locale: enable en_US.UTF-8 to silence perl/locale warnings
sed -i 's/^# *en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen
locale-gen >/dev/null
update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "[2/10] Create dev user '${DEV_USER}' if needed..."
if ! user_exists "$DEV_USER"; then
  adduser --disabled-password --gecos "" "$DEV_USER"
  usermod -aG sudo "$DEV_USER"
  chsh -s "$DEV_SHELL" "$DEV_USER"
fi
mkdir -p "/home/${DEV_USER}"
chown -R "${DEV_USER}:${DEV_GROUP}" "/home/${DEV_USER}"

echo "[3/10] Detect Conda for ${DEV_USER} (install Miniforge only if missing)..."
if RUNUSER "command -v conda >/dev/null 2>&1"; then
  CONDA_BASE="$(RUNUSER 'conda info --base 2>/dev/null' || true)"
else
  CONDA_BASE=""
fi

if [[ -z "${CONDA_BASE}" || ! -d "${CONDA_BASE}" ]]; then
  echo "No existing Conda found for ${DEV_USER} — installing Miniforge..."
  ARCH="$(arch_triplet)"
  INSTALLER="Miniforge3-Linux-${ARCH}.sh"
  INSTALL_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"
  RUNUSER "cd ~ && wget -q '${INSTALL_URL}' -O '${INSTALLER}' && bash '${INSTALLER}' -b -p '${MINIFORGE_DIR}' && rm -f '${INSTALLER}' && '${MINIFORGE_DIR}/bin/conda' init bash"
  CONDA_BASE="${MINIFORGE_DIR}"
else
  echo "Found Conda at: ${CONDA_BASE}"
  RUNUSER "'${CONDA_BASE}/bin/conda' init bash || true"
fi

echo "[4/10] Create Python ${PY_VERSION} env '${ENV_NAME}' (if missing)..."
RUNUSER "source '${CONDA_BASE}/etc/profile.d/conda.sh' && (conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}' || conda create -y -n '${ENV_NAME}' python='${PY_VERSION}')"

echo "[5/10] Configure Conda channels..."
RUNUSER "conda config --add channels conda-forge || true"
RUNUSER "conda config --add channels pytorch || true"
RUNUSER "conda config --add channels pyg || true"
RUNUSER "conda config --set channel_priority strict || true"

echo "[6/10] Install PyTorch (CUDA if available)..."
if has_nvidia; then
  RUNUSER "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1"
else
  RUNUSER "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && conda install -y -c pytorch pytorch torchvision torchaudio cpuonly"
fi

echo "[7/10] Install PyG + common libs..."
RUNUSER "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && conda install -y -c pyg pyg"
RUNUSER "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && conda install -y -c conda-forge numpy pandas scipy scikit-learn networkx tqdm matplotlib seaborn jupyterlab ipykernel sqlalchemy psycopg2-binary pydot graphviz"

echo "[8/10] Git identity..."
RUNUSER "git config --global user.name '${GIT_USER_NAME}'"
RUNUSER "git config --global user.email '${GIT_USER_EMAIL}'"

echo "[9/10] Granting ${DEV_USER} access to ${KAIROS_DIR}..."
mkdir -p "${KAIROS_DIR}"
chown -R "${DEV_USER}:${DEV_GROUP}" "${KAIROS_DIR}"

echo "[10/10] Switch into '${DEV_USER}' with env '${ENV_NAME}' active..."
exec runuser -l "$DEV_USER" -c "bash -lc 'source \"${CONDA_BASE}/etc/profile.d/conda.sh\" && conda activate \"${ENV_NAME}\" && echo && echo \"✅ Ready as ${DEV_USER} with Conda env ${ENV_NAME} active.\" && echo \"   python: \$(python -V)\" && echo \"   conda : \$(conda --version)\" && echo && exec bash -i'"
