#!/usr/bin/env bash
set -euo pipefail

# target directory
TARGET_DIR="./theia/gzip"
mkdir -p "$TARGET_DIR"

# install gdown if needed
command -v gdown >/dev/null 2>&1 || pip install gdown

# download folder contents into target
# --fuzzy allows full folder URLs
# --remaining-ok skips already-downloaded files
gdown --fuzzy --folder 'https://drive.google.com/drive/folders/13zdJvC62zsJc2nD7KWxtN9xkk05LdQGw' \
      -O "$TARGET_DIR" \
      --remaining-ok