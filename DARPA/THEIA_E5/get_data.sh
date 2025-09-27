#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Script: download_and_decompress.sh
# Purpose: Download all .gz files from Google Drive (via gdown) and decompress
# -----------------------------------------------------------------------------

FOLDER_URL="https://drive.google.com/drive/folders/13zdJvC62zsJc2nD7KWxtN9xkk05LdQGw"
OUTPUT_DIR="./theia"

# --- Check dependencies ---
for cmd in pip gdown pv pigz; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[*] Installing missing dependency: $cmd"
    if [[ "$cmd" == "gdown" ]]; then
      pip install gdown
    else
      sudo apt-get update && sudo apt-get install -y pv pigz
    fi
  fi
done

# --- Download the folder ---
echo "[*] Downloading from Google Drive..."
gdown --fuzzy --folder "$FOLDER_URL" -O ./ --remaining-ok

# --- Find and decompress all .gz files ---
echo "[*] Searching for .gz files in $OUTPUT_DIR ..."
find "$OUTPUT_DIR" -type f -name '*.gz' -print0 \
| while IFS= read -r -d '' f; do
    out="${f%.gz}"
    echo "Decompressing: $f  ->  $out"
    pv "$f" | pigz -d > "$out"
    # Uncomment the next line to remove .gz after success
    # rm "$f"
done |& tee decompress_progress.log

echo "------------------------------------------------------------"
echo "✅ All downloads complete. Decompression log: decompress_progress.log"
echo "------------------------------------------------------------"

