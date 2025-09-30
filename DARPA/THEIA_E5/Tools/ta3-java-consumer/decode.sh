#!/usr/bin/env bash
#
# decode_all_bins.sh
#
# Loop through all *.bin* files in $DATA_DIR, run the DARPA TC
# json_consumer.sh on each, and move non-empty JSON shards to $OUT_DIR.
#

set -euo pipefail

# --- Configuration ---
DATA_DIR="/home/kairos/DARPA/THEIA_E5/theia"
CONSUMER_DIR="/home/kairos/DARPA/THEIA_E5/Tools/ta3-java-consumer/tc-bbn-kafka"
OUT_DIR="/home/kairos/DARPA/THEIA_E5/theia"

cd "$CONSUMER_DIR"

for binfile in "$DATA_DIR"/*.bin*; do
    echo "[INFO] Processing $binfile"
    ./json_consumer.sh "$binfile"

    # Move generated JSON shards into OUT_DIR
    base=$(basename "$binfile")
    for shard in "${CONSUMER_DIR}/${base}".json*; do
        if [ -f "$shard" ]; then
            size=$(stat -c%s "$shard")
            if [ "$size" -gt 0 ]; then
                mv "$shard" "$OUT_DIR/"
                echo "  -> moved $(basename "$shard") ($size bytes)"
            else
                echo "  -> skipping empty shard $(basename "$shard")"
                rm -f "$shard"
            fi
        fi
    done
done

echo "[DONE] All .bin files processed. JSON shards are in $OUT_DIR"
