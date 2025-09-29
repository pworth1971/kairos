#!/usr/bin/env python3
"""
Decode DARPA Engagement 5 *.bin* Avro files into JSON.

- Reads *.bin* files from BIN_PATH
- Uses schema file (Avro .avsc)
- Writes one decoded JSON file per input into OUT_DIR

Requires:
    pip install avro-python3
"""

import avro.schema
from avro.io import DatumReader, BinaryDecoder
import io, json
from pathlib import Path

# --- Configuration ---
SCHEMA_FILE = "./Schema/TCCDMDatum.avsc"
BIN_PATH    = Path("./theia/")      # input directory with *.bin* files
OUT_DIR     = Path("./decoded/")    # output directory

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Avro schema ---
with open(SCHEMA_FILE, "r") as f:
    schema = avro.schema.parse(f.read())

def decode_avro_bin(bin_file: Path):
    """Stream decode an Avro .bin file → yields one record at a time."""
    with open(bin_file, "rb") as f:
        buf = f.read()
        decoder = BinaryDecoder(io.BytesIO(buf))
        reader = DatumReader(schema)
        while True:
            try:
                yield reader.read(decoder)
            except Exception:
                break  # reached EOF

def process_file(bin_file: Path):
    print(f"[INFO] Decoding {bin_file.name} ...")
    out_json = OUT_DIR / f"{bin_file.name}.json"

    # Decode and dump JSON lines
    with open(out_json, "w") as out:
        for rec in decode_avro_bin(bin_file):
            json.dump(rec, out)
            out.write("\n")

    print(f"  wrote decoded JSON {out_json}")

# --- Main loop ---
if __name__ == "__main__":
    bin_files = sorted(BIN_PATH.glob("*.bin*"))
    if not bin_files:
        print(f"No .bin files found in {BIN_PATH}")
    for bf in bin_files:
        try:
            process_file(bf)
        except Exception as e:
            print(f"[ERROR] Failed to process {bf}: {e}")
