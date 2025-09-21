#!/usr/bin/env python3
"""
THEIA E3 Data Preprocessing → Postgres → Temporal Graphs (PyG)

This script is a cleaned, well‑commented version of a Jupyter notebook that:
  1) Parses THEIA E3 JSON log shards (line‑delimited JSON-ish) to extract:
       - NetFlowObject nodes
       - Subject (process) nodes
       - FileObject nodes
       - Event edges (excluding EVENT_FLOWS_TO)
  2) Loads nodes/edges into a Postgres database.
  3) Builds a node2id table.
  4) Featurizes nodes using hierarchical string hashing.
  5) Generates per‑day TemporalData graphs and saves them.

It is designed for very large inputs: files are streamed line‑by‑line and DB
inserts are batched.

USAGE (examples)
---------------
# Minimal, assuming defaults and env vars:
python theia3_datapreprocess_clean.py \
  --raw-dir /raw_log \
  --out-dir ./train_graph

# With DB parameters:
PGDATABASE=tc_theia_dataset_db PGUSER=postgres PGPASSWORD=postgres \
PGHOST=/var/run/postgresql PGPORT=5432 \
python theia3_datapreprocess_clean.py --raw-dir /home/kairos/DARPA/THEIA_E3/raw

# Only run specific phases:
python theia3_datapreprocess_clean.py --raw-dir /raw_log --skip-events --skip-graphs

NOTES
-----
- The original code used a helper named stringtomd5 but implemented SHA‑256;
  we keep SHA‑256 because the pipeline expects 64‑hex hashes.
- Regexes are compiled once. Lines that don't match are skipped.
- Table schemas are created if missing (idempotent).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import psycopg2
from psycopg2 import extras as psql_extras

import numpy as np
import pytz
from datetime import datetime
from time import mktime
from tqdm import tqdm

import torch
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import TemporalData

# ----------------------------- Configuration -----------------------------

DEFAULT_RAW_DIR = os.environ.get("THEIA_E3_PATH", "/raw_log")
DEFAULT_OUT_DIR = "./train_graph"

# Postgres connection defaults; can be overridden by env vars or CLI
DEFAULT_PG = dict(
    database=os.environ.get("PGDATABASE", "tc_theia_dataset_db"),
    host=os.environ.get("PGHOST", "/var/run/postgresql/"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "postgres"),
    port=os.environ.get("PGPORT", "5432"),
)

# Relationship mapping and one‑hot vectors
REL2ID = {
    'EVENT_CONNECT': 1,
    'EVENT_EXECUTE': 2,
    'EVENT_OPEN':    3,
    'EVENT_READ':    4,
    'EVENT_RECVFROM':5,
    'EVENT_RECVMSG': 6,
    'EVENT_SENDMSG': 7,
    'EVENT_SENDTO':  8,
    'EVENT_WRITE':   9,
}
ID2REL = {v: k for k, v in REL2ID.items()}

# ------------------------- Helpers: hashing & time ------------------------

def sha256_hex(s: str) -> str:
    """Return 64‑hex SHA‑256 of input string (UTF‑8)."""
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

_TZ_US_EASTERN = pytz.timezone("US/Eastern")

def ns_time_to_datetime_str(ns: int) -> str:
    """Nanoseconds since epoch → 'YYYY-mm-dd HH:MM:SS.fffffffff' (local time)."""
    dt = datetime.fromtimestamp(int(ns) // 1_000_000_000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(ns % 1_000_000_000)).zfill(9)
    return s

def datetime_to_ns_time_US(s: str) -> int:
    """'YYYY-mm-dd HH:MM:SS' in US/Eastern → ns since epoch (int)."""
    tm = time.strptime(s, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(tm))
    ts = _TZ_US_EASTERN.localize(dt).timestamp()
    return int(ts * 1_000_000_000)

# --------------------------- DB schema helpers ---------------------------

SCHEMA_SQL = {
    "netflow_node_table": """
    CREATE TABLE IF NOT EXISTS netflow_node_table (
        uuid TEXT PRIMARY KEY,
        hashid TEXT NOT NULL,
        local_addr TEXT NOT NULL,
        local_port TEXT NOT NULL,
        remote_addr TEXT NOT NULL,
        remote_port TEXT NOT NULL
    );
    """,
    "subject_node_table": """
    CREATE TABLE IF NOT EXISTS subject_node_table (
        uuid TEXT PRIMARY KEY,
        hashid TEXT NOT NULL,
        cmdline TEXT NOT NULL,
        tgid TEXT NOT NULL,
        path TEXT NOT NULL
    );
    """,
    "file_node_table": """
    CREATE TABLE IF NOT EXISTS file_node_table (
        uuid TEXT PRIMARY KEY,
        hashid TEXT NOT NULL,
        path TEXT NOT NULL
    );
    """,
    "node2id": """
    CREATE TABLE IF NOT EXISTS node2id (
        node_hash TEXT PRIMARY KEY,
        node_type TEXT NOT NULL,
        node_msg  TEXT NOT NULL,
        index_id  INTEGER NOT NULL UNIQUE
    );
    """,
    "event_table": """
    CREATE TABLE IF NOT EXISTS event_table (
        src_hash TEXT NOT NULL,
        src_index INTEGER NOT NULL,
        rel_type TEXT NOT NULL,
        dst_hash TEXT NOT NULL,
        dst_index INTEGER NOT NULL,
        timestamp_rec BIGINT NOT NULL
    );
    """,
}

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_event_time ON event_table (timestamp_rec);",
    "CREATE INDEX IF NOT EXISTS idx_event_src  ON event_table (src_index);",
    "CREATE INDEX IF NOT EXISTS idx_event_dst  ON event_table (dst_index);",
]

def ensure_schema(conn) -> None:
    """Create tables and indexes if they don't already exist."""
    with conn, conn.cursor() as cur:
        for name, sql in SCHEMA_SQL.items():
            cur.execute(sql)
        for idx in INDEX_SQL:
            cur.execute(idx)

# ----------------------- Patterns & line extraction ----------------------

# Pre‑compile regex patterns used to extract fields
PAT_NETFLOW = re.compile(
    r'NetFlowObject":{"uuid":"(.*?)".*?"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),'
)
PAT_SUBJECT = re.compile(
    r'Subject":{"uuid":"(.*?)".*?"cmdLine":{"string":"(.*?)"}.*?"properties":{"map":{"tgid":"(.*?)"'
)
PAT_PATH = re.compile(r'"path":"(.*?)"')
PAT_FILEOBJ = re.compile(r'FileObject":{"uuid":"(.*?)".*?"filename":"(.*?)"')
PAT_EVENT = re.compile(
    r'"timestampNanos":(.*?),.*?"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)".*?"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)".*?"type":"(.*?)"'
)

# ------------------------------ Parsing ---------------------------------

def list_log_files(raw_dir: str) -> List[str]:
    """Return sorted list of shard files in raw_dir."""
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".json") or ".json." in f])
    if not files:
        logging.warning("No JSON log shards found in %s", raw_dir)
    return files

def parse_netflow(raw_dir: str, files: Iterable[str]) -> Tuple[Dict[str, Tuple[str,str]], List[List[str]]]:
    """
    Parse NetFlowObject lines.
    Returns:
      netobj2hash : map uuid->(hash, "laddr,lport,raddr,rport") and hash->uuid
      rows        : list of [uuid, hash, laddr, lport, raddr, rport] for DB insert
    """
    netobj2hash: Dict[str, Tuple[str, str]] = {}
    rows: List[List[str]] = []

    for fname in tqdm(files, desc="NetFlow", unit="file"):
        with open(os.path.join(raw_dir, fname), "r", errors="ignore") as f:
            for line in f:
                if '"NetFlowObject"' not in line:
                    continue
                m = PAT_NETFLOW.search(line)
                if not m:
                    continue
                uuid, laddr, lport, raddr, rport = m.groups()
                nodeprop = f"{laddr},{lport},{raddr},{rport}"
                h = sha256_hex(nodeprop)
                netobj2hash[uuid] = (h, nodeprop)
                netobj2hash[h] = uuid
                rows.append([uuid, h, laddr, lport, raddr, rport])
    return netobj2hash, rows

def parse_subjects(raw_dir: str, files: Iterable[str]) -> Tuple[Dict[str, Tuple[str,str,str,str]], List[List[str]]]:
    """
    Parse Subject lines.
    Returns:
      subject2hash : map uuid->(hash, cmdline, tgid, path) and hash->uuid
      rows         : list of [uuid, hash, cmdline, tgid, path]
    """
    subject2hash: Dict[str, Tuple[str,str,str,str]] = {}
    rows: List[List[str]] = []

    for fname in tqdm(files, desc="Subject", unit="file"):
        with open(os.path.join(raw_dir, fname), "r", errors="ignore") as f:
            for line in f:
                if '"Subject"' not in line:
                    continue
                ms = PAT_SUBJECT.search(line)
                if not ms:
                    continue
                uuid, cmdline, tgid = ms.groups()
                mp = PAT_PATH.search(line)
                path = mp.group(1) if mp else "null"
                nodeprop = f"{cmdline},{tgid},{path}"
                h = sha256_hex(nodeprop)
                subject2hash[uuid] = (h, cmdline, tgid, path)
                subject2hash[h] = uuid
                rows.append([uuid, h, cmdline, tgid, path])
    return subject2hash, rows

def parse_files(raw_dir: str, files: Iterable[str]) -> Tuple[Dict[str, Tuple[str,str]], List[List[str]]]:
    """
    Parse FileObject lines.
    Returns:
      file2hash : map uuid->(hash, path) and hash->uuid
      rows      : list of [uuid, hash, path]
    """
    file2hash: Dict[str, Tuple[str,str]] = {}
    rows: List[List[str]] = []

    for fname in tqdm(files, desc="FileObject", unit="file"):
        with open(os.path.join(raw_dir, fname), "r", errors="ignore") as f:
            for line in f:
                if '"FileObject"' not in line:
                    continue
                m = PAT_FILEOBJ.search(line)
                if not m:
                    continue
                uuid, path = m.groups()
                h = sha256_hex(path)
                file2hash[uuid] = (h, path)
                file2hash[h] = uuid
                rows.append([uuid, h, path])
    return file2hash, rows

# ----------------------------- DB utilities ------------------------------

def bulk_insert(cur, table: str, rows: List[List[object]], page_size: int = 10000) -> None:
    """Efficiently insert many rows via psycopg2.extras.execute_values."""
    if not rows:
        return
    placeholders = "(" + ",".join(["%s"] * len(rows[0])) + ")"
    sql = f"INSERT INTO {table} VALUES %s ON CONFLICT DO NOTHING"
    psql_extras.execute_values(cur, sql, rows, template=placeholders, page_size=page_size)

# ------------------------- Node2ID & lookups -----------------------------

def build_node2id(cur, file_rows, subject_rows, net_rows) -> Dict[str, int]:
    """
    Populate node2id table and return nodeid2msg lookup:
      - Maps node_hash -> index_id
      - And index_id  -> { node_type : node_msg }
    """
    node_list = {}

    # For file nodes: type="file", msg=path
    for uuid, h, path in file_rows:
        node_list[h] = ["file", path]

    # For subject nodes: type="subject", msg=path (or cmdline?), keep original behavior: use path in final map
    for uuid, h, cmd, tgid, path in subject_rows:
        node_list[h] = ["subject", path]

    # For netflow nodes: type="netflow", msg="laddr:lport->raddr:rport" (use "laddr:lport" if matching original)
    for uuid, h, laddr, lport, raddr, rport in net_rows:
        node_list[h] = ["netflow", f"{laddr}:{lport}"]

    # Build rows for node2id
    rows = []
    nodeid2msg = {}
    for idx, (node_hash, (typ, msg)) in enumerate(node_list.items()):
        rows.append([node_hash, typ, msg, idx])

    bulk_insert(cur, "node2id", rows, page_size=10000)

    # Load back into dict for lookups
    cur.execute("SELECT node_hash, node_type, node_msg, index_id FROM node2id ORDER BY index_id;")
    rows = cur.fetchall()
    for node_hash, node_type, node_msg, index_id in rows:
        nodeid2msg[node_hash] = index_id
        nodeid2msg[index_id] = {node_type: node_msg}

    return nodeid2msg

# ------------------------------ Events ----------------------------------

def ingest_events(cur, raw_dir: str, files: Iterable[str], subject2hash, file2hash, netobj2hash, nodeid2msg):
    """
    Stream Event records from logs and batch insert into event_table.
    Stores edges as (src_hash, src_idx, rel_type, dst_hash, dst_idx, timestamp).
    """
    batch = []
    BATCH_SIZE = 10000
    skipped = 0
    total = 0

    for fname in tqdm(files, desc="Events", unit="file"):
        with open(os.path.join(raw_dir, fname), "r", errors="ignore") as f:
            for line in f:
                if '"Event"' not in line or 'EVENT_FLOWS_TO' in line:
                    continue
                m = PAT_EVENT.search(line)
                if not m:
                    continue
                ts, subj_uuid, obj_uuid, rel = m.groups()
                try:
                    ts = int(ts)
                except Exception:
                    continue

                # Map UUIDs to 64‑hex node hashes using prior maps
                subj_h = subject2hash.get(subj_uuid, (None,))[0] if subj_uuid in subject2hash else None

                obj_h = None
                if obj_uuid in subject2hash:
                    obj_h = subject2hash[obj_uuid][0]
                elif obj_uuid in file2hash:
                    obj_h = file2hash[obj_uuid][0]
                elif obj_uuid in netobj2hash:
                    obj_h = netobj2hash[obj_uuid][0]

                if not subj_h or not obj_h:
                    skipped += 1
                    continue

                if len(subj_h) != 64 or len(obj_h) != 64:
                    skipped += 1
                    continue

                # Direction: if relation is a "read/recv" then data flows *to* subject
                if rel in ('EVENT_READ','EVENT_READ_SOCKET_PARAMS','EVENT_RECVFROM','EVENT_RECVMSG'):
                    src_h, dst_h = obj_h, subj_h
                else:
                    src_h, dst_h = subj_h, obj_h

                src_idx = nodeid2msg.get(src_h)
                dst_idx = nodeid2msg.get(dst_h)
                if src_idx is None or dst_idx is None:
                    skipped += 1
                    continue

                batch.append([src_h, src_idx, rel, dst_h, dst_idx, ts])
                total += 1

                if len(batch) >= BATCH_SIZE:
                    bulk_insert(cur, "event_table", batch, page_size=BATCH_SIZE)
                    batch.clear()

    if batch:
        bulk_insert(cur, "event_table", batch, page_size=BATCH_SIZE)

    logging.info("Events ingested: %d (skipped: %d)", total, skipped)

# ---------------------------- Featurization ------------------------------

def path_to_hierarchy(p: str, sep: str = '/') -> List[str]:
    parts = [x for x in p.strip().split(sep) if x]
    out = []
    for i, part in enumerate(parts):
        if i == 0:
            out.append(part)
        else:
            out.append(out[-1] + sep + part)
    return out

def ip_to_hierarchy(ip: str) -> List[str]:
    parts = [x for x in ip.strip().split('.') if x]
    out = []
    for i, part in enumerate(parts):
        if i == 0:
            out.append(part)
        else:
            out.append(out[-1] + '.' + part)
    return out

def build_node_feature_matrix(cur, rel2id=REL2ID, n_features: int = 16) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    """
    Build:
      - node2higvec: [N, n_features] float32
      - rel2vec:     dict[str -> onehot Tensor]
    """
    # Pull node2id table into memory
    cur.execute("SELECT node_hash, node_type, node_msg, index_id FROM node2id ORDER BY index_id;")
    rows = cur.fetchall()

    # Compose hierarchical strings per node and feature-hash them
    fh = FeatureHasher(n_features=n_features, input_type="string")
    msg_strings = []
    for node_hash, node_type, node_msg, index_id in rows:
        if node_type == 'netflow':
            s = "netflow" + "".join(ip_to_hierarchy(node_msg))
        elif node_type == 'file':
            s = "file"    + "".join(path_to_hierarchy(node_msg))
        elif node_type == 'subject':
            s = "subject" + "".join(path_to_hierarchy(node_msg))
        else:
            s = node_type + node_msg
        msg_strings.append(s)

    X = fh.transform(msg_strings).toarray().astype(np.float32)
    # Relationship one-hots
    relvec = torch.nn.functional.one_hot(
        torch.arange(0, len(rel2id)), num_classes=len(rel2id)
    )
    rel2vec = {rel: relvec[idx] for rel, idx in rel2id.items()}
    return X, rel2vec

# --------------------------- Graph generation ---------------------------

def generate_daily_graphs(cur, out_dir: str, node2higvec: np.ndarray, rel2vec: Dict[str, torch.Tensor]) -> None:
    """
    Iterate 2018-04-02..2018-04-13 (inclusive of start days) and build TemporalData graphs by day.
    Saves: {out_dir}/graph_4_{day}.TemporalData.simple
    """
    os.makedirs(out_dir, exist_ok=True)

    for day in tqdm(range(2, 14), desc="Graphs/day"):
        start_ns = datetime_to_ns_time_US(f'2018-04-{day:02d} 00:00:00')
        end_ns   = datetime_to_ns_time_US(f'2018-04-{day+1:02d} 00:00:00')

        cur.execute("""
            SELECT src_index, rel_type, dst_index, timestamp_rec
            FROM event_table
            WHERE timestamp_rec > %s AND timestamp_rec < %s
            ORDER BY timestamp_rec;
        """, (start_ns, end_ns))
        events = cur.fetchall()

        if not events:
            logging.info("No events for 2018-04-%02d", day)
            continue

        # Build TemporalData
        src_idx, dst_idx, msg_list, t_list = [], [], [], []
        for src_i, rel, dst_i, t in events:
            src_idx.append(int(src_i))
            dst_idx.append(int(dst_i))
            # Concatenate [node_feat[src], rel_onehot, node_feat[dst]]
            msg_list.append(torch.cat([
                torch.from_numpy(node2higvec[src_i]),
                rel2vec.get(rel, torch.zeros(len(REL2ID), dtype=torch.long)),
                torch.from_numpy(node2higvec[dst_i]),
            ]))
            t_list.append(int(t))

        data = TemporalData()
        data.src = torch.tensor(src_idx, dtype=torch.long)
        data.dst = torch.tensor(dst_idx, dtype=torch.long)
        data.t   = torch.tensor(t_list, dtype=torch.long)
        data.msg = torch.vstack(msg_list).to(torch.float32)

        torch.save(data, os.path.join(out_dir, f"graph_4_{day}.TemporalData.simple"))
        logging.info("Saved %s", os.path.join(out_dir, f"graph_4_{day}.TemporalData.simple"))

# --------------------------------- Main ----------------------------------

def run(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    if not os.path.isdir(raw_dir):
        logging.error("Raw directory does not exist: %s", raw_dir)
        sys.exit(1)

    files = list_log_files(raw_dir)
    if not files:
        logging.error("No input files found in %s", raw_dir)
        sys.exit(1)

    # Connect to Postgres
    conn = psycopg2.connect(
        database=args.dbname, host=args.dbhost, user=args.dbuser,
        password=args.dbpass, port=args.dbport
    )
    ensure_schema(conn)
    cur = conn.cursor()

    # 1) Parse & insert nodes
    if not args.skip_nodes:
        netobj2hash, net_rows     = parse_netflow(raw_dir, files)
        subject2hash, subj_rows   = parse_subjects(raw_dir, files)
        file2hash, file_rows      = parse_files(raw_dir, files)

        logging.info("Inserting %d netflow nodes...", len(net_rows))
        bulk_insert(cur, "netflow_node_table", net_rows)

        logging.info("Inserting %d subject nodes...", len(subj_rows))
        bulk_insert(cur, "subject_node_table", subj_rows)

        logging.info("Inserting %d file nodes...", len(file_rows))
        bulk_insert(cur, "file_node_table", file_rows)
        conn.commit()
    else:
        # If skipping nodes, we still need the maps:
        logging.info("Skipping node parsing; loading maps back from DB.")
        netobj2hash, subject2hash, file2hash = {}, {}, {}
        cur.execute("SELECT uuid, hashid, local_addr, local_port, remote_addr, remote_port FROM netflow_node_table;")
        for uuid, h, la, lp, ra, rp in cur.fetchall():
            netobj2hash[uuid] = (h, f"{la},{lp},{ra},{rp}")
            netobj2hash[h] = uuid
        cur.execute("SELECT uuid, hashid, cmdline, tgid, path FROM subject_node_table;")
        for uuid, h, cmd, tgid, path in cur.fetchall():
            subject2hash[uuid] = (h, cmd, tgid, path)
            subject2hash[h] = uuid
        cur.execute("SELECT uuid, hashid, path FROM file_node_table;")
        for uuid, h, path in cur.fetchall():
            file2hash[uuid] = (h, path)
            file2hash[h] = uuid

    # 2) Build node2id (and nodeid2msg lookup)
    nodeid2msg = build_node2id(cur,
                               file_rows if not args.skip_nodes else [(u,h,p) for u,(h,p) in [(k,v) for k,v in file2hash.items() if len(k)!=64]],
                               subj_rows if not args.skip_nodes else [(u,h,cmd,tgid,path) for u,(h,cmd,tgid,path) in [(k,v) for k,v in subject2hash.items() if len(k)!=64]],
                               net_rows  if not args.skip_nodes else [(u,h,*v[1].split(',')) for u,v in [(k,v) for k,v in netobj2hash.items() if len(k)!=64]]
                               )
    conn.commit()

    # 3) Ingest events
    if not args.skip_events:
        ingest_events(cur, raw_dir, files, subject2hash, file2hash, netobj2hash, nodeid2msg)
        conn.commit()
    else:
        logging.info("Skipping event ingestion.")

    # 4) Featurize nodes
    if not args.skip_feats:
        node2higvec, rel2vec = build_node_feature_matrix(cur, n_features=args.hash_dim)
        # Save side artifacts
        torch.save(node2higvec, os.path.join(out_dir, "node2higvec.pt"))
        torch.save(rel2vec,    os.path.join(out_dir, "rel2vec.pt"))
        logging.info("Saved node2higvec.pt and rel2vec.pt in %s", out_dir)
    else:
        node2higvec = torch.load(os.path.join(out_dir, "node2higvec.pt")).numpy() \
                      if os.path.exists(os.path.join(out_dir, "node2higvec.pt")) \
                      else None
        rel2vec = torch.load(os.path.join(out_dir, "rel2vec.pt")) \
                  if os.path.exists(os.path.join(out_dir, "rel2vec.pt")) \
                  else None
        if node2higvec is None or rel2vec is None:
            logging.error("Featurization skipped but artifacts not found in %s", out_dir)
            sys.exit(1)

    # 5) Build graphs per day
    if not args.skip_graphs:
        generate_daily_graphs(cur, out_dir, node2higvec, rel2vec)
    else:
        logging.info("Skipping graph generation.")

    cur.close()
    conn.close()
    logging.info("All done.")

# --------------------------------- CLI -----------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="THEIA E3 preprocessing → Postgres → TemporalData graphs")
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help=f"Directory of THEIA E3 raw logs (default: {DEFAULT_RAW_DIR})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory for graphs/artifacts (default: {DEFAULT_OUT_DIR})")

    # DB params
    p.add_argument("--dbname", default=DEFAULT_PG["database"])
    p.add_argument("--dbhost", default=DEFAULT_PG["host"])
    p.add_argument("--dbuser", default=DEFAULT_PG["user"])
    p.add_argument("--dbpass", default=DEFAULT_PG["password"])
    p.add_argument("--dbport", default=DEFAULT_PG["port"])

    # Feature hashing dimension
    p.add_argument("--hash-dim", type=int, default=16, help="FeatureHasher output dimension (default: 16)")

    # Phase toggles
    p.add_argument("--skip-nodes", action="store_true", help="Skip parsing/inserting nodes")
    p.add_argument("--skip-events", action="store_true", help="Skip event ingestion")
    p.add_argument("--skip-feats", action="store_true", help="Skip featurization (expects existing artifacts)")
    p.add_argument("--skip-graphs", action="store_true", help="Skip temporal graph generation")

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p

if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
