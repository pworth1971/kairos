#!/usr/bin/env python3
"""
THEIA E5 Data Preprocessing → Postgres → Temporal Graphs (PyG)

This script is a cleaned, well‑commented version of a Jupyter notebook that:
  1) Parses THEIA E5 JSON log shards (line‑delimited JSON-ish) to extract:
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
python theia5_datapreprocess.py --raw-dir /raw_log --out-dir ./train_graph

# With DB parameters:
PGDATABASE=tc_theia_dataset_db PGUSER=postgres PGPASSWORD=passwd PGHOST=/var/run/postgresql PGPORT=5432 python theia5_datapreprocess_clean.py --raw-dir /home/kairos/DARPA/THEIA_E5/raw_log

# Only run specific phases:
python theia3_datapreprocess.py --raw-dir /raw_log --skip-events --skip-graphs


NOTES
-----
- The original code used a helper named stringtomd5 but implemented SHA‑256;
  we keep SHA‑256 because the pipeline expects 64‑hex hashes.
- Regexes are compiled once. Lines that don't match are skipped.
- Table schemas are created if missing (idempotent).
"""



# ----------------------------- Imports -----------------------------


from __future__ import annotations

import argparse
import os
import re
import sys
import json
import hashlib
import logging
import time
import pytz

from time import mktime
from tqdm import tqdm

from datetime import datetime, timezone

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import psycopg2
from psycopg2 import extras as psql_extras

import numpy as np

#import torch
#from sklearn.feature_extraction import FeatureHasher
#from torch_geometric.data import TemporalData




# ----------------------------- Configuration -----------------------------

DEFAULT_RAW_DIR = "/home/kairos/DARPA/THEIA_E5/"
DEFAULT_OUT_DIR = "/home/kairos/DARPA/THEIA_E5/train_graph/"

# Postgres connection defaults; can be overridden by env vars or CLI
DEFAULT_PG = dict(
    database=os.environ.get("PGDATABASE", "tc_e5_theia_dataset_db"),
    host=os.environ.get("PGHOST", "/var/run/postgresql/"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", "Rafter9876!@"),
    port=os.environ.get("PGPORT", "5432"),
)

# ------------------------- Helpers: hashing & time ------------------------



def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s

def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)
    
import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest() 
    
# --------------------------- DB schema helpers ---------------------------

"""
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database tc_e5_theia_dataset_db;

# switch to the created database
postgres=# \connect tc_e5_theia_dataset_db;

# create the event table and grant the privileges to postgres
tc_e5_theia_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_e5_theia_dataset_db=# alter table event_table owner to postgres;
tc_e5_theia_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_e5_theia_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_theia_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_e5_theia_dataset_db=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_theia_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_e5_theia_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_e5_theia_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_e5_theia_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_e5_theia_dataset_db=# alter table node2id owner to postgres;
tc_e5_theia_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);

"""



# --------------------------------- Main ----------------------------------

import functools
import os
import json
import multiprocessing as mp
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *
import threading
import networkx as nx
import math


import psycopg2
from psycopg2.extras import DictCursor


# --- Compile once for speed (matches your current pattern/fields) ---
PAT_NETFLOW = re.compile(
    r'NetFlowObject":{"uuid":"(.*?)".*?'
    r'"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}'
)

def run(args):

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    print("raw_dir:" + raw_dir)
    print("out_dir:" + out_dir)

    if not os.path.isdir(raw_dir):
        logging.error("Raw directory does not exist: %s", raw_dir)
        sys.exit(1)
    
    #
    # create fileList
    #
    from pathlib import Path

    root = Path.cwd() / "raw_log"
    print(f"Scanning: {root.resolve()}")

    # Only files starting with 'ta1-theia' AND containing 'json' (case-insensitive)
    fileList = sorted(
        p.name for p in root.iterdir()
        if p.is_file() and p.name.startswith("ta1-theia") and "json" in p.name.lower()
    )

    print(f"Found {len(fileList)} files:")
    for name in fileList:
        print(name)

    # Connect to Postgres    
    try:
        conn = psycopg2.connect(
            dbname="tc_e5_theia_dataset_db",
            user="postgres",
            password="Rafter9876!@",      # <- set this to whatever you set with \password
            host="127.0.0.1",         # TCP loopback (bypasses Unix-socket peer auth)
            port=5432,
            connect_timeout=10,
            options="-c statement_timeout=60000"
        )
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT current_database(), inet_server_addr(), inet_server_port()")
        print(cur.fetchone())
    except psycopg2.Error as e:
        print("PSQL error:", e.pgcode, e)
        raise

    cur = conn.cursor()

    filePath = raw_dir
    print("filePath:" + filePath)


    # Expect these to be defined already:
    # - stringtomd5(s: str) -> 64-hex hash (your sha256 helper)
    # - filePath: str or Path to the folder
    # - fileList: list[str] of filenames inside filePath
    # - netobj2hash: dict; netobjset: set
    # If not, uncomment these:
    # netobj2hash = {}
    # netobjset = set()
    # filePath = Path.cwd() / "raw_log"

    # --- NetFlow extraction init (must be before the loop) ---
    import re, time
    from pathlib import Path

    PAT_NETFLOW = re.compile(
        r'NetFlowObject":{"uuid":"(.*?)".*?'
        r'"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}'
    )

    netobjset: set[str] = set()                 # tracks unique NetFlow hashes
    netobj2hash: dict[str, list[str]] = {}      # uuid -> [hash, props] and hash -> uuid

    tot_lines = tot_matches = tot_new = tot_errors = 0
    tot_new_hashes = new_hashes = errors = 0
    t0_all = time.perf_counter()

    for file in tqdm(fileList):
        fullPath = Path(filePath) / file
        print(f"processing file: {fullPath} ...")
        
        lines = matches = errors = 0
        before_unique = len(netobjset)
        t0 = time.perf_counter()

        with open(fullPath, "r", errors="ignore") as f:
            for line in f:
                lines += 1
                if "NetFlowObject" not in line:
                    continue

                matches += 1
                try:
                    m = PAT_NETFLOW.search(line)
                    if not m:
                        # pattern not satisfied; skip
                        continue

                    nodeid, srcaddr, srcport, dstaddr, dstport = m.groups()
                    nodeproperty = f"{srcaddr},{srcport},{dstaddr},{dstport}"
                    hashstr = stringtomd5(nodeproperty)

                    # Insert/overwrite maps (idempotent)
                    netobj2hash[nodeid] = [hashstr, nodeproperty]
                    netobj2hash[hashstr] = nodeid

                    # Track unique netflow nodes (by hash)
                    if hashstr not in netobjset:
                        netobjset.add(hashstr)

                except Exception:
                    errors += 1
                    # If you want, print the bad line once in a while:
                    # if errors < 5: print("parse error on line:", line[:200])
                    continue

        elapsed = time.perf_counter() - t0
        new_hashes = len(netobjset) - before_unique
        rate = (lines / elapsed) if elapsed > 0 else 0.0

        # Per-file summary
        print(
            f"[SUMMARY] {file} | lines={lines:,} | netflow_matches={matches:,} | "
            f"new_unique_netflows={new_hashes:,} | parse_errors={errors:,} | "
            f"time={elapsed:.1f}s | rate≈{rate:,.0f} lines/s"
        )

        # Accumulate totals
        tot_lines += lines
        tot_matches += matches
        tot_new_hashes += new_hashes
        tot_errors += errors

    # Grand total
    all_elapsed = time.perf_counter() - t0_all
    print(
        f"[TOTAL] files={len(fileList)} | lines={tot_lines:,} | "
        f"netflow_matches={tot_matches:,} | new_unique_netflows={tot_new_hashes:,} | "
        f"parse_errors={tot_errors:,} | time={all_elapsed:.1f}s | "
        f"rate≈{(tot_lines/all_elapsed if all_elapsed>0 else 0):,.0f} lines/s"
    )


    # Build rows for netflow_node_table and insert with execute_values.

    from psycopg2.extras import execute_values  # <- fixes the "ex is not defined" error

    # netobj2hash is expected to have BOTH:
    #   uuid -> [hash_id, "src_addr,src_port,dst_addr,dst_port"]
    #   hash_id -> uuid
    # We only want the UUID keys (len != 64). Example row order matches the table:
    #   (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port)

    datalist = []
    for k, v in netobj2hash.items():
        # keep only UUID keys (hash keys are 64-hex chars)
        if len(k) != 64:
            hash_id = v[0]
            try:
                src_addr, src_port, dst_addr, dst_port = v[1].split(",")
            except ValueError:
                # Skip malformed nodeproperty strings
                continue
            datalist.append([k, hash_id, src_addr, src_port, dst_addr, dst_port])

    # Optional: peek at a few rows for sanity
    from pprint import pprint
    pprint(datalist[:3])
    print(f"Prepared {len(datalist)} rows for insert into netflow_node_table")

    if datalist:
        # Always specify columns and a template; ON CONFLICT avoids duplicate PK errors
        sql = """
            INSERT INTO netflow_node_table
                (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        template = "(%s,%s,%s,%s,%s,%s)"
        execute_values(cur, sql, datalist, template=template, page_size=10000)

        # Commit using the correct connection variable (conn or connect)
        # If your variable is named `connect`, either rename it to conn or call connect.commit()
        conn.commit()
        print("Insert committed.")
    else:
        print("No rows to insert; skipping database write.")


    # del netobj2hash
    # del datalist



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
