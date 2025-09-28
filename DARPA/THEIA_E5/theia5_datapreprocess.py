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
from time import mktime
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

from pathlib import Path
from tqdm import tqdm
import numpy as np
import functools
import networkx as nx
import math

import psycopg2
from psycopg2 import extras as psql_extras
from psycopg2.extras import execute_values
from psycopg2.extras import DictCursor
import psycopg2.extras

from contextlib import contextmanager
import gc

import torch
import torch_geometric
from torch_geometric.transforms import NormalizeFeatures

from sklearn.feature_extraction import FeatureHasher
from sklearn import preprocessing 


# ----------------------------- Configuration -----------------------------

DEFAULT_RAW_DIR = "/home/kairos/DARPA/THEIA_E5/"
DEFAULT_OUT_DIR = "/home/kairos/DARPA/THEIA_E5/train_graph/"
DEFAULT_SUB_DIR = "theia/"
DEFAULT_EMB_DIR = "./embeddings/"


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




# ------------------------------------ BULK INSERT Handling -----------------------------------------

@contextmanager
def get_db_cursor(conn):
    """Context manager for database cursor with proper cleanup"""
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def bulk_insert(conn, table_name, datalist, batch_size=100000):
    """
    Optimized bulk insert with batching, memory management, and performance tuning
    """
    logging.info(f"bulk_insert() - table_name:{table_name}, rows:{len(datalist)}")

    if not datalist:
        logging.warning("No rows to insert; skipping database write.")
        return
    
    total_rows = len(datalist)

    try:
        with get_db_cursor(conn) as cur:
            # Optimize PostgreSQL settings for bulk insert
            cur.execute("SET synchronous_commit = OFF")
            cur.execute("SET maintenance_work_mem = '1GB'")
            
            sql = f'''INSERT INTO {table_name} VALUES %s'''
            inserted_count = 0
            batch_num = 1
            total_batches = (total_rows + batch_size - 1) // batch_size
            
            # Process in batches to manage memory
            for i in range(0, total_rows, batch_size):
                batch = datalist[i:i + batch_size]
                current_batch_size = len(batch)
                logging.info(f"Processing batch {batch_num}/{total_batches} "
                             f"({current_batch_size:,} rows)")
                
                # Use execute_values with optimized parameters
                psycopg2.extras.execute_values(
                    cur, 
                    sql, 
                    batch,
                    template=None,
                    page_size=10000,  # Smaller page size for better memory usage
                    fetch=False
                )
                inserted_count += current_batch_size
                batch_num += 1
                
                # Commit every few batches to avoid long transactions
                if batch_num % 5 == 0:  # Commit every 5 batches
                    conn.commit()
                    logging.info(f"Committed {inserted_count:,}/{total_rows:,} rows")
                
                # Force garbage collection to free memory
                if batch_num % 10 == 0:
                    gc.collect()
            
            # Final commit
            conn.commit()
            logging.info(f"Successfully inserted all {inserted_count:,} rows into {table_name}.")
    
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during bulk insert: {e}")
        raise
    
    finally:
        # Clear the datalist to free memory
        datalist.clear()
        gc.collect()

# ------------------------------------ BULK INSERT Handling -----------------------------------------



# --------------------------------- Main ----------------------------------

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
    

    # --------------------------------------------------------------------------------
    # create fileList
    #

    logging.info("creating fileList for parsing...")
    from pathlib import Path

    root = Path.cwd() / DEFAULT_SUB_DIR
    print(f"Scanning: {root.resolve()}")

    # Only files starting with 'ta1-theia' AND containing 'json' (case-insensitive)
    fileList = sorted(
        p.name for p in root.iterdir()
        if p.is_file() and p.name.startswith("ta1-theia") and "json" in p.name.lower()
    )

    logging.info(f"Found {len(fileList)} files")
    for name in fileList:
        print(name)
    # --------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------
    # Connect to Postgres    

    logging.info("\n\rconnecting to postgresql...")

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
    # --------------------------------------------------------------------------------
    

    # --------------------------------------------------------------------------------
    # 
    # Dataset Reset / Truncation 
    #

    if args.reset:

        logging.warning("RESET flag passed: truncating all tables in database %s", args.dbname)
        
        reset_sql = """
            TRUNCATE TABLE
                event_table,
                file_node_table,
                netflow_node_table,
                subject_node_table,
                node2id
            RESTART IDENTITY CASCADE;
        """
        
        #logging.info(f"truncate sql: {reset_sql}")

        try:
            cur.execute(reset_sql)
            conn.commit()
            logging.info("All tables truncated successfully.")
        except Exception as e:
            logging.error("Failed to reset tables: %s", e)
            conn.rollback()
            sys.exit(1)

    # --------------------------------------------------------------------------------    

    filePath = raw_dir + DEFAULT_SUB_DIR
    logging.info("filePath:" + filePath)


    # --------------------------------------------------------------------------
    # ----------------------------- NetFlow ------------------------------------
    #
    # --- NetFlow extraction init (must be before the loop) ---
    #
    # --------------------------------------------------------------------------

    if not args.skip_netflows:

        print("\n\tparsing netflow info...\n")

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

                # Get the total number of lines in the file for the progress bar
                total_lines_in_file = sum(1 for _ in f)
                f.seek(0)  # Reset file pointer to the beginning
                
                # Create a progress bar for the lines
                with tqdm(total=total_lines_in_file, desc="Processing lines", leave=False) as line_progress:

                    for line in f:
                        lines += 1
                        
                        line_progress.update(1)  # Update the progress bar for each line

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

        #
        # Build rows for netflow_node_table and insert with execute_values.
        #

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
        #from pprint import pprint
        #pprint(datalist[:3])
        logging.info(f"Prepared {len(datalist)} rows for insert into netflow_node_table")

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
            logging.info(f"Successfully inserted {len(datalist)} rows into netflow_node_table.")
        else:
            logging.warning("No rows to insert; skipping database write.")

        del netobj2hash
        del datalist
        gc.collect()
    else:
        logging.warning("\r--skipping netflow info parsing--")


    # --------------------------------------------------------------------------
    # ----------------------------- Process ------------------------------------
    #
    # --- NetFlow extraction init (must be before the loop) ---
    #
    # --------------------------------------------------------------------------

    if not args.skip_processes:

        print("\n\tparsing process info from fileList...\n")
        
        scusess_count=0
        fail_count=0
        subject_objset=set()
        subject_obj2hash={}# 

        for file in tqdm(fileList):
            fullPath = Path(filePath) / file
            logging.info(f"processing file: {fullPath} ...")

            with open(fullPath, "r") as f:
                # Get the total number of lines in the file for the progress bar
                total_lines_in_file = sum(1 for _ in f)
                f.seek(0)  # Reset file pointer to the beginning
                
                # Create a progress bar for the lines
                with tqdm(total=total_lines_in_file, desc="Processing lines", leave=False) as line_progress:

                    for line in (f):

                        line_progress.update(1)  # Update the progress bar for each line

                        if "schema.avro.cdm20.Subject" in line:
        #                     print(line)
                            subject_uuid=re.findall('avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"',line)
        #                
                            try:
        #                         (subject_uuid[0][-1])
                                subject_obj2hash[subject_uuid[0][0]]=subject_uuid[0][-1]
                                scusess_count+=1
                            except:
                                try:
                                    subject_obj2hash[subject_uuid[0][0]]="null"
                                except:
                                    pass
        #                             print(line)
        #                         print(line)                        
                                fail_count+=1

        procdatalist=[]
        for i in subject_obj2hash.keys():
            if len(i)!=64:
                procdatalist.append([i]+[stringtomd5(subject_obj2hash[i]),subject_obj2hash[i]])

        # Optional: peek at a few rows for sanity
        #pprint(procdatalist[:3])
        logging.info(f"Prepared {len(procdatalist)} rows for insert into subject_node_table")

        if procdatalist:
            # Always specify columns and a template; ON CONFLICT avoids duplicate PK errors
            sql = '''
                INSERT INTO subject_node_table
                    VALUES %s
                '''
            execute_values(cur,sql, procdatalist, page_size=10000)
            conn.commit()
            logging.info(f"Successfully inserted {len(procdatalist)} rows in subject_node_table.")
        else:
            logging.warning("No rows to insert; skipping database write.")

        del procdatalist
        gc.collect()
    else:
        logging.warning("\r--skipping process info parsing--")


    # --------------------------------------------------------------------------
    # ----------------------------- File Data ----------------------------------
    #
    #
    # --------------------------------------------------------------------------

    if not args.skip_files:

        print("\n\tparsing file info...\n")

        file_node=set()
        file_obj2hash={}
        fail_count=0

        for file in tqdm(fileList):    

            fullPath = Path(filePath) / file
            logging.info(f"processing file: {fullPath} ...")

            with open(fullPath, "r") as f:

                # Get the total number of lines in the file for the progress bar
                total_lines_in_file = sum(1 for _ in f)
                f.seek(0)  # Reset file pointer to the beginning
                
                # Create a progress bar for the lines
                with tqdm(total=total_lines_in_file, desc="Processing lines", leave=False) as line_progress:

                    for line in f:

                        line_progress.update(1)  # Update the progress bar for each line

                        if "avro.cdm20.FileObject" in line:
        #                     print(line)
                            Object_uuid=re.findall('avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"',line) 
                            try:
                                file_node.add(Object_uuid[0])
                                file_obj2hash[Object_uuid[0][0]]=Object_uuid[0][-1]
                            except:
                                fail_count+=1
        #                         print(line)
        
        filedatalist=[]
        for i in file_obj2hash.keys():
            if len(i)!=64:
                filedatalist.append([i]+[stringtomd5(file_obj2hash[i]),file_obj2hash[i]])

        # Optional: peek at a few rows for sanity
        #pprint(filedatalist[:3])
        logging.info(f"Prepared {len(filedatalist)} rows for insert into file_node_table")

        if filedatalist:
            sql = '''insert into file_node_table
                                values %s
                    '''
            execute_values(cur, sql, filedatalist, page_size=10000)
            conn.commit() 
            logging.info(f"Successfully inserted {len(filedatalist)} rows into file_node_table.")
        else:
            logging.warning("No rows to insert; skipping database write.")

        del filedatalist
        gc.collect()
    
    else:
        logging.warning("\r--skipping file info parsing--")

    # --------------------------------------------------------------------------
    #
    # --------------------------- Event / Node2ID Data --------------------------
    #
    # Build a unified node catalog (node2id):
    #   - Each row links a stable node hash_id to:
    #       * node_type  ∈ {"file","subject","netflow"}
    #       * node_value (human-friendly attribute; e.g., filename or "ip:port")
    #       * index_id   (dense integer id for graph building)
    #
    # Design notes:
    #   - We read only the columns we need (no SELECT *) for clarity and stability.
    #   - We make the insert idempotent via ON CONFLICT DO NOTHING.
    #   - We assign fresh index_id values *only* to hashes not already in node2id.
    #   - We also construct convenience dicts:
    #       file_uuid2hash, subject_uuid2hash, net_uuid2hash
    #       (uuid → hash_id) to translate THEIA UUIDs during edge construction.
    # --------------------------------------------------------------------------

    
    # --------------------------------------------------------------------------
    #
    # Generate the data for node2id table
    #
    # --------------------------------------------------------------------------
    
    if not args.skip_node2ids:

        print("\r\nbuilding node2id catalog from node tables...")

        node_list={}

        ##################################################################################################
        cur.execute("""
            SELECT * FROM file_node_table
        """)
        file_nodes = cur.fetchall()    
        
        for i in file_nodes:    
            node_list[i[1]]=["file",i[-1]]
        
        file_uuid2hash={}
        for i in file_nodes:
            file_uuid2hash[i[0]]=i[1]
        
        ##################################################################################################
        cur.execute("""
            SELECT * FROM subject_node_table
        """)
        subject_nodes = cur.fetchall()
        
        for i in subject_nodes:
            node_list[i[1]]=["subject",i[-1]]

        subject_uuid2hash={}
        for i in subject_nodes:
            subject_uuid2hash[i[0]]=i[1]
        
        ##################################################################################################
        cur.execute("""
            SELECT * FROM netflow_node_table
        """)
        netflow_nodes = cur.fetchall()
        
        for i in netflow_nodes:
            node_list[i[1]]=["netflow",i[-2]+":"+i[-1]]
        
        net_uuid2hash={}
        for i in netflow_nodes:
            net_uuid2hash[i[0]]=i[1]
        
        ##################################################################################################
        node_list_database=[]
        node_index=0
        for i in node_list:
            node_list_database.append([i]+node_list[i]+[node_index])
            node_index+=1
        
        logging.info("New node2id rows to insert: %d", len(node_list_database))

        if node_list_database:
            sql = '''insert into node2id values %s'''

            execute_values(cur, sql, node_list_database, page_size=10000)
            conn.commit()  
            logging.info(f"Successfully inserted {len(node_list_database)} rows in node2id table.")
        else:
            logging.warning("No rows to insert; skipping database write.")

        del node_list_database
        gc.collect()

    else:
        logging.warning("skipped node2id parsing...")


    # -----------------------------------------------------------------------------------------------
    # ------------------------------------- event table data ----------------------------------------
    #

    if not args.skip_events:

        print("\r\nparsing event_table info...")

        # Constructing the map for nodeid to msg
        sql="select * from node2id ORDER BY index_id;"
        cur.execute(sql)
        rows = cur.fetchall()

        nodeid2msg={}  # nodeid => msg and node hash => nodeid
        for i in rows:
            nodeid2msg[i[0]]=i[-1]
            nodeid2msg[i[-1]]={i[1]:i[2]}

        #print(nodeid2msg)
        
        include_edge_type=[
            'EVENT_CLOSE',
            'EVENT_OPEN',
            'EVENT_READ',
            'EVENT_WRITE',
            'EVENT_EXECUTE',
            'EVENT_RECVFROM',
            'EVENT_RECVMSG',
            'EVENT_SENDMSG',
            'EVENT_SENDTO',
        ]
        
        datalist=[]
        edge_type=set()
        reverse=["EVENT_RECVFROM","EVENT_RECVMSG","EVENT_READ"]        

        for file in tqdm(fileList):

            file_name = filePath + file 
            logging.info(f"processing file: {file_name}...")

            with open(file_name, "r") as f:

                # Get the total number of lines in the file for the progress bar
                total_lines_in_file = sum(1 for _ in f)
                f.seek(0)  # Reset file pointer to the beginning
                
                # Create a progress bar for the lines
                with tqdm(total=total_lines_in_file, desc="Processing lines", leave=False) as line_progress:

                    for line in (f):
                        
                        line_progress.update(1)  # Update the progress bar for each line

                        if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
        #                     print(line)
                            subject_uuid=re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"',line)
                            predicateObject_uuid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"',line)
                            if len(subject_uuid) >0 and len(predicateObject_uuid)>0:
                                if subject_uuid[0] in subject_uuid2hash\
                                and (predicateObject_uuid[0] in file_uuid2hash or predicateObject_uuid[0] in net_uuid2hash):
                                    relation_type=re.findall('"type":"(.*?)"',line)[0]
                                    time_rec=re.findall('"timestampNanos":(.*?),',line)[0]
                                    time_rec=int(time_rec) 
                                    subjectId=subject_uuid2hash[subject_uuid[0]]
                                    if predicateObject_uuid[0] in file_uuid2hash:
                                        objectId=file_uuid2hash[predicateObject_uuid[0]]
                                    else:
                                        objectId=net_uuid2hash[predicateObject_uuid[0]]
        #                                 print(line)
                                    edge_type.add(relation_type)
                                    if relation_type in reverse:
                                        datalist.append([objectId,nodeid2msg[objectId],relation_type,subjectId,nodeid2msg[subjectId],time_rec])
                                    else:
                                        datalist.append([subjectId,nodeid2msg[subjectId],relation_type,objectId,nodeid2msg[objectId],time_rec])

        logging.info(f"Prepared {len(datalist)} rows for insert into event_table")
        
        bulk_insert(conn, 'event_table', datalist, batch_size=100000)
    else:
        logging.warning("skipping event parsing...")


    # -----------------------------------------------------------------------------------------------
    # -------------------------------------- featurization ------------------------------------------
    #

    # Initialize FeatureHasher for string and dictionary inputs
    FH_string=FeatureHasher(n_features=16,input_type="string")
    FH_dict=FeatureHasher(n_features=16,input_type="dict")

    def path2higlist(p):
        """Convert a file path into a hierarchical list."""
        l=[]
        spl=p.strip().split('/')
        for i in spl:
            if len(l)!=0:
                l.append(l[-1]+'/'+i)
            else:
                l.append(i)
    #     print(l)
        return l

    def ip2higlist(p):
        """Convert an IP address into a hierarchical list."""
        l=[]
        spl=p.strip().split('.')
        for i in spl:
            if len(l)!=0:
                l.append(l[-1]+'.'+i)
            else:
                l.append(i)
    #     print(l)
        return l

    def subject2higlist(p):
        """Convert a subject string into a hierarchical list."""
        l=[]
        spl=p.strip().split('/')
        for i in spl:
            if len(l)!=0:
                l.append(l[-1]+'/'+i)
            else:
                l.append(i)
    #     print(l)
        return l

    def list2str(l):
        """Convert a list into a single string by concatenating elements."""
        s=''
        for i in l:
            s+=i
        return s

    # Initialize lists for node message vectors and dictionaries
    node_msg_vec=[]
    node_msg_dic_list=[]

    # Process each node ID in nodeid2msg
    for i in tqdm(nodeid2msg.keys()):
        if type(i)==int:                                # ensure key is an int
            # Check for 'netflow' key and build hierarchical list
            if 'netflow' in nodeid2msg[i].keys():
                higlist=['netflow']
                higlist+=ip2higlist(nodeid2msg[i]['netflow'])
            
            # Check for 'file' key and build hierarchical list
            if 'file' in nodeid2msg[i].keys():
                higlist=['file']
                higlist+=path2higlist(nodeid2msg[i]['file'])

            # Check for 'subject' key and build hierarchical list
            if 'subject' in nodeid2msg[i].keys():
                higlist=['subject']
                higlist+=subject2higlist(nodeid2msg[i]['subject'])
    
            # Convert the hierarchical list to string and append to the message dictionary list
            node_msg_dic_list.append(list2str(higlist))

    # Initialize a list to store hierarchical vectors for nodes
    node2higvec=[]

    # Convert each message dictionary to a high-dimensional vector
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)

    # Reshape the node vector array
    node2higvec=np.array(node2higvec).reshape([-1,16])

    # Define relation to ID mapping
    rel2id = {
        1: 'EVENT_CONNECT', 'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE', 'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN', 'EVENT_OPEN': 3,
        4: 'EVENT_READ', 'EVENT_READ': 4,
        5: 'EVENT_RECVFROM', 'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG', 'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG', 'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO', 'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE', 'EVENT_WRITE': 9
    }
    
    # Generate edge type one-hot
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)

    # Map different relation types to their one-hot encoding
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:                  # skip int keys
            rel2vec[i]= relvec[rel2id[i]-1]     # map relation type to vector
            rel2vec[relvec[rel2id[i]-1]]=i      # reverse mapping

    # Create the embeddings directory if it does not exist
    if not os.path.exists(DEFAULT_EMB_DIR):
        os.makedirs(DEFAULT_EMB_DIR)
        logging.info(f"Created directory: {DEFAULT_EMB_DIR}")

    # save the results (embeddings) to files
    torch.save(node2higvec, DEFAULT_EMB_DIR + "node2higvec")
    torch.save(rel2vec, DEFAULT_EMB_DIR + "rel2vec")

    # -----------------------------------------------------------------------------------------------


# --------------------------------- CLI -----------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="THEIA E5 preprocessing → Postgres → TemporalData graphs")

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
    p.add_argument("--skip-netflows", action="store_true", help="Skip parsing/inserting netflow data")
    p.add_argument("--skip-processes", action="store_true", help="Skip parsing/inserting process data")
    p.add_argument("--skip-files", action="store_true", help="Skip parsing/inserting file data ")
    p.add_argument("--skip-node2ids", action="store_true", help="Skip parsing/inserting node2id data ")
    p.add_argument("--skip-events", action="store_true", help="Skip parsing/inserting event data ")

    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    p.add_argument(
        "-r", "--reset",
        action="store_true",
        help="Reset (truncate) all known tables in the database before processing"
    )

    return p

if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
