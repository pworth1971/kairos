import os
import re
import torch
import hashlib
from tqdm import tqdm
import psycopg2
from psycopg2 import extras as ex

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Connection settings for PostgreSQL 16 (macOS/Homebrew or Linux)
PG_CONFIG = {
    "dbname": "tc_cadet_dataset_db",                # target database
    "user": "postgres",                             # owner user (default for your setup)
    "password": "Rafter9876!@",                     # optional, leave blank for local trust
    "host": "localhost",
    "port": 5432
}


from pathlib import Path


# Path to your raw JSON files
RAW_DIR = "./data/"                 

# Path containing your decoded CADETS json shards
DATA_DIR = Path(RAW_DIR)   

# Dynamically gather all *.json files (non-recursive)
filelist = sorted(
    f.name for f in DATA_DIR.iterdir()
    if f.is_file() and f.suffix == ".json"
)

print(f"[INFO] Found {len(filelist)} JSON files in {DATA_DIR}:")
for name in filelist:
    print("   ", name)



# Reverse edge types if applicable (as per your original logic)
edge_reversed = set()



# ------------------------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------------------------

def init_database_connection():
    """Initialize PostgreSQL connection using psycopg2"""
    try:
        connect = psycopg2.connect(**PG_CONFIG)
        connect.autocommit = False
        cur = connect.cursor()
        print(f"[+] Connected to PostgreSQL: {PG_CONFIG['dbname']} on {PG_CONFIG['host']}:{PG_CONFIG['port']}")
        return cur, connect
    except Exception as e:
        raise SystemExit(f"[!] Database connection failed: {e}")


# ------------------------------------------------------------------------------
# DATABASE RESET
# ------------------------------------------------------------------------------

def clear_database(cur, connect):
    """Delete all rows from all relevant tables before re-ingestion."""
    tables = [
        "event_table",
        "file_node_table",
        "subject_node_table",
        "netflow_node_table",
        "node2id"
    ]
    print("\n⚠️  Clearing all existing rows from database tables...")

    for table in tables:
        try:
            cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
            connect.commit()
            print(f"   - Cleared: {table}")
        except Exception as e:
            connect.rollback()
            print(f"[!] Failed to clear {table}: {e}")

    print("✅ Database tables successfully reset.\n")
    
    
    
# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def stringtomd5(originstr):
    """Return SHA-256 hash of string (for consistency with previous pipeline)"""
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()


# ------------------------------------------------------------------------------
# INGESTION FUNCTIONS
# ------------------------------------------------------------------------------


# Files to process
"""
filelist = [
    'ta1-cadets-e3-official.bin.json',
    'ta1-cadets-e3-official.bin.json.1',
    'ta1-cadets-e3-official.bin.json.2',
    'ta1-cadets-e3-official-1.bin.json',
    'ta1-cadets-e3-official-1.bin.json.1',
    'ta1-cadets-e3-official-1.bin.json.2',
    'ta1-cadets-e3-official-1.bin.json.3',
    'ta1-cadets-e3-official-1.bin.json.4',
    'ta1-cadets-e3-official-2.bin.json',
    'ta1-cadets-e3-official-2.bin.json.1'
]
"""


def store_netflow(file_path, cur, connect):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        hashstr = stringtomd5(nodeproperty)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(","))

    sql = '''insert into netflow_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()



def store_subject(file_path, cur, connect):
    # Parse data from logs
    scusess_count = 0
    fail_count = 0
    subject_objset = set()
    subject_obj2hash = {}  #
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "Event" in line:
                    subject_uuid = re.findall(
                        '"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"', line)
                    try:
                        subject_obj2hash[subject_uuid[0][0]] = subject_uuid[0][-1]
                        scusess_count += 1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]] = "null"
                        except:
                            pass
                        fail_count += 1
    # Store into database
    datalist = []
    for i in subject_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(subject_obj2hash[i]), subject_obj2hash[i]])
    sql = '''insert into subject_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()


def store_file(file_path, cur, connect):
    file_node = set()
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                    Object_uuid = re.findall('FileObject":{"uuid":"(.*?)",', line)
                    try:
                        file_node.add(Object_uuid[0])
                    except:
                        print(line)

    file_obj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}',
                                                      line)
                    if len(predicateObject_uuid) > 0:
                        if predicateObject_uuid[0] in file_node:
                            if '"predicateObjectPath":null,' not in line and '<unknown>' not in line:
                                path_name = re.findall('"predicateObjectPath":{"string":"(.*?)"', line)
                                file_obj2hash[predicateObject_uuid[0]] = path_name

    datalist = []
    for i in file_obj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [stringtomd5(file_obj2hash[i][0]), file_obj2hash[i][0]])
    sql = '''insert into file_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()
    
    

def create_node_list(cur, connect):
    node_list = {}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        node_list[i[1]] = ["file", i[-1]]
    file_uuid2hash = {}
    for i in records:
        file_uuid2hash[i[0]] = i[1]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["subject", i[-1]]
    subject_uuid2hash = {}
    for i in records:
        subject_uuid2hash[i[0]] = i[1]

    # netflow
    sql = """
    select * from netflow_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["netflow", i[-2] + ":" + i[-1]]

    net_uuid2hash = {}
    for i in records:
        net_uuid2hash[i[0]] = i[1]

    node_list_database = []
    node_index = 0
    for i in node_list:
        node_list_database.append([i] + node_list[i] + [node_index])
        node_index += 1

    sql = '''insert into node2id
                         values %s
            '''
    ex.execute_values(cur, sql, node_list_database, page_size=10000)
    connect.commit()

    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash):
    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                    subject_uuid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                    if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                        if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in file_uuid2hash or predicateObject_uuid[0] in net_uuid2hash):
                            relation_type = re.findall('"type":"(.*?)"', line)[0]
                            time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                            time_rec = int(time_rec)
                            subjectId = subject_uuid2hash[subject_uuid[0]]
                            if predicateObject_uuid[0] in file_uuid2hash:
                                objectId = file_uuid2hash[predicateObject_uuid[0]]
                            else:
                                objectId = net_uuid2hash[predicateObject_uuid[0]]
                            if relation_type in reverse:
                                datalist.append(
                                    [objectId, nodeid2msg[objectId], relation_type, subjectId, nodeid2msg[subjectId],
                                     time_rec])
                            else:
                                datalist.append(
                                    [subjectId, nodeid2msg[subjectId], relation_type, objectId, nodeid2msg[objectId],
                                     time_rec])

    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()



# --------------------------------------------------------------------------
#  Database statistics printer
# --------------------------------------------------------------------------
def show_database_statistics():
    """Print row counts for all tables and event_table breakdown by operation."""
    cur, connect = init_database_connection()

    print("\n=== Database Table Row Counts ===")
    tables = [
        "event_table",
        "file_node_table",
        "subject_node_table",
        "netflow_node_table",
        "node2id"
    ]

    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            count = cur.fetchone()[0]
            print(f"  {table:25s}: {count:,}")
        except Exception as e:
            print(f"  [ERROR] Could not count {table}: {e}")

    print("\n=== event_table Breakdown by Operation (alphabetical) ===")
    try:
        cur.execute("""
            SELECT operation, COUNT(*) AS count
            FROM event_table
            GROUP BY operation
            ORDER BY operation ASC;
        """)
        rows = cur.fetchall()
        if rows:
            for op, count in rows:
                print(f"  {op:25s}: {count:,}")
        else:
            print("  [!] No events found in event_table.")
    except Exception as e:
        print(f"  [ERROR] Could not query event_table breakdown: {e}")

    connect.close()
    print("\n📊 Database statistics collection complete.")
    
    
    

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="CADETS E3 database loader and statistics tool.")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show database table counts and event_table breakdown without ingestion.")
    args = parser.parse_args()


    if args.show:
        show_database_statistics()
    else:
        cur, connect = init_database_connection()

        # 🚨 Clear the database first
        print("\n=== Clearing CADETS E3 Dataset ===")
        clear_database(cur, connect)

        print("\n=== Ingesting CADETS E3 Dataset ===")
        
        # There will be 155322 netflow nodes stored in the table
        print("Processing netflow data")
        store_netflow(file_path=RAW_DIR, cur=cur, connect=connect)
        
        # There will be 224146 subject nodes stored in the table
        print("Processing subject data")
        store_subject(file_path=RAW_DIR, cur=cur, connect=connect)
        
        # There will be 234245 file nodes stored in the table
        print("Processing file data")
        store_file(file_path=RAW_DIR, cur=cur, connect=connect)
        
        # There will be 268242 entities stored in the table
        print("Extracting the node list")
        nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash = create_node_list(cur=cur, connect=connect)
        
        # There will be 29727441 events stored in the table
        print("Processing the events")
        store_event(
            file_path=RAW_DIR,
            cur=cur,
            connect=connect,
            reverse=edge_reversed,
            nodeid2msg=nodeid2msg,
            subject_uuid2hash=subject_uuid2hash,
            file_uuid2hash=file_uuid2hash,
            net_uuid2hash=net_uuid2hash
        )
        connect.close()
        
        print("\n✅ Ingestion complete: tc_cadet_dataset_db populated successfully.")
        
        # --------------------------------------------------------------------------
        # ✅ Post-ingestion summary statistics
        # --------------------------------------------------------------------------

        show_database_statistics()

        
        