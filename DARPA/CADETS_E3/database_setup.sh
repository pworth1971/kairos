#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Script: setup_tc_cadet_dataset_db.sh
# Purpose: Connect to system 'postgres' database, create tc_cadet_dataset_db
#          and the required tables.
# Works on macOS (Homebrew PostgreSQL) and Linux (system PostgreSQL)
# ------------------------------------------------------------------------------

# === Configuration ===
SYSTEM_DB="${SYSTEM_DB:-postgres}"             # default system DB
TARGET_DB="${TARGET_DB:-tc_cadet_dataset_db}"  # name of database to create
DB_USER="${DB_USER:-postgres}"                 # database owner
DB_PORT="${DB_PORT:-5432}"                     # PostgreSQL port

# --- Detect environment (macOS vs Linux) ---
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "[*] Detected macOS environment (Homebrew PostgreSQL)..."
  PSQL_CMD="psql"
else
  echo "[*] Detected Linux environment (system PostgreSQL)..."
  if id "postgres" &>/dev/null; then
    PSQL_CMD="sudo -u postgres psql"
  else
    echo "[!] System user 'postgres' not found. Falling back to direct psql."
    PSQL_CMD="psql"
  fi
fi

# --- Check connection to system DB ---
echo "[*] Connecting to system database '${SYSTEM_DB}'..."
$PSQL_CMD -d "${SYSTEM_DB}" -p "${DB_PORT}" -v ON_ERROR_STOP=1 -c "\conninfo" >/dev/null

# --- Create target database if missing ---
echo "[*] Checking if target database '${TARGET_DB}' exists..."
DB_EXISTS=$($PSQL_CMD -d "${SYSTEM_DB}" --tuples-only --no-align -c "SELECT 1 FROM pg_database WHERE datname='${TARGET_DB}';" | tr -d ' ')
if [[ "${DB_EXISTS}" != "1" ]]; then
  echo "[*] Creating target database '${TARGET_DB}' owned by ${DB_USER}..."
  $PSQL_CMD -d "${SYSTEM_DB}" -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${TARGET_DB} OWNER ${DB_USER};"
else
  echo "[*] Database '${TARGET_DB}' already exists. Skipping creation."
fi

# --- Create tables inside tc_cadet_dataset_db ---
echo "[*] Creating tables inside '${TARGET_DB}'..."

$PSQL_CMD -d "${TARGET_DB}" -v ON_ERROR_STOP=1 <<EOSQL
-- Event table
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
ALTER TABLE event_table OWNER TO ${DB_USER};
CREATE UNIQUE INDEX IF NOT EXISTS event_table__id_uindex ON event_table (_id);
GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO ${DB_USER};

-- File node table
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO ${DB_USER};

-- Netflow node table
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO ${DB_USER};

-- Subject node table
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO ${DB_USER};

-- node2id table
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL CONSTRAINT node2id_pk PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO ${DB_USER};
CREATE UNIQUE INDEX IF NOT EXISTS node2id_hash_id_uindex ON node2id (hash_id);
EOSQL

echo "[*] Table creation complete!"

cat <<EOF

------------------------------------------------------------
✅ PostgreSQL setup complete

System DB: ${SYSTEM_DB}
Target DB: ${TARGET_DB}
Owner:     ${DB_USER}
Port:      ${DB_PORT}

Tables created:
  - event_table
  - file_node_table
  - netflow_node_table
  - subject_node_table
  - node2id

To connect manually:
  psql -d ${TARGET_DB}
  \dt        # list tables
------------------------------------------------------------
EOF
