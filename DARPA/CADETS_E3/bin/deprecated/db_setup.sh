#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Script: setup_cadets_e3_schema.sh
# Purpose: Create the tc_cadet_dataset_db database, user, and tables for CADETS E3.
# Works on macOS (brew Postgres) and Ubuntu (systemd or pg_ctlcluster environments).
# ------------------------------------------------------------------------------

DB_NAME="${DB_NAME:-tc_cadet_dataset_db}"
DB_USER="${DB_USER:-appuser}"
DB_PASS="${DB_PASS:-changeme-strong}"
DB_PORT="${DB_PORT:-5432}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

echo "------------------------------------------------------------"
echo "[*] CADETS E3 Database Schema Setup"
echo "------------------------------------------------------------"

# === macOS (brew Postgres) ==========================================
if [[ "$OS" == "darwin" ]]; then
  echo "[*] Detected macOS (Homebrew) environment."
  export PATH="$(brew --prefix)/opt/postgresql@16/bin:$PATH"

  echo "[*] Starting PostgreSQL (brew)..."
  brew services start postgresql@16 || true
  sleep 2

  echo "[*] Creating database user and database if missing..."
  psql -U postgres -h localhost -p "$DB_PORT" -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 \
    || psql -U postgres -h localhost -p "$DB_PORT" -d postgres -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

  psql -U postgres -h localhost -p "$DB_PORT" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 \
    || psql -U postgres -h localhost -p "$DB_PORT" -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

  echo "[*] Creating schema tables..."
  psql -U postgres -h localhost -p "$DB_PORT" -d "$DB_NAME" <<'EOSQL'
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
EOSQL

# === Ubuntu / Debian environments ==================================
elif [[ -f /etc/debian_version ]]; then
  echo "[*] Detected Ubuntu/Debian environment."

  # Start PostgreSQL depending on whether systemd is available
  if pidof systemd >/dev/null 2>&1; then
    echo "[*] Starting PostgreSQL via systemd..."
    sudo systemctl start postgresql
  else
    echo "[*] Starting PostgreSQL via pg_ctlcluster..."
    sudo pg_ctlcluster 16 main start || true
  fi

  echo "[*] Creating user and database..."
  sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 \
    || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

  sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 \
    || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

  echo "[*] Creating schema tables..."
  sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'EOSQL'
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
EOSQL

else
  echo "[!] Unsupported OS — this script supports macOS and Ubuntu/Debian only."
  exit 1
fi

# ------------------------------------------------------------------------------
# Verification summary
# ------------------------------------------------------------------------------
echo "[*] Verifying tables created..."
if [[ "$OS" == "darwin" ]]; then
  psql -h localhost -p "$DB_PORT" -U postgres -d "$DB_NAME" -c "\dt"
else
  sudo -u postgres psql -d "$DB_NAME" -c "\dt"
fi

cat <<EOF

------------------------------------------------------------
✅ CADETS E3 schema created successfully in database: ${DB_NAME}

Connection details:
  Host: localhost
  Port: ${DB_PORT}
  Database: ${DB_NAME}
  User: ${DB_USER}
  Password: ${DB_PASS}

Tables created:
  - event_table
  - file_node_table
  - netflow_node_table
  - subject_node_table
  - node2id
------------------------------------------------------------
