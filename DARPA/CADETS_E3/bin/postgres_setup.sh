#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Script: setup_postgres16.sh
# Purpose: Install and configure PostgreSQL 16 on Ubuntu (20.04+)
# - Installs from PGDG repo
# - Enables/starts service
# - Optionally opens remote access
# - Creates a database user, a database, and the schema for CADETS dataset
# ------------------------------------------------------------------------------

# === Customize these ===
DB_USER="${DB_USER:-appuser}"
DB_PASS="${DB_PASS:-changeme-strong}"
DB_NAME="${DB_NAME:-tc_cadet_dataset_db}"
DB_PORT="${DB_PORT:-5432}"
LISTEN_ALL="${LISTEN_ALL:-true}"        # true to listen on all interfaces
ENABLE_UFW="${ENABLE_UFW:-false}"       # true to open 5432 in UFW

CODENAME="$(lsb_release -cs)"

echo "[*] Installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y curl gnupg lsb-release ca-certificates

echo "[*] Adding PostgreSQL (PGDG) APT repository..."
sudo install -d -m 0755 /usr/share/keyrings
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/postgres.gpg
echo "deb [signed-by=/usr/share/keyrings/postgres.gpg] http://apt.postgresql.org/pub/repos/apt ${CODENAME}-pgdg main" \
  | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null

echo "[*] Installing PostgreSQL 16 server and client..."
sudo apt-get update -y
sudo apt-get install -y postgresql-16 postgresql-client-16

echo "[*] Ensuring service is enabled and started..."
sudo systemctl enable postgresql
sudo systemctl start postgresql

PG_CONF_DIR="/etc/postgresql/16/main"
PG_CONF="${PG_CONF_DIR}/postgresql.conf"
PG_HBA="${PG_CONF_DIR}/pg_hba.conf"

echo "[*] Configuring postgresql.conf (port=${DB_PORT})..."
sudo sed -i "s/^[# ]*port *= *.*/port = ${DB_PORT}/" "${PG_CONF}"

if [[ "${LISTEN_ALL}" == "true" ]]; then
  echo "[*] Enabling listen_addresses='*'..."
  sudo sed -i "s/^[# ]*listen_addresses *= *.*/listen_addresses = '*'/;" "${PG_CONF}"
fi

echo "[*] Configuring pg_hba.conf authentication..."
if ! grep -q "0.0.0.0/0" "${PG_HBA}"; then
  echo "host    all             all             0.0.0.0/0               scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
fi
if ! grep -q "::/0" "${PG_HBA}"; then
  echo "host    all             all             ::/0                    scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
fi

echo "[*] Reloading PostgreSQL configs..."
sudo systemctl reload postgresql

echo "[*] Creating role and database..."
sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}';" | grep -q 1 \
  || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}';" | grep -q 1 \
  || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

echo "[*] Creating schema in ${DB_NAME}..."

sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'EOSQL'
-- Event table
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial PRIMARY KEY
);
ALTER TABLE event_table OWNER TO postgres;
CREATE UNIQUE INDEX IF NOT EXISTS event_table__id_uindex ON event_table (_id);

-- File node table
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

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
ALTER TABLE netflow_node_table OWNER TO postgres;

-- Subject node table
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO postgres;

-- node2id table
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL CONSTRAINT node2id_pk PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO postgres;
CREATE UNIQUE INDEX IF NOT EXISTS node2id_hash_id_uindex ON node2id (hash_id);
EOSQL

# Optional firewall opening
if [[ "${ENABLE_UFW}" == "true" ]]; then
  if command -v ufw >/dev/null 2>&1; then
    echo "[*] Opening ${DB_PORT}/tcp in UFW..."
    sudo ufw allow "${DB_PORT}/tcp" || true
  fi
fi

echo "[*] Verifying local connection..."
sudo -u postgres psql -p "${DB_PORT}" -c "SELECT current_database(), current_user;" >/dev/null

cat <<EOF

------------------------------------------------------------
✅ PostgreSQL 16 with tc_cadet_dataset_db schema is ready.

Connection details:
  Host:        $(hostname -I | awk '{print $1}') (or 127.0.0.1)
  Port:        ${DB_PORT}
  Database:    ${DB_NAME}
  User:        ${DB_USER}
  Password:    ${DB_PASS}

To connect:
  PGPASSWORD=${DB_PASS} psql -h localhost -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME}

Schema tables created:
  - event_table
  - file_node_table
  - netflow_node_table
  - subject_node_table
  - node2id
------------------------------------------------------------
EOF
