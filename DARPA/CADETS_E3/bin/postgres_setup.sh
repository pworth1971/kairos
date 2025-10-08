#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Script: setup_postgres16.sh
# Purpose: Install and configure PostgreSQL 16 on Ubuntu (with or without systemd)
#           and macOS (Homebrew)
# - Installs PostgreSQL
# - Starts the service via systemd, pg_ctlcluster, or brew as appropriate
# - Configures pg_hba.conf and postgresql.conf
# - Creates database, user, and CADETS schema
# ------------------------------------------------------------------------------

# === Customize these ===
DB_USER="${DB_USER:-appuser}"
DB_PASS="${DB_PASS:-changeme-strong}"
DB_NAME="${DB_NAME:-tc_cadet_dataset_db}"
DB_PORT="${DB_PORT:-5432}"
LISTEN_ALL="${LISTEN_ALL:-true}"        # true to listen on all interfaces
ENABLE_UFW="${ENABLE_UFW:-false}"       # true to open 5432 in UFW

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

# ------------------------------------------------------------------------------
# macOS PATH: Use brew installation
# ------------------------------------------------------------------------------
if [[ "$OS" == "darwin" ]]; then
  echo "[*] Detected macOS system."
  echo "[*] Ensuring Homebrew PostgreSQL@16 is installed..."
  brew list postgresql@16 >/dev/null 2>&1 || brew install postgresql@16

  echo "[*] Starting PostgreSQL service (brew)..."
  brew services start postgresql@16 || true

  PGDATA="$(brew --prefix)/var/postgresql@16"
  export PATH="$(brew --prefix)/opt/postgresql@16/bin:$PATH"

  echo "[*] Waiting for PostgreSQL to become ready..."
  sleep 3

  echo "[*] Creating database and role if missing..."
  psql -U postgres -h localhost -p "$DB_PORT" -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 \
    || psql -U postgres -h localhost -p "$DB_PORT" -d postgres -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

  psql -U postgres -h localhost -p "$DB_PORT" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 \
    || psql -U postgres -h localhost -p "$DB_PORT" -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

  echo "[*] macOS PostgreSQL@16 setup complete."

# ------------------------------------------------------------------------------
# Linux PATH: Ubuntu container (no systemd)
# ------------------------------------------------------------------------------
elif [[ -f /etc/debian_version ]]; then
  CODENAME="$(lsb_release -cs)"
  echo "[*] Detected Ubuntu/Debian system."

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

  PG_CONF_DIR="/etc/postgresql/16/main"
  PG_CONF="${PG_CONF_DIR}/postgresql.conf"
  PG_HBA="${PG_CONF_DIR}/pg_hba.conf"

  # Check if systemd is available
  if pidof systemd >/dev/null 2>&1; then
    echo "[*] Starting PostgreSQL via systemd..."
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
  else
    echo "[*] systemd not available — using pg_ctlcluster..."
    sudo pg_ctlcluster 16 main start || true
  fi

  echo "[*] Configuring PostgreSQL..."
  sudo sed -i "s/^[# ]*port *= *.*/port = ${DB_PORT}/" "${PG_CONF}"
  if [[ "${LISTEN_ALL}" == "true" ]]; then
    sudo sed -i "s/^[# ]*listen_addresses *= *.*/listen_addresses = '*'/;" "${PG_CONF}"
  fi

  echo "[*] Configuring pg_hba.conf authentication..."
  if ! grep -q "0.0.0.0/0" "${PG_HBA}"; then
    echo "host    all             all             0.0.0.0/0               scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
  fi
  if ! grep -q "::/0" "${PG_HBA}"; then
    echo "host    all             all             ::/0                    scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
  fi

  echo "[*] Reloading configs..."
  if pidof systemd >/dev/null 2>&1; then
    sudo systemctl reload postgresql
  else
    sudo pg_ctlcluster 16 main reload
  fi

  echo "[*] Creating user and database..."
  sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}';" | grep -q 1 \
    || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

  sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}';" | grep -q 1 \
    || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

  echo "[*] Creating schema for ${DB_NAME}..."
  sudo -u postgres psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'EOSQL'
CREATE TABLE IF NOT EXISTS event_table (
    src_node varchar, src_index_id varchar, operation varchar,
    dst_node varchar, dst_index_id varchar, timestamp_rec bigint,
    _id serial PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL, hash_id varchar NOT NULL, path varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid varchar NOT NULL, hash_id varchar NOT NULL,
    src_addr varchar, src_port varchar, dst_addr varchar, dst_port varchar,
    CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar, hash_id varchar, exec varchar
);
CREATE TABLE IF NOT EXISTS node2id (
    hash_id varchar NOT NULL PRIMARY KEY,
    node_type varchar, msg varchar, index_id bigint
);
EOSQL

  echo "[*] PostgreSQL 16 setup complete."

else
  echo "[!] Unsupported platform. Please run on macOS or Ubuntu/Debian."
  exit 1
fi

# ------------------------------------------------------------------------------
# Verification and summary
# ------------------------------------------------------------------------------
echo "[*] Verifying local connection..."
if [[ "$OS" == "darwin" ]]; then
  psql -h localhost -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT current_database(), current_user;" || true
else
  sudo -u postgres psql -p "$DB_PORT" -c "SELECT current_database(), current_user;" || true
fi

cat <<EOF

------------------------------------------------------------
✅ PostgreSQL 16 with tc_cadet_dataset_db schema is ready.

Connection details:
  Host:        $(hostname -I 2>/dev/null | awk '{print $1}') (or 127.0.0.1)
  Port:        ${DB_PORT}
  Database:    ${DB_NAME}
  User:        ${DB_USER}
  Password:    ${DB_PASS}

To connect manually:
  PGPASSWORD=${DB_PASS} psql -h localhost -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME}

------------------------------------------------------------
EOF
