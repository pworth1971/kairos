#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Script: setup_postgres18.sh
# Purpose: Install and configure PostgreSQL 18 on Ubuntu (20.04+)
# - Installs from PGDG repo
# - Enables/starts service
# - (Optional) Opens remote access on 0.0.0.0:5432
# - Creates a database user and database
# ------------------------------------------------------------------------------

# === Customize these ===
DB_USER="${DB_USER:-appuser}"
DB_PASS="${DB_PASS:-changeme-strong}"
DB_NAME="${DB_NAME:-appdb}"
DB_PORT="${DB_PORT:-5432}"
LISTEN_ALL="${LISTEN_ALL:-true}"        # true to listen on all interfaces
ENABLE_UFW="${ENABLE_UFW:-false}"       # true to open 5432 in UFW

# --- Detect distro codename (focal, jammy, noble, etc.)
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

echo "[*] Installing PostgreSQL 18 server and client..."
sudo apt-get update -y
sudo apt-get install -y postgresql-18 postgresql-client-18

echo "[*] Ensuring service is enabled and started..."
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Paths for the default cluster
PG_CONF_DIR="/etc/postgresql/18/main"
PG_CONF="${PG_CONF_DIR}/postgresql.conf"
PG_HBA="${PG_CONF_DIR}/pg_hba.conf"

echo "[*] Configuring postgresql.conf (port=${DB_PORT})..."
sudo sed -i "s/^[# ]*port *= *.*/port = ${DB_PORT}/" "${PG_CONF}"

if [[ "${LISTEN_ALL}" == "true" ]]; then
  echo "[*] Enabling listen_addresses='*' for remote access..."
  sudo sed -i "s/^[# ]*listen_addresses *= *.*/listen_addresses = '*'/;" "${PG_CONF}"
fi

echo "[*] Configuring pg_hba.conf authentication rules..."
# Prefer scram-sha-256 (default in modern Postgres); add IPv4/IPv6 lines if not present.
if ! grep -qE "^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+0\.0\.0\.0/0" "${PG_HBA}"; then
  echo "host    all             all             0.0.0.0/0               scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
fi
if ! grep -qE "^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+::/0" "${PG_HBA}"; then
  echo "host    all             all             ::/0                    scram-sha-256" | sudo tee -a "${PG_HBA}" >/dev/null
fi

echo "[*] Reloading PostgreSQL to apply config..."
sudo systemctl reload postgresql

# Create role and database (idempotent)
echo "[*] Creating database role and database (if not exist)..."
sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}';" | grep -q 1 \
  || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASS}';"

sudo -u postgres psql --tuples-only --no-align -c "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}';" | grep -q 1 \
  || sudo -u postgres psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

# Optional UFW rule
if [[ "${ENABLE_UFW}" == "true" ]]; then
  if command -v ufw >/dev/null 2>&1; then
    echo "[*] Opening TCP ${DB_PORT}/tcp in UFW..."
    sudo ufw allow "${DB_PORT}/tcp" || true
  else
    echo "[!] UFW not installed; skipping firewall rule."
  fi
fi

# Connection test (local)
echo "[*] Verifying local connectivity..."
sudo -u postgres psql -p "${DB_PORT}" -c "SELECT version();" >/dev/null

cat <<EOF

------------------------------------------------------------
✅ PostgreSQL 18 is installed and configured.

Connection details:
  Host:        $(hostname -I | awk '{print $1}') (or 127.0.0.1)
  Port:        ${DB_PORT}
  Database:    ${DB_NAME}
  User:        ${DB_USER}
  Password:    ${DB_PASS}

Local test:
  sudo -u postgres psql -p ${DB_PORT} -d ${DB_NAME} -U ${DB_USER}

Files:
  ${PG_CONF}
  ${PG_HBA}

To apply future config changes:
  sudo systemctl reload postgresql
To check status/logs:
  sudo systemctl status postgresql
  sudo journalctl -u postgresql -f
------------------------------------------------------------
EOF
