#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Create THEIA DB + tables + indexes (idempotent)
# Defaults can be overridden via env vars:
#   DB_NAME=tc_e5_theia_dataset_db DB_OWNER=postgres ./create_theia_db.sh
# ============================================================

DB_NAME="${DB_NAME:-tc_e5_theia_dataset_db}"
DB_OWNER="${DB_OWNER:-postgres}"

# --- Helpers to run psql/createdb as 'postgres' regardless of caller ----
run_psql() {
  if [ "$(id -un)" = "postgres" ]; then
    psql "$@"
  else
    sudo -u postgres psql "$@"
  fi
}

run_createdb() {
  if [ "$(id -un)" = "postgres" ]; then
    createdb "$@"
  else
    sudo -u postgres createdb "$@"
  fi
}

echo "==> Ensuring database '${DB_NAME}' exists (owner: ${DB_OWNER}) ..."
if run_psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  echo "    Database already exists."
else
  run_createdb -O "${DB_OWNER}" "${DB_NAME}"
  echo "    Database created."
fi

echo "==> Applying schema to '${DB_NAME}' ..."
run_psql -v ON_ERROR_STOP=1 -d "${DB_NAME}" <<'SQL'
-- ===================== event_table ======================
CREATE TABLE IF NOT EXISTS event_table (
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
ALTER TABLE event_table OWNER TO postgres;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE c.relname = 'event_table__id_uindex'
       AND c.relkind = 'i'
  ) THEN
    CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
  END IF;
END $$;

GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO postgres;

-- ===================== file_node_table ===================
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid varchar NOT NULL,
    hash_id   varchar NOT NULL,
    path      varchar,
    CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

-- ===================== netflow_node_table =================
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

-- ===================== subject_node_table =================
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
ALTER TABLE subject_node_table OWNER TO postgres;

-- ===================== node2id ============================
CREATE TABLE IF NOT EXISTS node2id (
    hash_id   varchar NOT NULL PRIMARY KEY,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
ALTER TABLE node2id OWNER TO postgres;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE c.relname = 'node2id_hash_id_uindex'
       AND c.relkind = 'i'
  ) THEN
    CREATE UNIQUE INDEX node2id_hash_id_uindex ON node2id (hash_id);
  END IF;
END $$;
SQL

echo "✅ Done. Schema ensured in database '${DB_NAME}'."
