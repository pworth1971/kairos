#!/bin/bash
# db_check.sh — show row counts and sizes for all tables in each listed database

USER="postgres"
HOST="localhost"
PORT="5432"

DATABASES=("cadets_e3" "clearscope_e3" "optc_051" "optc_201" "optc_501" "theia_e3")

for DB in "${DATABASES[@]}"; do
  echo "==============================="
  echo "📊 Row counts for database: $DB"
  echo "==============================="

  psql -U "$USER" -h "$HOST" -p "$PORT" -d "$DB" -X -q -t -c "
    SELECT
      c.relname AS table_name,
      pg_size_pretty(pg_total_relation_size(c.oid)) AS size,
      c.reltuples::bigint AS estimated_rows
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
    ORDER BY estimated_rows DESC;
  " | column -t

  echo ""
done
