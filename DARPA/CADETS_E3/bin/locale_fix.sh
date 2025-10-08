#!/usr/bin/env bash
set -euo pipefail

# Prefer setting LANG/LC_CTYPE; only set LC_ALL if you truly need to override all LC_*.
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
# export LC_ALL=en_US.UTF-8   # optional; usually leave unset

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y locales

# Ensure en_US.UTF-8 is enabled in /etc/locale.gen (idempotent)
sed -i 's/^[#[:space:]]*en_US\.UTF-8[[:space:]]\+UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen

# Generate the locale
locale-gen en_US.UTF-8

# Set system defaults (affects future login shells)
update-locale LANG=en_US.UTF-8 LC_CTYPE=en_US.UTF-8
# If you really want LC_ALL system-wide, uncomment:
# update-locale LC_ALL=en_US.UTF-8

# Also export for the current shell/session
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
# export LC_ALL=en_US.UTF-8

