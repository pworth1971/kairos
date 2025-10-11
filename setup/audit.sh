#!/usr/bin/env bash
# ==============================================================================
# Script: audit.sh
# Purpose: Enable and configure Linux auditd to capture ALL events (systemd or Docker)
# Author: Athena Security Group
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Function helpers
# ------------------------------------------------------------------------------

is_systemd() {
  pidof systemd >/dev/null 2>&1
}

is_root() {
  [ "$(id -u)" -eq 0 ]
}

# ------------------------------------------------------------------------------
# 1. Ensure root privileges
# ------------------------------------------------------------------------------

if ! is_root; then
  echo "[INFO] Elevating privileges..."
  exec sudo bash "$0" "$@"
  exit 0
fi

# ------------------------------------------------------------------------------
# 2. Install audit packages
# ------------------------------------------------------------------------------

echo "[INFO] Installing auditd..."
apt update -y && apt install -y auditd audispd-plugins

# ------------------------------------------------------------------------------
# 3. Start auditd (systemd-aware)
# ------------------------------------------------------------------------------

if is_systemd; then
  echo "[INFO] Starting auditd via systemd..."
  systemctl enable auditd
  systemctl restart auditd
else
  echo "[INFO] Starting auditd manually (non-systemd environment)..."
  if ! /usr/sbin/auditd; then
    echo "[WARN] Could not start auditd automatically, trying fallback..."
    pkill auditd || true
    /usr/sbin/auditd || echo "[ERROR] Auditd failed to start. Check /var/log/audit/."
  fi
fi

# ------------------------------------------------------------------------------
# 4. Create comprehensive audit rules
# ------------------------------------------------------------------------------

AUDIT_RULES_FILE="/etc/audit/rules.d/full_audit.rules"
echo "[INFO] Creating comprehensive audit rules at $AUDIT_RULES_FILE"

cat > "$AUDIT_RULES_FILE" <<'EOF'
## ============================================================================
## Full-system audit rules for comprehensive provenance capture
## ============================================================================

# Audit all syscalls (32-bit and 64-bit)
-a always,exit -F arch=b64 -S all -k syscall_all
-a always,exit -F arch=b32 -S all -k syscall_all

# Process creation/execution
-w /usr/bin/ -p x -k exec_bin
-w /bin/ -p x -k exec_bin
-w /sbin/ -p x -k exec_sbin
-w /usr/sbin/ -p x -k exec_sbin

# Configuration files
-w /etc/ -p wa -k etc_config

# User and group management
-w /etc/passwd -p wa -k user_mgmt
-w /etc/shadow -p wa -k user_mgmt
-w /etc/group -p wa -k user_mgmt
-w /etc/gshadow -p wa -k user_mgmt

# Network and sockets
-w /etc/hosts -p wa -k network
-a always,exit -F arch=b64 -S connect,accept,bind,listen,sendto,recvfrom -k netops

# File read/write/delete
-a always,exit -F arch=b64 -S open,openat,creat,unlink,rename,truncate,ftruncate,chmod,chown -k file_io

# Kernel modules and integrity
-w /sbin/insmod -p x -k kernel_mod
-w /sbin/rmmod -p x -k kernel_mod
-w /sbin/modprobe -p x -k kernel_mod
-a always,exit -F arch=b64 -S init_module,delete_module -k kernel_mod

# Privilege use
-a always,exit -F arch=b64 -S setuid,setgid,setreuid,setregid -k privilege

# Time changes
-a always,exit -F arch=b64 -S adjtimex,settimeofday,clock_settime -k time_change
-w /etc/localtime -p wa -k time_change

# Root and admin actions
-a always,exit -F euid=0 -S all -k root_activity

# Lock audit configuration
-e 1
EOF

# ------------------------------------------------------------------------------
# 5. Load and verify rules
# ------------------------------------------------------------------------------

echo "[INFO] Loading rules..."
augenrules --load || echo "[WARN] augenrules may need manual reload."

# ------------------------------------------------------------------------------
# 6. Display status
# ------------------------------------------------------------------------------

echo "[INFO] Checking auditd status..."
auditctl -s || echo "[WARN] Could not retrieve auditctl status. Check permissions."

echo "
✅ [SUCCESS] Linux auditing fully configured.
Logs: /var/log/audit/audit.log
Useful commands:
  sudo ausearch -ts recent
  sudo aureport --summary
  sudo tail -f /var/log/audit/audit.log
"
