#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"
SESSION="${3:-bimc_cifar}"

ssh "$REMOTE" "cd ${REMOTE_DIR} && \
  if [ ! -f logs/${SESSION}.pid ]; then \
    echo 'No PID file for ${SESSION}'; \
    exit 0; \
  fi && \
  PID=\$(cat logs/${SESSION}.pid) && \
  if ps -p \$PID -f; then \
    echo '${SESSION} is running'; \
  else \
    echo '${SESSION} is not running'; \
  fi"
