#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".vscode/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "data/" \
  --exclude "datasets/*/" \
  --exclude "outputs/" \
  --exclude "checkpoints/" \
  --exclude "logs/" \
  --rsync-path="mkdir -p ${REMOTE_DIR} && rsync" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "Synced to ${REMOTE}:${REMOTE_DIR}"
