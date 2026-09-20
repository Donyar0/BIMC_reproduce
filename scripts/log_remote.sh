#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"
SESSION="${3:-bimc_cifar}"

ssh "$REMOTE" "tail -f ${REMOTE_DIR}/logs/${SESSION}.log"
