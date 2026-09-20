#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"
ENV_NAME="${3:-bimc}"
PYTHON_VERSION="${4:-3.9}"

./scripts/sync_to_server.sh "$REMOTE" "$REMOTE_DIR"

ssh "$REMOTE" "cd $REMOTE_DIR && \
  if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then . ~/miniconda3/etc/profile.d/conda.sh; \
  elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then . ~/anaconda3/etc/profile.d/conda.sh; \
  elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then . /opt/conda/etc/profile.d/conda.sh; \
  elif [ -x ~/miniconda3/bin/conda ]; then export PATH=~/miniconda3/bin:\$PATH; \
  elif [ -x ~/anaconda3/bin/conda ]; then export PATH=~/anaconda3/bin:\$PATH; \
  fi && \
  if ! command -v conda >/dev/null 2>&1; then \
    echo 'conda not found on remote PATH. Try: ssh $REMOTE \"find ~ -maxdepth 3 -name conda -type f\"' >&2; \
    exit 1; \
  fi && \
  if conda env list | awk '{print \$1}' | grep -qx '$ENV_NAME'; then \
    echo 'Conda env $ENV_NAME already exists'; \
  else \
    conda create -n '$ENV_NAME' python='$PYTHON_VERSION' -y; \
  fi && \
  conda run -n '$ENV_NAME' python -m pip install -r requirements.txt"

echo "Remote conda env '$ENV_NAME' is ready on $REMOTE"
