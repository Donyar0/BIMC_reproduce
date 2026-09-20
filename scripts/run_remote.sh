#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"
SESSION="${3:-bimc_exp}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-bimc}"
shift $(( $# < 3 ? $# : 3 ))

if [ "$#" -eq 0 ]; then
  set -- python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml
fi

./scripts/sync_to_server.sh "$REMOTE" "$REMOTE_DIR"

REMOTE_CMD=$(printf "%q " "$@")
if [ -n "$REMOTE_CONDA_ENV" ]; then
  REMOTE_CMD="conda run -n $(printf "%q" "$REMOTE_CONDA_ENV") --no-capture-output $REMOTE_CMD"
fi

CONDA_INIT='if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then . ~/miniconda3/etc/profile.d/conda.sh; elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then . ~/anaconda3/etc/profile.d/conda.sh; elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then . /opt/conda/etc/profile.d/conda.sh; elif [ -x ~/miniconda3/bin/conda ]; then export PATH=~/miniconda3/bin:$PATH; elif [ -x ~/anaconda3/bin/conda ]; then export PATH=~/anaconda3/bin:$PATH; fi'

ssh "$REMOTE" "cd $REMOTE_DIR && mkdir -p logs && $CONDA_INIT && if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s $SESSION '$REMOTE_CMD 2>&1 | tee -a logs/$SESSION.log'; else nohup bash -lc '$CONDA_INIT && $REMOTE_CMD' > logs/$SESSION.log 2>&1 & echo \$! > logs/$SESSION.pid; fi"

echo "Started '$SESSION' on $REMOTE"
echo "Attach if tmux is available: ssh $REMOTE 'tmux attach -t $SESSION'"
echo "If nohup was used, PID file: ${REMOTE_DIR}/logs/${SESSION}.pid"
echo "Log file: ${REMOTE_DIR}/logs/${SESSION}.log"
echo "Conda env: ${REMOTE_CONDA_ENV:-disabled}"
