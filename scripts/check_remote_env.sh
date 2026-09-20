#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-miner@219.224.3.245}"
REMOTE_DIR="${2:-/home/miner/zdy/cloned_repos/BiMC}"

./scripts/sync_to_server.sh "$REMOTE" "$REMOTE_DIR"

ssh "$REMOTE" "cd $REMOTE_DIR && python - <<'PY'
import importlib
import sys

print('python:', sys.executable)
print('version:', sys.version.replace('\n', ' '))

packages = [
    'torch',
    'torchvision',
    'yacs',
    'tqdm',
    'ftfy',
    'regex',
    'sklearn',
]

missing = []
for name in packages:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        missing.append((name, exc))
        print(f'{name}: MISSING ({exc})')
    else:
        print(f'{name}: {getattr(module, \"__version__\", \"installed\")}')

try:
    import torch
except Exception:
    pass
else:
    print('cuda_available:', torch.cuda.is_available())
    print('cuda_device_count:', torch.cuda.device_count())

if missing:
    raise SystemExit(1)
PY"
