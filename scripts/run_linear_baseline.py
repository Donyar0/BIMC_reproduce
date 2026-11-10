#!/usr/bin/env python3
"""One-run BiMC text calibration evaluation. Defaults: CIFAR100 seed 1, linear.

Examples:
  python scripts/run_linear_baseline.py
  python scripts/run_linear_baseline.py --dataset cub200 --seed 1
  python scripts/run_linear_baseline.py --dataset miniimagenet --seed 1
  python scripts/run_linear_baseline.py --mode original  # paired baseline
No visual description weighting, attention or pseudo-validation.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cifar100": "configs/datasets/cifar100.yaml",
    "cub200": "configs/datasets/cub200.yaml",
    "miniimagenet": "configs/datasets/miniimagenet.yaml",
}
FINAL = re.compile(r"Final acc:\s*(\[[^\r\n]*\])")
TASK = re.compile(r"task\s+\d+,\s+acc:\s*(\[[^\r\n]*\])")
BIAS = re.compile(r"BIAS_METRICS_JSON:(\{[^\r\n]*\})")
INC_UPDATE = re.compile(r"VDT_INCREMENTAL_UPDATE_JSON:(\{[^\r\n]*\})")


def parse_run(log):
    matches = FINAL.findall(log)
    if not matches:
        return None
    final_acc = ast.literal_eval(matches[-1])
    tasks = [ast.literal_eval(a)
             for a in TASK.findall(log.rsplit("Task-wise acc:", 1)[-1])]
    if len(tasks) != len(final_acc) or not final_acc:
        return None
    bias = [json.loads(a) for a in BIAS.findall(log)]
    incremental_updates = [json.loads(a) for a in INC_UPDATE.findall(log)]
    return {
        "final_acc": final_acc, "task_wise_acc": tasks,
        "last_acc": final_acc[-1],
        "average_acc": sum(final_acc) / len(final_acc),
        "final_base_acc": tasks[-1][0],
        "final_novel_task_mean": (
            sum(tasks[-1][1:]) / len(tasks[-1][1:]) if len(tasks[-1]) > 1 else None
        ),
        "bias_by_session": bias,
        "incremental_updates": incremental_updates,
    }


def provenance():
    def git(*parts):
        try:
            p = subprocess.run(["git", *parts], cwd=ROOT,
                               capture_output=True, text=True)
            return p.stdout.strip() if p.returncode == 0 else None
        except OSError:
            return None

    paths = ["main.py", "models/bimc.py", "models/vdt_linear.py",
             "engine/engine.py", "scripts/run_linear_baseline.py"]
    digest = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "source_sha256": digest}


def run(args):
    if args.inc_epochs < 0 or args.inc_lr <= 0:
        raise ValueError("Incremental epochs must be nonnegative and LR positive")
    if args.mode == "original" and args.inc_epochs > 0:
        raise ValueError("Incremental updates require --mode linear")
    train_cfg = ROOT / "configs/trainers/bimc.yaml"
    dataset_cfg = ROOT / DATASETS[args.dataset]
    for path in (train_cfg, dataset_cfg):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / args.output_dir / stamp
    output.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable, "main.py", "--data_cfg", DATASETS[args.dataset],
        "--train_cfg", "configs/trainers/bimc.yaml",
        "--opts", "SEED", str(args.seed), "TRAINER.BiMC.VDT_MODE", args.mode,
        "TRAINER.BiMC.VDT_INC_EPOCHS", str(args.inc_epochs),
        "TRAINER.BiMC.VDT_INC_LR", str(args.inc_lr),
    ]
    manifest = {
        "dataset": args.dataset, "seed": args.seed, "mode": args.mode,
        "incremental_epochs": args.inc_epochs, "incremental_lr": args.inc_lr,
        "config_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (train_cfg, dataset_cfg)
        }, "command": cmd, **provenance()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = output / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        code = process.wait()
    metrics = parse_run(log_path.read_text(encoding="utf-8"))
    report = {
        "status": "success" if code == 0 and metrics is not None else "failed",
        "exit_code": code, "elapsed_seconds": round(time.monotonic()-start, 2),
        "dataset": args.dataset, "seed": args.seed, "mode": args.mode,
        "incremental_epochs": args.inc_epochs, "incremental_lr": args.inc_lr,
        "metrics": metrics
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Results directory:", output)
    print("Final Acc:", metrics["last_acc"] if metrics else None)
    return 0 if report["status"] == "success" else 1


def main():
    p = argparse.ArgumentParser(description="Clean linear-only BiMC text calibration")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cifar100")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", choices=("linear", "original"), default="linear")
    p.add_argument("--inc-epochs", type=int, default=0,
                   help="5-shot epochs per incremental session; default 0 keeps Adapter frozen")
    p.add_argument("--inc-lr", type=float, default=1e-5,
                   help="Incremental AdamW learning rate, independent of base LR")
    p.add_argument("--output-dir", default="output/vdt_linear")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
