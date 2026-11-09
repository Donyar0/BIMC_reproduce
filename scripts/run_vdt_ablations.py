#!/usr/bin/env python3
"""Run A-E CLIP-A-self ablations separately and produce CSV/JSON summaries.

Example:
    python scripts/run_vdt_ablations.py
    python scripts/run_vdt_ablations.py --data-cfg configs/datasets/cub200.yaml
    python scripts/run_vdt_ablations.py --swanlab
"""
import argparse
import csv
from datetime import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = (
    ("A_mean_no_training", "mean", 0.2, 0),
    ("B_ratio_0.1", "attention", 0.1, 20),
    ("C_ratio_0.3", "attention", 0.3, 20),
    ("D_epochs_5", "attention", 0.2, 5),
    ("E_epochs_50", "attention", 0.2, 50),
)

FINAL_PATTERN = re.compile(r"Final acc:\s*(\[[^\r\n]*\])")
TASK_PATTERN = re.compile(r"task\s+\d+,\s+acc:\s*(\[[^\r\n]*\])")


def parse_results(log_text):
    import ast
    finals = FINAL_PATTERN.findall(log_text)
    if not finals:
        return None, []
    final_acc = ast.literal_eval(finals[-1])
    task_lines = log_text.split("Task-wise acc:")[-1]
    task_wise = [ast.literal_eval(x) for x in TASK_PATTERN.findall(task_lines)]
    return final_acc, task_wise


def make_command(args, name, mode, ratio, epochs):
    cmd = [
        sys.executable, "main.py",
        "--data_cfg", args.data_cfg,
        "--train_cfg", args.train_cfg,
    ]
    if args.swanlab:
        cmd.extend([
            "--use_swanlab",
            "--swanlab_project", args.swanlab_project,
            "--swanlab_experiment", f"vdt-ablation-{name}",
            "--swanlab_mode", args.swanlab_mode,
        ])
    cmd.extend([
        "--opts",
        "TRAINER.BiMC.VDT_MODE", mode,
        "TRAINER.BiMC.VDT_RATIO", str(ratio),
        "TRAINER.BiMC.VDT_EPOCHS", str(epochs),
    ])
    return cmd


def run_all(args):
    root = Path(__file__).resolve().parents[1]
    for config in (args.data_cfg, args.train_cfg):
        if not (root / config).is_file():
            raise FileNotFoundError(f"Missing config under repository root: {config}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / args.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    results = []
    for idx, (name, mode, ratio, epochs) in enumerate(EXPERIMENTS, 1):
        cmd = make_command(args, name, mode, ratio, epochs)
        log_path = out_dir / f"{name}.log"
        print(f"\n[{idx}/5] {name} | mode={mode}, ratio={ratio}, epochs={epochs}", flush=True)
        print(" ".join(cmd), flush=True)

        # Separate process = fresh model/optimizer/data-seed every experiment.
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            exit_code = process.wait()

        raw = log_path.read_text(encoding="utf-8")
        finals, task_wise = parse_results(raw)
        status = "success" if exit_code == 0 and finals is not None else "failed"
        last_acc = finals[-1] if finals else None
        avg_acc = round(sum(finals) / len(finals), 4) if finals else None
        entry = {
            "experiment": name, "mode": mode, "ratio": ratio, "epochs": epochs,
            "status": status, "exit_code": exit_code,
            "final_acc": finals, "task_wise_acc": task_wise,
            "last_session_acc": last_acc, "average_acc": avg_acc,
            "log": str(log_path.relative_to(root)),
            "command": cmd,
        }
        results.append(entry)
        rows.append({k: entry[k] for k in (
            "experiment", "mode", "ratio", "epochs", "status", "exit_code",
            "last_session_acc", "average_acc", "log"
        )})
        # Incremental persistence also covers interruption midway through the sweep.
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Result: {status}, last={last_acc}, avg={avg_acc}", flush=True)

    print(f"\nAll ablation outputs: {out_dir}", flush=True)
    for r in rows:
        print("{experiment:20s}  {status:8s}  final={last_session_acc}  avg={average_acc}".format(**r))
    return 1 if any(r["status"] != "success" for r in rows) else 0


def main():
    parser = argparse.ArgumentParser(description="Run the five VDT adapter ablations A-E.")
    parser.add_argument("--data-cfg", default="configs/datasets/cifar100.yaml")
    parser.add_argument("--train-cfg", default="configs/trainers/bimc.yaml",
                        help="Use bimc.yaml for Ensemble=False and bimc_ensemble.yaml for True")
    parser.add_argument("--output-dir", default="output/vdt_ablations")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="BIMC_reproduce")
    parser.add_argument("--swanlab-mode", default="online",
                        choices=["online", "offline", "local", "disabled"])
    args = parser.parse_args()
    sys.exit(run_all(args))


if __name__ == "__main__":
    main()
