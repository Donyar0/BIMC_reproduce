#!/usr/bin/env python3
"""Run a paired, fixed-seed visual CE vs fused CE experiment.

Each selected arm uses exactly the same data/seed/base-visual-adapter
architecture and visual training hyperparameters, differing *only* in
the visual episodic CE classification logits. Default arm: visual-only.
Use --ablation both to compare with frozen base-trained Linear text.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Paired visual objective ablation")
    parser.add_argument("--dataset", choices=("cub200", "cifar100", "miniimagenet"),
                        default="cub200")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ablation", choices=("visual", "both"), default="visual")
    parser.add_argument("--visual-epochs", type=int, default=20)
    parser.add_argument("--visual-steps", type=int, default=5)
    parser.add_argument("--visual-max-support", type=int, default=0)
    parser.add_argument("--output-dir", default="output/visual_objective_ablation")
    args = parser.parse_args()
    output = (ROOT / args.output_dir /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for objective in ("visual_ce", "fusion_ce"):
        cmd = [
            sys.executable, "scripts/run_linear_baseline.py",
            "--dataset", args.dataset, "--seed", str(args.seed),
            "--ablation", args.ablation,
            "--visual-objective", objective,
            "--visual-epochs", str(args.visual_epochs),
            "--visual-steps", str(args.visual_steps),
            "--visual-max-support", str(args.visual_max_support),
        ]
        logfile = output / (objective + ".log")
        print("Run {} | ablation={} dataset={} seed={}".format(
            objective, args.ablation, args.dataset, args.seed), flush=True)
        with logfile.open("w", encoding="utf-8") as stream:
            p = subprocess.run(cmd, cwd=ROOT, stdout=stream,
                               stderr=subprocess.STDOUT, check=False)
        raw = logfile.read_text(encoding="utf-8")
        folders = [line.split("Results directory:", 1)[-1].strip()
                   for line in raw.splitlines()
                   if line.startswith("Results directory:")]
        result_path = Path(folders[-1]) / "result.json" if folders else None
        report = (json.loads(result_path.read_text(encoding="utf-8"))
                  if result_path is not None and result_path.is_file() else {})
        metrics = report.get("metrics") or {}
        row = {
            "objective": objective, "ablation": args.ablation,
            "dataset": args.dataset, "seed": args.seed,
            "status": report.get("status", "failed"),
            "return_code": p.returncode,
            "final_acc": metrics.get("last_acc"),
            "avg_acc": metrics.get("average_acc"),
            "base_acc": metrics.get("final_base_acc"),
            "novel_task_mean": metrics.get("final_novel_task_mean"),
            "result_path": str(result_path) if result_path else None,
            "log_path": str(logfile),
        }
        rows.append(row)
        (output / "summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print("{}: Final={}, Base={}, NovelMean={}".format(
            objective, row["final_acc"], row["base_acc"],
            row["novel_task_mean"]
        ), flush=True)
        if p.returncode != 0 or row["status"] != "success":
            print("Failed: see " + str(logfile), flush=True)
            return 1
    print("Summary: " + str(output / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
