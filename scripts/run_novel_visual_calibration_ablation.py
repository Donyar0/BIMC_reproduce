#!/usr/bin/env python3
"""Paired CUB FSCIL test: with vs without novel BiMC visual recalibration.

Default: visual Adapter only, pure visual CE, CUB200 seed 1.
Both runs share identical training settings; only novel-class second
stage visual recalibration is switched. No sweep.
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
    p = argparse.ArgumentParser(description="Novel visual calibration on/off ablation")
    p.add_argument("--dataset", choices=("cub200", "cifar100", "miniimagenet"),
                   default="cub200")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ablation", choices=("visual", "both"), default="visual")
    p.add_argument("--visual-objective", choices=("visual_ce", "fusion_ce"),
                   default="visual_ce")
    p.add_argument("--visual-epochs", type=int, default=20)
    p.add_argument("--visual-steps", type=int, default=5)
    p.add_argument("--visual-max-support", type=int, default=0)
    p.add_argument("--output-dir", default="output/novel_visual_calibration_ablation")
    args = p.parse_args()
    folder = (ROOT / args.output_dir /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    for use_calibration in (True, False):
        mode = "with_calibration" if use_calibration else "without_calibration"
        cmd = [
            sys.executable, "scripts/run_linear_baseline.py",
            "--dataset", args.dataset, "--seed", str(args.seed),
            "--ablation", args.ablation,
            "--visual-objective", args.visual_objective,
            "--visual-epochs", str(args.visual_epochs),
            "--visual-steps", str(args.visual_steps),
            "--visual-max-support", str(args.visual_max_support),
        ]
        if not use_calibration:
            cmd.append("--no-novel-visual-calibration")
        log_path = folder / (mode + ".log")
        print("Running {} ({}/{})".format(mode, args.dataset, args.ablation),
              flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, cwd=ROOT, stdout=handle,
                                    stderr=subprocess.STDOUT, check=False)
        log = log_path.read_text(encoding="utf-8")
        found = [v.split("Results directory:", 1)[-1].strip()
                 for v in log.splitlines()
                 if v.startswith("Results directory:")]
        result_path = Path(found[-1]) / "result.json" if found else None
        report = (json.loads(result_path.read_text(encoding="utf-8"))
                  if result_path is not None and result_path.is_file() else {})
        metrics = report.get("metrics") or {}
        row = {
            "calibration": mode,
            "dataset": args.dataset,
            "seed": args.seed,
            "ablation": args.ablation,
            "visual_objective": args.visual_objective,
            "status": report.get("status", "failed"),
            "exit_code": result.returncode,
            "final_acc": metrics.get("last_acc"),
            "average_acc": metrics.get("average_acc"),
            "final_base_acc": metrics.get("final_base_acc"),
            "final_novel_task_mean": metrics.get("final_novel_task_mean"),
            "log": str(log_path),
            "result_path": str(result_path) if result_path else None,
        }
        results.append(row)
        (folder / "summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(results)
        print("{} | Final={} | Base={} | NovelMean={}".format(
            mode, row["final_acc"], row["final_base_acc"],
            row["final_novel_task_mean"]), flush=True)
        if result.returncode or row["status"] != "success":
            print("Run failed; see " + str(log_path), flush=True)
            return 1
    print("Summary:", folder / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
