#!/usr/bin/env python3
"""One command for matched BiMC vs CLIP-A-self FSCIL verification.

Defaults: CIFAR100 seeds 1,2,3, CUB200 seed 1, miniImageNet seed 1,
each paired with original BiMC. Both arms use the SAME split/seed/config.
Each run is a separate process; all outputs are saved incrementally.
"""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

FINAL_RE = re.compile(r"Final acc:\s*(\[[^\r\n]*\])")
TASK_RE = re.compile(r"task\s+\d+,\s+acc:\s*(\[[^\r\n]*\])")
DATASETS = {
    "cifar100": "configs/datasets/cifar100.yaml",
    "cub200": "configs/datasets/cub200.yaml",
    "miniimagenet": "configs/datasets/miniimagenet.yaml",
}


def parse_result(log):
    matches = FINAL_RE.findall(log)
    if not matches:
        return None
    acc = ast.literal_eval(matches[-1])
    section = log.rsplit("Task-wise acc:", 1)[-1]
    tasks = [ast.literal_eval(v) for v in TASK_RE.findall(section)]
    if len(tasks) != len(acc):
        return None
    last_tasks = tasks[-1]
    base = float(last_tasks[0])
    novel = statistics.mean(last_tasks[1:]) if len(last_tasks) > 1 else None
    return {
        "final_acc": acc,
        "task_wise_acc": tasks,
        "last_acc": float(acc[-1]),
        "average_acc": statistics.mean(acc),
        "final_base_acc": base,
        "final_novel_task_mean": novel,
        "final_harmonic": (2 * base * novel / (base + novel)) if novel is not None and base + novel else None,
    }


def git_info(root):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(seeds, datasets):
    plan = []
    for dataset in datasets:
        matched_seeds = seeds if dataset == "cifar100" else seeds[:1]
        for seed in matched_seeds:
            for mode in ("original", "attention"):
                plan.append((dataset, int(seed), mode))
    return plan


def command_for(dataset, seed, mode, args, run_name):
    cmd = [
        sys.executable, "main.py",
        "--data_cfg", DATASETS[dataset],
        "--train_cfg", args.train_cfg,
    ]
    if args.swanlab:
        cmd += [
            "--use_swanlab", "--swanlab_project", args.swanlab_project,
            "--swanlab_experiment", run_name,
            "--swanlab_mode", args.swanlab_mode,
        ]
    cmd += [
        "--opts",
        "SEED", str(seed),
        "TRAINER.BiMC.VDT_MODE", mode,
        "TRAINER.BiMC.VDT_RATIO", str(args.ratio),
        "TRAINER.BiMC.VDT_EPOCHS", str(args.epochs),
        "TRAINER.BiMC.VDT_ORIGINAL_LAMBDA_T", str(args.original_lambda_t),
    ]
    return cmd


def write_summary(out, records):
    (out / "results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    columns = [
        "dataset", "seed", "mode", "status", "last_acc", "average_acc",
        "final_base_acc", "final_novel_task_mean", "final_harmonic",
        "elapsed_seconds", "log",
    ]
    with (out / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: r.get(key) for key in columns} for r in records])

    pairs = {}
    for r in records:
        if r["status"] == "success":
            pairs.setdefault((r["dataset"], r["seed"]), {})[r["mode"]] = r
    comparisons = []
    for (dataset, seed), arms in sorted(pairs.items()):
        if set(arms) != {"original", "attention"}:
            continue
        before, after = arms["original"], arms["attention"]
        comparisons.append({
            "dataset": dataset, "seed": seed,
            **{
                f"delta_{metric}": round(after[metric] - before[metric], 5)
                for metric in ("last_acc", "average_acc", "final_base_acc",
                               "final_novel_task_mean", "final_harmonic")
                if after[metric] is not None and before[metric] is not None
            },
        })
    (out / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    with (out / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["dataset", "seed", "delta_last_acc", "delta_average_acc",
                   "delta_final_base_acc", "delta_final_novel_task_mean",
                   "delta_final_harmonic"]
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(comparisons)

    # Across-seed variability is reported only when >=2 matched pairs exist.
    aggregate = []
    for dataset in sorted({p["dataset"] for p in comparisons}):
        items = [p for p in comparisons if p["dataset"] == dataset]
        row = {"dataset": dataset, "paired_seeds": len(items)}
        for metric in ["delta_last_acc", "delta_average_acc",
                       "delta_final_base_acc", "delta_final_novel_task_mean",
                       "delta_final_harmonic"]:
            values = [p[metric] for p in items if metric in p]
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_std"] = statistics.stdev(values) if len(values) >= 2 else None
        aggregate.append(row)
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return comparisons, aggregate


def run(args):
    root = Path(__file__).resolve().parents[1]
    dataset_names = [x.strip().lower() for x in args.datasets.split(",")]
    invalid = set(dataset_names) - set(DATASETS)
    if invalid:
        raise ValueError("Unknown dataset(s): " + ", ".join(sorted(invalid)))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    plan = build_plan(seeds, dataset_names)
    if not Path(root / args.train_cfg).exists():
        raise FileNotFoundError(args.train_cfg)
    for d in dataset_names:
        if not (root / DATASETS[d]).exists():
            raise FileNotFoundError(DATASETS[d])

    out = root / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    config_paths = [root / args.train_cfg] + [root / DATASETS[d] for d in dataset_names]
    manifest = {
        "git": git_info(root),
        "python": sys.version,
        "settings": vars(args),
        "config_sha256": {str(p.relative_to(root)): file_sha256(p) for p in config_paths},
        "plan": [{"dataset": d, "seed": seed, "mode": mode} for d, seed, mode in plan],
        "note": "Identical seed/config inside each paired original vs attention comparison."
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = []
    import time
    for i, (dataset, seed, mode) in enumerate(plan, 1):
        name = f"{dataset}_seed{seed}_{mode}"
        cmd = command_for(dataset, seed, mode, args, name)
        print(f"\n[{i}/{len(plan)}] {name}: {' '.join(cmd)}", flush=True)
        log_path = out / (name + ".log")
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
            exit_code = proc.wait()
        elapsed = round(time.monotonic() - started, 2)
        parsed = parse_result(log_path.read_text(encoding="utf-8"))
        record = {
            "dataset": dataset, "seed": seed, "mode": mode,
            "status": "success" if exit_code == 0 and parsed is not None else "failed",
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "log": str(log_path.relative_to(root)),
            "command": cmd,
        }
        if parsed:
            record.update(parsed)
        records.append(record)
        write_summary(out, records)
        print(f"Completed {name}: {record['status']} final={record.get('last_acc')}", flush=True)

    comparisons, aggregate = write_summary(out, records)
    print("\nResults directory:", out, flush=True)
    for p in comparisons:
        print(f"{p['dataset']} seed={p['seed']}: "
              f"delta Final={p['delta_last_acc']:+.3f}, "
              f"delta Base={p['delta_final_base_acc']:+.3f}, "
              f"delta Novel={p['delta_final_novel_task_mean']:+.3f}", flush=True)
    for row in aggregate:
        print(f"{row['dataset']}: paired={row['paired_seeds']}, "
              f"Final delta mean={row['delta_last_acc_mean']}, "
              f"std={row['delta_last_acc_std']}", flush=True)
    return int(any(r["status"] != "success" for r in records))


def main():
    p = argparse.ArgumentParser(description="Matched original BiMC vs VDT verification")
    p.add_argument("--datasets", default="cifar100,cub200,miniimagenet")
    p.add_argument("--seeds", default="1,2,3",
                   help="CIFAR repeats. Cross-dataset runs use the first seed.")
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--original-lambda-t", type=float, default=0.5)
    p.add_argument("--train-cfg", default="configs/trainers/bimc.yaml",
                   help="Default Ensemble=False; use bimc_ensemble.yaml for True")
    p.add_argument("--output-dir", default="output/vdt_validation")
    p.add_argument("--swanlab", action="store_true")
    p.add_argument("--swanlab-project", default="BIMC_reproduce")
    p.add_argument("--swanlab-mode", default="online",
                   choices=("online", "offline", "local", "disabled"))
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
