#!/usr/bin/env python3
"""Focused follow-up experiments for CLIP-A-self / BiMC.

Default plan:
  CIFAR100 seed 1: original / attention / linear / mlp (4 runs)
  CUB200 & miniImageNet seed 2,3: original / attention (8 runs)

All modes use the same frozen CLIP, base image feature CE, residual ratio,
epoch budget and BiMC classifier. Attention vs linear vs MLP have equal
trainable parameter counts at ViT-B/16 D=512 and attention reduction=4.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

# Include repository root when invoked directly as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.validate_vdt_fscil import DATASETS, git_info, file_sha256, parse_result

BIAS_RE = re.compile(r"BIAS_METRICS_JSON:(\{[^\r\n]*\})")


def build_plan(args):
    plan = []
    if args.adapters:
        for seed in args.adapter_seeds:
            for mode in ("original", "attention", "linear", "mlp"):
                plan.append(("cifar100", seed, mode))
    if args.transfer:
        for dataset in ("cub200", "miniimagenet"):
            for seed in args.transfer_seeds:
                for mode in ("original", "attention"):
                    plan.append((dataset, seed, mode))
    return plan


def make_command(args, dataset, seed, mode):
    cmd = [sys.executable, "main.py",
           "--data_cfg", DATASETS[dataset], "--train_cfg", args.train_cfg]
    if args.swanlab:
        cmd += ["--use_swanlab", "--swanlab_project", args.swanlab_project,
                "--swanlab_experiment", f"vdt-followup-{dataset}-s{seed}-{mode}",
                "--swanlab_mode", args.swanlab_mode]
    cmd += ["--opts", "SEED", str(seed),
            "TRAINER.BiMC.VDT_MODE", mode,
            "TRAINER.BiMC.VDT_RATIO", str(args.ratio),
            "TRAINER.BiMC.VDT_EPOCHS", str(args.epochs),
            "TRAINER.BiMC.VDT_REDUCTION", "4"]
    return cmd


def parse_bias(log):
    entries = [json.loads(x) for x in BIAS_RE.findall(log)]
    if not entries:
        return None, []
    return entries[-1], entries


def summarize(records, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    fields = ["dataset", "seed", "mode", "status", "last_acc", "average_acc",
              "final_base_acc", "final_novel_task_mean",
              "novel_to_base_pct", "base_to_novel_pct", "params",
              "elapsed_seconds", "log"]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k) for k in fields} for r in records])

    # Every comparison is matched against the original of the same dataset/seed.
    pairs = {(r["dataset"], r["seed"], r["mode"]): r
             for r in records if r["status"] == "success"}
    comparisons = []
    metrics = ("last_acc", "average_acc", "final_base_acc",
               "final_novel_task_mean", "novel_to_base_pct", "base_to_novel_pct")
    for (dataset, seed, mode), r in sorted(pairs.items()):
        if mode == "original":
            continue
        original = pairs.get((dataset, seed, "original"))
        if original is None:
            continue
        comparison = {"dataset": dataset, "seed": seed, "mode": mode}
        for metric in metrics:
            a, b = r.get(metric), original.get(metric)
            comparison["delta_" + metric] = round(a-b, 5) if a is not None and b is not None else None
        comparisons.append(comparison)
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    with (output / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["dataset", "seed", "mode"] + ["delta_" + m for m in metrics]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparisons)

    aggregate = []
    for dataset, mode in sorted({(x["dataset"], x["mode"]) for x in comparisons}):
        items = [x for x in comparisons if x["dataset"] == dataset and x["mode"] == mode]
        row = {"dataset": dataset, "mode": mode, "paired_seeds": len(items)}
        for metric in metrics:
            name = "delta_" + metric
            values = [x[name] for x in items if x[name] is not None]
            row[name + "_mean"] = statistics.mean(values) if values else None
            row[name + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(row)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return comparisons, aggregate


def parse_seeds(value):
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds or any(x < 0 for x in seeds):
        raise argparse.ArgumentTypeError("Pass nonnegative comma-separated seeds")
    return seeds


def run(args):
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(args)
    if not plan:
        raise ValueError("Enable --adapters or --transfer")
    configs = [root / args.train_cfg] + [root / DATASETS[d] for d in sorted({d for d, _, _ in plan})]
    for path in configs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = root / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    manifest = {
        "git": git_info(root), "python": sys.version, "args": vars(args),
        "config_sha256": {str(p.relative_to(root)): file_sha256(p) for p in configs},
        "plan": [{"dataset": d, "seed": seed, "mode": mode} for d, seed, mode in plan],
        "note": "Only training-free original or base-only CE-trained text adapters; no incremental training.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for index, (dataset, seed, mode) in enumerate(plan, 1):
        name = f"{dataset}_seed{seed}_{mode}"
        cmd = make_command(args, dataset, seed, mode)
        logfile = output / (name + ".log")
        print(f"\n[{index}/{len(plan)}] {name}", flush=True)
        print(" ".join(cmd), flush=True)
        started = time.monotonic()
        with logfile.open("w", encoding="utf-8") as f:
            process = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    f.write(line)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            code = process.wait()
        log = logfile.read_text(encoding="utf-8")
        result = parse_result(log)
        bias, bias_by_session = parse_bias(log)
        params_match = re.findall(r"trainable adapter parameters=(\d+)", log)
        success = code == 0 and result is not None and bias is not None
        record = {
            "dataset": dataset, "seed": seed, "mode": mode,
            "status": "success" if success else "failed", "exit_code": code,
            "elapsed_seconds": round(time.monotonic()-started, 2),
            "log": str(logfile.relative_to(root)), "command": cmd,
            "params": int(params_match[-1]) if params_match else None,
            "bias_by_session": bias_by_session,
        }
        if result:
            record.update(result)
        if bias:
            record["novel_to_base_pct"] = bias["novel_to_base_pct"]
            record["base_to_novel_pct"] = bias["base_to_novel_pct"]
            record["final_bias"] = bias
        records.append(record)
        summarize(records, output)
        print(f"{name}: {record['status']}, final={record.get('last_acc')}, "
              f"novel->base={record.get('novel_to_base_pct')}", flush=True)
    comparisons, aggregate = summarize(records, output)
    print("\nSaved:", output, flush=True)
    for item in comparisons:
        print(f"{item['dataset']} seed={item['seed']} {item['mode']}: "
              f"delta Final={item['delta_last_acc']}, "
              f"delta Novel->Base={item['delta_novel_to_base_pct']}", flush=True)
    return int(any(x["status"] != "success" for x in records))


def main():
    parser = argparse.ArgumentParser(description="BiMC CLIP-A-self follow-up experiments")
    parser.add_argument("--adapters", dest="adapters", action="store_true", default=True)
    parser.add_argument("--no-adapters", dest="adapters", action="store_false")
    parser.add_argument("--transfer", dest="transfer", action="store_true", default=True)
    parser.add_argument("--no-transfer", dest="transfer", action="store_false")
    parser.add_argument("--adapter-seeds", type=parse_seeds, default=[1])
    parser.add_argument("--transfer-seeds", type=parse_seeds, default=[2, 3])
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-cfg", default="configs/trainers/bimc.yaml")
    parser.add_argument("--output-dir", default="output/vdt_followup")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="BIMC_reproduce")
    parser.add_argument("--swanlab-mode", default="online",
                        choices=["online", "offline", "local", "disabled"])
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
