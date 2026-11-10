#!/usr/bin/env python3
"""Targeted adapter-comparison and fine-grained CUB200 confusion experiments.

Default: paired original/attention/linear/mlp for CIFAR100 seeds 2,3 and
CUB200 seeds 1,2,3 (20 independent runs). Existing validation files are
never overwritten. Each CUB run exports class-level diagnostic JSON.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_vdt_fscil import DATASETS, git_info, file_sha256, parse_result
from scripts.run_vdt_followup import make_command, parse_bias, summarize, parse_seeds

DIAGNOSTIC_RE = re.compile(r"^FINEGRAINED_DIAGNOSTICS_JSON:(\{.*\})$", re.MULTILINE)
MODES = ("original", "attention", "linear", "mlp")


def build_plan(cifar_seeds=(2, 3), cub_seeds=(1, 2, 3)):
    return [(dataset, seed, mode)
            for dataset, seeds in (("cifar100", cifar_seeds), ("cub200", cub_seeds))
            for seed in seeds for mode in MODES]


def extract_diagnostics(log):
    found = DIAGNOSTIC_RE.findall(log)
    if not found:
        return None
    return json.loads(found[-1])


def compare_classwise(records, output):
    """Compare same-seed CUB class errors vs the original's test predictions.

    Confidence changes are NOT inferred; only error counts/accuracies are compared.
    """
    output.mkdir(parents=True, exist_ok=True)
    originals = {
        (r["dataset"], r["seed"]): r for r in records
        if r.get("mode") == "original" and r.get("status") == "success"
        and r.get("diagnostics") is not None
    }
    class_deltas = []
    pair_deltas = []
    for run in records:
        if run.get("dataset") != "cub200" or run.get("mode") == "original":
            continue
        control = originals.get((run["dataset"], run["seed"]))
        if not control or run.get("status") != "success" or not run.get("diagnostics"):
            continue
        before = control["diagnostics"]
        after = run["diagnostics"]
        before_classes = {c["class_id"]: c for c in before["per_class"]}
        after_classes = {c["class_id"]: c for c in after["per_class"]}
        if set(before_classes) != set(after_classes):
            raise ValueError("Different class sets for paired diagnostics")
        for cid in sorted(before_classes):
            a, b = before_classes[cid], after_classes[cid]
            if a["support"] != b["support"]:
                raise ValueError("Different test support within a paired class")
            class_deltas.append({
                "seed": run["seed"], "mode": run["mode"],
                "class_id": cid, "class_name": a["class_name"],
                "group": a["group"], "support": a["support"],
                "original_accuracy_pct": a["accuracy_pct"],
                "adapter_accuracy_pct": b["accuracy_pct"],
                "delta_accuracy_pct": round(b["accuracy_pct"]-a["accuracy_pct"], 4),
            })
        old_pairs = {(p["true_id"], p["pred_id"]): p
                     for p in before["all_confusion_pairs"]}
        new_pairs = {(p["true_id"], p["pred_id"]): p
                     for p in after["all_confusion_pairs"]}
        for key in sorted(set(old_pairs) | set(new_pairs)):
            a = old_pairs.get(key)
            b = new_pairs.get(key)
            info = b or a
            delta = (b["count"] if b else 0) - (a["count"] if a else 0)
            if delta == 0:
                continue
            pair_deltas.append({
                "seed": run["seed"], "mode": run["mode"],
                "true_id": key[0], "true_name": info["true_name"],
                "true_group": info["true_group"],
                "pred_id": key[1], "pred_name": info["pred_name"],
                "pred_group": info["pred_group"],
                "original_count": a["count"] if a else 0,
                "adapter_count": b["count"] if b else 0,
                "delta_count": delta,
            })

    def csv_write(name, rows, cols):
        with (output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    csv_write("cub_class_accuracy_deltas.csv", class_deltas,
              ["seed", "mode", "class_id", "class_name", "group", "support",
               "original_accuracy_pct", "adapter_accuracy_pct", "delta_accuracy_pct"])
    pair_deltas.sort(key=lambda x: (-abs(x["delta_count"]), x["seed"], x["mode"],
                                           x["true_id"], x["pred_id"]))
    csv_write("cub_confusion_pair_deltas.csv", pair_deltas,
              ["seed", "mode", "true_id", "true_name", "true_group",
               "pred_id", "pred_name", "pred_group",
               "original_count", "adapter_count", "delta_count"])
    return class_deltas, pair_deltas


def run(args):
    plan = build_plan(args.cifar_seeds if args.cifar else (),
                      args.cub_seeds if args.cub else ())
    if not plan:
        raise ValueError("Both datasets disabled")
    configs = [ROOT / args.train_cfg] + [
        ROOT / DATASETS[name] for name in sorted({d for d, _, _ in plan})
    ]
    for f in configs:
        if not f.is_file():
            raise FileNotFoundError(str(f))
    output = ROOT / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "diagnostics").mkdir()
    manifest = {
        "git": git_info(ROOT), "python": sys.version,
        "args": vars(args),
        "config_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in configs},
        "plan": [{"dataset": d, "seed": s, "mode": m} for d, s, m in plan],
        "interpretation": "Directed true->pred confusion; no taxonomic family inference.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = []
    for index, (dataset, seed, mode) in enumerate(plan, 1):
        name = f"{dataset}_seed{seed}_{mode}"
        cmd = make_command(args, dataset, seed, mode)
        logfile = output / (name + ".log")
        print(f"\n[{index}/{len(plan)}] {name}", flush=True)
        print(" ".join(cmd), flush=True)
        started = time.monotonic()
        with logfile.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
            code = proc.wait()
        log = logfile.read_text(encoding="utf-8")
        accuracy = parse_result(log)
        bias, bias_by_session = parse_bias(log)
        diag = extract_diagnostics(log) if dataset == "cub200" else None
        diagnostic_expected = dataset == "cub200"
        success = (code == 0 and accuracy is not None and bias is not None
                   and (not diagnostic_expected or diag is not None))
        params = re.findall(r"trainable adapter parameters=(\d+)", log)
        record = {
            "dataset": dataset, "seed": seed, "mode": mode,
            "status": "success" if success else "failed",
            "exit_code": code, "elapsed_seconds": round(time.monotonic()-started, 2),
            "log": str(logfile.relative_to(ROOT)),
            "command": cmd, "params": int(params[-1]) if params else None,
            "bias_by_session": bias_by_session,
        }
        if accuracy:
            record.update(accuracy)
        if bias:
            record.update({
                "novel_to_base_pct": bias["novel_to_base_pct"],
                "base_to_novel_pct": bias["base_to_novel_pct"],
                "final_bias": bias,
            })
        if diag is not None:
            diagnostic_path = output / "diagnostics" / (name + ".json")
            diagnostic_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False)+"\n",
                                       encoding="utf-8")
            record["diagnostics"] = diag
            record["diagnostics_file"] = str(diagnostic_path.relative_to(ROOT))
        records.append(record)
        summarize(records, output)
        compare_classwise(records, output)
        print(f"Saved {name}: {record['status']} | Final={record.get('last_acc')}", flush=True)

    paired, aggregate = summarize(records, output)
    class_deltas, pair_deltas = compare_classwise(records, output)
    print("\nAll results:", output, flush=True)
    print("Paired comparisons:", len(paired), "CUB class deltas:", len(class_deltas),
          "changed directed confusions:", len(pair_deltas), flush=True)
    return int(any(r["status"] != "success" for r in records))


def main():
    parser = argparse.ArgumentParser(description="CIFAR+CUB adapter ablations and CUB confusions")
    parser.add_argument("--cifar", dest="cifar", action="store_true", default=True)
    parser.add_argument("--no-cifar", dest="cifar", action="store_false")
    parser.add_argument("--cub", dest="cub", action="store_true", default=True)
    parser.add_argument("--no-cub", dest="cub", action="store_false")
    parser.add_argument("--cifar-seeds", type=parse_seeds, default=[2, 3])
    parser.add_argument("--cub-seeds", type=parse_seeds, default=[1, 2, 3])
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-cfg", default="configs/trainers/bimc.yaml")
    parser.add_argument("--output-dir", default="output/vdt_finegrained")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="BIMC_reproduce")
    parser.add_argument("--swanlab-mode", default="online",
                        choices=("online", "offline", "local", "disabled"))
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
