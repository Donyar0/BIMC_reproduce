#!/usr/bin/env python3
"""Contrastive description selection vs visual-only and learned adapters.

Defaults: CUB200 seeds 1,2,3 and CIFAR100/miniImageNet seed 1, with
original/mean/attention/visual_weighted/contrastive_fixed/contrastive_cv.
30 isolated paired runs, no tuning on real FSCIL test data.
"""
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_vdt_fscil import DATASETS, git_info, file_sha256, parse_result
from scripts.run_vdt_followup import parse_bias, parse_seeds, summarize
from scripts.run_vdt_finegrained import compare_classwise, extract_diagnostics

MODES = ("original", "mean", "attention", "visual_weighted",
         "contrastive_fixed", "contrastive_cv")
GEOMETRY_RE = re.compile(r"^VDT_GEOMETRY_JSON:(\{.*\})$", re.MULTILINE)
WEIGHT_RE = re.compile(r"^VDT_WEIGHT_DIAGNOSTICS_JSON:(\{.*\})$", re.MULTILINE)
CV_RE = re.compile(r"^VDT_PSEUDO_CV_JSON:(\{.*\})$", re.MULTILINE)


def plan_from_args(args):
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if not datasets or any(d not in DATASETS for d in datasets):
        raise ValueError("Unknown or empty datasets list")
    plan = []
    for dataset in datasets:
        seeds = args.cub_seeds if dataset == "cub200" else args.other_seeds
        for seed in seeds:
            for mode in MODES:
                plan.append((dataset, seed, mode))
    return plan


def make_cmd(args, dataset, seed, mode):
    cmd = [sys.executable, "main.py", "--data_cfg", DATASETS[dataset],
           "--train_cfg", args.train_cfg]
    if args.swanlab:
        cmd += ["--use_swanlab", "--swanlab_project", args.swanlab_project,
                "--swanlab_experiment", f"vdt-visual-{dataset}-s{seed}-{mode}",
                "--swanlab_mode", args.swanlab_mode]
    cmd += ["--opts", "SEED", str(seed), "TRAINER.BiMC.VDT_MODE", mode,
            "TRAINER.BiMC.VDT_EPOCHS", str(args.epochs),
            "TRAINER.BiMC.VDT_RATIO", str(args.ratio),
            "TRAINER.BiMC.VDT_VISUAL_TEMP", str(args.temp),
            "TRAINER.BiMC.VDT_VISUAL_BLEND", str(args.blend),
            "TRAINER.BiMC.VDT_NEGATIVE_STRENGTH", str(args.negative_strength),
            "TRAINER.BiMC.VDT_CV_EPISODES", str(args.cv_episodes),
            "TRAINER.BiMC.VDT_CV_QUERIES", str(args.cv_queries)]
    return cmd


def parse_geometry(text):
    matches = GEOMETRY_RE.findall(text)
    return json.loads(matches[-1]) if matches else None


def parse_named_json(text, pattern):
    found = pattern.findall(text)
    return json.loads(found[-1]) if found else None


def compare_weight_diagnostics(records, output):
    rows = []
    for r in records:
        w = r.get("weight_diagnostics")
        if not w:
            continue
        for group in ("base", "novel"):
            stats = w[group]
            rows.append({
                "dataset": r["dataset"], "seed": r["seed"],
                "mode": r["mode"], "group": group,
                "selected_strength": r.get("selected_strength"),
                **stats,
            })
    columns = ["dataset","seed","mode","group","selected_strength",
               "num_classes","normalized_entropy_mean","top1_weight_mean",
               "intra_description_cos_mean","weighted_margin_mean",
               "unweighted_margin_mean"]
    with (output / "weight_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows([{col:row.get(col) for col in columns} for row in rows])
    return rows


def compare_geometry(records, output):
    """For each adapter, report alignment/separation vs the SAME raw mean."""
    rows = []
    for r in records:
        g = r.get("geometry")
        if not g:
            continue
        alignment = g["support_alignment"]
        mean_separation = lambda key: (
            sum(pair["delta_separation"] for pair in g[key])/len(g[key])
            if g[key] else None
        )
        rows.append({
            "dataset": r["dataset"], "seed": r["seed"], "mode": r["mode"],
            "delta_base_alignment": alignment["delta_base_mean"],
            "delta_novel_alignment": alignment["delta_novel_mean"],
            "mean_delta_most_similar_pair_separation": mean_separation("most_similar_raw_pairs"),
        })
    columns = ["dataset","seed","mode","delta_base_alignment",
               "delta_novel_alignment","mean_delta_most_similar_pair_separation"]
    with (output/"geometry_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return rows


def run(args):
    plan = plan_from_args(args)
    configs = [ROOT/args.train_cfg] + [ROOT/DATASETS[d] for d in sorted({p[0] for p in plan})]
    for cfg in configs:
        if not cfg.is_file():
            raise FileNotFoundError(str(cfg))
    output = ROOT / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output/"diagnostics").mkdir()
    (output/"geometry").mkdir()
    (output/"weights").mkdir()
    (output/"pseudo_cv").mkdir()
    manifest = {
        "git": git_info(ROOT), "python": sys.version,
        "args": vars(args),
        "config_sha256": {str(p.relative_to(ROOT)):file_sha256(p) for p in configs},
        "plan": [{"dataset":d,"seed":s,"mode":m} for d,s,m in plan],
        "control": ("All methods use identical class split and BiMC vision branch. "
                    "Visual weighting uses TRAINING support visual features only.")
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    records=[]
    for index,(dataset,seed,mode) in enumerate(plan,1):
        name=f"{dataset}_seed{seed}_{mode}"
        cmd=make_cmd(args,dataset,seed,mode)
        logpath=output/(name+".log")
        print(f"\n[{index}/{len(plan)}] {name}\n{' '.join(cmd)}",flush=True)
        start=time.monotonic()
        with logpath.open("w",encoding="utf-8") as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in proc.stdout:
                    print(line,end="",flush=True)
                    f.write(line)
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
            exitcode=proc.wait()
        log=logpath.read_text(encoding="utf-8")
        scores=parse_result(log)
        bias,bias_sessions=parse_bias(log)
        diag=extract_diagnostics(log) if dataset=="cub200" else None
        geometry=parse_geometry(log) if dataset=="cub200" and mode!="original" else None
        weight=parse_named_json(log,WEIGHT_RE) if dataset=="cub200" and mode in (
            "visual_weighted","contrastive_fixed","contrastive_cv") else None
        cv_report=parse_named_json(log,CV_RE) if mode=="contrastive_cv" else None
        success=(exitcode==0 and scores is not None and bias is not None and
                 (mode!="contrastive_cv" or cv_report is not None) and
                 (dataset!="cub200" or
                  (diag is not None and (mode=="original" or geometry is not None) and
                   (mode not in ("visual_weighted","contrastive_fixed",
                                 "contrastive_cv") or weight is not None))))
        params=re.findall(r"trainable adapter parameters=(\d+)",log)
        row={
            "dataset":dataset,"seed":seed,"mode":mode,"status":"success" if success else "failed",
            "exit_code":exitcode,"command":cmd,"log":str(logpath.relative_to(ROOT)),
            "elapsed_seconds":round(time.monotonic()-start,2),
            "params":int(params[-1]) if params else None,"bias_by_session":bias_sessions,
        }
        if scores:
            row.update(scores)
        if bias:
            row["novel_to_base_pct"]=bias["novel_to_base_pct"]
            row["base_to_novel_pct"]=bias["base_to_novel_pct"]
            row["final_bias"]=bias
        if diag is not None:
            path=output/"diagnostics"/(name+".json")
            path.write_text(json.dumps(diag,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            row["diagnostics"]=diag
            row["diagnostics_file"]=str(path.relative_to(ROOT))
        if geometry is not None:
            path=output/"geometry"/(name+".json")
            path.write_text(json.dumps(geometry,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            row["geometry"]=geometry
            row["geometry_file"]=str(path.relative_to(ROOT))
        if weight is not None:
            path=output/"weights"/(name+".json")
            path.write_text(json.dumps(weight,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            row["weight_diagnostics"]=weight
            row["weight_diagnostics_file"]=str(path.relative_to(ROOT))
        if cv_report is not None:
            path=output/"pseudo_cv"/(name+".json")
            path.write_text(json.dumps(cv_report,indent=2)+"\n",encoding="utf-8")
            row["pseudo_cv"]=cv_report
            row["pseudo_cv_file"]=str(path.relative_to(ROOT))
            row["selected_strength"]=cv_report["selected_strength"]
        elif mode=="contrastive_fixed":
            row["selected_strength"]=args.negative_strength
        records.append(row)
        summarize(records,output)
        compare_classwise(records,output)
        compare_geometry(records,output)
        compare_weight_diagnostics(records,output)
        print(f"{name}: {row['status']} | Final={row.get('last_acc')}",flush=True)
    comparisons,aggregate=summarize(records,output)
    compare_classwise(records,output)
    compare_geometry(records,output)
    compare_weight_diagnostics(records,output)
    print("\nOutputs:",output,flush=True)
    print("Paired comparisons:",len(comparisons),flush=True)
    return int(any(r["status"]!="success" for r in records))


def main():
    p=argparse.ArgumentParser(description="Contrastive VDT + base-only pseudo-incremental validation")
    p.add_argument("--datasets",default="cub200,cifar100,miniimagenet")
    p.add_argument("--cub-seeds",type=parse_seeds,default=[1,2,3])
    p.add_argument("--other-seeds",type=parse_seeds,default=[1])
    p.add_argument("--temp",type=float,default=10.0)
    p.add_argument("--blend",type=float,default=0.5)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--ratio",type=float,default=0.2)
    p.add_argument("--negative-strength",type=float,default=1.0)
    p.add_argument("--cv-episodes",type=int,default=3)
    p.add_argument("--cv-queries",type=int,default=5)
    p.add_argument("--train-cfg",default="configs/trainers/bimc.yaml")
    p.add_argument("--output-dir",default="output/vdt_contrastive")
    p.add_argument("--swanlab",action="store_true")
    p.add_argument("--swanlab-project",default="BIMC_reproduce")
    p.add_argument("--swanlab-mode",default="online",
                   choices=["online","offline","local","disabled"])
    sys.exit(run(p.parse_args()))


if __name__=="__main__":
    main()
