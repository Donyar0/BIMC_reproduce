#!/usr/bin/env python3
"""Three-arm FSCIL adapter ablation, one dataset/seed at a time.

text   = base-trained Linear text Adapter + ordinary BiMC visual mean
visual = original BiMC text calibration + base-trained visual attention
both   = same Linear text Adapter + same visual attention architecture

Optional --include-original runs a fourth genuine BiMC control.
This script does not add new losses or perform a hyperparameter sweep.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("text", "visual", "both")


def run(args):
    folder = ROOT / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True,exist_ok=False)
    arms = (("original",) + MODES) if args.include_original else MODES
    results=[]
    for index, arm in enumerate(arms, 1):
        cmd = [
            sys.executable,"scripts/run_linear_baseline.py",
            "--dataset",args.dataset,"--seed",str(args.seed),
            "--ablation",arm,
            "--objective",args.objective if arm in ("text","both") else "text_ce",
            "--visual-objective",
            args.visual_objective if arm in ("visual","both") else "visual_ce",
            "--visual-epochs",str(args.visual_epochs),
            "--visual-steps",str(args.visual_steps),
            "--visual-max-support",str(args.visual_max_support),
            "--output-dir",args.run_output_dir,
        ]
        log = folder / (arm+".log")
        print("[{}/{}] {} | dataset={} seed={}".format(
            index,len(arms),arm,args.dataset,args.seed),flush=True)
        with log.open("w",encoding="utf-8") as stream:
            process = subprocess.Popen(
                cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT
            )
            try:
                code=process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        raw = log.read_text(encoding="utf-8")
        matched = [line.split("Results directory:",1)[-1].strip()
                   for line in raw.splitlines()
                   if line.startswith("Results directory:")]
        result_path = Path(matched[-1]) / "result.json" if matched else None
        report = json.loads(result_path.read_text(encoding="utf-8")) if (
            result_path is not None and result_path.is_file()
        ) else None
        metrics = report.get("metrics") if report else None
        row = {
            "arm":arm,"dataset":args.dataset,"seed":args.seed,
            "visual_objective":args.visual_objective if arm in ("visual","both") else None,
            "status":report.get("status") if report else "failed",
            "return_code":code,
            "last_acc":metrics.get("last_acc") if metrics else None,
            "average_acc":metrics.get("average_acc") if metrics else None,
            "final_base_acc":metrics.get("final_base_acc") if metrics else None,
            "final_novel_task_mean":metrics.get("final_novel_task_mean") if metrics else None,
            "run_folder":str(result_path.parent) if result_path else None,
            "log":str(log.relative_to(ROOT))
        }
        results.append(row)
        (folder / "results.json").write_text(
            json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
        )
        with (folder / "summary.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print("    {} Final={} Base={} NovelTaskMean={}".format(
            row["status"],row["last_acc"],row["final_base_acc"],
            row["final_novel_task_mean"]
        ),flush=True)
        if code != 0 or row["status"] != "success":
            print("Run failed. Check", log,flush=True)
            return 1
    print("Summary:",folder / "summary.csv")
    return 0


def main():
    p=argparse.ArgumentParser(description="Three-way visual/text adapter ablation")
    p.add_argument("--dataset",choices=("cub200","cifar100","miniimagenet"),
                   default="cub200")
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--include-original",action="store_true")
    p.add_argument("--objective",choices=("text_ce","fusion_ce"),default="text_ce")
    p.add_argument("--visual-objective",choices=("visual_ce","fusion_ce"),
                   default="visual_ce")
    p.add_argument("--visual-epochs",type=int,default=20)
    p.add_argument("--visual-steps",type=int,default=5)
    p.add_argument("--visual-max-support",type=int,default=0)
    p.add_argument("--output-dir",default="output/adapter_triplet")
    p.add_argument("--run-output-dir",default="output/vdt_linear")
    sys.exit(run(p.parse_args()))


if __name__=="__main__":
    main()
