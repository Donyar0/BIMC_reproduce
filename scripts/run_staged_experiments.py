#!/usr/bin/env python3
"""Run the balanced ridge ablations in order and summarize their results."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    'stage1': 'balanced_ridge_mlp.yaml',
    'stage2': 'balanced_ridge_kernel.yaml',
    'stage3': 'balanced_ridge_kernel_local.yaml',
}


def completed_run(directory, dataset):
    """Find the newest complete run for this experiment and dataset."""
    candidates = sorted((directory / dataset).glob('seed_*/metrics.json'),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            metrics = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if metrics.get('complete') and metrics.get('completed_sessions') == metrics.get('total_sessions'):
            return path.parent, metrics
    return None


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'stage', 'dataset', 'ridge_lambda', 'final_acc', 'avg_acc', 'run_dir'))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_for(stage, dataset, ridge_lambda, output_dir, use_swanlab):
    command = [sys.executable, '-u', 'main.py',
               '--data_cfg', f'configs/datasets/{dataset}.yaml',
               '--train_cfg', f'configs/trainers/{TRAINERS[stage]}']
    if use_swanlab:
        command.append('--use_swanlab')
    command += ['--opts', 'RIDGE.LAMBDA', str(float(ridge_lambda)),
                'FLOW.OUTPUT_DIR', str(output_dir)]
    return command


def run_one(stage, dataset, ridge_lambda, output_root, use_swanlab, dry_run):
    label = f'{stage}_lambda{ridge_lambda:g}'
    output_dir = output_root / label
    command = command_for(stage, dataset, ridge_lambda, output_dir, use_swanlab)
    print(f'[{stage} | {dataset} | lambda={ridge_lambda:g}]', flush=True)
    print(' '.join(command), flush=True)
    if dry_run:
        return None
    existing = completed_run(output_dir, dataset)
    if existing is not None:
        print(f'Skipping completed run: {existing[0]}', flush=True)
        return existing
    log_dir = output_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{label}_{dataset}.log'
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND: ' + ' '.join(command) + '\n')
        handle.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
            code = process.wait()
    if code:
        raise RuntimeError(f'{stage}/{dataset} exited with code {code}; see {log_path}')
    result = completed_run(output_dir, dataset)
    if result is None:
        raise RuntimeError(f'{stage}/{dataset} finished without complete metrics; see {log_path}')
    return result


def add_result(rows, stage, dataset, ridge_lambda, result, summary_path):
    run_dir, metrics = result
    rows.append(dict(stage=stage, dataset=dataset, ridge_lambda=f'{ridge_lambda:g}',
                     final_acc=metrics['final_acc'], avg_acc=metrics['avg_acc'],
                     run_dir=str(run_dir)))
    write_summary(summary_path, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['cub200', 'miniimagenet', 'cifar100'],
                        choices=['cub200', 'miniimagenet', 'cifar100'])
    parser.add_argument('--lambdas', nargs='+', type=float,
                        default=[10.0, 50.0, 100.0, 500.0],
                        help='CUB200 stage-1 sweep; best average session accuracy is reused')
    parser.add_argument('--output-root', default='output/staged_balanced_kernel')
    parser.add_argument('--use-swanlab', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(value <= 0 for value in args.lambdas):
        parser.error('All ridge lambdas must be positive')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Datasets must not be repeated')
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    summary_path = output_root / 'summary.csv'
    rows = []

    # Sweep CUB first. The remaining stages use the selected value so the
    # change from one stage to the next is only the method component.
    sweep_results = []
    if 'cub200' in args.datasets:
        for ridge_lambda in args.lambdas:
            result = run_one('stage1', 'cub200', ridge_lambda, output_root,
                             args.use_swanlab, args.dry_run)
            if result is not None:
                add_result(rows, 'stage1', 'cub200', ridge_lambda, result, summary_path)
                sweep_results.append((ridge_lambda, result[1]['avg_acc']))
    if args.dry_run:
        print('Following commands use the best CUB200 lambda; preview uses the first candidate.')
        selected = args.lambdas[0]
    elif sweep_results:
        selected = max(sweep_results, key=lambda pair: pair[1])[0]
        print(f'Selected lambda={selected:g} by CUB200 stage-1 average session accuracy')
    else:
        selected = args.lambdas[0]
        print(f'CUB200 omitted; using lambda={selected:g}')

    for dataset in args.datasets:
        if dataset != 'cub200':
            result = run_one('stage1', dataset, selected, output_root,
                             args.use_swanlab, args.dry_run)
            if result is not None:
                add_result(rows, 'stage1', dataset, selected, result, summary_path)
    for stage in ('stage2', 'stage3'):
        for dataset in args.datasets:
            result = run_one(stage, dataset, selected, output_root,
                             args.use_swanlab, args.dry_run)
            if result is not None:
                add_result(rows, stage, dataset, selected, result, summary_path)
    if not args.dry_run:
        print(f'All experiments complete. Comparison: {summary_path}')


if __name__ == '__main__':
    main()
