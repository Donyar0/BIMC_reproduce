"""Re-evaluate a saved image-to-text Flow with four ODE solvers."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from yacs.config import CfgNode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Completed image-to-text Flow run')
    parser.add_argument('--data_cfg', default='configs/datasets/cifar100.yaml')
    parser.add_argument('--use_swanlab', action='store_true')
    parser.add_argument('--trusted_checkpoints', action='store_true',
                        help='Allow pickle loading only for checkpoints you trust')
    parser.add_argument('--solvers', nargs='+', choices=('euler', 'midpoint', 'heun', 'rk4'),
                        default=('euler', 'midpoint', 'heun', 'rk4'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    with (run_dir / 'config.yaml').open(encoding='utf-8') as handle:
        original = CfgNode.load_cfg(handle)
    if not hasattr(original.FLOW, 'PREDICTION'):
        raise ValueError('A completed image-to-text Flow run is required')
    output = run_dir / 'ode_solver_ablation'
    output.mkdir(exist_ok=True)
    rows = []
    for solver in args.solvers:
        cfg = original.clone()
        cfg.defrost()
        # Older checkpoints may have saved augmentation settings that the
        # clean-feature runner no longer accepts. They are irrelevant to eval.
        if 'AUGMENT' in cfg.FLOW:
            cfg.FLOW.pop('AUGMENT')
        cfg.FLOW.SOLVER = solver
        cfg.FLOW.OUTPUT_DIR = str(output / solver)
        cfg.freeze()
        config_path = output / f'{solver}.yaml'
        config_path.write_text(cfg.dump(), encoding='utf-8')
        before = set((output / solver / cfg.DATASET.NAME.lower()).glob('eval_*')) if (output / solver / cfg.DATASET.NAME.lower()).exists() else set()
        command = [sys.executable, '-u', str(repo / 'main.py'), '--data_cfg',
                   str((repo / args.data_cfg).resolve()), '--train_cfg', str(config_path),
                   '--eval_run', str(run_dir)]
        if args.use_swanlab:
            command.append('--use_swanlab')
        if args.trusted_checkpoints:
            command.append('--trusted_checkpoints')
        print(f'Running {solver}: {" ".join(command)}', flush=True)
        log_path = output / f'{solver}.log'
        try:
            with log_path.open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            # The child process writes its traceback to the per-solver log.
            # Surface it so a foreground run explains the failure immediately.
            lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            print(f'\n{solver} failed; last lines of {log_path}:', file=sys.stderr)
            print('\n'.join(lines[-80:]), file=sys.stderr)
            raise RuntimeError(f'{solver} evaluation failed (exit {error.returncode}); '
                               f'see {log_path}') from None
        after = set((output / solver / cfg.DATASET.NAME.lower()).glob('eval_*'))
        created = after - before
        if len(created) != 1:
            raise RuntimeError(f'Expected one evaluation directory for {solver}, got {len(created)}')
        result_dir = created.pop()
        metrics = json.loads((result_dir / 'metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(solver=solver, prediction=cfg.FLOW.PREDICTION,
                         steps=cfg.FLOW.STEPS, final_acc=metrics['final_acc'],
                         avg_acc=metrics['avg_acc'],
                         eval_seconds=sum(session['elapsed_time'] for session in metrics['sessions']),
                         result_dir=str(result_dir)))
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'{solver}: final_acc={metrics["final_acc"]:.2f}, avg_acc={metrics["avg_acc"]:.2f}', flush=True)
    print(f'Summary: {output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
