import subprocess
from urllib.parse import urlparse
import re


_SWANLAB_RUN = None


def _cfg_to_dict(cfg):
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()

    if isinstance(cfg, dict):
        return {key: _cfg_to_dict(value) for key, value in cfg.items()}

    return cfg


def _git_value(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return ""


def _public_github_remote():
    """Never send credentials from a configured Git remote to experiment logs."""
    remote = _git_value(["git", "remote", "get-url", "origin"])
    if remote.startswith("git@github.com:"):
        path = remote.split(":", 1)[1]
    else:
        parsed = urlparse(remote)
        if parsed.hostname != "github.com":
            return ""
        path = parsed.path.lstrip("/")
    path = path.removesuffix(".git").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        return ""
    return "https://github.com/" + path


def collect_git_info():
    branch = _git_value(["git", "branch", "--show-current"])
    commit = _git_value(["git", "rev-parse", "HEAD"])
    remote = _public_github_remote()
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": bool(_git_value(["git", "status", "--short"])),
        "git_remote_url": remote,
        "git_commit_url": f"{remote}/commit/{commit}" if remote and commit else "",
    }


def init_swanlab(cfg, args):
    global _SWANLAB_RUN

    if not getattr(args, "use_swanlab", False):
        return False

    try:
        import swanlab
    except ImportError:
        print("SwanLab is not installed. Run `pip install swanlab` or remove --use_swanlab.")
        return False

    config = _cfg_to_dict(cfg)
    config["command"] = {
        "data_cfg": args.data_cfg,
        "train_cfg": args.train_cfg,
    }
    config["git"] = collect_git_info()
    print("SwanLab Git:", config["git"])

    experiment_name = args.swanlab_experiment
    if not experiment_name:
        dataset = getattr(cfg.DATASET, "NAME", "dataset")
        method = getattr(cfg, "METHOD", "") or "BiMC"
        seed = getattr(cfg, "SEED", "seed")
        experiment_name = f"{dataset}-{method}-seed{seed}"

    _SWANLAB_RUN = swanlab.init(
        project=args.swanlab_project,
        experiment_name=experiment_name,
        config=config,
        mode=args.swanlab_mode,
    )
    return True


def log_metrics(metrics, step=None):
    if _SWANLAB_RUN is None:
        return

    import swanlab

    if step is None:
        swanlab.log(metrics)
    else:
        swanlab.log(metrics, step=step)


def finish():
    if _SWANLAB_RUN is None:
        return

    try:
        import swanlab

        swanlab.finish()
    except Exception:
        pass
