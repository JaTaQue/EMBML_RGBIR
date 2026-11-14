"""Train a YOLO model on a data yaml and keep a record of the run.

    python tools/train.py --data-yaml <data yaml> --model weights/yolo26n_4ch.pt \\
        --name fusion_v3_s0 imgsz=640 epochs=30 batch=8 seed=0
    python tools/train.py --data-yaml <data yaml> --model weights/yolo26n.pt --name rgb_v3_s0 --dry-run

This script knows nothing about splits, image kinds or datasets. The data yaml comes from
tools/build_lists.py (which checks the split and writes the lists), or from any other
Ultralytics dataset. What it does:
  1. check that the data channels (`channels:` in the yaml, default 3) match the model input
  2. refuse a run name that already exists, so a result is never overwritten
  3. train with Ultralytics; every key=value after the options goes to it unchanged
  4. if a split_info.yaml sits next to the data yaml, copy it into the run folder, and write
     run_info.yaml (command, settings, code commit, host, Slurm job)

--dry-run does steps 1 and 2 and stops. Works the same on the laptop and on the HPC.
"""

import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def value(text):
    """'30' -> 30, '0.5' -> 0.5, 'True' -> True, 'None' -> None, anything else stays a string."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_overrides(items):
    """['epochs=30', 'batch=8'] -> {'epochs': 30, 'batch': 8}; the Python API needs real types."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"extra argument must be key=value, got: {item}")
        k, v = item.split("=", 1)
        out[k] = value(v)
    return out


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_channels(data_yaml):
    with open(data_yaml) as f:
        return int(yaml.safe_load(f).get("channels", 3))


def model_channels(model_path):
    from ultralytics import YOLO

    return YOLO(model_path).model.model[0].conv.in_channels


def run_record(args, overrides):
    return "\n".join([
        f"name: {args.name}", f"data_yaml: {args.data_yaml}", f"model: {args.model}",
        "settings: " + (str(overrides) if overrides else "{}"),
        f"command: {' '.join(sys.argv)}", f"code_commit: {git('rev-parse', 'HEAD')}",
        f"host: {platform.node()}", f"slurm_job: {os.environ.get('SLURM_JOB_ID', 'none')}",
        f"started: {datetime.datetime.now().isoformat(timespec='seconds')}"]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-yaml", required=True, help="Ultralytics data yaml (from build_lists.py or any dataset)")
    p.add_argument("--model", required=True, help="weights file, e.g. weights/yolo26n.pt")
    p.add_argument("--name", required=True, help="run name; must not exist yet")
    p.add_argument("--project", default="runs", help="folder for the runs (default: runs)")
    p.add_argument("--dry-run", action="store_true", help="check the channels and the run name, then stop")
    a, extra = p.parse_known_args()  # everything left over must be an Ultralytics key=value
    overrides = parse_overrides(extra)
    data_yaml = Path(a.data_yaml).expanduser().resolve()
    a.data_yaml = data_yaml

    have, need = model_channels(a.model), data_channels(data_yaml)
    if have != need:
        raise SystemExit(f"channels do not match: the model takes {have}, the data has {need} "
                         f"(`channels:` in {data_yaml.name}). Use a model with {need} input channels.")
    run_dir = Path(a.project).resolve() / a.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists, pick a new name or delete it")
    print(f"data: {data_yaml}  channels: {need}  model: {a.model}")
    if a.dry_run:
        print("dry run: stopping before training")
        return

    from ultralytics import YOLO

    YOLO(a.model).train(data=str(data_yaml), project=str(Path(a.project).resolve()), name=a.name,
                        exist_ok=True, **overrides)
    info = data_yaml.parent / "split_info.yaml"
    if info.exists():
        shutil.copy(info, run_dir / "split_info.yaml")
    (run_dir / "run_info.yaml").write_text(run_record(a, overrides))
    print(f"record: {run_dir / 'run_info.yaml'}" + (" and split_info.yaml" if info.exists() else ""))


if __name__ == "__main__":
    main()
