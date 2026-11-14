"""Train a YOLO model on a split version. One command: check, build lists, train, record.

    export EMBML_DATA=~/ias0360/data       # once per computer, instead of --data
    python tools/train.py v3 --data ~/ias0360/data --images fusion \\
        --model weights/yolo26n_4ch.pt --name fusion_v3_s0 imgsz=640 epochs=30 batch=8 seed=0
    python tools/train.py v3 --data ~/ias0360/data --images fusion --datasets rgbt \\
        --model weights/yolo26n_4ch.pt --name fusion_rgbt_v3_s0 epochs=30 --dry-run

Steps (each one is a function you can import):
  1. check_split.check: stop if the split has a leak (group or recording in two splits)
  2. build_lists.build: make the image lists and yaml for --images / --datasets
  3. refuse a run name that already exists, so a result is never overwritten
  4. train with Ultralytics; every key=value after the options goes to it unchanged
  5. copy split_info.yaml into the run folder and write run_info.yaml (command, settings,
     code commit, host, Slurm job), so the model keeps a record of how it was trained

Works the same on the laptop and on the HPC. Nothing here is specific to a model, an image
kind or a dataset: those come from the arguments and from assignment.csv.
"""

import argparse
import datetime
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_lists
import check_split


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


def run_record(args, overrides, built):
    return "\n".join([
        f"name: {args.name}", f"split: {args.version}", f"images: {args.images}",
        f"datasets: {' '.join(args.datasets) if args.datasets else 'all in assignment.csv'}",
        f"model: {args.model}", f"data_yaml: {built['yaml']}",
        "settings: " + (str(overrides) if overrides else "{}"),
        f"command: {' '.join(sys.argv)}", f"code_commit: {build_lists.git('rev-parse', 'HEAD')}",
        f"host: {platform.node()}", f"slurm_job: {os.environ.get('SLURM_JOB_ID', 'none')}",
        f"started: {datetime.datetime.now().isoformat(timespec='seconds')}"]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version", help="split version, a folder in splits/ (v3, v4, ...)")
    p.add_argument("--data", help="folder with rgbt-tiny/ and anti-uav/ (default: environment variable EMBML_DATA)")
    p.add_argument("--images", choices=["rgb", "ir", "fusion"], required=True)
    p.add_argument("--datasets", nargs="+", help="keep only these datasets (default: all in assignment.csv)")
    p.add_argument("--model", required=True, help="weights file, e.g. weights/yolo26n.pt")
    p.add_argument("--name", required=True, help="run name; must not exist yet")
    p.add_argument("--project", default="runs", help="folder for the runs (default: runs)")
    p.add_argument("--dry-run", action="store_true", help="check and build the lists, then stop")
    a, extra = p.parse_known_args()  # everything left over must be an Ultralytics key=value
    overrides = parse_overrides(extra)

    errors = check_split.check(build_lists.read_assignment(a.version))
    if errors:
        raise SystemExit("split check failed:\n  " + "\n  ".join(errors))
    built = build_lists.build(a.version, a.data, a.images, a.datasets)
    print(f"lists: {built['yaml']}  frames {built['frames']}")

    run_dir = Path(a.project).resolve() / a.name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists, pick a new name or delete it")
    if a.dry_run:
        print("dry run: stopping before training")
        return

    from ultralytics import YOLO

    YOLO(a.model).train(data=str(built["yaml"]), project=str(Path(a.project).resolve()), name=a.name,
                        exist_ok=True, **overrides)
    shutil.copy(built["info"], run_dir / "split_info.yaml")
    (run_dir / "run_info.yaml").write_text(run_record(a, overrides, built))
    print(f"record: {run_dir / 'split_info.yaml'}, {run_dir / 'run_info.yaml'}")


if __name__ == "__main__":
    main()
