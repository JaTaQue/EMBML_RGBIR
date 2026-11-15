"""Save a finished run into the repo: one row in results/runs.csv, and (for important runs) the model file.

    python tools/save_run.py runs/<name> [--note "why this run, what it showed"]
    python tools/save_run.py runs/<name> --model [--weights last|best]

Reads the files the run folder already has (run_info.yaml, split_info.yaml, args.yaml, results.csv) and

  - copies the three small ones into results/<name>/  (run_info.yaml, split_info.yaml, results.csv)
  - adds one row to results/runs.csv with the data version, the settings and the last-epoch metrics
  - with --model, also copies the model to models/<name>_<last|best>.pt and writes it into the row

A row is added once and never edited: if a number was wrong, make a new run. The metrics are the
Ultralytics ones on the val set of the split, taken from the last epoch. Run the debug runs without this tool.
"""

import argparse
import csv
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["run", "date", "split", "labels", "images", "datasets", "model", "epochs", "batch", "imgsz",
           "precision", "recall", "mAP50", "mAP50-95", "metrics_on", "model_file", "note"]
FILES = ("run_info.yaml", "split_info.yaml", "results.csv")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def last_metrics(path):
    """(number of epochs, metrics of the last epoch) from an Ultralytics results.csv."""
    with open(path, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"{path} has no epochs")
    return len(rows), rows[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run folder, e.g. runs/fusion_v3_s0")
    p.add_argument("--note", default="", help="one sentence: why this run, what it showed")
    p.add_argument("--model", action="store_true", help="also copy the model file into models/")
    p.add_argument("--weights", choices=["last", "best"], default="last", help="which weights to copy (default last)")
    a = p.parse_args()
    run = Path(a.run).resolve()
    missing = [f for f in FILES + ("args.yaml",) if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run} is missing {', '.join(missing)}: was it made with tools/train.py?")
    name = run.name

    table = ROOT / "results" / "runs.csv"
    if table.exists():
        with open(table, newline="") as f:
            if any(r["run"] == name for r in csv.DictReader(f)):
                raise SystemExit(f"{name} is already in {table.name}. A row is never edited: use a new run name.")

    info, split, args = load_yaml(run / "run_info.yaml"), load_yaml(run / "split_info.yaml"), load_yaml(run / "args.yaml")
    epochs, m = last_metrics(run / "results.csv")
    if split.get("uncommitted_changes_in_splits_or_tools") == "yes":
        print("warning: the lists were built with uncommitted changes in splits/ or tools/")

    model_file = ""
    if a.model:
        src = run / "weights" / f"{a.weights}.pt"
        if not src.exists():
            raise SystemExit(f"{src} not found")
        dst = ROOT / "models" / f"{name}_{a.weights}.pt"
        if dst.exists():
            raise SystemExit(f"{dst} exists: a published model is never replaced, use a new run name")
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        model_file = f"models/{dst.name}"

    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(run / f, out / f)

    row = {"run": name, "date": str(info.get("started", ""))[:10], "split": split.get("split", ""),
           "labels": split.get("labels", ""), "images": split.get("images", ""),
           "datasets": split.get("datasets", ""), "model": Path(str(info.get("model", ""))).name,
           "epochs": epochs, "batch": args.get("batch", ""), "imgsz": args.get("imgsz", ""),
           "precision": m.get("metrics/precision(B)", ""), "recall": m.get("metrics/recall(B)", ""),
           "mAP50": m.get("metrics/mAP50(B)", ""), "mAP50-95": m.get("metrics/mAP50-95(B)", ""),
           "metrics_on": "val", "model_file": model_file, "note": a.note}
    for k in ("precision", "recall", "mAP50", "mAP50-95"):
        row[k] = f"{float(row[k]):.4f}" if row[k] != "" else ""
    new = not table.exists()
    with open(table, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"saved {name}: row in {table.relative_to(ROOT)}, files in {out.relative_to(ROOT)}/"
          + (f", model {model_file}" if model_file else ""))
    print("  " + ", ".join(f"{k} {row[k]}" for k in ("split", "labels", "images", "epochs", "mAP50", "mAP50-95")))


if __name__ == "__main__":
    main()
