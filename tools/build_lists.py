"""Build YOLO image lists and a data yaml from a split version's assignment.csv.

    export EMBML_DATA=~/ias0360/data       # once per computer, instead of --data
    python tools/build_lists.py v3 --images fusion
    python tools/build_lists.py v3 --data ~/ias0360/data --images fusion
    python tools/build_lists.py v3 --data ~/ias0360/data --images fusion --datasets rgbt

The split is checked first (tools/check_split.py) and the build stops if it has a leak. Every row
of splits/<version>/assignment.csv goes to the split it names. --images is the
kind of image: rgb or ir (RGBT-Tiny JPGs) or fusion (4-channel PNGs, R, G, B, IR).
--datasets keeps only some datasets (default: all rows in the file). --labels is the label version
(default v0, the authors' labels; v1 and later come from tools/apply_fixes.py). RGBT-Tiny images and
labels are read from yolo_<labels>/ and fusion_<labels>/ (see convert_rgbt.py).

Writes <data>/rgbt-tiny/splits/<version>/labels_<labels>/<name>/{train,val,test}.txt, <name>.yaml and
split_info.yaml, where name is the image kind, plus _<datasets> if --datasets is used. Lines
are written relative to the list file, so the lists are the same on every computer; only the
yaml has a full path. Ultralytics finds a label by swapping /images/ for /labels/ in the image
path, so images and labels must sit under the same root (rgbt-tiny/yolo_<labels>,
rgbt-tiny/fusion_<labels>, anti-uav/fusion). Anti-UAV (auav) only has fusion images (left out of rgb and ir); its kept
frames are listed in anti-uav/fusion_all.txt.

As a module: build(version, data, images, datasets=None) -> dict with the output folder, the
yaml path, the split_info path and the frame counts.
"""

import argparse
import csv
import datetime
import hashlib
import os
import subprocess
from pathlib import Path

import check_split

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
MOD = {"rgb": "00", "ir": "01"}


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_dir(arg):
    """The data folder: the --data argument, else the EMBML_DATA environment variable."""
    value = arg or os.environ.get("EMBML_DATA")
    if not value:
        raise SystemExit("no data folder: pass --data or set the environment variable EMBML_DATA")
    return Path(value).expanduser().resolve()


def read_assignment(version):
    with open(ROOT / "splits" / version / "assignment.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build(version, data, images, datasets=None, labels="v0"):
    data = data_dir(data)
    rows = read_assignment(version)
    errors = check_split.check(rows)  # never build lists from a split with a leak
    if errors:
        raise SystemExit("split check failed:\n  " + "\n  ".join(errors))
    if datasets:
        rows = [r for r in rows if r["dataset"] in datasets]
    if images != "fusion" and any(r["dataset"] == "auav" for r in rows):
        print("note: auav has only fusion images, so its rows are left out")
        rows = [r for r in rows if r["dataset"] != "auav"]
    name = images + ("_" + "+".join(datasets) if datasets else "")
    out = data / "rgbt-tiny" / "splits" / version / f"labels_{labels}" / name
    out.mkdir(parents=True, exist_ok=True)

    tiny = data / "rgbt-tiny"
    lists = {"train": [], "val": [], "test": []}
    for root in ("yolo" if images != "fusion" else "fusion",):
        if any(r["dataset"] == "rgbt" for r in rows) and not (tiny / f"{root}_{labels}" / "labels").is_dir():
            raise SystemExit(f"labels {labels} not found at {tiny / f'{root}_{labels}'}: run tools/convert_rgbt.py "
                             f"(v0) or tools/apply_fixes.py {labels}")
    for r in rows:
        if r["dataset"] == "rgbt":
            d = (tiny / f"yolo_{labels}" / "images" / r["clip"] / MOD[images] if images != "fusion"
                 else tiny / f"fusion_{labels}" / "images" / r["clip"])
            lists[r["split"]] += sorted(d.glob("*.jpg" if images != "fusion" else "*.png"))
    videos = {r["clip"]: r["split"] for r in rows if r["dataset"] == "auav"}
    if videos:  # only the frames we kept (every 10th, RGB-IR error <= 10 px)
        for line in open(data / "anti-uav" / "fusion_all.txt"):
            f = Path(line.strip())
            if line.strip() and f.parts[-2] in videos:
                lists[videos[f.parts[-2]]].append(f)

    for split, files in lists.items():
        (out / f"{split}.txt").write_text("\n".join("./" + os.path.relpath(f, out) for f in files) + "\n")
    yaml = [f"# built by tools/build_lists.py from splits/{version}/assignment.csv, images {images}",
            f"path: {out}", "train: train.txt", "val: val.txt", "test: test.txt"]
    if images == "fusion":
        yaml.append("channels: 4")
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / f"{name}.yaml").write_text("\n".join(yaml) + "\n")

    # a record of exactly what was built, to be copied into the run folder when training starts
    sha = hashlib.sha256((ROOT / "splits" / version / "assignment.csv").read_bytes()).hexdigest()
    dirty = "yes" if git("status", "--porcelain", "splits", "tools") not in ("", "unknown") else "no"
    label_info = tiny / f"{'yolo' if images != 'fusion' else 'fusion'}_{labels}" / "labels_info.txt"
    fixes_sha = next((l.split(":", 1)[1].strip() for l in label_info.read_text().splitlines()
                      if l.startswith("fixes_sha256")), "none (authors' labels)") if label_info.exists() else "unknown"
    info = [f"split: {version}", f"images: {images}", f"labels: {labels}", f"fixes_sha256: {fixes_sha}",
            f"datasets: {' '.join(datasets) if datasets else 'all in assignment.csv'}",
            f"assignment_sha256: {sha}", f"code_commit: {git('rev-parse', 'HEAD')}",
            f"uncommitted_changes_in_splits_or_tools: {dirty}",
            f"built: {datetime.datetime.now().isoformat(timespec='seconds')}",
            "frames:"] + [f"  {k}: {len(v)}" for k, v in lists.items()]
    (out / "split_info.yaml").write_text("\n".join(info) + "\n")
    return {"dir": out, "yaml": out / f"{name}.yaml", "info": out / "split_info.yaml",
            "frames": {k: len(v) for k, v in lists.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version")
    p.add_argument("--data", help="folder with rgbt-tiny/ and anti-uav/ (default: environment variable EMBML_DATA)")
    p.add_argument("--images", choices=["rgb", "ir", "fusion"], required=True)
    p.add_argument("--datasets", nargs="+", help="keep only these datasets (default: all in assignment.csv)")
    p.add_argument("--labels", default="v0", help="label version: v0 = the authors' labels, v1 ... = with our fixes")
    a = p.parse_args()
    r = build(a.version, a.data, a.images, a.datasets, a.labels)
    print(f"{r['yaml'].stem} {a.version}: " + ", ".join(f"{k} {v}" for k, v in r["frames"].items())
          + f" frames -> {r['dir']}")


if __name__ == "__main__":
    main()
