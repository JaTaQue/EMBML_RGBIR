"""Build a label version from the authors' labels (v0) plus our fixes.

    python tools/apply_fixes.py v1 [--data <data folder>] [--seal]

Reads annotations/<version>/fixes.csv and the v0 labels written by convert_rgbt.py, and writes the
labels of <version> next to them:

  <data>/rgbt-tiny/yolo_<version>/    images link, labels/<clip>/<00|01>/<frame>.txt   (RGB and IR)
  <data>/rgbt-tiny/fusion_<version>/  images link, labels/<clip>/<frame>.txt           (the IR labels)

The authors' labels are never edited. v0 is read only. With an empty fixes file the result is
identical to v0.

fixes.csv has one row per change:  box_id,action,old_box,new_value,reason,reviewer,date
  box_id    rgbt/<clip>/<frame>/<rgb|ir>/<n>, n = line number (from 0) in the v0 label file of that
            frame and modality. For `add` use n = new1, new2, ... (any text starting with "new").
  old_box   the box as it is in v0, a copy of its label line (e.g. 5 0.696875 0.818359 0.034375 0.019531).
            Required for every action except `add`. The script stops if it is not the box in v0, so a
            fix can never change a different box than the one that was reviewed.
  action    ok       reviewed, no change (kept so a review leaves a trace)
            remove   delete the box
            relabel  new_value = class name (ship car cyclist pedestrian bus drone plane)
            resize   new_value = "x y w h", the new box in pixels (top-left corner, width, height)
            add      new_value = "class x y w h", a new box
  A fix on an `ir` box also changes the fusion labels, since those are the IR labels. Only RGBT-Tiny
  boxes (rgbt/...) are supported; Anti-UAV labels are not versioned yet.

A version is never edited after it is used. info.yaml in the version folder keeps the hash of
fixes.csv; --seal writes it, and after that this script stops if fixes.csv has changed.
"""

import argparse
import csv
import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from common import data_dir
from convert_rgbt import NAMES, link, yolo_line

ROOT = Path(__file__).resolve().parents[1]
CLASS_ID = {name: i for i, name in NAMES.items()}
ACTIONS = {"ok", "remove", "relabel", "resize", "add"}
COLUMNS = ["box_id", "action", "old_box", "new_value", "reason", "reviewer", "date"]
W, H = 640, 512  # all RGBT-Tiny frames


def read_info(path):
    info = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def parse_box(text):
    parts = text.split()
    if len(parts) != 4:
        raise ValueError(f'expected "x y w h", got "{text}"')
    return [float(p) for p in parts]


def load_fixes(path):
    """{(clip, frame, mod): {n: (action, new_value)}} and the problems found."""
    problems, out, seen = [], {}, set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path.name} must have the columns {','.join(COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            where = f"line {line}"
            parts = row["box_id"].split("/")
            if len(parts) != 5 or parts[0] != "rgbt" or parts[3] not in ("rgb", "ir"):
                problems.append(f"{where}: box_id must be rgbt/<clip>/<frame>/<rgb|ir>/<n>, got {row['box_id']}")
                continue
            if row["action"] not in ACTIONS:
                problems.append(f"{where}: unknown action {row['action']}")
                continue
            if row["box_id"] in seen and not parts[4].startswith("new"):
                problems.append(f"{where}: {row['box_id']} appears twice")
            seen.add(row["box_id"])
            try:
                if row["action"] == "relabel" and row["new_value"] not in CLASS_ID:
                    raise ValueError(f"unknown class {row['new_value']}")
                if row["action"] == "resize":
                    parse_box(row["new_value"])
                if row["action"] == "add":
                    cls, *box = row["new_value"].split()
                    if cls not in CLASS_ID:
                        raise ValueError(f"unknown class {cls}")
                    parse_box(" ".join(box))
            except ValueError as e:
                problems.append(f"{where}: {e}")
                continue
            out.setdefault((parts[1], parts[2], parts[3]), {})[parts[4]] = (row["action"], row["new_value"], row["old_box"])
    return out, problems


def apply_to_lines(lines, fixes, key, problems):
    """The label lines of one frame after the fixes. Counts go in `done`."""
    done, adds, result = Counter(), [], []
    for n, line in enumerate(lines):
        action, value, old = fixes.get(str(n), ("keep", "", ""))
        if action != "keep" and old.strip() != line.strip():
            problems.append(f"{'/'.join(key)}/{n}: old_box does not match v0 (v0 has: {line})")
            continue
        done[action] += 1
        if action == "remove":
            continue
        if action == "relabel":
            line = f"{CLASS_ID[value]} " + line.split(" ", 1)[1]
        elif action == "resize":
            line = yolo_line(int(line.split()[0]), parse_box(value), W, H)
            if line is None:
                problems.append(f"{'/'.join(key)}/{n}: the new box is outside the image")
                continue
        result.append(line)
    for n, (action, value, old) in fixes.items():
        if n.startswith("new"):
            cls, *box = value.split()
            line = yolo_line(CLASS_ID[cls], parse_box(" ".join(box)), W, H)
            if line is None:
                problems.append(f"{'/'.join(key)}/{n}: the new box is outside the image")
            else:
                result.append(line)
                done["add"] += 1
        elif not n.isdigit() or int(n) >= len(lines):
            problems.append(f"{'/'.join(key)}/{n}: the frame has {len(lines)} boxes in v0")
    return result, done


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version", help="label version, a folder in annotations/ (v1, v2, ...)")
    p.add_argument("--data", help="folder with rgbt-tiny/ (default: environment variable EMBML_DATA)")
    p.add_argument("--seal", action="store_true", help="write the hash of fixes.csv into info.yaml")
    a = p.parse_args()
    tiny = data_dir(a.data) / "rgbt-tiny"
    folder = ROOT / "annotations" / a.version
    fixes_file = folder / "fixes.csv"
    sha = hashlib.sha256(fixes_file.read_bytes()).hexdigest()
    sealed = read_info(folder / "info.yaml").get("fixes_sha256")
    if sealed and sealed != sha:
        raise SystemExit(f"{fixes_file} has changed since {a.version} was sealed. A used version is never edited: "
                         f"make a new version instead.")
    for root in ("yolo_v0", "fusion_v0"):
        if not (tiny / root / "labels").is_dir():
            raise SystemExit(f"{tiny / root} not found: run tools/convert_rgbt.py first")

    fixes, problems = load_fixes(fixes_file)
    pairs = {(k[0], k[1], k[2]) for k in fixes}
    for clip, frame, mod in sorted(pairs):
        src = tiny / "yolo_v0" / "labels" / clip / ("00" if mod == "rgb" else "01") / f"{frame}.txt"
        if not src.exists():
            problems.append(f"rgbt/{clip}/{frame}/{mod}: no such frame in v0")
    if problems:
        raise SystemExit("problems in fixes.csv:\n  " + "\n  ".join(problems))

    new_yolo, new_fusion = tiny / f"yolo_{a.version}", tiny / f"fusion_{a.version}"
    for old, new in (("yolo_v0", new_yolo), ("fusion_v0", new_fusion)):
        if new.exists():
            raise SystemExit(f"{new} already exists: a label version is built once. Delete it to rebuild.")
        shutil.copytree(tiny / old / "labels", new / "labels")
    link(new_yolo / "images", "../raw/images")
    link(new_fusion / "images", "../fusion/images")

    done = Counter()
    for (clip, frame, mod), fx in sorted(fixes.items()):
        targets = [new_yolo / "labels" / clip / ("00" if mod == "rgb" else "01") / f"{frame}.txt"]
        if mod == "ir":  # the fusion labels are the IR labels
            targets.append(new_fusion / "labels" / clip / f"{frame}.txt")
        for i, path in enumerate(targets):
            lines = path.read_text().splitlines()
            result, counts = apply_to_lines(lines, fx, (clip, frame, mod), problems if i == 0 else [])
            path.write_text("\n".join(result) + ("\n" if result else ""))
            if i == 0:
                done.update(counts)
    if problems:
        raise SystemExit("problems while applying:\n  " + "\n  ".join(problems))

    summary = {k: v for k, v in done.items() if k != "keep"}
    source = read_info(tiny / "yolo_v0" / "labels_info.txt").get("source_sha256", "unknown")
    text = "\n".join([f"labels version {a.version}", "parent: v0 (the authors' labels, RGBT-Tiny)",
                      f"source_sha256: {source}", f"fixes_sha256: {sha}", f"fixes_rows: {sum(len(v) for v in fixes.values())}",
                      f"applied: {summary or 'none'}", f"code_commit: {git('rev-parse', 'HEAD')}"])
    for root in (new_yolo, new_fusion):
        (root / "labels_info.txt").write_text(text + "\n")
    if a.seal:
        info = read_info(folder / "info.yaml")
        info["fixes_sha256"] = sha
        (folder / "info.yaml").write_text("".join(f"{k}: {v}\n" for k, v in info.items()))
    print(text)
    print(f"-> {new_yolo}\n-> {new_fusion}" + ("" if sealed or a.seal else "\nnot sealed yet: run with --seal when the fixes are final"))


if __name__ == "__main__":
    main()
