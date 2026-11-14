"""Check a split version and write its checks.txt.

    python tools/check_split.py v3                     # structure checks only
    python tools/check_split.py v3 --data ~/ias0360/data --write   # + frames and boxes, writes checks.txt

Reads splits/<version>/assignment.csv (columns dataset, clip, group, split). Fails
(exit 1) if:
  - a row is missing a column, or (dataset, clip) appears twice
  - a split name is not train, val or test
  - a group is in more than one split
  - a recording (rgbt clip name without its last _N) is in more than one split
  - train, val or test has no clip
  - val or test contains a dataset other than rgbt
With --data (the folder that holds rgbt-tiny/ and anti-uav/) it also counts frames and
IR boxes per split and class. Run it after making a split and before every training.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]


def recording(clip):
    return clip.rsplit("_", 1)[0]


def check(rows):
    errors = []
    for k in ("dataset", "clip", "group", "split"):
        if any(not r.get(k) for r in rows):
            errors.append(f"empty column: {k}")
    dup = [k for k, n in Counter((r["dataset"], r["clip"]) for r in rows).items() if n > 1]
    if dup:
        errors.append(f"clips listed twice: {dup[:5]}")
    bad = {r["split"] for r in rows} - {"train", "val", "test"}
    if bad:
        errors.append(f"unknown split names: {sorted(bad)}")
    groups = defaultdict(set)
    recs = defaultdict(set)
    for r in rows:
        groups[(r["dataset"], r["group"])].add(r["split"])
        if r["dataset"] == "rgbt":
            recs[recording(r["clip"])].add(r["split"])
    cut = [g for g, s in groups.items() if len(s) > 1]
    if cut:
        errors.append(f"groups in more than one split: {cut[:5]} ({len(cut)} in all)")
    cut = [k for k, s in recs.items() if len(s) > 1]
    if cut:
        errors.append(f"recordings in more than one split: {cut[:5]} ({len(cut)} in all)")
    for sp in ("train", "val", "test"):
        if not any(r["split"] == sp for r in rows):
            errors.append(f"split {sp} is empty")
    if any(r["split"] in ("val", "test") and r["dataset"] != "rgbt" for r in rows):
        errors.append("val or test contains a dataset other than rgbt")
    return errors


def counts(rows, data):
    out = []
    tiny = Path(data) / "rgbt-tiny" / "yolo" / "labels"
    for sp in ("train", "val", "test"):
        sel = [r for r in rows if r["dataset"] == "rgbt" and r["split"] == sp]
        frames, boxes = 0, Counter()
        for r in sel:
            for f in (tiny / r["clip"] / "01").glob("*.txt"):
                frames += 1
                for line in f.read_text().splitlines():
                    boxes[NAMES[int(line.split()[0])]] += 1
        out.append(f"rgbt {sp:5}: {len(sel):3d} clips, {len({r['group'] for r in sel}):2d} groups, "
                   f"{frames:6d} frames, boxes (IR) {dict((n, boxes[n]) for n in NAMES)}")
    sa = [r for r in rows if r["dataset"] == "auav"]
    if sa:
        out.append(f"auav train: {len(sa)} videos")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version")
    p.add_argument("--data", help="folder with rgbt-tiny/ and anti-uav/ (adds frame and box counts)")
    p.add_argument("--write", action="store_true", help="write splits/<version>/checks.txt")
    a = p.parse_args()
    folder = ROOT / "splits" / a.version
    rows = list(csv.DictReader(open(folder / "assignment.csv", encoding="utf-8-sig")))
    errors = check(rows)
    lines = counts(rows, a.data) if a.data else [f"{Counter(r['split'] for r in rows if r['dataset'] == 'rgbt')} rgbt clips"]
    lines += [f"checks: {'FAILED' if errors else 'passed'} (no group or recording in two splits, no clip twice, "
              "no empty split, val and test only rgbt)"] + [f"ERROR: {e}" for e in errors]
    print("\n".join(lines))
    if a.write and not errors:
        (folder / "checks.txt").write_text("\n".join(lines) + "\n")
        print(f"wrote {folder / 'checks.txt'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
