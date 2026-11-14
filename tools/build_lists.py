"""Build YOLO image lists and a data yaml from a split version's assignment.csv.

    python tools/build_lists.py v3 --data ~/ias0360/data --images fusion
    python tools/build_lists.py v3 --data ~/ias0360/data --images fusion --datasets rgbt

Every row of splits/<version>/assignment.csv goes to the split it names. --images is the
kind of image: rgb or ir (RGBT-Tiny JPGs) or fusion (4-channel PNGs, R, G, B, IR).
--datasets keeps only some datasets (default: all rows in the file).

Writes <data>/rgbt-tiny/splits/<version>/<name>/{train,val,test}.txt and <name>.yaml, where
name is the image kind, plus _<datasets> if --datasets is used. Lines are written relative
to the list file, so the lists are the same on every computer; only the yaml has a full path.
Ultralytics finds a label by swapping /images/ for /labels/ in the image path, so images
and labels must sit under the same root (rgbt-tiny/yolo, rgbt-tiny/fusion, anti-uav/fusion).
Anti-UAV (auav) only has fusion images (left out of rgb and ir); its kept frames are listed in anti-uav/fusion_all.txt.
"""

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
MOD = {"rgb": "00", "ir": "01"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version")
    p.add_argument("--data", required=True, help="folder with rgbt-tiny/ and anti-uav/")
    p.add_argument("--images", choices=["rgb", "ir", "fusion"], required=True)
    p.add_argument("--datasets", nargs="+", help="keep only these datasets (default: all in assignment.csv)")
    a = p.parse_args()
    data = Path(a.data).expanduser().resolve()
    rows = list(csv.DictReader(open(ROOT / "splits" / a.version / "assignment.csv", encoding="utf-8-sig")))
    if a.datasets:
        rows = [r for r in rows if r["dataset"] in a.datasets]
    if a.images != "fusion" and any(r["dataset"] == "auav" for r in rows):
        print("note: auav has only fusion images, so its rows are left out")
        rows = [r for r in rows if r["dataset"] != "auav"]
    name = a.images + ("_" + "+".join(a.datasets) if a.datasets else "")
    out = data / "rgbt-tiny" / "splits" / a.version / name
    out.mkdir(parents=True, exist_ok=True)

    tiny = data / "rgbt-tiny"
    lists = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["dataset"] == "rgbt":
            d = tiny / "yolo" / "images" / r["clip"] / MOD[a.images] if a.images != "fusion" else tiny / "fusion" / "images" / r["clip"]
            lists[r["split"]] += sorted(d.glob("*.jpg" if a.images != "fusion" else "*.png"))
    videos = {r["clip"]: r["split"] for r in rows if r["dataset"] == "auav"}
    if videos:  # only the frames we kept (every 10th, RGB-IR error <= 10 px)
        for line in open(data / "anti-uav" / "fusion_all.txt"):
            f = Path(line.strip())
            if line.strip() and f.parts[-2] in videos:
                lists[videos[f.parts[-2]]].append(f)

    for split, files in lists.items():
        (out / f"{split}.txt").write_text("\n".join("./" + os.path.relpath(f, out) for f in files) + "\n")
    yaml = [f"# built by tools/build_lists.py from splits/{a.version}/assignment.csv, images {a.images}",
            f"path: {out}", "train: train.txt", "val: val.txt", "test: test.txt"]
    if a.images == "fusion":
        yaml.append("channels: 4")
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / f"{name}.yaml").write_text("\n".join(yaml) + "\n")
    print(f"{name} {a.version}: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items()) + f" frames -> {out}")


if __name__ == "__main__":
    main()
