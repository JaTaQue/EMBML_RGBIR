"""Build YOLO image lists and data yaml files from a split version.

    python tools/build_lists.py v3 --data ~/ias0360/data --recipe fusion
    python tools/build_lists.py v3 --data ~/ias0360/data --recipe fusion_antiuav
    python tools/build_lists.py v3 --data ~/ias0360/data --recipe rgb     # or ir

Reads splits/<version>/assignment.csv and writes, per recipe,
<data>/rgbt-tiny/splits/<version>/<recipe>/{train,val,test}.txt and <recipe>.yaml.

Recipes (see config.yaml): rgb and ir use the RGBT-Tiny JPGs; fusion uses the 4-channel
PNGs (R, G, B, IR); fusion_antiuav is fusion plus the Anti-UAV frames in train.
List lines are written relative to the list file ("./../../../yolo/images/..."), so the
lists are the same on every computer. Only the yaml has an absolute `path:`, made here.
Ultralytics finds a label by swapping /images/ for /labels/ in the image path, so images
and labels must sit under the same root as in the data folder (rgbt-tiny/yolo, rgbt-tiny/fusion).
"""

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
MOD = {"rgb": "00", "ir": "01"}


def frames(data, recipe, clip):
    t = Path(data) / "rgbt-tiny"
    if recipe in ("rgb", "ir"):
        return sorted((t / "yolo" / "images" / clip / MOD[recipe]).glob("*.jpg"))
    return sorted((t / "fusion" / "images" / clip).glob("*.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version")
    p.add_argument("--data", required=True, help="folder with rgbt-tiny/ and anti-uav/")
    p.add_argument("--recipe", choices=["rgb", "ir", "fusion", "fusion_antiuav"], required=True)
    a = p.parse_args()
    data = Path(a.data).expanduser().resolve()
    rows = list(csv.DictReader(open(ROOT / "splits" / a.version / "assignment.csv", encoding="utf-8-sig")))
    out = data / "rgbt-tiny" / "splits" / a.version / a.recipe
    out.mkdir(parents=True, exist_ok=True)
    base = "fusion" if a.recipe.startswith("fusion") else a.recipe

    lists = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["dataset"] == "rgbt":
            lists[r["split"]] += frames(data, base, r["clip"])
    n = {k: len(v) for k, v in lists.items()}
    if a.recipe == "fusion_antiuav":
        videos = {r["clip"] for r in rows if r["dataset"] == "auav"}
        kept = [Path(l.strip()) for l in open(data / "anti-uav" / "fusion_all.txt") if l.strip()]
        lists["train"] += [f for f in kept if f.parts[-2] in videos]

    for split, files in lists.items():
        lines = ["./" + os.path.relpath(f, out) for f in files]
        (out / f"{split}.txt").write_text("\n".join(lines) + "\n")
    yaml = [f"# built by tools/build_lists.py from splits/{a.version}/assignment.csv, recipe {a.recipe}",
            f"path: {out}", "train: train.txt", "val: val.txt", "test: test.txt"]
    if base == "fusion":
        yaml.append("channels: 4")
    yaml += ["names:"] + [f"  {i}: {name}" for i, name in enumerate(NAMES)]
    (out / f"{a.recipe}.yaml").write_text("\n".join(yaml) + "\n")
    print(f"{a.recipe} {a.version}: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items()) + f" frames -> {out}")


if __name__ == "__main__":
    main()
