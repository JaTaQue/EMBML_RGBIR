"""Build Metadata/boxes_drone.csv: one row per drone box and modality (see Metadata/readme.md).

    python tools/make_boxes_drone.py [--labels v0] [--data <data folder>]

Run tools/make_metadata.py first: the clips of Metadata/clips.csv with a drone are the ones read here, and
Metadata/info.yaml must say that clips.csv was built from the same label version as --labels.
After it ran, info.yaml has an entry for boxes_drone.csv.

box_id is the same in every label version. A fix in annotations/vN/fixes.csv names a box by its line number in the v0
label file, but a `remove` shifts the lines of the same frame in vN. So the lines of vN are mapped back by replaying
fixes.csv (the order is the one apply_fixes.py writes): a box that survives keeps its v0 number, an added box is called
new1, new2, ... as in fixes.csv. Track IDs follow the same mapping (a box relabelled to drone, or added, has none: NA).

Boxes come from the YOLO labels the models train on, so sizes are on the 640 x 512 picture and the
counts equal clips.csv:
  RGBT-Tiny  <data>/rgbt-tiny/yolo_v0/labels/<clip>/{00,01}/<frame>.txt   (every frame)
  Anti-UAV   <data>/anti-uav/yolo/labels/<video>/{00,01}/<frame>.txt      (the kept frames; RGB is warped onto the IR grid)
00 = RGB, 01 = IR. The script stops if the drone boxes of a clip differ from drone_boxes_ir / drone_boxes_rgb in clips.csv.

Columns:
  box_id       <dataset>/<clip>/<frame>/<rgb or ir>/<n>. n is the line number (from 0) of the box in the YOLO label file of
               that frame and modality (all classes counted), the same numbering as annotations/vN/fixes.csv
  track_id     the authors' tracking_id (RGBT-Tiny only; found by replaying the label conversion, and checked against the
               label file line of every box no fix touched). NA for Anti-UAV and for boxes a fix made drones
  review       from fixes.csv: not_reviewed, reviewed_ok (action ok), fixed (relabel or resize) or added. Anti-UAV: not_reviewed
  x1 y1 x2 y2  the box in pixels, size_px = sqrt(width x height)
  ir_contrast, ir_clutter     IR rows: the brightest pixel in the box minus the mean of a ring around it, and the standard
                              deviation of the ring. The ring is the box grown by its longer side on every side
  rgb_contrast                RGB rows: the largest absolute difference between a pixel in the box and the ring mean
                              (grey picture). Anti-UAV RGB is the warped picture, so rows near the edge of the RGB view
                              can have black pixels in the ring
The contrast columns of the other modality are NA. Not here yet: the RGB-IR pairing and offset_px.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from apply_fixes import load_fixes
from common import data_dir
from convert_rgbt import yolo_line

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "Metadata"
DRONE, W, H = 5, 640, 512
MOD = {"00": "rgb", "01": "ir"}

HEADER = ["box_id", "dataset", "clip", "frame", "modality", "track_id", "x1", "y1", "x2", "y2", "size_px",
          "ir_contrast", "ir_clutter", "rgb_contrast", "review"]


def regions(gray, box):
    """(pixels inside the box, pixels of the ring around it), or None if one of them is empty."""
    x0, y0, x1, y1 = box
    m = max(x1 - x0, y1 - y0)
    X0, Y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    X1, Y1 = min(gray.shape[1], int(np.ceil(x1 + m))), min(gray.shape[0], int(np.ceil(y1 + m)))
    big = gray[Y0:Y1, X0:X1].astype(float)
    mask = np.ones(big.shape, bool)
    iy0, ix0 = int(y0) - Y0, int(x0) - X0
    mask[iy0:max(iy0 + 1, int(np.ceil(y1)) - Y0), ix0:max(ix0 + 1, int(np.ceil(x1)) - X0)] = False
    if not (~mask).any() or not mask.any():
        return None
    return big[~mask], big[mask]


def measure(gray, box, modality):
    """(ir_contrast, ir_clutter, rgb_contrast) for one box; the ones of the other modality are NA."""
    r = regions(gray, box)
    if r is None:
        return "NA", "NA", "NA"
    inner, ring = r
    if modality == "ir":
        return round(float(inner.max() - ring.mean()), 2), round(float(ring.std()), 2), "NA"
    return "NA", "NA", round(float(np.abs(inner - ring.mean()).max()), 2)


def track_ids(raw):
    """{(clip, frame, modality, n): (tracking_id, label line)} for the RGBT-Tiny drones.

    Replays convert_rgbt.py: boxes marked ignore or crowd are skipped, a box with no area left after the cut
    to the picture is dropped, and n counts the lines that remain in the frame, in file order."""
    out = {}
    for mod in MOD:
        for split in ("train", "test"):
            with open(raw / "annotations_coco" / f"instances_{mod}_{split}2017.json") as f:
                d = json.load(f)
            size = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in d["images"]}
            n_of = {}
            for a in d["annotations"]:
                if a.get("ignore", 0) or a.get("iscrowd", 0):
                    continue
                width, height, name = size[a["image_id"]]
                line = yolo_line(a["category_id"], a["bbox"], width, height)
                if line is None:
                    continue
                n = n_of.get(a["image_id"], 0)
                n_of[a["image_id"]] = n + 1
                if a["category_id"] == DRONE:
                    clip, _, frame = name.split("/")
                    out[(clip, Path(frame).stem, MOD[mod], n)] = (a.get("tracking_id", "NA"), line)
            del d
    return out


def frame_ids(n_v0, fx):
    """[(id, review)] for the lines of the label file of one frame, in file order.

    n_v0 = number of lines of the frame in v0, fx = the fixes of the frame ({n: (action, value, old_box)}) or None.
    Same order as apply_fixes.apply_to_lines: the boxes of v0 without the removed ones, then the added ones."""
    if not fx:
        return [(str(n), "not_reviewed") for n in range(n_v0)]
    out = []
    for n in range(n_v0):
        action = fx.get(str(n), ("keep", "", ""))[0]
        if action != "remove":
            out.append((str(n), {"keep": "not_reviewed", "ok": "reviewed_ok"}.get(action, "fixed")))
    return out + [(k, "added") for k in fx if k.startswith("new")]


def read_drones(label_dir, v0_dir, fixes, clip, name):
    """[(frame, id, review, label line)] for the drone lines of the label files in label_dir.

    v0_dir and fixes are given for RGBT-Tiny (ids are mapped back to v0), None for Anti-UAV (unversioned)."""
    out = []
    for f in sorted(label_dir.glob("*.txt")):
        lines = f.read_text().splitlines()
        if fixes is None:
            ids = [(str(n), "not_reviewed") for n in range(len(lines))]
        else:
            fx = fixes.get((clip, f.stem, name))
            n_v0 = len((v0_dir / f.name).read_text().splitlines()) if fx else len(lines)
            ids = frame_ids(n_v0, fx)
        if len(ids) != len(lines):
            raise SystemExit(f"{clip}/{name}/{f.stem}: {len(lines)} lines, but the fixes give {len(ids)}")
        for (i, review), line in zip(ids, lines):
            if line.split()[0] == str(DRONE):
                out.append((f.stem, i, review, line))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="v0", help="RGBT-Tiny label version (default v0, the authors' labels)")
    p.add_argument("--data", help="data folder (else the EMBML_DATA environment variable)")
    a = p.parse_args()
    info_path = META / "info.yaml"
    info = yaml.safe_load(info_path.read_text()) if info_path.exists() else None
    if not info or info["labels"]["rgbt"] != a.labels:
        raise SystemExit(f"Metadata/info.yaml does not say that clips.csv was built from labels {a.labels}: "
                         f"run tools/make_metadata.py --labels {a.labels} first")
    data = data_dir(a.data)
    tiny, anti = data / "rgbt-tiny", data / "anti-uav"
    fixes = {}
    if a.labels != "v0":
        fixes, problems = load_fixes(ROOT / "annotations" / a.labels / "fixes.csv")
        if problems:
            raise SystemExit("problems in fixes.csv:\n  " + "\n  ".join(problems))

    with open(META / "clips.csv", newline="") as f:
        clips = [r for r in csv.DictReader(f) if int(r["drone_boxes_ir"]) + int(r["drone_boxes_rgb"]) > 0]
    print("replaying the RGBT-Tiny label conversion for the track IDs ...", flush=True)
    tracks = track_ids(tiny / "raw")

    out = []
    for c in clips:
        clip, ds = c["clip"], c["dataset"]
        label_root = tiny / f"yolo_{a.labels}" / "labels" if ds == "rgbt" else anti / "yolo" / "labels"
        image_root = tiny / "raw" / "images" if ds == "rgbt" else anti / "yolo" / "images"
        want = {"ir": int(c["drone_boxes_ir"]), "rgb": int(c["drone_boxes_rgb"])}
        for mod, name in MOD.items():
            drones = (read_drones(label_root / clip / mod, tiny / "yolo_v0" / "labels" / clip / mod, fixes, clip, name)
                      if ds == "rgbt" else read_drones(label_root / clip / mod, None, None, clip, name))
            if len(drones) != want[name]:
                raise SystemExit(f"{clip} {name}: {len(drones)} drone boxes here, {want[name]} in clips.csv")
            cache = (None, None)
            for frame, n, review, line in drones:
                if cache[0] != frame:
                    img = cv2.imread(str(image_root / clip / mod / f"{frame}.jpg"), cv2.IMREAD_GRAYSCALE)
                    if img is None or img.shape != (H, W):
                        raise SystemExit(f"{clip}/{mod}/{frame}.jpg is missing or not {W} x {H}")
                    cache = (frame, img)
                _, cx, cy, w, h = map(float, line.split())
                box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
                track = "NA"
                if ds == "rgbt" and n.isdigit():
                    entry = tracks.get((clip, frame, name, int(n)))
                    if review in ("not_reviewed", "reviewed_ok"):  # no fix changed the line: it must be the v0 box
                        if entry is None or entry[1] != line:
                            raise SystemExit(f"{clip}/{mod}/{frame} box {n} does not match the COCO replay")
                    if entry is not None:
                        track = entry[0]
                ir_c, ir_cl, rgb_c = measure(cache[1], box, name)
                out.append([f"{ds}/{clip}/{frame}/{name}/{n}", ds, clip, frame, name, track,
                            *(round(v, 2) for v in box), round((w * W * h * H) ** 0.5, 2), ir_c, ir_cl, rgb_c, review])

    with open(META / "boxes_drone.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(out)
    info["built"]["boxes_drone.csv"] = {"script": "tools/make_boxes_drone.py", "labels": a.labels,
                                        "date": datetime.date.today().isoformat(), "rows": len(out)}
    with open(info_path, "w") as f:
        f.write("# Which label version the label-dependent tables in this folder were built from. Written by\n"
                "# tools/make_metadata.py (it clears the entry of boxes_drone.csv) and tools/make_boxes_drone.py.\n")
        yaml.safe_dump(info, f, sort_keys=False)
    for ds in ("rgbt", "auav"):
        for name in ("ir", "rgb"):
            print(f"{ds} {name}: {sum(1 for r in out if r[1] == ds and r[4] == name)} drone boxes")
    print(f"Metadata/boxes_drone.csv: {len(out)} rows")


if __name__ == "__main__":
    main()
