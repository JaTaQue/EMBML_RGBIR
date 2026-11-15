"""Convert the RGBT-Tiny COCO labels to YOLO labels. No splits here, only the conversion.

    python tools/convert_rgbt.py [--version v0] [--data <data folder>]

Reads <data>/rgbt-tiny/raw/annotations_coco/instances_{00,01}_{train,test}2017.json (00 = RGB,
01 = IR; the official train and test are only read to get all 115 clips, the split is made
elsewhere) and writes two label roots for the same version:

  <data>/rgbt-tiny/yolo_<version>/images  -> ../raw/images            (a link, no copy)
  <data>/rgbt-tiny/yolo_<version>/labels/<clip>/<00|01>/<frame>.txt   RGB and IR labels
  <data>/rgbt-tiny/fusion_<version>/images -> ../fusion/images         (4-channel PNGs, see
                                                                       convert_fusion_images.py)
  <data>/rgbt-tiny/fusion_<version>/labels/<clip>/<frame>.txt          the IR labels (RGB is
                                                                       warped onto the IR grid)

Why a folder per version: YOLO finds a label by swapping /images/ for /labels/ in the image path,
so each label version needs its own root with an `images` link next to its `labels`. The images
are shared by all versions. v0 is the authors' labels as they are; fixed labels get a new version.

A label line is `class cx cy w h`, centre and size scaled 0 to 1 by the image size. Boxes are cut
to the image; a box with no area left is dropped; boxes the authors marked `ignore` or `iscrowd` are
skipped. A frame with no objects gets an empty file (a background image).

The summary and the box counts per class are printed, and written to labels_info.txt in each root, with a hash of
the authors' files. If annotations/authors_files.sha256 exists, the hash must match it, so everybody builds v0 from the same files.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from common import data_dir

ROOT = Path(__file__).resolve().parents[1]

NAMES = {0: "ship", 1: "car", 2: "cyclist", 3: "pedestrian", 4: "bus", 5: "drone", 6: "plane"}
MODS = {"00": "rgb", "01": "ir"}


def yolo_line(cls, bbox, width, height):
    """COCO [x, y, w, h] (top-left, pixels) -> 'cls cx cy w h' scaled 0..1, cut to the image, or None."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"


def frames_of(raw, mod):
    """{file_name: (width, height, [(class, bbox), ...])} over the official train and test of one modality."""
    out, skipped = {}, 0
    for split in ("train", "test"):
        with open(raw / "annotations_coco" / f"instances_{mod}_{split}2017.json") as f:
            d = json.load(f)
        images = {im["id"]: im for im in d["images"]}
        for im in d["images"]:
            out[im["file_name"]] = (im["width"], im["height"], [])
        for a in d["annotations"]:
            if a.get("ignore", 0) or a.get("iscrowd", 0):
                skipped += 1
                continue
            out[images[a["image_id"]]["file_name"]][2].append((a["category_id"], a["bbox"]))
    return out, skipped


def link(path, target):
    """Make path a link to target (relative); keep it if it already points there."""
    if path.is_symlink() or path.exists():
        if path.is_symlink() and os.readlink(path) == target:
            return
        raise SystemExit(f"{path} exists and is not the expected link to {target}")
    path.symlink_to(target)


def write_labels(frames, root, flat):
    """Write one label file per frame. flat: labels/<clip>/<frame>.txt, else labels/<clip>/<mod>/<frame>.txt."""
    boxes, dropped, empty = Counter(), 0, 0
    for fn, (width, height, objects) in frames.items():
        clip, mod, frame = fn.split("/")
        path = root / "labels" / clip / (Path(frame).stem + ".txt" if flat else f"{mod}/{Path(frame).stem}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for cls, bbox in objects:
            line = yolo_line(cls, bbox, width, height)
            if line is None:
                dropped += 1
            else:
                lines.append(line)
                boxes[NAMES[cls]] += 1
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        empty += not lines
    return boxes, dropped, empty


def source_hash(raw):
    """One hash over the four COCO files of the authors, so we can tell that two copies are the same."""
    h = hashlib.sha256()
    for mod in MODS:
        for split in ("train", "test"):
            h.update((raw / "annotations_coco" / f"instances_{mod}_{split}2017.json").read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v0", help="label version (default v0, the authors' labels)")
    p.add_argument("--data", help="folder with rgbt-tiny/ (default: environment variable EMBML_DATA)")
    a = p.parse_args()
    tiny = data_dir(a.data) / "rgbt-tiny"
    raw = tiny / "raw"
    sha = source_hash(raw)
    recorded = ROOT / "annotations" / "authors_files.sha256"
    if recorded.exists() and recorded.read_text().split()[0] != sha:
        raise SystemExit(f"the authors' COCO files on this computer differ from the ones the team used\n"
                         f"  here:   {sha}\n  team:   {recorded.read_text().split()[0]} (annotations/authors_files.sha256)\n"
                         "Labels built from different files are not the same version. Ask the others which download they have.")
    yolo, fusion = tiny / f"yolo_{a.version}", tiny / f"fusion_{a.version}"
    for root in (yolo, fusion):
        root.mkdir(parents=True, exist_ok=True)
    link(yolo / "images", "../raw/images")
    link(fusion / "images", "../fusion/images")

    report = []
    ir_frames = None
    for mod, name in MODS.items():
        frames, ignored = frames_of(raw, mod)
        boxes, dropped, empty = write_labels(frames, yolo, flat=False)
        report.append(f"{name}: {len(frames)} frames, {sum(boxes.values())} boxes, {empty} empty frames, "
                      f"{ignored} ignored or crowd, {dropped} cut to nothing; per class {dict(boxes)}")
        if mod == "01":
            ir_frames = frames
    # fusion labels are the IR labels, with the clip/<frame> layout of the 4-channel PNGs
    write_labels(ir_frames, fusion, flat=True)
    text = "\n".join([f"labels version {a.version}", "source: RGBT-Tiny COCO json, official train and test pooled",
                      f"source_sha256: {sha}"] + report)
    for root in (yolo, fusion):
        (root / "labels_info.txt").write_text(text + "\n")
    print(text)
    print(f"-> {yolo}\n-> {fusion}")
    if not recorded.exists():
        print(f"note: no hash recorded in the repo yet; to make these files the team reference, write the source_sha256 "
              f"above into {recorded}")


if __name__ == "__main__":
    main()
