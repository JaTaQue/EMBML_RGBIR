"""Anti-UAV RGBT (Anti-UAV300) videos -> RGBT-Tiny style YOLO folders, RGB warped onto the IR grid.

    python tools/convert_antiuav.py [--step 10] [--max-err 10] [--data <data folder>] [--videos <name> ...]

Reads <data>/anti-uav/raw/{train,val,test}/<video>/{infrared,visible}.{mp4,json} and
<data>/anti-uav/raw/label_new/<split>.json (challenge tags).

Per video, IR = s * RGB + t (one scale, a shift) is fitted by least squares on the centres of the RGB
and IR drone boxes of all frames where both exist. Videos with a median residual above --max-err px
are dropped (mostly ones where the RGB camera zooms during the video, so one transform does not fit).
Every --step-th frame is written: IR as is (640x512), RGB warped to 640x512 (rows the RGB camera does
not see are black). Labels use the RGBT-Tiny class ids (drone = 5); frames where the drone is not
visible get an empty label file (a background image).

Writes, under <data>/anti-uav/:
    yolo/images/<video>/{00,01}/<frame>.jpg   (00 = RGB, 01 = IR, as RGBT-Tiny)
    yolo/labels/<video>/{00,01}/<frame>.txt
    yolo/{rgb,ir}_{train,val,test}.txt        the official Anti-UAV split; our own split ignores it
    videos.csv                                fit, error, kept, challenge tags per video

The 4-channel images are made next: python tools/convert_fusion_images.py --dataset auav
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from common import data_dir

DRONE, W, H = 5, 640, 512


def fit(ir, vi):
    """(s, tx, ty, median residual px) of IR = s * RGB + t from the box centres."""
    ok = [i for i in range(min(len(ir["exist"]), len(vi["exist"]))) if ir["exist"][i] and vi["exist"][i]
          and len(ir["gt_rect"][i]) == 4 and len(vi["gt_rect"][i]) == 4]
    if len(ok) < 20:
        return None
    c = lambda r: [r[0] + r[2] / 2, r[1] + r[3] / 2]
    a = np.array([c(vi["gt_rect"][i]) for i in ok])
    b = np.array([c(ir["gt_rect"][i]) for i in ok])
    A = np.c_[a.reshape(-1), np.tile([1, 0], len(a)), np.tile([0, 1], len(a))]
    s, tx, ty = np.linalg.lstsq(A, b.reshape(-1), rcond=None)[0]
    res = np.linalg.norm((A @ [s, tx, ty] - b.reshape(-1)).reshape(-1, 2), axis=1)
    return s, tx, ty, float(np.median(res))


def yolo_line(r):
    """YOLO label line for an x, y, w, h box in px on the 640x512 grid, clipped; None if outside."""
    x0, y0, x1, y1 = max(r[0], 0), max(r[1], 0), min(r[0] + r[2], W), min(r[1] + r[3], H)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return f"{DRONE} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


def convert(d, s, tx, ty, ir, vi, step, out):
    M = np.float32([[s, 0, tx], [0, s, ty]])
    for m in ("00", "01"):
        (out / "images" / d.name / m).mkdir(parents=True, exist_ok=True)
        (out / "labels" / d.name / m).mkdir(parents=True, exist_ok=True)
    cap_ir, cap_vi = cv2.VideoCapture(str(d / "infrared.mp4")), cv2.VideoCapture(str(d / "visible.mp4"))
    n = min(len(ir["exist"]), len(vi["exist"]))
    for i in range(n):
        ok1, im_ir = cap_ir.read()
        ok2, im_vi = cap_vi.read()
        if not (ok1 and ok2):
            break
        if i % step:
            continue
        name = f"{i:05d}"
        cv2.imwrite(str(out / "images" / d.name / "01" / f"{name}.jpg"), cv2.resize(im_ir, (W, H)))
        cv2.imwrite(str(out / "images" / d.name / "00" / f"{name}.jpg"), cv2.warpAffine(im_vi, M, (W, H)))
        r_ir = ir["gt_rect"][i] if ir["exist"][i] and len(ir["gt_rect"][i]) == 4 else None
        r_vi = vi["gt_rect"][i] if vi["exist"][i] and len(vi["gt_rect"][i]) == 4 else None
        if r_vi:
            r_vi = [r_vi[0] * s + tx, r_vi[1] * s + ty, r_vi[2] * s, r_vi[3] * s]
        for m, r in (("01", r_ir), ("00", r_vi)):
            line = yolo_line(r) if r else None
            (out / "labels" / d.name / m / f"{name}.txt").write_text(line + "\n" if line else "")
    return n


def main(step, max_err, data, only):
    root = data_dir(data) / "anti-uav"
    raw, out = root / "raw", root / "yolo"
    rows, lists = [], {}
    for split in ("train", "val", "test"):
        tags = json.load(open(raw / "label_new" / f"{split}.json"))
        for d in sorted((raw / split).iterdir()):
            if only and d.name not in only:
                continue
            ir, vi = json.load(open(d / "infrared.json")), json.load(open(d / "visible.json"))
            f = fit(ir, vi)
            keep = f is not None and f[3] <= max_err
            row = dict(video=d.name, split=split, kept=int(keep), tags=" ".join(tags.get(d.name, [])),
                       scale="", tx="", ty="", err="", frames=len(ir["exist"]))
            if f:
                row.update(scale=round(f[0], 4), tx=round(f[1], 1), ty=round(f[2], 1), err=round(f[3], 1))
            rows.append(row)
            if keep:
                convert(d, *f[:3], ir, vi, step, out)
                for m, mod in (("00", "rgb"), ("01", "ir")):
                    lists.setdefault((mod, split), []).extend(
                        str(p) for p in sorted((out / "images" / d.name / m).glob("*.jpg")))
            print(f"{split:5} {d.name}  err {row['err']}  {'kept' if keep else 'dropped'}", flush=True)
    for (mod, split), paths in lists.items():
        (out / f"{mod}_{split}.txt").write_text("\n".join(paths) + "\n")
    with open(root / "videos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kept = [r for r in rows if r["kept"]]
    print(f"\n{len(kept)} of {len(rows)} videos kept, "
          + ", ".join(f"{s} {len(lists.get(('ir', s), []))} frames" for s in ("train", "val", "test")))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=int, default=10, help="keep every n-th frame")
    p.add_argument("--max-err", type=float, default=10, help="drop videos whose alignment error is above this (px)")
    p.add_argument("--data", help="folder with anti-uav/ (default: environment variable EMBML_DATA)")
    p.add_argument("--videos", nargs="+", help="only these videos (default: all)")
    a = p.parse_args()
    main(a.step, a.max_err, a.data, a.videos)
