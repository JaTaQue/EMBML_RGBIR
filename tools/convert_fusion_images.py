"""Stack RGB and IR frames into 4-channel PNG files (R, G, B, IR), for RGBT-Tiny or Anti-UAV.

    python tools/convert_fusion_images.py [--dataset rgbt|auav] [--data <data folder>] [--clips ...] [--workers 8]

For every frame of every clip (or only --clips), reads <clip>/00/<frame>.jpg (RGB) and
<clip>/01/<frame>.jpg (IR) from the input folder of the dataset and writes <clip>/<frame>.png with the
channels in the order R, G, B, IR:

  rgbt (default): <data>/rgbt-tiny/raw/images   ->  <data>/rgbt-tiny/fusion/images
  auav:           <data>/anti-uav/yolo/images   ->  <data>/anti-uav/fusion/images
                  (the output of the Anti-UAV converter, where RGB is already warped onto the IR view)

A frame whose PNG already exists is skipped, so a run can be repeated or resumed. The two images must
be on the same grid so the channels line up: RGBT-Tiny is aligned by the authors, Anti-UAV by our converter.

Ultralytics reads these PNGs with `channels: 4` in the data yaml and does not swap the colour order
for 4 channels, so the order on disk is the order the network sees. A PNG viewer shows odd colours
and treats IR as transparency; that is expected. The images are the same for every label version,
so they are built once. The label roots (see convert_rgbt.py) link to this folder.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from common import data_dir


def stack(job):
    src, dst = job
    if dst.exists():
        return 0
    rgb = cv2.imread(str(src / "00" / dst.with_suffix(".jpg").name), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(src / "01" / dst.with_suffix(".jpg").name), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        raise SystemExit(f"cannot read the frame pair for {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")  # write first, then rename, so an interrupted run leaves no broken file
    cv2.imwrite(str(tmp), np.dstack([cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), ir]))
    os.replace(tmp, dst)
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["rgbt", "auav"], default="rgbt")
    p.add_argument("--data", help="folder with rgbt-tiny/ and anti-uav/ (default: environment variable EMBML_DATA)")
    p.add_argument("--clips", nargs="+", help="only these clips (default: all)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    root = data_dir(a.data) / ("rgbt-tiny" if a.dataset == "rgbt" else "anti-uav")
    images = root / ("raw" if a.dataset == "rgbt" else "yolo") / "images"
    out = root / "fusion" / "images"
    clips = a.clips or sorted(d.name for d in images.iterdir() if d.is_dir())
    jobs = [(images / c, out / c / (f.stem + ".png"))
            for c in clips for f in sorted((images / c / "01").glob("*.jpg"))]
    done = 0
    with Pool(a.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(stack, jobs, chunksize=64)):
            done += n
            if i and i % 5000 == 0:
                print(f"{i}/{len(jobs)} frames")
    print(f"{len(jobs)} frames, {done} written, {len(jobs) - done} already there -> {out}")


if __name__ == "__main__":
    main()
