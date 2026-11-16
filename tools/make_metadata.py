"""Build Metadata/clips.csv: one row per RGBT-Tiny clip and per kept Anti-UAV video (see Metadata/readme.md).

    python tools/make_metadata.py [--data <data folder>] [--frames 12]

RGBT-Tiny: reads the authors' IR and RGB labels (<data>/rgbt-tiny/raw/annotations_coco/instances_{01,00}_*.json)
and the RGB pictures (for brightness).
Anti-UAV: reads <data>/anti-uav/videos.csv (the 192 videos with kept = 1), our YOLO labels
(<data>/anti-uav/yolo/labels/<video>/{01,00}/*.txt, RGB warped onto the IR grid, one label file per kept frame)
and the original visible video (for brightness: the warped pictures have black rows, so they are not used).
Both: Metadata/sessions.csv says which recordings form a session. Writes Metadata/clips.csv.

Where each column comes from:
  clip, recording, n_frames                   the IR label files (n_frames = IR pictures of the clip). For Anti-UAV the
                                              clip is the video, the recording is its date_time part (video name without the
                                              last two fields) and n_frames = kept frames (every 10th). The authors' train
                                              and test split is not in the table: it is in their label files
  session                                     Metadata/sessions.csv: the recording of the clip is listed there
  drone_boxes_ir, drone_boxes_rgb             drone boxes in the IR and in the RGB labels (they differ: the RGB-only model
                                              trains on the RGB labels, the fusion model on the IR labels)
  n_plane, n_other_boxes                      IR boxes. n_other_boxes = every box that is not a drone or a plane
  drone_size_median                           median of sqrt(width x height) of the IR drone boxes, in pixels
                                              (on the 640 x 512 picture; Anti-UAV boxes are the ones in our YOLO labels)
  brightness_p25, brightness_mean             from --frames RGB pictures spread over the clip: the median over the
                                              pictures of the 25th percentile and of the mean of the grey picture.
                                              There is no light column: day or night is set per session in sessions.csv
  align_err_px                                Anti-UAV only (NA for RGBT-Tiny): median error of the RGB-IR alignment of the
                                              video (videos.csv). The authors' challenge tags are not in the table

The script stops if a recording is in no session or in two, or if the clip, frame or drone-box counts
of a session do not match Metadata/sessions.csv.
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import data_dir

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "Metadata"
DRONE, PLANE = 5, 6

HEADER = ["clip", "dataset", "recording", "session", "n_frames", "drone_boxes_ir",
          "drone_boxes_rgb", "drone_size_median", "n_plane", "n_other_boxes",
          "brightness_p25", "brightness_mean", "align_err_px"]
W, H = 640, 512   # the picture grid of RGBT-Tiny and of our Anti-UAV labels


def read_labels(raw, mod):
    """Per clip from the labels of one modality (01 = IR, 00 = RGB): official split, frame count, drone sizes, box counts."""
    split_of, frames, sizes = {}, Counter(), defaultdict(list)
    counts = defaultdict(Counter)
    for split in ("train", "test"):
        with open(raw / "annotations_coco" / f"instances_{mod}_{split}2017.json") as f:
            d = json.load(f)
        clip_of = {}
        for im in d["images"]:
            clip = im["file_name"].split("/")[0]
            clip_of[im["id"]] = clip
            frames[clip] += 1
            if split_of.setdefault(clip, split) != split:
                raise SystemExit(f"clip {clip} is in both the official train and test")
        for a in d["annotations"]:
            if a.get("ignore", 0) or a.get("iscrowd", 0):
                continue
            clip = clip_of[a["image_id"]]
            k = a["category_id"]
            counts[clip]["drone" if k == DRONE else "plane" if k == PLANE else "other"] += 1
            if k == DRONE:
                _, _, w, h = a["bbox"]
                sizes[clip].append((w * h) ** 0.5)
    return split_of, frames, sizes, counts


def read_sessions():
    """(dataset, recording) -> session, and the session rows, from Metadata/sessions.csv."""
    rec_session, rows = {}, {}
    with open(META / "sessions.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["session"]] = r
            for rec in r["recordings"].split():
                key = (r["dataset"], rec)
                if key in rec_session:
                    raise SystemExit(f"recording {rec} is in two sessions: {rec_session[key]} and {r['session']}")
                rec_session[key] = r["session"]
    return rec_session, rows


def brightness(raw, clip, n):
    """Median over n evenly spaced RGB pictures of the grey 25th percentile and the grey mean."""
    files = sorted((raw / "images" / clip / "00").glob("*.jpg"))
    if not files:
        raise SystemExit(f"no RGB pictures for {clip}")
    p25, mean = [], []
    for i in np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int):
        grey = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        p25.append(float(np.percentile(grey, 25)))
        mean.append(float(grey.mean()))
    return statistics.median(p25), statistics.median(mean)


def check_sessions(rows, out):
    """Clips, frames and IR drone boxes of every session must equal the ones in Metadata/sessions.csv."""
    got = defaultdict(lambda: [0, 0, 0])
    for r in out:
        g = got[r["session"]]
        g[0] += 1
        g[1] += r["n_frames"]
        g[2] += r["drone_boxes_ir"]
    bad = []
    for s, r in rows.items():
        want = [int(r["n_clips"]), int(r["n_frames"]), int(r["n_drone_boxes"])]
        if got[s] != want:
            bad.append(f"  {s}: clips, frames, drone boxes {got[s]} here, {want} in sessions.csv")
    if bad:
        raise SystemExit("counts do not match Metadata/sessions.csv:\n" + "\n".join(bad))


def read_yolo(folder):
    """{frame: [(class, width_px, height_px), ...]} from the YOLO label files of one folder."""
    out = {}
    for f in folder.glob("*.txt"):
        boxes = []
        for line in f.read_text().splitlines():
            if line.strip():
                k, _, _, w, h = line.split()
                boxes.append((int(k), float(w) * W, float(h) * H))
        out[f.stem] = boxes
    return out


def video_brightness(path, n):
    """Median over n evenly spaced pictures of a video of the grey 25th percentile and the grey mean."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise SystemExit(f"cannot read {path}")
    p25, mean = [], []
    for i in np.linspace(0, total - 1, min(n, total)).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p25.append(float(np.percentile(grey, 25)))
        mean.append(float(grey.mean()))
    if not p25:
        raise SystemExit(f"no picture could be read from {path}")
    return statistics.median(p25), statistics.median(mean)


def auav_rows(anti, nframes, rec_session):
    """One row per kept Anti-UAV video."""
    out = []
    with open(anti / "videos.csv", newline="") as f:
        videos = [r for r in csv.DictReader(f) if r["kept"] == "1"]
    for v in sorted(videos, key=lambda r: r["video"]):
        name = v["video"]
        ir = read_yolo(anti / "yolo" / "labels" / name / "01")
        rgb = read_yolo(anti / "yolo" / "labels" / name / "00")
        if not ir or set(ir) != set(rgb):
            raise SystemExit(f"{name}: the RGB and IR label files do not cover the same frames")
        rec = name.rsplit("_", 2)[0]
        if ("auav", rec) not in rec_session:
            raise SystemExit(f"recording {rec} (video {name}) is in no session of Metadata/sessions.csv")
        drones = [(w * h) ** 0.5 for b in ir.values() for k, w, h in b if k == DRONE]
        p25, mean = video_brightness(anti / "raw" / v["split"] / name / "visible.mp4", nframes)
        out.append(dict(
            clip=name, dataset="auav", recording=rec, session=rec_session[("auav", rec)], n_frames=len(ir), drone_boxes_ir=len(drones),
            drone_boxes_rgb=sum(1 for b in rgb.values() for k, _, _ in b if k == DRONE),
            drone_size_median=round(statistics.median(drones), 2) if drones else "NA",
            n_plane=sum(1 for b in ir.values() for k, _, _ in b if k == PLANE),
            n_other_boxes=sum(1 for b in ir.values() for k, _, _ in b if k not in (DRONE, PLANE)),
            brightness_p25=round(p25, 1), brightness_mean=round(mean, 1),
            align_err_px=v["err"]))
    return out


def rgbt_rows(raw, nframes, rec_session):
    """One row per RGBT-Tiny clip."""
    split_of, frames, sizes, counts = read_labels(raw, "01")
    split_rgb, frames_rgb, _, counts_rgb = read_labels(raw, "00")
    if split_rgb != split_of or frames_rgb != frames:
        raise SystemExit("the RGB and IR label files do not have the same clips, splits and frame counts")
    out = []
    for clip in sorted(split_of):
        rec = clip.rsplit("_", 1)[0]
        if ("rgbt", rec) not in rec_session:
            raise SystemExit(f"recording {rec} (clip {clip}) is in no session of Metadata/sessions.csv")
        c = counts[clip]
        p25, mean = brightness(raw, clip, nframes)
        out.append(dict(
            clip=clip, dataset="rgbt", recording=rec, session=rec_session[("rgbt", rec)],
            n_frames=frames[clip], drone_boxes_ir=c["drone"],
            drone_boxes_rgb=counts_rgb[clip]["drone"],
            drone_size_median=round(statistics.median(sizes[clip]), 2) if sizes[clip] else "NA",
            n_plane=c["plane"], n_other_boxes=c["other"],
            brightness_p25=round(p25, 1), brightness_mean=round(mean, 1), align_err_px="NA"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="data folder (else the EMBML_DATA environment variable)")
    p.add_argument("--frames", type=int, default=12, help="RGB pictures per clip for the brightness")
    a = p.parse_args()
    data = data_dir(a.data)
    rec_session, rows = read_sessions()

    out = rgbt_rows(data / "rgbt-tiny" / "raw", a.frames, rec_session)
    out += auav_rows(data / "anti-uav", a.frames, rec_session)
    check_sessions(rows, out)

    with open(META / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(out)

    for ds in ("rgbt", "auav"):
        r = [x for x in out if x["dataset"] == ds]
        print(f"{ds}: {len(r)} clips, {sum(x['n_frames'] for x in r)} frames, "
              f"{sum(x['drone_boxes_ir'] for x in r)} IR and {sum(x['drone_boxes_rgb'] for x in r)} RGB drone boxes")
    print(f"Metadata/clips.csv: {len(out)} rows")


if __name__ == "__main__":
    main()
