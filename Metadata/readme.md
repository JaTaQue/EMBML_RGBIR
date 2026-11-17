# Metadata

Tables **about** the data (not the pictures or the labels). They are used to
1. decide the split (which clips belong together, are train, val and test alike?),
2. analyse results (where does the model fail: night, tiny drones, one place?),
3. find bad data (odd boxes, odd clips).

Training does not read them. Training reads `splits/vN/assignment.csv` (the decision).
The data is RGBT-Tiny (`dataset = rgbt`) and Anti-UAV (`dataset = auav`), in the same tables.

## Status

`sessions.csv` was rebuilt on 2026-11-15 from a visual review of the sessions (film strips) and covers **both datasets**: 39 RGBT-Tiny
sessions (`S###`) and 37 Anti-UAV sessions (`A###`, one per recording). Session IDs were renumbered, so the IDs in
`splits/v3/assignment.csv` (a finished version) are the **old** ones.

`clips.csv` is made by `tools/make_metadata.py` and covers the **115 RGBT-Tiny clips and the 192 kept Anti-UAV videos** (307 rows). Its
counts are written into `sessions.csv` by the same script. It replaces `clips_rgbt_tiny.csv` (Tanishq's first version, deleted; it is in the git
history). `boxes_drone.csv` is made by `tools/make_boxes_drone.py`. Not done yet: the RGB-IR pairing and `offset_px`.

## Levels and links

```
boxes_drone.csv -- clip -->  clips.csv   -- session -->  sessions.csv
(one row per drone box)      (one row per clip)         (one row per group of clips)
```

Each fact is stored once, at the level where it is true. A box finds its session through its clip.

## sessions.csv: one row per group of clips

| Column | Who fills it | Meaning |
|---|---|---|
| `session` (key), `dataset` | script | group ID, `S###` for `rgbt`, `A###` for `auav` |
| `recordings`, `n_clips`, `n_frames` | script | counted from the clips (for auav: the recording, the videos, the kept frames) |
| `official_split` | script | `train`, `test` or `both` (both = the authors put clips of this place on both sides). `NA` for auav |
| `light` | script, checked by eye | `day`, `night`, `dusk` or `mixed` (the clips differ). Rule on the clips, see `clips.csv`. Checked blind on 103 clips and videos, 0 day/night contradictions |
| `scene_type` | **by hand** | one of `city street`, `junction`, `city area`, `plaza`, `sports field`, `highway`, `waterside`, `sea`, `open land`, `sky only` |
| `background` | **by hand** | what is behind the objects. One or more of `sky`, `buildings`, `vegetation`, `water`, `ground`, `mountains`, joined with `+` (`sky+buildings`) |
| `scene_check` | by hand | who checked `scene_type` and `background`: `corrected by Markkus`, `no correction by Markkus` and so on |
| `n_drone_boxes` | script | drone boxes in the group (IR labels) |
| `evidence`, `comment_visual_check` | by hand | what the grouping is based on, and what was checked by eye |

A session is a group of clips from the same place and time. We made the grouping ourselves, the authors give none.
For Anti-UAV we start with one session per recording (the `date_time` part of the video name).

## clips.csv: one row per clip (RGBT-Tiny clips and Anti-UAV videos). All made by the script

| Column | Meaning |
|---|---|
| `clip` (key), `dataset` | for example `DJI_0257_2` |
| `recording`, `session` | the flight (`DJI_0257`) and the group. Clips of one recording are always in one session |
| `n_frames` | frames in the clip |
| `drone_boxes_ir`, `drone_boxes_rgb`, `drone_size_median` | drone boxes in the IR labels and in the RGB labels, and the median size (sqrt(width x height), pixels) of the IR drone boxes. The two counts differ (13,148 IR against 11,582 RGB in total), and the RGB-only model trains on the RGB labels, the fusion model on the IR labels. The core columns: our focus is drones. Contrast is not in this table: it is per box (`boxes_drone.csv`), where it can be joined to hits and misses |
| `n_plane`, `n_other_boxes` | planes (the object most like a drone, for hard-negative checks) and all other non-drone boxes together. No count per class: the label files hold those |
| `brightness_p25`, `brightness_mean` | median over 12 RGB pictures spread over the clip of the 25th percentile and of the mean of the grey picture. There is no `light` column in `clips.csv`: day, night, dusk or mixed is set per session in `sessions.csv`. Rule used for it: `night` if `brightness_p25` is below 40, or below 60 while `brightness_mean` is below 90; else `day` (the old rule, mean below 60, called lit night streets day). Use these two numbers to find the dark clips of a `mixed` session |
| `align_err_px` | only auav (`NA` for rgbt): median error in pixels of the RGB-IR alignment of the video (`videos.csv`) |

## boxes_drone.csv: one row per drone box and modality (made by `tools/make_boxes_drone.py`)

Our focus is drones, so this is the box table that matters. It has 58 906 rows (6.5 MB) and is a local build output (see below). Boxes come from the YOLO labels the
models train on (RGBT-Tiny: every frame; Anti-UAV: the kept frames, every 10th, RGB warped onto the IR grid), so sizes are on the 640 x 512 picture
and the counts per clip equal `drone_boxes_ir` and `drone_boxes_rgb` in `clips.csv` (the script stops otherwise).

| Column | Meaning |
|---|---|
| `box_id` (key), `dataset`, `clip`, `frame`, `modality` | `box_id` is `<dataset>/<clip>/<frame>/<rgb or ir>/<n>`. `n` is the line number (from 0) of the box in the YOLO label file of that frame and modality, all classes counted, **in the v0 file**: the numbering of `annotations/vN/fixes.csv`, kept in every label version. A box added by a fix is `new1`, `new2`, ... All rows are drones, so there is no class column |
| `track_id` | the authors' tracking ID (rgbt only, `NA` for auav and for boxes a fix made drones). It is only unique inside a clip (the values are 0 and 1), so use it with `clip` |
| `review` | from `fixes.csv`: `not_reviewed`, `reviewed_ok` (action `ok`), `fixed` (relabel or resize) or `added`. Anti-UAV: `not_reviewed` |
| `x1`, `y1`, `x2`, `y2`, `size_px` | the box in pixels, and sqrt(width x height) |
| `ir_contrast`, `ir_clutter` | IR rows only: the brightest pixel in the box minus the mean of a ring around it, and the spread of the ring. The ring is the box grown by its longer side on every side. Contrast can be below zero (13 of 30 385 IR boxes) |
| `rgb_contrast` | RGB rows only: the largest absolute difference between a pixel in the box and the ring mean (grey picture). Not comparable with `ir_contrast`: it has no sign |

The columns of the other modality are `NA`. **Not here yet:** the pairing of the RGB and IR box of one drone, and `offset_px`. In RGBT-Tiny the
`track_id` is a start: in 8 590 of the 8 890 frames that have drones in both modalities the IDs agree. Every frame with an RGB drone also has an
IR drone, but 1 574 frames have an IR drone only.

**`boxes.csv` (all classes) is optional.** It is too big (about 1.2 million RGBT-Tiny rows) and nothing in the drone analysis needs it.
There is no script for it in the repo yet.

## What is in git

`sessions.csv` is in git: the grouping and the hand columns are our work, and the script only refreshes its counts. `clips.csv`,
`boxes_drone.csv` and `info.yaml` are **not** in git (see `.gitignore`). They are outputs of `tools/build_all.py` for one label version, they
describe the pictures and labels on one computer, and they can be rebuilt at any time. Everyone builds them after getting the data:
`python tools/build_all.py v0` (about 2 minutes). Nobody edits them by hand. To share a result, share the label version and the commit.

## Label versions: `info.yaml`

Counts, sizes, boxes, contrast and track IDs come from the labels, so they are only true for the label version they were built from. The authors'
labels are v0; our corrections make v1, v2, ... (`annotations/vN/fixes.csv`, applied by `tools/apply_fixes.py`). `Metadata/info.yaml` records the
version (`labels: rgbt: v0`), the hash of the authors' files, and which script built each table and when. Anti-UAV labels are not versioned yet
(`auav: unversioned`).

**One command builds everything for a version:** `python tools/build_all.py vN`. It makes the labels of the version if they are missing, then
`make_metadata.py --labels vN` (`clips.csv`, the counts in `sessions.csv`, `info.yaml`) and `make_boxes_drone.py --labels vN` (`boxes_drone.csv`).
Re-running it for the same version gives the same files. `make_metadata.py` clears the `boxes_drone.csv` entry of `info.yaml`, so a box table
from an older version is easy to spot, and `make_boxes_drone.py` stops if `info.yaml` does not match its `--labels`.

- **`box_id` is the same in every version.** A fix names a box by its line number in the **v0** label file, but a `remove` shifts the lines of that
  frame in vN. `make_boxes_drone.py` maps the lines back by replaying `fixes.csv`: a box that survives keeps its v0 number (and its track ID), an added
  box is `new1`, `new2`, ... as in `fixes.csv`. So a box found in the table can be fixed by copying its `box_id`, and tables of two versions can be
  joined on it.
- **The counts in `sessions.csv`** (`n_clips`, `n_frames`, `n_drone_boxes`) are written by `make_metadata.py`, in place, and the changes are printed.
  The hand columns are kept. It stops if a recording or a session has no clip.
- **Not label dependent**, so unchanged by a new version: the session grouping, brightness, alignment error.
- Tested on a throwaway version with one `remove`, `resize`, `ok`, `relabel` and `add`: the IDs, the `review` column, the track IDs, the
  `clips.csv` counts and the `sessions.csv` counts all came out as expected.

## Rules

- Columns made by the script are **never edited by hand**. The script rebuilds them and stops if a file no longer matches the data.
  The hand columns of `sessions.csv` are kept when the script runs.
- The split is **not** copied here. `splits/vN/assignment.csv` is the only place for it.
- Label fixes are not here either. They are in `annotations/vN/fixes.csv`.
- The authors' pictures and labels are not in git, and these tables hold no copy of them.

## Checks

- Every `clip` in a `boxes_drone.csv` row exists in `clips.csv`, and every `session` in `clips.csv` exists in `sessions.csv`.
- Every session has at least one clip. Counts in `sessions.csv` and `clips.csv` equal what the label files say.
- No empty `light` or `n_frames`.
