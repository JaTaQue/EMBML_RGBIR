# Metadata

Tables **about** the data (not the pictures or the labels). They are used to
1. decide the split (which clips belong together, are train, val and test alike?),
2. analyse results (where does the model fail: night, tiny drones, one place?),
3. find bad data (odd boxes, odd clips).

Training does not read them. Training reads `splits/vN/assignment.csv` (the decision).
The data is RGBT-Tiny (`dataset = rgbt`) and Anti-UAV (`dataset = auav`), in the same tables.

## Status

`sessions.csv` was rebuilt on 2026-11-15 from a visual review of the sessions (film strips) and covers **both datasets**: 39 RGBT-Tiny
sessions (`S###`) and 37 Anti-UAV sessions (`A###`, one per recording). In `clips_rgbt_tiny.csv` only the `session` column was updated to the
new session IDs; its other columns are still Tanishq's first version, and the Anti-UAV videos are not in it yet. Session IDs were renumbered,
so the IDs in `splits/v3/assignment.csv` (a finished version) are the **old** ones. Not done yet: the script (`tools/make_metadata.py`),
the box table, the class counts and drone columns of `clips.csv`.

## Levels and links

```
boxes.csv   -- clip    -->  clips.csv   -- session -->  sessions.csv
(one row per box)           (one row per clip)         (one row per group of clips)
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
| `confidence`, `evidence`, `comment_visual_check` | by hand | how sure the grouping is (`sure` or `unsure`), what it is based on, and what was checked by eye |

A session is a group of clips from the same place and time. We made the grouping ourselves, the authors give none.
For Anti-UAV we start with one session per recording (the `date_time` part of the video name).

## clips.csv: one row per clip (RGBT-Tiny clips and Anti-UAV videos). All made by the script

| Column | Meaning |
|---|---|
| `clip` (key), `dataset` | for example `DJI_0257_2` |
| `recording`, `session` | the flight (`DJI_0257`) and the group. Clips of one recording are always in one session |
| `official_split` | the authors' train or test (`NA` for auav) |
| `n_frames` | frames in the clip |
| `n_ship` ... `n_plane` | boxes per class |
| `drone_boxes`, `drone_size_median`, `drone_contrast_median` | how many drones, how big, how visible |
| `brightness`, `light` | `night` if the 25th percentile of the grey RGB picture is below 40, or below 60 while the mean is below 90; else `day` (the old rule, mean below 60, called lit night streets day) |
| `align_err_px`, `tags` | only auav: error of the RGB-IR alignment, challenge tags |

## boxes: one row per annotation

| Column | Meaning |
|---|---|
| `box_id` (key), `dataset`, `clip`, `frame`, `modality`, `class` | `box_id` is `rgbt/<clip>/<frame>/<rgb or ir>/<n>` |
| `track_id` | the same object over the frames (rgbt only) |
| `size_px` | sqrt(width x height) of the box |
| `ir_contrast`, `ir_clutter`, `rgb_contrast`, `offset_px` | how visible the drone is; distance between its RGB and IR box. Measured for drones only |

There are two files. `boxes_drone.csv` (drones only: about 25 000 RGBT-Tiny rows, plus the Anti-UAV drones) is small and goes in git.
The full `boxes.csv` (all classes, about 1.2 million RGBT-Tiny rows) is too big, so each person builds it locally.

## Rules

- Columns made by the script are **never edited by hand**. The script rebuilds them and stops if a file no longer matches the data.
  The hand columns of `sessions.csv` are kept when the script runs.
- The split is **not** copied here. `splits/vN/assignment.csv` is the only place for it.
- Label fixes are not here either. They are in `annotations/vN/fixes.csv`.
- The authors' pictures and labels are not in git, and these tables hold no copy of them.

## Checks

- Every `clip` in a box row exists in `clips.csv`, and every `session` in `clips.csv` exists in `sessions.csv`.
- Every session has at least one clip. Counts in `sessions.csv` and `clips.csv` equal what the label files say.
- No empty `light` or `n_frames`.
