# EMBML_RGBIR

RGB + IR drone detection on small hardware. This page explains the data, split, training and results part of the repo.

## What is where

| Folder | What it holds |
|---|---|
| `Metadata/` | tables about the data: `sessions.csv` (our grouping, in git) and the tables built from the labels (`clips.csv`, `boxes_drone.csv`, local only) |
| `splits/` | which clip is train, val or test: one folder per split version |
| `annotations/` | our fixes to the authors' labels: one folder per label version |
| `results/`, `models/` | what we trained: one row per run, and the models that matter |
| `tools/` | scripts to convert the data, fix labels, build the metadata tables and lists, and train |
| `raspberry/` | tracker and stream simulator for the Raspberry Pi |

The pictures and the authors' labels are **not** in git (research use only). Each person gets them from the
authors. Only small text files are in git.

## The idea in six lines

1. The authors' labels (**annotations**) are converted once into YOLO **labels** (v0). They are never edited.
2. Our corrections are a small file, `annotations/vN/fixes.csv`. v0 plus the fixes gives labels vN.
3. The split is a table, `splits/vN/assignment.csv`: one row per clip with its group and split.
   A group (clips of one place) is never cut, so the same place is not in train and test.
4. A script turns the split and the label version into the lists that YOLO needs (`train.txt`, ...).
5. Training takes only a data yaml, and every run saves a record of the split and labels it used.
6. Runs that matter are added to a shared table, `results/runs.csv`, and the important models go to `models/`.

```
authors' data -> convert -> labels v0 -> (+ fixes) -> labels vN
                                                          |
assignment.csv ---------------------------> build_lists --+--> train.txt, val.txt, test.txt, data yaml
                                                                                  |
                                                      results/runs.csv, models/ <-+-- train
```

Pictures and labels sit side by side (`images/` and `labels/`), and YOLO finds a label by swapping `images`
for `labels` in the path. Nothing is moved into train, val or test folders: the split is only lists of names.

## Quick start

Install the packages first: `pip install -r requirements.txt` (Python 3.12).

```
export EMBML_DATA=<your data folder>          # holds rgbt-tiny/ and anti-uav/
python tools/convert_rgbt.py                  # authors' labels -> YOLO labels v0
python tools/convert_antiuav.py               # Anti-UAV videos -> frames and labels
python tools/convert_fusion_images.py         # 4-channel PNGs (R, G, B, IR); --dataset auav for Anti-UAV
python tools/apply_fixes.py v1                # labels v1 = v0 + annotations/v1/fixes.csv
python tools/build_all.py v1                  # labels (if missing) and the Metadata tables for v1: clips.csv, boxes_drone.csv
python tools/make_4ch_weights.py --model weights/yolo26n.pt --out weights/yolo26n_4ch.pt   # fusion only
python tools/build_lists.py v3 --images fusion --labels v1
python tools/train.py --data-yaml $EMBML_DATA/rgbt-tiny/splits/v3/labels_v1/fusion/fusion.yaml \
    --model weights/yolo26n_4ch.pt --name fusion_v3_v1 imgsz=640 epochs=30
python tools/save_run.py runs/fusion_v3_v1 --model --note "why this run, what it showed"   # share it
```

`--images` is `rgb`, `ir` or `fusion`. `--datasets rgbt` leaves Anti-UAV out of the lists.
Get `yolo26n.pt` first (Ultralytics downloads it the first time you load it by name) and put it in `weights/`.
Fusion needs the 4-channel copy, made by `make_4ch_weights.py`; `train.py` stops if the model and the data differ in channels.

`build_all.py vN` builds the metadata tables of a label version (it needs the Anti-UAV conversion first). They are **not in git**: everyone builds
them locally, and `Metadata/info.yaml` records which label version they came from. See `Metadata/readme.md`.

## Rules

- A split version or a label version is never edited after use. A change makes a new version.
- The tables in `Metadata/` that come from the labels are rebuilt with `tools/build_all.py` after a new label version, never edited by hand.
- Compare models only when they used the same split version and the same label version.
- `build_lists.py` stops if a group or recording is in two splits (`tools/check_split.py` does the same check), or if a picture has no label file.
- Each run folder gets `run_info.yaml` and `split_info.yaml`, with the settings, versions and hashes.
- A row in `results/runs.csv` is never edited and a model in `models/` is never replaced. A correction is a new run.

## Results so far

Two baselines on the official RGBT-Tiny split (val = test, so these are test numbers; last epoch, 30 epochs, trained on the HPC):

| Run | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| `rgb_all_n640_e30_official_s0` | RGB | 0.371 | 0.161 |
| `fusion_n640_e30_official_s0` | RGB + IR | 0.423 | 0.193 |

These are overall numbers for all 7 classes. The official split leaks places between train and test (by our session grouping, 93% of the
test frames are in places that also hold train clips), so they are a reference and not a measure of how well drones are found in new places.
The full table is `results/runs.csv`.

More details: `splits/README.md`, `annotations/README.md`, `results/README.md`, `tools/README.md`.
