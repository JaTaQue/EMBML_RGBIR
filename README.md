# EMBML_RGBIR

RGB + IR drone detection on small hardware. This page explains the data, split and training part of the repo.

## What is where

| Folder | What it holds |
|---|---|
| `Metadata/` | tables about the data: sessions and clips (made by the team) |
| `splits/` | which clip is train, val or test: one folder per split version |
| `annotations/` | our fixes to the authors' labels: one folder per label version |
| `tools/` | scripts to convert the data, fix labels, build lists and train |
| `raspberry/` | tracker and stream simulator for the Raspberry Pi |

The pictures and the authors' labels are **not** in git (research use only). Each person gets them from the
authors. Only small text files are in git.

## The idea in five lines

1. The authors' labels (**annotations**) are converted once into YOLO **labels** (v0). They are never edited.
2. Our corrections are a small file, `annotations/vN/fixes.csv`. v0 plus the fixes gives labels vN.
3. The split is a table, `splits/vN/assignment.csv`: one row per clip with its group and split.
   A group (clips of one place) is never cut, so the same place is not in train and test.
4. A script turns the split and the label version into the lists that YOLO needs (`train.txt`, ...).
5. Training takes only a data yaml, and every run saves a record of the split and labels it used.

```
authors' data -> convert -> labels v0 -> (+ fixes) -> labels vN
                                                          |
assignment.csv ---------------------------> build_lists --+--> train.txt, val.txt, test.txt, data yaml -> train
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
python tools/build_lists.py v3 --images fusion --labels v1
python tools/train.py --data-yaml $EMBML_DATA/rgbt-tiny/splits/v3/labels_v1/fusion/fusion.yaml \
    --model weights/yolo26n_4ch.pt --name fusion_v3_v1 imgsz=640 epochs=30
```

`--images` is `rgb`, `ir` or `fusion`. `--datasets rgbt` leaves Anti-UAV out of the lists.

## Rules

- A split version or a label version is never edited after use. A change makes a new version.
- Compare models only when they used the same split version and the same label version.
- `build_lists.py` stops if a group or recording is in two splits (`tools/check_split.py` does the same check), or if a picture has no label file.
- Each run folder gets `run_info.yaml` and `split_info.yaml`, with the settings, versions and hashes.

More details: `splits/README.md`, `annotations/README.md`, `tools/README.md`.
