# tools/

Run from the repo root. Set the data folder once per computer: `export EMBML_DATA=<data folder>`
(it holds `rgbt-tiny/` and `anti-uav/`; the data itself is not in git).

## Convert (once per computer, from the authors' files)

| Tool | Does |
|---|---|
| `convert_rgbt.py` | RGBT-Tiny COCO labels to YOLO labels v0, the authors' labels (`yolo_v0/`, `fusion_v0/`) |
| `convert_antiuav.py` | Anti-UAV videos to JPG frames and YOLO labels: aligns RGB to IR, keeps every 10th frame |
| `convert_fusion_images.py` | RGB and IR frames to 4-channel PNGs (`--dataset rgbt` or `auav`) |

Order: `convert_rgbt.py`, `convert_antiuav.py`, then `convert_fusion_images.py` for each dataset
(only needed for fusion).

## Model for fusion

| Tool | Does |
|---|---|
| `make_4ch_weights.py` | copies a pretrained 3-channel YOLO model into one whose first layer takes 4 channels (R, G, B, IR) |

## Share a run

| Tool | Does |
|---|---|
| `save_run.py` | adds one row to `results/runs.csv` and copies the run's small record files; `--model` also copies the model into `models/` |

## Fix labels

| Tool | Does |
|---|---|
| `apply_fixes.py` | the authors' labels (v0) plus `annotations/<version>/fixes.csv` gives the labels of that version (`yolo_v1/`, `fusion_v1/`). See `annotations/README.md` |

## Split, build and train

| Tool | Does |
|---|---|
| `check_split.py` | checks `splits/<version>/assignment.csv` for leaks and writes `checks.txt` |
| `build_lists.py` | checks the split, then writes the image lists, the data yaml and `split_info.yaml`. `--labels` picks the label version (default v0) |
| `train.py` | trains YOLO on a data yaml, checks the channels, writes `run_info.yaml` |

`common.py` holds the shared helper for the data folder.

## Not connected yet

Anti-UAV labels are not versioned: `build_lists.py` takes its frames from `anti-uav/fusion_all.txt`, which
the old script `antiuav_fusion.py` made (it is not in this repo).
