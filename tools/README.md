# tools/

Run from the repo root. Set the data folder once per computer: `export EMBML_DATA=<data folder>`
(it holds `rgbt-tiny/` and `anti-uav/`; the data itself is not in git).

## Convert (once per computer, from the authors' files)

| Tool | Does |
|---|---|
| `convert_rgbt.py` | RGBT-Tiny COCO labels to YOLO labels, in a folder per label version (`yolo_v0/`, `fusion_v0/`) |
| `convert_antiuav.py` | Anti-UAV videos to JPG frames and YOLO labels: aligns RGB to IR, keeps every 10th frame |
| `convert_fusion_images.py` | RGB and IR frames to 4-channel PNGs (`--dataset rgbt` or `auav`) |

Order: `convert_rgbt.py`, `convert_antiuav.py`, then `convert_fusion_images.py` for each dataset
(only needed for fusion).

## Split, build and train

| Tool | Does |
|---|---|
| `check_split.py` | checks `splits/<version>/assignment.csv` for leaks and writes `checks.txt` |
| `build_lists.py` | checks the split, then writes the image lists, the data yaml and `split_info.yaml` |
| `train.py` | trains YOLO on a data yaml, checks the channels, writes `run_info.yaml` |

`common.py` holds the shared helper for the data folder.

## Not connected yet

`build_lists.py` still reads labels from the old folders (`yolo/labels`, `fusion/labels`), not from
the versioned ones (`yolo_v0`, ...). A `--labels` option is the next step.
