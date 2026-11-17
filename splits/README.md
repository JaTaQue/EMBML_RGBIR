# splits/

One folder per split version. A finished version is never changed; a new split is a new folder.

Each version has:

- `assignment.csv`: one row per clip. Columns: `dataset` (rgbt or auav), `clip`, `group`, `split`. This is the only real content. Facts about a clip (frames, drone counts, size, brightness, session) are in the metadata tables (`Metadata/`, built locally by `tools/build_all.py`), not here.
- `config.yaml`: how the split was made (script, seed, rules).
- `checks.txt`: frames, clips and boxes per split, and the checks (no group or recording in two splits).

The image lists (`train.txt`, ...) and the YOLO yaml files are not stored here. They are built from `assignment.csv`.

A group is the unit that is never cut. Clips of the same recording are always in the same group.

Versions:

- `v3`: our session split (seed 481), with Anti-UAV in train only.

Tools (in `tools/`). Set the data folder once per computer with `export EMBML_DATA=<data>` (add it to `~/.bashrc`), then `--data` is not needed:

- `python tools/check_split.py v3 --data <data> --write` checks a version and writes `checks.txt`. It fails if a group or recording is in two splits.
- `python tools/build_lists.py v3 --images fusion --labels v1` builds the image lists and the yaml file for training. `--images` is `rgb`, `ir` or `fusion`. Every row of `assignment.csv` goes to its split. `--datasets rgbt` leaves out Anti-UAV. The lists are built on each computer and are not stored in git.

Each build also writes `split_info.yaml` next to the lists: split version, image kind, datasets, a hash of `assignment.csv`, the git commit of this repo and the frame counts. When you start a training, copy it into the run folder, so the model keeps a record of its data:

```
cp <data>/rgbt-tiny/splits/v3/fusion/split_info.yaml runs/<run name>/
```

Training is a separate step. `build_lists.py` checks the split and writes a data yaml; `train.py` only takes that yaml and a model, so it works for any dataset. It checks that the model and the data have the same number of channels (fusion needs the 4-channel model), saves `run_info.yaml` in the run folder, and copies `split_info.yaml` if it sits next to the data yaml:

```
python tools/build_lists.py v3 --images fusion
python tools/train.py --data-yaml $EMBML_DATA/rgbt-tiny/splits/v3/fusion/fusion.yaml \
    --model weights/yolo26n_4ch.pt --name fusion_v3_s0 imgsz=640 epochs=30 batch=8 seed=0
```
