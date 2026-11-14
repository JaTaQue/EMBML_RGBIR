# splits/

One folder per split version. A finished version is never changed; a new split is a new folder.

Each version has:

- `assignment.csv`: one row per clip. Columns: `dataset` (rgbt or auav), `clip`, `group`, `split`. This is the only real content. Facts about a clip (frames, official split, classes) are in the metadata tables, not here.
- `config.yaml`: how the split was made (script, seed, rules).
- `checks.txt`: frames, clips and boxes per split, and the checks (no group or recording in two splits).

The image lists (`train.txt`, ...) and the YOLO yaml files are not stored here. They are built from `assignment.csv`.

A group is the unit that is never cut. Clips of the same recording are always in the same group.

Versions:

- `v3`: our session split (seed 481), with Anti-UAV in train only.

Tools (in `tools/`):

- `python tools/check_split.py v3 --data <data> --write` checks a version and writes `checks.txt`. It fails if a group or recording is in two splits.
- `python tools/build_lists.py v3 --data <data> --images fusion` builds the image lists and the yaml file for training. `--images` is `rgb`, `ir` or `fusion`. Every row of `assignment.csv` goes to its split. `--datasets rgbt` leaves out Anti-UAV. The lists are built on each computer and are not stored in git.
