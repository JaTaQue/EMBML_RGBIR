# splits/

One folder per split version. A finished version is never changed; a new split is a new folder.

Each version has:

- `assignment.csv`: one row per clip. Columns: `dataset` (rgbt or auav), `clip`, `group`, `split`, `n_frames`, `official_split`. This is the only real content.
- `config.yaml`: how the split was made (script, seed, rules) and which recipes (which datasets go into train).
- `checks.txt`: frames, clips and boxes per split, and the checks (no group or recording in two splits).

The image lists (`train.txt`, ...) and the YOLO yaml files are not stored here. They are built from `assignment.csv`.

A group is the unit that is never cut. Clips of the same recording are always in the same group.

Versions:

- `v3`: our session split (seed 481), with Anti-UAV in train only.
