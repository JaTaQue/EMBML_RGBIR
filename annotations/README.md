# annotations/

Our fixes to the authors' labels. The authors' labels are never edited, and they are not in git
(research use only). A label version is the authors' labels (v0) plus the fixes of that version.

```
annotations/
  v1/
    fixes.csv    all our fixes up to v1 (one row per change)
    info.yaml    version, parent, date, hash of fixes.csv once sealed
```

A finished version is never edited. New fixes make a new folder (v2) that holds all the rows of v1
plus the new ones. v1 is empty: with no fixes it gives labels identical to v0.

## fixes.csv

`box_id,action,new_value,reason,reviewer,date`

| Column | Meaning |
|---|---|
| `box_id` | `rgbt/<clip>/<frame>/<rgb or ir>/<n>`. n is the line number (from 0) in the v0 label file of that frame and modality. For `add` use `new1`, `new2`, ... |
| `action` | `ok` reviewed, no change; `remove`; `relabel` (new_value = class name); `resize` (new_value = `x y w h` in pixels, top-left corner); `add` (new_value = `class x y w h`) |
| `reason`, `reviewer`, `date` | why, who, when. Fill them in, so a change can be discussed later |

A fix on an `ir` box also changes the fusion labels, because those are the IR labels. Only RGBT-Tiny
boxes are supported; Anti-UAV labels are not versioned yet.

## Use

```
python tools/convert_rgbt.py                    # once: the authors' labels as YOLO labels (v0)
python tools/apply_fixes.py v1                  # v0 + annotations/v1/fixes.csv -> labels v1
python tools/apply_fixes.py v1 --seal           # when the fixes are final: stores the hash of fixes.csv
python tools/build_lists.py v3 --images fusion --labels v1
```

After `--seal`, the script stops if `fixes.csv` of that version has changed.
Compare models only when they used the same split version and the same label version. Both are
written into `split_info.yaml` and the run folder.
