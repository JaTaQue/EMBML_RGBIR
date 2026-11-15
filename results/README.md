# results/

What the team has trained, one row per run, and the records of those runs. The models themselves
(about 5 MB each) are in `../models/`, only for the runs that matter.

```
results/
  runs.csv         one row per saved run
  <run name>/      run_info.yaml, split_info.yaml, results.csv of that run
models/
  <run name>_last.pt    only the important models
```

## Add a run

```
python tools/save_run.py runs/<name> --note "why this run, what it showed"
python tools/save_run.py runs/<name> --model --note "..."     # also copies the model into models/
```

Then commit `results/` (and `models/`) on a branch and open a pull request.

## Rules

- A row is never edited, and a model is never replaced. If something was wrong, make a new run with a new name.
- Compare rows only when `split` and `labels` are the same. `images` and `datasets` can differ: that is the experiment.
- `precision`, `recall`, `mAP50` and `mAP50-95` are the Ultralytics numbers on the **val** set of the split, from the last
  epoch. They are overall numbers for all classes. Drone-only numbers are not in this table yet.
- Debug and tiny test runs are not saved here.
