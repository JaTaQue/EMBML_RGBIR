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

## The two official-split baselines

`rgb_all_n640_e30_official_s0` and `fusion_n640_e30_official_s0` were trained on the HPC on 2026-11-11, before `train.py`
existed. Their `run_info.yaml` and `split_info.yaml` were written afterwards from `args.yaml` and the job log, and say
`reconstructed: yes`. Hashes and the code commit are unknown. Split `official` means the authors' train and test, with val = test,
so their metrics are test numbers. It is not a version in `splits/` (one clip cannot be both val and test in `assignment.csv`).
