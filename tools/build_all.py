"""Build the labels of one version and the metadata tables from them, in one command.

    python tools/build_all.py v1 [--data <data folder>] [--frames 12]

The version is the only thing you choose. In order, stopping at the first step that fails:

  1. convert_rgbt.py            the authors' labels as YOLO labels (v0), only if v0 does not exist yet
  2. apply_fixes.py <version>   v0 + annotations/<version>/fixes.csv -> labels of <version>, only if it does not exist yet
                                (skipped for v0). To rebuild a version, delete its yolo_<version> and fusion_<version> folders
  3. make_metadata.py           Metadata/clips.csv, the counts in Metadata/sessions.csv, Metadata/info.yaml
  4. make_boxes_drone.py        Metadata/boxes_drone.csv

Anti-UAV is not versioned and not converted here: run convert_antiuav.py first (and convert_fusion_images.py for fusion).
Building the training lists is a separate step that also needs a split: build_lists.py <split> --labels <version>.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import data_dir

TOOLS = Path(__file__).resolve().parent


def run(script, *args):
    print(f"\n== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(TOOLS / script), *args], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version", help="RGBT-Tiny label version: v0, or a folder in annotations/ (v1, v2, ...)")
    p.add_argument("--data", help="data folder (else the EMBML_DATA environment variable)")
    p.add_argument("--frames", default="12", help="RGB pictures per clip for the brightness")
    a = p.parse_args()
    data = data_dir(a.data)
    where = ["--data", str(data)]

    if not (data / "anti-uav" / "yolo" / "labels").is_dir():
        raise SystemExit(f"no Anti-UAV labels at {data / 'anti-uav' / 'yolo' / 'labels'}: run tools/convert_antiuav.py first")
    try:
        if not (data / "rgbt-tiny" / "yolo_v0" / "labels").is_dir():
            run("convert_rgbt.py", *where)
        if a.version != "v0" and not (data / "rgbt-tiny" / f"yolo_{a.version}" / "labels").is_dir():
            run("apply_fixes.py", a.version, *where)
        run("make_metadata.py", "--labels", a.version, "--frames", a.frames, *where)
        run("make_boxes_drone.py", "--labels", a.version, *where)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"\nstopped: {Path(e.cmd[1]).name} failed (exit code {e.returncode})")
    print(f"\nlabels {a.version}: labels, Metadata/clips.csv, sessions.csv counts, boxes_drone.csv are built")


if __name__ == "__main__":
    main()
