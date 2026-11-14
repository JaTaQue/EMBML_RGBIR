"""Small helpers shared by the tools."""

import os
from pathlib import Path


def data_dir(arg=None):
    """The data folder: the --data argument, else the EMBML_DATA environment variable."""
    value = arg or os.environ.get("EMBML_DATA")
    if not value:
        raise SystemExit("no data folder: pass --data or set the environment variable EMBML_DATA")
    return Path(value).expanduser().resolve()
