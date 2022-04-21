import pandas as pd
from pathlib import Path
import inspect
import shutil
import sys

from .util import create_or_overwrite
import toolbox


def profile(df: pd.DataFrame, outdir: Path):
    fname = inspect.currentframe().f_code.co_name
    import pdb; pdb.set_trace()

    outdir = outdir / fname
    create_or_overwrite(outdir)
