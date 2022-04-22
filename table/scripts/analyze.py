from pathlib import Path
import inspect

import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import sweetviz as sv

from .util import remove_and_create, load_dataframe, get_logger
import toolbox
from toolbox import bokehplotlib as blt


def profile(filepath: Path, outdir: Path, config):
    func_name = inspect.currentframe().f_code.co_name
    outdir = outdir / func_name
    remove_and_create(outdir)

    df = load_dataframe(filepath)

    """Sweetviz
    Tips:
        セグメント分割されたデータセットの比較する場合
        report = sv.compare([seg1, 'seg1'], [seg2, 'seg2'])
    """
    report = sv.analyze(df)
    report.show_html(
        filepath=(outdir / "sweetviz.html").as_posix(),
        open_browser=False,
        )

    """ 相関分析
    """
    if config.mode == "reg":
        logger = get_logger("correlation", outdir)
        corrs = df.corr()[config.target_name].sort_values(ascending=False)
        logger.info(corrs)


    """ 単相関プロット
    """

    figs = []
    yname = config.target_name
    for colname in df.columns:
        #pandas_bokeh.output_notebook()
        if config.mode == "reg":
            pass
        elif config.mode == "class":
            if df[colname].dtype != "O":
                p = blt.hist(df, x=colname, hue=yname)
                figs.append(p)
            else:
                df.groupby([yname, colname]).count()
                p = blt.barplot(df, x=colname, y)
        else:
            raise NotImplementedError(config.mode)

    fig = blt.gridplot(figs, cols=2)
    fig.savefig(outdir / "single.html")

    """ nC2相関プロット
    """
    if config.nc2:
        pass
