from pathlib import Path
import itertools
import inspect

import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import sweetviz as sv

from .util import remove_and_create, load_dataframe, get_logger
import toolbox
from toolbox import bokehplotlib as blt
from toolbox import transform


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
    if config.target_type == "numerical":
        logger = get_logger("correlation", outdir)
        corrs = df.corr()[config.target_name].sort_values(ascending=False)
        logger.info(corrs)


    """ 単相関プロット
    """
    yname = config.target_name
    if config.target_type == "numerical":
        figs = []
        for colname in df.columns:
            if colname == config.target_name:
                continue

            if colname not in config.categorical_cols:
                p = blt.scatter(df, x=colname, y=yname)
            else:
                p = blt.barplot(df, x=colname, y=yname)
            figs.append(p)

        fig = blt.gridplot(figs, cols=2)
        fig.savefig(outdir / "singleplot_num.html")


    if config.target_type == "numerical":
        df = transform.to_discrete(df, yname)

    figs = []
    for colname in df.columns:
        if colname in config.target_name:
            continue

        if colname not in config.categorical_cols:
            try:
                p = blt.hist(df, x=colname, hue=yname)
            except:
                import pdb; pdb.set_trace()
        else:
            p = blt.countplot(df, x=colname, hue=yname)
        figs.append(p)

    fig = blt.gridplot(figs, cols=2)
    fig.savefig(outdir / "singleplot_cat.html")


    """ NC2相関プロット
        Y-categorical only
    """

    colnames = [col for col in df.columns if col != config.target_name]
    colnames = [col for col in colnames if col not in config.categorical_cols]

    figs = []
    for col1, col2 in itertools.combinations(colnames, 2):
        p = blt.scatter(df, x=col1, y=col2, hue=yname)
        p.legend()
        figs.append(p)

    fig = blt.gridplot(figs, cols=2)
    fig.savefig(outdir / "combinationplot.html")
