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
from toolbox import transform, feature_selection, xai, fileio


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
        df = transform.numerical_to_category(df, yname)

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


def select_by_boruta(filepath: Path, outdir: Path, config):
    func_name = inspect.currentframe().f_code.co_name
    outdir = outdir / func_name
    remove_and_create(outdir)

    df = load_dataframe(filepath)
    df, encoder = transform.category_to_ordinal(df, config.categorical_cols)

    cmap = {}
    for d in encoder.category_mapping:
        cmap[d["col"]] = d["mapping"]
    fileio.save_dict_as_json(cmap, outdir / "categorymap.json")

    y = df[config.target_name]
    X = df.drop([config.target_name], axis=1)

    info = feature_selection.by_boruta(
        y, X, p=config.p, target_type=config.target_type
        )

    selected_cols = info["confirmed"]
    fileio.save_dict_as_json(info, outdir / "result.txt")

    X_selected = X[selected_cols]
    viz = xai.explainable_tree(y, X_selected, target_type=config.target_type)
    viz.save(str(outdir / "tree.svg"))


def explain_by_tree():
    pass
