from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import sys
sys.path.append("../")

import click
import pandas as pd
from sklearn.datasets import load_boston

from scripts import loads, cleans, feats, analyze


HOME = Path(__file__).parent

@dataclass
class BaseConfig:
    mode: Literal['reg', 'class']
    #: 目的変数(1つのみ)
    target_name: str
    #: 対象外の目的変数
    nontarget_names: List[str]
    #: カテゴリ変数
    categorical_cols: List[str]
    encoding: Literal["none", "int", "onehot"]
    #: データクレンジングで落とす列
    drop_cols: List[str]
    #: 最終モデル構築に使う列
    use_cols: List[str]

    """ 基本設定 """
    datadir: Path = HOME / "data"
    outdir: Path = HOME / "result"
    suffix: str = "csv" #or "hdf"


SELECTED_COLS = []

config = BaseConfig(
    mode = "class",
    target_name = "species",
    nontarget_names = [],
    categorical_cols = ["species", "island", "sex"],
    encoding = "none",
    drop_cols = [],
    use_cols= SELECTED_COLS,
    )


@click.group()
def cli():
    pass


@cli.command()
@click.option("--out_file", "-o", default="raw.csv", type=str)
@click.option("--sample", is_flag=True)
def load(out_file, sample):
    """ 生tableを取得する

    Args:
        out (Path): path to output dataset
        sample (bool): load sample dataset boston housing
    """
    out_path = config.datadir / "raw" / out_file
    loads.load(out_path, sample)


@cli.command()
@click.option("--input_file", "-i", default="raw.csv", type=str)
@click.option("--out_file", "-o", default="clean.csv", type=str)
def clean(input_file, out_file):
    """ 最低限のデータクレンジングを行う
        - 型異常への対処
        - テーブルの結合  など
    """
    input_path = config.datadir / "raw" / input_file
    out_path = config.datadir / "clean" / out_file
    cleans.clean(input_path, out_path)


@cli.command()
@click.option("--input_file", "-i", default="clean.csv", type=str)
def feat(input_file):
    """ 特徴量の追加
        - 不要な行の削除
        - ラベル無し列の削除
        - 多項式特徴量の追加
        - 特徴量の追加  など
    """
    input_path = config.datadir / "clean" / input_file
    out_dir = config.datadir / "processed"
    feats.feat(input_path, out_dir, config)


@cli.command()
@click.option("--filename", "-f", type=str, required=True)
@click.option("--profile", is_flag=True)
@click.option("--boruta", is_flag=True)
@click.option("--ga", is_flag=True)
@click.option("--xai", is_flag=True)
@click.option("--cluster", is_flag=True)
@click.option("--run_all", is_flag=True)
def eda(filename, profile, boruta, ga, xai, cluster, run_all):
    """
       この時点で特徴量数が多すぎるなら変数選択だけして(feature_selection)
       次iterにいったほうがよい
    """

    file_path = config.datadir / "processed" / filename
    assert file_path.exists()

    out_dir = config.outdir / file_path.stem
    if not out_dir.exists():
        out_dir.mkdir()

    if profile or run_all:
        analyze.profile(file_path, out_dir, config)

    if boruta or run_all:
        analyze.select_by_boruta(file_path, out_dir, config)

    if ga or run_all:
        analyze.select_by_ga(file_path, out_dir, config)

    if xai or run_all:
        analyze.xai(file_path, out_dir, config)

    if cluster or run_all:
        analyze.cluster(file_path, out_dir, config)


def modeling(filename):

    file_path = config.datadir / "processed" / filename
    assert file_path.exists()

    outdir = config.outdir / file_path.name
    if not outdir.exists():
        outdir.mkdir()

    train_model(filepath, outdir, config)

if __name__ == "__main__":
    cli()
