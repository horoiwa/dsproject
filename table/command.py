from dataclasses import dataclass
from pathlib import Path
from typing import List

import click
import pandas as pd
from sklearn.datasets import load_boston

from scripts import loads, cleans, feats, analysis


HOME = Path(__file__).parent

@dataclass
class BaseConfig:
    target_name: str
    mode: str
    nontarget_names: List[str]
    drop_cols: List[str]
    selected_cols: List[str]
    datadir: Path = HOME / "data"
    outdir: Path = HOME / "result"
    suffix: str = "csv" #or "hdf"


SELECTED_COLS = []

config = BaseConfig(
    target_name = "Price",
    mode = "class"
    nontarget_names = [],
    drop_cols = [],
    selected_cols = SELECTED_COLS,
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
def analysis(filename, profile, boruta, ga, xai, cluster):
    """
       この時点で特徴量数が多すぎるなら変数選択だけして(feature_selection)
       次iterにいったほうがよい
    """

    file_path = config.datadir / "processed" / filename
    assert file_path.exists()

    outdir = config.outdir / file_path.name
    if not outdir.exists():
        outdir.mkdir()

    if profile:
        analysis.profile(file_path, out_dir, config)

    if boruta:
        analysis.select_by_boruta(file_path, out_dir, config)

    if ga:
        analysis.select_by_ga(file_path, out_dir, config)

    if xai:
        analysis.xai(file_path, out_dir, config)

    if cluster:
        analysis.cluster(file_path, out_dir, config)


def modeling(filename):

    file_path = config.datadir / "processed" / filename
    assert file_path.exists()

    outdir = config.outdir / file_path.name
    if not outdir.exists():
        outdir.mkdir()

    train_model(filepath, outdir, config)

if __name__ == "__main__":
    cli()
