from dataclasses import dataclass
from pathlib import Path
from typing import List

import click
import pandas as pd
from sklearn.datasets import load_boston

from scripts import loads, cleans


DATADIR = Path(__file__).parent / "data"


@dataclass
class BaseConfig:
    target_names: List[str]
    drop_cols: List[str]


Config = BaseConfig(
    target_names=["Price"],
    drop_cols = [],
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
    out_path = DATADIR / "raw" / out_file
    loads.load(out_path, sample)


@cli.command()
@click.option("--input_file", "-i", default="raw.csv", type=str)
@click.option("--out_file", "-o", default="clean.csv", type=str)
def clean(input_file, out_file):
    """ 最低限のデータクレンジングを行う
        - 型異常への対処
        - テーブルの結合  など
    """
    input_path = DATADIR / "raw" / input_file
    out_path = DATADIR / "clean" / out_file
    cleans.clean(input_path, out_path)


@cli.command()
@click.option("--input_file", "-i", default="raw.csv", type=str)
@click.option("--out_file", "-o", default="clean.csv", type=str)
def feat(input_file, out_file):
    """ 最低限のデータクレンジングを行う
        - 型異常への対処
        - テーブルの結合  など
    """
    input_path = DATADIR / "raw" / input_file
    out_path = DATADIR / "clean" / out_file
    cleans.clean(input_path, out_path)



if __name__ == "__main__":
    cli()
