from pathlib import Path

import pandas as pd
from sklearn.datasets import load_boston

from .util import save_dataframe


def load(out_path: Path, sample: bool):

    if sample:
        df = load_sample()
    else:
        df = load_dataset()

    save_dataframe(out_path, df)


def load_dataset():
    """ Add your code
    """
    return None


def load_sample(dataset="penguin") -> pd.DataFrame:

    if dataset == "boston":
        bos = load_boston()
        X = pd.DataFrame(bos.data, columns=bos.feature_names)
        y = pd.DataFrame(bos.target, columns=["Price"])
        df = pd.concat([y, X], axis=1)
    elif dataset == "penguin":
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    else:
        raise NotImplementedError(dataset)

    return df

