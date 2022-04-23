import copy
import numpy as np


def to_discrete(df, colname):
    """
    連続値フィールドを適当な分位で離散化する
    25, 50, 25
    """
    df = copy.deepcopy(df)
    values = df[colname].dropna().values
    q75, q25 = np.percentile(values, [75 ,25])
    df[colname] = df[colname].apply(
        lambda v: "q75" if v >= q75 else "q25-75" if v >= q25 else "q25" if v < q25 else np.nan
        )
    return df
