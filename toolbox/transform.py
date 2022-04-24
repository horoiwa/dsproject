import copy
import functools

import numpy as np
import category_encoders as ce


def numerical_to_category(df, colnames):
    """
    連続値フィールドを適当な分位で離散化する
    25, 50, 25
    """
    df = copy.deepcopy(df)
    colnames = colnames if type(colnames) == list else [colnames]

    for colname in colnames:
        values = df[colname].dropna().values
        q75, q25 = np.percentile(values, [75 ,25])
        df[colname] = df[colname].apply(
            lambda v: "q75" if v >= q75 else "q25-75" if v >= q25 else "q25" if v < q25 else np.nan
            )
    return df


def category_to_xxx(df, colnames, encoder_cls):
    df = copy.deepcopy(df)
    colnames = colnames if type(colnames) == list else [colnames]
    encoder = encoder_cls(cols=colnames)
    df = encoder.fit_transform(df)
    return df


category_to_integer = functools.partial(category_to_xxx, encoder_cls=ce.OrdinalEncoder)
category_to_onehot = functools.partial(category_to_xxx, encoder_cls=ce.OneHotEncoder)



def category_to_baseN(df, colnames, n=10):
    df = copy.deepcopy(df)
    colnames = colnames if type(colnames) == list else [colnames]
    encoder = ce.BaseNEncoder(cols=colnames, base=n)
    df = encoder.fit_transform(df)
    return df
