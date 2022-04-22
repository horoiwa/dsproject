import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .util import load_dataframe, save_dataframe, get_logger


def feat(input_path, out_dir, config):

    yname = config.target_name
    df = load_dataframe(input_path)

    #: Get logger
    logger = get_logger("feat", out_dir)
    logger.info("Feat process:")
    logger.info(df.shape)

    #: Drop unuse cols
    drop_cols = config.nontarget_names + config.drop_cols
    for colname in drop_cols:
        df = df.drop([colname], axis=1)

    logger.info("無関係なcolumnsをdrop:")
    logger.info(f"Dropped columns: {drop_cols}")
    logger.info(df.shape)


    #: Drop unlabeled records
    df = df.dropna(subset=[yname], axis=0)

    logger.info("ラベルの無いrecordをdrop:")
    logger.info(df.columns)
    logger.info(df.shape)

    for funcname, feat_func in CUSTOM_FEATURIZERS.items():
        logger.info(f"Apply featurizer: {funcname}")
        df = feat_func(df)

    y = df.loc[:, [config.target_name]]
    if config.target_type == "numerical":
        y = to_categorical(y, config.target_name, config.target_name_cat)

    X = df.drop([config.target_name], axis=1)
    X_cat = X.loc[:, config.categorical_cols]
    X_numerical = X.drop(config.categorical_cols, axis=1)

    #: 最後に多項式特徴量の追加
    poly1 = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly1 = pd.DataFrame(
        poly1.fit_transform(X_numerical),
        columns=poly1.get_feature_names(X_numerical.columns)
        )

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X_poly2 = pd.DataFrame(
        poly2.fit_transform(X_numerical),
        columns=poly2.get_feature_names(X_numerical.columns)
        )

    df = pd.concat([y, X_cat, X_numerical], axis=1)
    save_dataframe(out_dir / f"base.{config.suffix}", df)

    df = pd.concat([y, X_cat, X_poly1], axis=1)
    save_dataframe(out_dir / f"poly1.{config.suffix}", df)

    df = pd.concat([y, X_cat, X_poly2], axis=1)
    save_dataframe(out_dir / f"poly2.{config.suffix}", df)


""" Add your featurizers
"""
CUSTOM_FEATURIZERS = {}

def register(func):
    if func.__name__ not in CUSTOM_FEATURIZERS:
        CUSTOM_FEATURIZERS[func.__name__] = func
    return func

@register
def feat_dummy(df):
    return df

def to_categorical(df, colname, colname2):
    """
    連続値フィールドを適当な分位で離散化する
    25, 50, 25
    """
    values = df[colname].dropna().values
    q75, q25 = np.percentile(values, [75 ,25])
    df[colname2] = df[colname].apply(
        lambda v: "q75" if v >= q75 else "q75-25" if v >= q25 else "q25" if v < q25 else np.nan
        )
    return df
