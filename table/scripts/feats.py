from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

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


    save_dataframe(out_dir / f"base.{config.suffix}", df)

    if config.poly:
        #: 最後に多項式特徴量の追加
        poly1 = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        df_poly1 = pd.DataFrame(
            poly1.fit_transform(df),
            columns=poly1.get_feature_names_out()
            )

        poly2 = PolynomialFeatures(degree=2, include_bias=False)
        df_poly2 = pd.DataFrame(
            poly2.fit_transform(df),
            columns=poly2.get_feature_names_out()
            )

        save_dataframe(out_dir / f"poly1.{config.suffix}", df_poly1)
        save_dataframe(out_dir / f"poly2.{config.suffix}", df_poly2)


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

@register
def to_discrete(df):
    """
    連続値フィールドを適当な分位で離散化する
    25, 50, 25
    """
    return df
