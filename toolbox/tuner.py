import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from ray import tune
import category_encoders as ce


class AutoModelBase:

    MODEL_CLS = None
    PARAMETERS = None

    def __init__(
        self, test_ratio: float,
        scaling: bool=None, categorical_features: list=None,
        ordinal_threshold: int=10,
        skip_category_encoding=False,
        ):

        self.test_ratio = test_ratio
        self.scaling = scaling
        self.categorical_features = categorical_features
        self.ordinal_threshold = ordinal_threshold
        self.skip_ce_encoding = skip_category_encoding

        self.transformers = None
        self.feature_names = None


    def fit(self, X: pd.DataFrame, y: pd.Series):

        X_post = self.transform(X, fit=True)
        y_post = y.values

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def transform(self, X: pd.DataFrame, fit=False):

        if fit:
            self._create_pipeline(X)

        tmp = self.transformer.transform(X)
        X_post = pd.DataFrame(tmp, columns=self._get_feature_names(self.transformer))

        return X_post

    def inv_transform(self, X_post: np.ndarray):
        pass

    def _create_pipeline(self, X: pd.DataFrame):

        self.feature_names = [col for col in X.columns]

        transformers = []
        if self.scaling:
            numeric_features = [col for col in X.columns if col not in self.categorical_features]
            transformers.append(("st", StandardScaler(), numeric_features))


        if self.categorical_features and not self.skip_ce_encoding:
            to_ordinal_features, to_onehot_features = [], []
            for col in self.categorical_features:
                nunique = X[col].nunique()
                if nunique > self.ordinal_threshold:
                    to_ordinal_features.append(col)
                else:
                    to_onehot_features.append(col)

            ord_encoder = ce.OrdinalEncoder(cols=to_ordinal_features)
            transformers.append(("ord", ord_encoder, to_ordinal_features))

            oh_encoder = ce.OneHotEncoder(cols=to_onehot_features)
            transformers.append(("oh", oh_encoder, to_onehot_features))

        self.transformer = ColumnTransformer(transformers)
        self.transformer.fit_transform(X)


    @staticmethod
    def _get_feature_names(column_transformer):
        """
        https://johaupt.github.io/blog/columnTransformer_feature_names.html

        Get feature names from all transformers.
        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        # Remove the internal helper function
        #check_is_fitted(column_transformer)

        # Turn loopkup into function for better handling with pipeline later
        def get_names(trans):
            # >> Original get_feature_names() method
            if trans == 'drop' or (
                    hasattr(column, '__len__') and not len(column)):
                return []
            if trans == 'passthrough':
                if hasattr(column_transformer, '_df_columns'):
                    if ((not isinstance(column, slice))
                            and all(isinstance(col, str) for col in column)):
                        return column
                    else:
                        return column_transformer._df_columns[column]
                else:
                    indices = np.arange(column_transformer._n_features)
                    return ['x%d' % i for i in indices[column]]
            if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
                # Turn error into a warning
                warnings.warn("Transformer %s (type %s) does not "
                                    "provide get_feature_names. "
                                    "Will return input column names if available"
                                    % (str(name), type(trans).__name__))
                # For transformers without a get_features_names method, use the input
                # names to the column transformer
                if column is None:
                    return []
                else:
                    return [name + "__" + f for f in column]

            return [name + "__" + f for f in trans.get_feature_names()]

        ### Start of processing
        feature_names = []

        # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
        if type(column_transformer) == sklearn.pipeline.Pipeline:
            l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
        else:
            # For column transformers, follow the original method
            l_transformers = list(column_transformer._iter(fitted=True))


        for name, trans, column, _ in l_transformers:
            if type(trans) == sklearn.pipeline.Pipeline:
                # Recursive call on pipeline
                _names = get_feature_names(trans)
                # if pipeline has no transformer that returns names
                if len(_names)==0:
                    _names = [name + "__" + f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans))

        return feature_names


class AutoSVR(AutoModelBase):
    MODEL_CLS = SVR
    PARAMETERS = None


class AutoLGBM(AutoModelBase):
    MODEL_CLS = None
    PARAMETERS = None


if __name__ == "__main__":
    """
    cols:
        species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
    """

    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    y = df["species"]
    X = df.drop("species", axis=1)
    model = AutoSVR(test_ratio=0.3, categorical_features=["island", "sex"],
                    scaling=True, ordinal_threshold=2)

    y = df["species"]
    X = df.drop(["species", "island", "sex"], axis=1)
    model = AutoSVR(test_ratio=0.3)
    model.fit(X, y)
