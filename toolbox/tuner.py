import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from ray import tune
import category_encoders as ce


class DummyTransformer(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y=None):
        # return the dataframe with the specified features
        return X.values


class AutoModelBase:

    MODEL_CLS = None
    PARAMETERS = None

    def __init__(
        self,
        test_ratio: float,
        scaling: bool=False,
        category_encoding=False,
        categorical_features: list=None,
        ordinal_threshold: int=10,
        ):

        self.test_ratio = test_ratio
        self.scaling = scaling
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.ordinal_threshold = ordinal_threshold
        self.category_encoding = category_encoding

        self.transformers = None
        self.feature_names_in = None
        self.feature_names_out = None
        self.model = None


    def fit(self, X: pd.DataFrame, y: pd.Series):

        X_post = self.transform(X, fit=True)
        y_post = y.values

    def predict(self, X):
        X_post = self.transform(X)
        y_pred = self.model.predict(X_post.values)
        return y_pred

    def score(self, X, y):
        X_post = self.transform(X)
        y_post = y.values
        score = self.model.score(X_post, y_post)
        return score

    def transform(self, X: pd.DataFrame, fit=False):

        if fit:
            self._create_pipeline(X)

        _tmp = self.transformer.transform(X)
        X_post = pd.DataFrame(_tmp, columns=self.get_feature_names())

        return X_post

    def _create_pipeline(self, X: pd.DataFrame):

        self.feature_names_in = [col for col in X.columns]

        transformers = []

        numeric_features = [col for col in X.columns if col not in self.categorical_features]
        if self.scaling:
            transformers.append(("st", StandardScaler(), numeric_features))
        else:
            transformers.append(("dm", DummyTransformer(), numeric_features))


        if self.categorical_features:
            if self.category_encoding:
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

            else:
                transformers.append(("dm", DummyTransformer(), self.categorical_features))

        self.transformer = ColumnTransformer(transformers)
        self.transformer.fit(X)
        self.feature_names_out = self._get_feature_names(self.transformer)

    def get_feature_names(self):
        return self.feature_names_out

    @staticmethod
    def _get_feature_names(column_transformer):
        """
        ColumnTransformerに保存されている列名情報を取得する
        ただし、Transformerに変換後の列名取得メソッド(get_feature_names)があるならそちらを利用
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
    y = ce.OrdinalEncoder(cols=["species"]).fit_transform(y)

    #X = df.drop(["species", "island", "sex"], axis=1)
    X = df.drop("species", axis=1)
    model = AutoSVR(test_ratio=0.3, scaling=True, category_encoding=False,
                    categorical_features=["island", "sex"], ordinal_threshold=2,)
    model.fit(X, y)
