import warnings
from typing import Callable
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, matthews_corrcoef
import optuna
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


class AutoModelBase(metaclass=ABCMeta):

    MODEL_CLS = None
    N_TRIALS = 30
    METRIC = None

    def __init__(
        self,
        test_ratio: float,
        scaling: bool=False,
        category_encoding=False,
        categorical_features: list=None,
        ordinal_threshold: int=10,
        n_trials: int = None,
        n_repeats: int = 10,
        metric: Callable = None
        ):

        self.test_ratio = test_ratio
        self.scaling = scaling
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.ordinal_threshold = ordinal_threshold
        self.category_encoding = category_encoding

        self.n_trials = n_trials if n_trials is not None else self.N_TRIALS
        self.n_repeats = n_repeats
        self.metric = metric if metric is not None else self.METRIC

        self.transformers = None
        self.feature_names_in = None
        self.feature_names_out = None
        self.model = None
        self.best_trial = None
        self.best_params = None

    @abstractmethod
    def define_search_space(self, trial):
        raise NotImplementedError()

    def fit(self, X: pd.DataFrame, y: pd.Series, n_jobs=1):

        X_post = self.transform(X, fit=True).values
        y_post = y.values

        best_params, best_trial = self._param_search(X_post, y_post)

        print()
        print("==== Best Trial =====")
        print(best_trial)
        print()
        print(best_params)
        print()

        model = self.MODEL_CLS(**best_params)
        model.fit(X_post, y_post)

    def _param_search(self, X, y) -> dict:

        def objective(trial):

            params = self.define_search_space(trial)

            model = self.MODEL_CLS(**params)

            if self.metric == "mse":
                metric = mean_squared_error
            elif self.metric == "r2":
                metric = r2_score
            elif self.metric == "matthews":
                metric = matthews_corrcoef
            elif self.metric == "accuracy":
                metric = accuracy_score
            else:
                raise NotImplementedError(self.metric)

            scores = []
            kf = RepeatedKFold(n_splits=3, n_repeats=self.n_repeats)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = metric(y_test, y_pred)
                scores.append(score)

            avg_score = sum(scores) / len(scores)

            return avg_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.best_trial = study.best_trial
        self.best_params = study.best_trial.params

        return self.best_params, self.best_trial

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
                #warnings.warn("Transformer %s (type %s) does not "
                #              "provide get_feature_names. "
                #              "Will return input column names if available"
                #              % (str(name), type(trans).__name__))
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


class AutoRidge(AutoModelBase):
    MODEL_CLS = Ridge
    N_TRIALS = 10
    METRIC = "mse"

    def define_search_space(self, trial):
        params = {}
        params["alpha"] = trial.suggest_loguniform('alpha', 1e-3, 1e2),
        return params


class AutoSVR(AutoModelBase):
    MODEL_CLS = SVR
    PARAMETERS = 50
    METRIC = "mse"


class AutoLGBMReg(AutoModelBase):
    MODEL_CLS = None
    PARAMETERS = 100
    METRIC = mean_squared_error


class AutoSVM(AutoModelBase):
    MODEL_CLS = SVR
    PARAMETERS = 50
    METRIC = matthews_corrcoef


class AutoLGBM(AutoModelBase):
    MODEL_CLS = None
    PARAMETERS = 100
    METRIC = matthews_corrcoef


"""
Notes:
    - マシュー相関係数はラベルに影響されないので使いやすい
    - 二値分類でaccを指標にするなら少数クラスを正にする
"""


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
    #model = AutoSVR(test_ratio=0.3, scaling=True, category_encoding=False,
    #                categorical_features=["island", "sex"], ordinal_threshold=5,)
    model = AutoRidge(test_ratio=0.3, scaling=True, category_encoding=True,
                      categorical_features=["island", "sex"], ordinal_threshold=5,)
    model.fit(X, y)
