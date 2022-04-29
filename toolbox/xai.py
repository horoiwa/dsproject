from typing import Literal
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtreeviz.trees import dtreeviz


def explainable_tree(y, X, target_type: Literal["numerical", "categorical"],
                     class_names=None, max_depth=4):
    """
        とくに回帰の場合は浅いDecisionTreeRegなどで
        ざっくりセグメンテーションしてからRidgeがよい
    Note:
      - jupyteでの可視化はdisplay(viz)
      - 保存する場合はviz.save("path.svg")

    """
    if target_type == "numerical":
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X.values, y.values)
        viz = dtreeviz(
            model,
            X,
            y,
            feature_names=[col for col in X.columns],
            target_name=y.name,
        )

    elif target_type == "categorical":

        if class_names is None:
            class_names = [str(i) for i in sorted(y.unique().tolist())]

        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X.values, y.values)
        viz = dtreeviz(
            model,
            X,
            y,
            feature_names=[col for col in X.columns],
            target_name=y.name,
            class_names=class_names
        )

    else:
        raise NotImplementedError(target_type)

    return viz


def shap_svr():
    pass
