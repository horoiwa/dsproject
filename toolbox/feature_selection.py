from typing import Literal
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from boruta import BorutaPy


def by_boruta(y, X, p=95.0, target_type=Literal["numerical", "categorical"]):

    if target_type == "categorical":
        rf = RandomForestClassifier(n_jobs=-1, max_depth=5)
    elif target_type == "numerical":
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    else:
        raise NotImplementedError(target_type)

    selector = BorutaPy(rf, n_estimators='auto', perc=p, verbose=2)
    selector.fit(X.values, y.values)

    selected_cols = [colname for colname, is_selected in zip(X.columns, selector.support_) if is_selected]
    rejected_cols = [colname for colname, is_selected in zip(X.columns, selector.support_) if not is_selected]

    tmp = {name: rank for name, rank in zip(X.columns, selector.ranking_)}
    ranking = sorted(tmp.items(), key=lambda item: item[1])

    info = {
        "p": p,
        "confirmed": selected_cols,
        "rejected": rejected_cols,
        "ranking": ranking,
    }
    return info



def by_ga(df, yname: str, target_type: Literal["reg", "classs"],
          min=5, max=10, strategy=Literal["tree", "linear"]):
    if strategy == "tree":
        model_cls = DesictionTreeClassifier
    pass
