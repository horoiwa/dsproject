from typing import Literal
import functools
import random

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from boruta import BorutaPy
from deap import algorithms, base, creator, tools
import ray

import sys
sys.path.append("..")
import toolbox


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


def _evalIndividual(ind, y, X, model, max_features, K=5):

    #: 変数の数がmax_features以上の場合は失格
    if sum(ind) == 0 or sum(ind) > max_features:
        return (float('inf'), -1.)

    X = X[:, [bool(i) for i in ind]]
    y = y.reshape(-1, 1)
    kf = KFold(n_splits=K)

    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    score_avg = sum(scores) / K

    return (sum(ind), score_avg)

@ray.remote
def worker(individuals: list, evalfunc):

    ind_fits = []
    for ind in individuals:
        fitness = evalfunc(ind)
        ind_fits.append((ind, fitness))

    return ind_fits


def split_population(population, k):
    l = [list(ind) for ind in population]
    chunked_list = []
    for i in range(k):
        chunked_list.append(l[i * len(l) // k:(i + 1) * len(l) // k])
    return chunked_list


def by_ga(y, X, max_features=10,
          max_depth=5, n_jobs=1,
          model_type=Literal["DTC", "DTR", "RFC", "RFR", "Ridge"],
          ngen=500, mu=100, lam=300):

    num_features = X.shape[1]

    if model_type == "DTC":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "DTR":
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_type == "RFC":
        model = RandomForestClassifier(max_depth=5, n_estimators=50)
    elif model_type == "RFR":
        model = RandomForestRegressor(max_depth=5, n_estimators=50)
    elif model_type == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        raise NotImplementedError(model_type)

    """GA with Deap
        Mu + lambda (毎世代fitnessを再評価)
    """
    #: weights: (num_features, score)
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    evalfunc= functools.partial(
        _evalIndividual,
        y=y.values, X=X.values, model=model, max_features=max_features,
        )

    def createIndividual():
        return creator.Individual(
            [random.randint(0, 1) for i in range(num_features)]
            )

    pop = [createIndividual() for i in range(mu)]
    ray.init()
    for gen in range(ngen):
        wip_jobs = [worker.remote(pop_subset, evalfunc) for pop_subset in split_population(pop, n_jobs)]
        import pdb; pdb.set_trace()
        #pop = functools.reduce(lambda l1, l2: l1+l2, ray.get(wip_jobs), [])

    #toolbox.register("evaluate", evalIndividual)
    #toolbox.register("mate", tools.cxTwoPoint)
    #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    #toolbox.register("select", tools.selNSGA2)
    # LAMBDA = lam
    # CXPB = 0.7
    # MUTPB = 0.1

    ray.shutdown()


