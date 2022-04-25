from typing import Literal
import copy
import functools
import random

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from boruta import BorutaPy
from deap import algorithms, base, creator, tools
#import ray

import toolbox
from toolbox import fileio


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


#@ray.remote
def worker(individuals: list, evalfunc):
    fits = [evalfunc(ind) for ind in individuals]
    return fits


def split_population(population, k):
    l = [list(ind) for ind in population]
    chunked_list = []
    for i in range(k):
        chunked_list.append(l[i * len(l) // k:(i + 1) * len(l) // k])
    return chunked_list


def by_ga(y, X, max_features=10,
          max_depth=5, n_jobs=1,
          model_type=Literal["DTC", "DTR", "RFC", "RFR", "Ridge"],
          ngen=500, mu=100, lam=300, cxpb=0.4):

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


    num_features = X.shape[1]
    indpb =  0.5 * max_features / num_features

    population = [
        creator.Individual(
            [random.choices([0, 1], weights=[1.0-indpb, indpb])[0]
             for i in range(num_features)])
        for i in range(mu)]

    logger = fileio.get_logger("ga")
    for gen in range(ngen + 1):

        #: 次世代個体の生産
        offspring = []
        for i in range(lam):
            if random.random() < cxpb:
                # Apply crossover
                ind1, ind2 = map(copy.deepcopy, random.sample(population, 2))
                ind1, ind2 = tools.cxTwoPoint(ind1, ind2)
                offspring.append(ind1)
            else:
                # Apply mutation
                ind = copy.deepcopy(random.choice(population))
                ind, = tools.mutFlipBit(ind, indpb=indpb)
                offspring.append(ind)

        #: 全個体の適合度を計算(前世代も再計算)
        population += offspring
        _fits = [worker(pop_subset, evalfunc) for pop_subset in split_population(population, n_jobs)]
        fits = functools.reduce(lambda l1, l2: l1+l2, _fits, [])
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit

        #: NSGA2による淘汰
        population = tools.selNSGA2(population, k=mu, nd="log")

        features = np.array([sum(ind) for ind in population])
        scores = np.array([ind.fitness.values[1] for ind in population])

        logger.info(f"==== GEN {gen} =====")
        logger.info(f"Score -- avg: {scores.mean():.2f} max: {scores.max():2f} min: {scores.min():2f}")
        logger.info(f"Feature -- avg: {features.mean(): 2f}  max: {features.max()} min: {features.min()}")




