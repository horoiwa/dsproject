from typing import Literal
import copy
import functools
import random
import multiprocessing as mp

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from boruta import BorutaPy
from deap import algorithms, base, creator, tools
import ray

from toolbox.fileio import get_logger


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


def _evalIndividual(ind, y, X, model, max_features, K=5, N=30):

    #: 変数の数がmax_features以上の場合は失格
    if sum(ind) == 0 or sum(ind) > max_features:
        return (float('inf'), 0.)

    X = X[:, [bool(i) for i in ind]]
    y = y.ravel()

    if X.shape[0] >= 1000:
        kf = KFold(n_splits=K)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)

        score_avg = sum(scores) / K
    else:
        #: スモールデータセットの場合はKFOLDを避ける
        scores = []
        for _ in range(N):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        score_avg = sum(scores) / N

    return (sum(ind), score_avg)


def worker(individuals: list, evalfunc):
    fits = [evalfunc(ind) for ind in individuals]
    return fits

ray_worker = ray.remote(worker)


def split_population(population, k):
    l = [list(ind) for ind in population]
    chunked_list = []
    for i in range(k):
        chunked_list.append(l[i * len(l) // k:(i + 1) * len(l) // k])
    return chunked_list


def by_ga(y, X, max_features=10,
          model_type=Literal["DTC", "DTR", "RFC", "RFR", "Ridge"],
          max_depth=5, n_jobs=1, background: Literal["mp", "ray"]="ray",
          ngen=500, mu=100, lam=300, cxpb=0.4, logfile=None):

    if model_type == "DTC":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "DTR":
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_type == "RFC":
        model = RandomForestClassifier(max_depth=5, n_estimators=50)
    elif model_type == "RFR":
        model = RandomForestRegressor(max_depth=5, n_estimators=50)
    elif model_type == "LM":
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
    indpb = max_features / num_features

    population = [
        creator.Individual(
            [random.choices([0, 1], weights=[1.0-indpb, indpb])[0]
             for i in range(num_features)])
        for i in range(mu)]


    logger = get_logger()
    logger.info("=========")
    logger.info(f"Start selection by GA ; N_JOBS: {n_jobs}")

    if n_jobs > 1:
        if background == "ray":
            ray.init(include_dashboard=False)
            logger.info(f"Use ray")
        elif background == "mp":
            p = mp.Pool(n_jobs)
            logger.info(f"Use multiprocessing")
        else:
            raise Exception(background)

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
        if n_jobs > 1:
            if background == "ray":
                _fits = ray.get([ray_worker.remote(pop_subset, evalfunc) for pop_subset in split_population(population, n_jobs)])
            elif background == "mp":
                _worker = functools.partial(worker, evalfunc=evalfunc)
                _fits = p.map(_worker, split_population(population, n_jobs))
            else:
                raise Exception(background)
        else:
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

    logger.info(f"==== Gracefully finished =====")
    pf = tools.ParetoFront()
    pf.update(population)
    pareto_front = pf.items

    info = []
    for ind in pareto_front:
        fit = ind.fitness.values
        selected_features = [colname for i, colname in zip(ind, X.columns) if i == 1]
        info.append(
            {"N": fit[0], "Score": fit[1], "Selected": selected_features}
            )
        logger.info("=====")
        logger.info(f"Score {fit[1]:.2f} | N {fit[0]}")
        logger.info(f"{selected_features}")

    info = sorted(info, key=lambda item: item["Score"], reverse=True)
    return info


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    bos = load_boston()
    scaler = StandardScaler()
    data = scaler.fit_transform(bos.data)
    X = pd.DataFrame(data, columns=bos.feature_names)
    y = pd.DataFrame(bos.target, columns=["Price"])
    by_ga(y, X, model_type="LM", ngen=30,
          max_features=5, n_jobs=5, background="mp")
