"""The classifier that reads the pairs.

Anything with ``fit(X, y)`` and ``predict_proba(X)`` works. ``X`` is a
float32 NumPy array that may contain NaN.

The scorer is ``mojolearn.GradientBoosting`` with oblivious trees.
mojolearn has no Windows wheel, so on platforms without one the scorer
is CatBoost, the library mojolearn's boosting follows, with the same
settings. Which scorer ran, and with what parameters, is recorded in
every result.
"""
from __future__ import annotations

import copy
import functools
import inspect
from dataclasses import dataclass
from typing import Callable

MOJOLEARN_PARAMS = dict(loss="Logloss", n_estimators=300, learning_rate=0.05,
                        max_depth=6, grow_policy="SymmetricTree",
                        boosting_type="Plain")
CATBOOST_PARAMS = dict(loss_function="Logloss", iterations=300,
                       learning_rate=0.05, depth=6,
                       grow_policy="SymmetricTree", boosting_type="Plain")


def _mojolearn(seed: int, threads: int = 1):
    import mojolearn
    return mojolearn.GradientBoosting(random_state=seed, **MOJOLEARN_PARAMS)


def _catboost(seed: int, threads: int = 1):
    import catboost
    return catboost.CatBoostClassifier(random_seed=seed, thread_count=threads,
                                       verbose=0, allow_writing_files=False,
                                       **CATBOOST_PARAMS)


def _from_estimator(estimator, seed: int, threads: int = 1):
    try:
        from sklearn.base import clone
        est = clone(estimator)
    except Exception:
        est = copy.deepcopy(estimator)
    if hasattr(est, "get_params") and "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est


def _from_factory(factory, seed: int, threads: int = 1):
    return factory(seed)


@dataclass
class Scorer:
    name: str
    params: dict
    make: Callable        # make(seed, threads) -> a fresh estimator

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params}


def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _named(name: str) -> Scorer:
    if name == "mojolearn":
        import mojolearn
        return Scorer("mojolearn.GradientBoosting",
                      {**MOJOLEARN_PARAMS, "version": mojolearn.__version__},
                      _mojolearn)
    if name == "catboost":
        import catboost
        return Scorer("catboost.CatBoostClassifier",
                      {**CATBOOST_PARAMS, "version": catboost.__version__},
                      _catboost)
    raise ValueError(f"unknown scorer {name!r}; use 'mojolearn', "
                     f"'catboost', an estimator, or a factory taking a seed")


def _describe_estimator(est) -> tuple[str, dict]:
    name = f"{type(est).__module__.split('.')[0]}.{type(est).__name__}"
    params = {}
    if hasattr(est, "get_params"):
        params = {k: v for k, v in est.get_params(deep=False).items()
                  if isinstance(v, (str, int, float, bool, type(None)))}
    return name, params


def resolve(scorer=None) -> Scorer:
    """Turn the ``scorer`` argument into a recorded, seedable scorer."""
    if scorer is None:
        if _have("mojolearn"):
            return _named("mojolearn")
        if _have("catboost"):
            return _named("catboost")
        raise ImportError(
            "no scorer is installed: pip install mojolearn (macOS arm64, "
            "Linux x86-64) or catboost")
    if isinstance(scorer, Scorer):
        return scorer
    if isinstance(scorer, str):
        return _named(scorer)
    if hasattr(scorer, "fit") and not inspect.isclass(scorer):
        if not hasattr(scorer, "predict_proba"):
            raise TypeError("the scorer needs a predict_proba method")
        name, params = _describe_estimator(scorer)
        return Scorer(name, params,
                      functools.partial(_from_estimator, scorer))
    if callable(scorer):
        name, params = _describe_estimator(scorer(0))
        return Scorer(name, params, functools.partial(_from_factory, scorer))
    raise TypeError(f"cannot use {scorer!r} as a scorer")


def fit_score(scorer: Scorer, seed: int, threads: int, X_train, y_train,
              X_test):
    """Fit a fresh estimator and return the positive-class score."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    est = scorer.make(seed, threads)
    # A column with a missing value among the scored rows but none among
    # the training rows: the boosting libraries refuse it. They treat a
    # missing value as below every observed one, so say that directly.
    X_test = np.asarray(X_test)
    gap = np.isnan(X_test).any(axis=0) & ~np.isnan(X_train).any(axis=0)
    if gap.any():
        X_test = X_test.copy()
        low = X_train[:, gap].min(axis=0)
        low = (low - np.abs(low) - 1).astype(X_test.dtype)
        block = X_test[:, gap]
        block[np.isnan(block)] = np.broadcast_to(low, block.shape)[
            np.isnan(block)]
        X_test[:, gap] = block
    # Workers share the machine: each fit gets its share of the cores,
    # whichever threading library the estimator uses underneath.
    with threadpool_limits(limits=threads):
        est.fit(X_train, y_train)
        p = np.asarray(est.predict_proba(X_test))
    return p[:, 1] if p.ndim == 2 else p
