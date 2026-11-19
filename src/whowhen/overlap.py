"""Overlap check: does training on other rows of the test units inflate the score?

Each replicate scores the same test rows twice, with the same number of
training rows of each class:

* disjoint: training holds no row of any test unit;
* overlap: some training rows are swapped for other rows of the test
  units (rows that are not themselves test rows).

The two fits differ only in whether training shares units with the test
rows. The gap in AUC (overlap minus disjoint), over many replicates with
fresh unit splits, is the result. A gap whose interval excludes zero
means a model evaluated on a row-random split would look better than it
will on new units.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._stats import auc
from .audit import n_workers
from .panel import Panel
from .scorers import Scorer, fit_score, resolve

TIME_SPLITS = (None, "calendar")


def overlap_masks(unit: np.ndarray, time: np.ndarray, rng: np.random.Generator,
                  test_frac: float = 0.2, time_split: str | None = None,
                  embargo: float = 0.0):
    """(test, leak, base) boolean row masks for one replicate.

    A fraction ``test_frac`` of units are test units. Without a time
    split, test rows are a random half of each test unit's rows and leak
    rows are the other half. With ``time_split="calendar"``, the cut is
    the ``1 - test_frac`` quantile of time over all rows: test rows are
    test-unit rows after the cut, and leak and base rows are at or before
    ``cut - embargo``. Base rows are rows of units that are not test
    units. Leak and base never contain a test row; base never contains a
    row of a test unit.
    """
    n_units = int(unit.max()) + 1 if len(unit) else 0
    tu = (rng.random(n_units) < test_frac)[unit]
    if time_split is None:
        half = rng.random(len(unit)) < 0.5
        return tu & half, tu & ~half, ~tu
    if time_split == "calendar":
        cut = float(np.quantile(time, 1.0 - test_frac))
        early = time <= cut - embargo
        return tu & (time > cut), tu & early, ~tu & early
    raise ValueError(f"time_split must be one of {TIME_SPLITS}; "
                     f"got {time_split!r}")


def matched_training(y: np.ndarray, leak: np.ndarray, base: np.ndarray,
                     rng: np.random.Generator):
    """Two training index sets with identical per-class counts.

    disjoint = every base row. overlap = leak rows plus base rows drawn
    to the same per-class count (leak rows are subsampled if there are
    more of them than base rows of that class). The test set is shared,
    so the two fits see the same test rows and the same amount of
    training data, and differ only in the leak rows.
    Returns (overlap, disjoint, leak_share per class).
    """
    ov, dj, share = [], [], []
    for c in (0, 1):
        b = np.flatnonzero(base & (y == c))
        lk = np.flatnonzero(leak & (y == c))
        n = len(b)
        if len(lk) >= n:
            lk = rng.choice(lk, size=n, replace=False)
            fill = np.array([], dtype=np.int64)
        else:
            fill = rng.choice(b, size=n - len(lk), replace=False)
        ov.append(np.concatenate([lk, fill]))
        dj.append(b)
        share.append(len(lk) / max(n, 1))
    return np.concatenate(ov), np.concatenate(dj), share


def _two_classes(y: np.ndarray) -> bool:
    return len(y) > 0 and y.min() != y.max()


def _replicate(X, y, unit, time, b: int, seed: int, test_frac: float,
               time_split, embargo: float, scorer: Scorer,
               threads: int) -> dict | None:
    """One unit split, one matched pair of fits. None if a class is absent."""
    rng = np.random.default_rng([seed, b])
    te, leak, base = overlap_masks(unit, time, rng, test_frac, time_split,
                                   embargo)
    ov, dj, share = matched_training(y, leak, base, rng)
    tei = np.flatnonzero(te)
    if not (_two_classes(y[tei]) and _two_classes(y[dj])):
        return None
    fit_seed = seed + b
    s_ov = fit_score(scorer, fit_seed, threads, X[ov], y[ov], X[tei])
    s_dj = fit_score(scorer, fit_seed, threads, X[dj], y[dj], X[tei])
    a_ov, a_dj = auc(y[tei], s_ov), auc(y[tei], s_dj)
    return {"replicate": b, "auc_overlap": a_ov, "auc_disjoint": a_dj,
            "gap": a_ov - a_dj, "n_test": int(len(tei)),
            "n_train": int(len(dj)), "n_test_units": int(len(np.unique(unit[tei]))),
            "leak_share_0": float(share[0]), "leak_share_1": float(share[1])}


@dataclass
class OverlapResult:
    """What the overlap check found.

    gap : mean over replicates of overlap AUC minus disjoint AUC.
    lo, hi : 2.5 and 97.5 percentiles of the per-replicate gap.
    auc_overlap, auc_disjoint : mean AUC of each arm.
    replicates : replicates that produced a gap.
    scorer : the classifier that produced the numbers.
    per_replicate : one row per replicate.
    info : what was checked (feature set, split, counts).
    """
    gap: float
    lo: float
    hi: float
    auc_overlap: float
    auc_disjoint: float
    replicates: int
    scorer: dict
    per_replicate: pd.DataFrame
    info: dict

    def to_dict(self) -> dict:
        return {"gap": self.gap, "lo": self.lo, "hi": self.hi,
                "auc_overlap": self.auc_overlap,
                "auc_disjoint": self.auc_disjoint,
                "replicates": self.replicates, "scorer": self.scorer,
                "info": self.info,
                "per_replicate": self.per_replicate.to_dict(orient="records")}

    def to_json(self, path=None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w") as f:
                f.write(s + "\n")
        return s

    def __repr__(self) -> str:
        split = self.info.get("time_split") or "random half of test-unit rows"
        return (f"overlap check ({self.replicates} replicates, "
                f"split: {split}, scorer: {self.scorer['name']})\n"
                f"  AUC overlap  {self.auc_overlap:.3f}\n"
                f"  AUC disjoint {self.auc_disjoint:.3f}\n"
                f"  gap          {self.gap:+.3f}  "
                f"[{self.lo:+.3f}, {self.hi:+.3f}]")


def overlap_check(panel: Panel, features, *, model=None,
                  replicates: int = 200, test_frac: float = 0.2,
                  time_split: str | None = None, embargo=None,
                  seed: int = 0, n_jobs: int | None = None) -> OverlapResult:
    """Measure how much sharing units between training and test inflates AUC.

    Parameters
    ----------
    panel : a ``Panel`` with an event column; the per-row event flag is
        the label.
    features : list of feature columns (one feature set).
    model : None for the library default, 'mojolearn', 'lightgbm', an
        estimator with fit and predict_proba, or a function from a seed
        to one.
    replicates : independent unit splits; each gives one gap.
    test_frac : fraction of units used as test units. With
        ``time_split="calendar"`` it is also the fraction of rows after
        the time cut.
    time_split : None, or "calendar" to take test rows from after a cut
        time and training rows from before it, in both arms.
    embargo : with "calendar", training rows must be at or before
        ``cut - embargo``; an offset in the time column's scale (a string
        such as '30D' for datetimes).
    n_jobs : parallel workers over replicates; every core by default, or
        the WHOWHEN_JOBS environment variable.
    """
    if not panel.event:
        raise ValueError("the overlap check needs a panel with an event "
                         "column; pass event= to Panel")
    if time_split not in TIME_SPLITS:
        raise ValueError(f"time_split must be one of {TIME_SPLITS}; "
                         f"got {time_split!r}")
    if embargo is not None and time_split != "calendar":
        raise ValueError("embargo applies only with time_split='calendar'")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    features = list(features)
    if not features:
        raise ValueError("the feature set needs at least one column")
    emb = panel.offset(embargo) if embargo is not None else 0.0
    if emb < 0:
        raise ValueError("embargo must be non-negative")
    sc = resolve(model)
    frame = panel.frame(features, keep_event_units=True)
    if frame.n_units < 10:
        raise ValueError(f"only {frame.n_units} units; the overlap check "
                         f"needs many units")
    y = frame.y
    workers = n_workers(n_jobs, replicates)
    threads = max(1, (os.cpu_count() or 1) // workers)
    runs = Parallel(n_jobs=workers)(
        delayed(_replicate)(frame.X, y, frame.unit, frame.time, b, seed,
                            test_frac, time_split, emb, sc, threads)
        for b in range(replicates))
    runs = [r for r in runs if r is not None]
    if not runs:
        raise ValueError("no replicate had both classes in its test and "
                         "training rows; the panel has too few events")
    per = pd.DataFrame(runs)
    lo, hi = np.percentile(per["gap"], [2.5, 97.5])
    info = {"features": features, "n_units": frame.n_units,
            "n_rows": int(len(y)), "event_rate": float(y.mean()),
            "test_frac": test_frac, "time_split": time_split,
            "embargo": None if embargo is None else str(embargo),
            "replicates_requested": replicates, "seed": seed,
            "workers": workers}
    return OverlapResult(gap=float(per["gap"].mean()), lo=float(lo),
                         hi=float(hi),
                         auc_overlap=float(per["auc_overlap"].mean()),
                         auc_disjoint=float(per["auc_disjoint"].mean()),
                         replicates=len(per), scorer=sc.describe(),
                         per_replicate=per, info=info)
