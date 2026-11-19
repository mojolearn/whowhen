"""Clock baseline: how much of the task does elapsed time alone predict?

Many event tasks on panel data can be scored well above chance from one
number, the time since the unit was first observed. A model's score
means little until it is set beside that.

Each replicate holds out a fresh fifth of the units and fits the same
scorer on several inputs: the clock alone, the features alone, and the
features with the clock. When rows have missing values, the count of
observed values per row is added as a second cheap input.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._stats import auc
from .audit import n_workers, pick_seed
from .overlap import forecast_labels
from .panel import Panel
from .scorers import Scorer, fit_score, resolve


def _replicate(sets: dict, y, unit, b: int, seed: int, test_frac: float,
               scorer: Scorer, threads: int) -> dict | None:
    rng = np.random.default_rng([seed, b])
    te = (rng.random(int(unit.max()) + 1) < test_frac)[unit]
    if y[te].min() == y[te].max() or y[~te].min() == y[~te].max():
        return None
    return {name: auc(y[te], fit_score(scorer, seed + b, threads, X[~te],
                                       y[~te], X[te]))
            for name, X in sets.items()}


@dataclass
class ClockResult:
    """What the clock baseline found.

    table : one row per input, with the mean held-out AUC and the 2.5
        and 97.5 percentiles over replicates.
    gain, gain_lo, gain_hi : AUC of features with the clock minus AUC of
        the clock alone, per replicate: what the features add once the
        clock is known.
    """
    table: pd.DataFrame
    gain: float
    gain_lo: float
    gain_hi: float
    replicates: int
    scorer: dict
    info: dict

    def to_dict(self) -> dict:
        return {"table": self.table.to_dict(orient="records"),
                "gain": self.gain, "gain_lo": self.gain_lo,
                "gain_hi": self.gain_hi, "replicates": self.replicates,
                "scorer": self.scorer, "info": self.info}

    def __repr__(self) -> str:
        return (f"clock baseline ({self.replicates} replicates, units held "
                f"out, scorer: {self.scorer['name']})\n"
                f"{self.table.round(3).to_string(index=False)}\n"
                f"  features add {self.gain:+.3f} "
                f"[{self.gain_lo:+.3f}, {self.gain_hi:+.3f}] to the clock")


def clock_baseline(panel: Panel, features, *, horizon=None, model=None,
                   replicates: int = 50, test_frac: float = 0.2,
                   seed: int | None = 0, n_jobs: int | None = None) -> ClockResult:
    """Score the event task from elapsed time alone, and beside the features.

    Parameters are as in ``overlap_check``. The clock is the time since
    the unit's first row. ``features`` should not themselves include a
    clock column if the comparison is to mean anything.
    """
    if not panel.event:
        raise ValueError("the clock baseline needs a panel with an event "
                         "column; pass event= to Panel")
    features = list(features)
    seed = pick_seed(seed)
    sc = resolve(model)
    frame = panel.frame(features, keep_event_units=True)
    X, y, unit, time = frame.X, frame.y, frame.unit, frame.time
    first = np.full(frame.n_units, np.inf)
    np.minimum.at(first, unit, time)
    clock = (time - first[unit]).astype(np.float32)
    if panel.datetime:
        clock = (clock / 3.6e12).astype(np.float32)      # hours
    clock = clock[:, None]
    if horizon is not None:
        keep, y = forecast_labels(unit, time, y, panel.offset(horizon))
        X, unit, clock = X[keep], unit[keep], clock[keep]
    sets = {"clock": clock, "features": X,
            "features + clock": np.hstack([X, clock])}
    if np.isnan(X).any():
        counts = (~np.isnan(X)).sum(1, keepdims=True).astype(np.float32)
        sets = {"clock": clock, "clock + counts": np.hstack([clock, counts]),
                "features": X, "features + clock": sets["features + clock"]}
    workers = n_workers(n_jobs, replicates)
    threads = max(1, (os.cpu_count() or 1) // workers)
    runs = Parallel(n_jobs=workers)(
        delayed(_replicate)(sets, y, unit, b, seed, test_frac, sc, threads)
        for b in range(replicates))
    per = pd.DataFrame([r for r in runs if r is not None])
    if per.empty:
        raise ValueError("no replicate had both classes in its test and "
                         "training rows; the panel has too few events")
    table = pd.DataFrame([{"input": c, "auc": per[c].mean(),
                           "lo": np.percentile(per[c], 2.5),
                           "hi": np.percentile(per[c], 97.5)}
                          for c in per.columns])
    g = per["features + clock"] - per["clock"]
    info = {"features": features, "n_units": frame.n_units,
            "n_rows": int(len(y)), "event_rate": float(y.mean()),
            "horizon": None if horizon is None else str(horizon),
            "replicates_requested": replicates, "seed": seed}
    return ClockResult(table=table, gain=float(g.mean()),
                       gain_lo=float(np.percentile(g, 2.5)),
                       gain_hi=float(np.percentile(g, 97.5)),
                       replicates=len(per), scorer=sc.describe(), info=info)
