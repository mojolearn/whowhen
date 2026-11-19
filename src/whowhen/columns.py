"""Which columns carry the signal: a per-column ranking."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._stats import auc
from .audit import KINDS, Draw, draw, n_workers, pick_seed
from .panel import Frame, Panel
from .scorers import Scorer, resolve


def _fit_auc(frame: Frame, d: Draw, cols: tuple, scorer: Scorer, seed: int,
             threads: int) -> float:
    """Held-out AUC of one column set on one shared draw."""
    return auc(d.y[~d.tr], d.fit(frame, list(cols), scorer, seed, threads))


def rank_columns(panel: Panel, columns, offset, *, scorer=None,
                 which=("identity", "timing"),
                 n_pairs: int = 20000, max_pairs_per_unit: int = 20,
                 repeats: int = 4, random: bool = False,
                 seed: int | None = None,
                 n_jobs: int | None = None) -> dict[str, pd.DataFrame]:
    """Rank columns by how much identity or timing signal each carries.

    For each audit in ``which`` and each seed, the pairs and the unit
    split are drawn once and shared by every fit, so differences between
    columns are not pair noise. The draws are the ones audit() makes
    for the same seed.

    Returns a dict from audit name to a table with one row per column,
    sorted by ``alone`` descending:

    column : the column.
    alone : AUC using that column only, mean over repeats.
    drop : AUC with all columns minus AUC without that column, mean over
        repeats. Near zero means the other columns cover what it carries.
        With a single column, the set without it scores 0.5 (chance).
    all : AUC with all columns, the same on every row.

    The scorer description is in each table's ``attrs["scorer"]``. No
    bootstrap is run: this is a ranking, not an interval estimate; use
    audit() for intervals on any set this points to.

    Parameters are as in audit(). n_jobs : parallel workers over
    (audit, seed, column set) fits; every core by default, or the
    WHOWHEN_JOBS environment variable.
    """
    columns = list(dict.fromkeys(columns))
    seed = pick_seed(seed, random)
    if not columns:
        raise ValueError("rank_columns needs at least one column")
    which = (which,) if isinstance(which, str) else tuple(which)
    bad = [w for w in which if w not in KINDS]
    if bad:
        raise ValueError(f"unknown audit {bad}; choose from {KINDS}")
    lo, hi = panel.offset(offset[0]), panel.offset(offset[1])
    if not 0 < lo <= hi:
        raise ValueError("offset needs 0 < min <= max")
    sc = resolve(scorer)
    frame = panel.frame(columns)
    frame.repeats, frame.random = repeats, random
    if frame.n_units < 10:
        raise ValueError(f"only {frame.n_units} units without an event; "
                         f"the audits need many units")
    cores = os.cpu_count() or 1

    # Phase 1: one draw per (audit, seed), in parallel.
    keys = [(kind, k) for kind in which for k in range(repeats)]
    workers = n_workers(n_jobs, len(keys))
    draws = dict(zip(keys, Parallel(n_jobs=workers)(
        delayed(draw)(frame, kind, "random", k, seed, lo, hi, n_pairs,
                      max_pairs_per_unit) for kind, k in keys)))

    # Phase 2: every (audit, seed, column set) fit, in parallel. Column
    # sets are deduplicated, so one column's "alone" and "all" share a fit.
    idx = list(range(len(columns)))
    full = tuple(idx)
    sets = {full} | {(j,) for j in idx}
    sets |= {tuple(i for i in idx if i != j) for j in idx}
    sets.discard(())
    sets = sorted(sets)
    tasks = [(key, cols) for key in keys for cols in sets]
    workers = n_workers(n_jobs, len(tasks))
    threads = max(1, cores // workers)
    aucs = dict(zip(tasks, Parallel(n_jobs=workers)(
        delayed(_fit_auc)(frame, draws[key], cols, sc, seed, threads)
        for key, cols in tasks)))

    out = {}
    for kind in which:
        ks = [(kind, k) for k in range(repeats)]
        per_seed = lambda cols: np.array(
            [aucs[key, cols] if cols else 0.5 for key in ks])
        a_full = per_seed(full)
        rows = []
        for j, c in enumerate(columns):
            rest = tuple(i for i in idx if i != j)
            rows.append({"column": c,
                         "alone": float(per_seed((j,)).mean()),
                         "drop": float((a_full - per_seed(rest)).mean()),
                         "all": float(a_full.mean())})
        t = (pd.DataFrame(rows).sort_values("alone", ascending=False,
                                            kind="stable")
             .reset_index(drop=True))
        t.attrs["scorer"] = sc.describe()
        out[kind] = t
    return out
