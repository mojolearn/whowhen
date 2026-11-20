"""Run the audits and report."""
from __future__ import annotations

import json
import os

import joblib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import pairs as P
from ._stats import auc, bootstrap_auc
from .panel import Frame, Panel
from .scorers import Scorer, fit_score, resolve

KINDS = ("identity", "timing", "placebo")


def cores() -> int:
    """Cores this process may use. Inside a container this is the
    container's share, not the host's core count."""
    return joblib.cpu_count()


def n_workers(n_jobs: int | None, n_tasks: int) -> int:
    """Workers for ``n_tasks`` independent tasks: every core by default."""
    if n_jobs is None:
        n_jobs = int(os.environ.get("WHOWHEN_JOBS", 0)) or cores()
    return max(1, min(n_jobs, n_tasks))


@dataclass
class Draw:
    """One pair draw and one unit split for one (audit, seed).

    Every feature set fitted on a Draw sees the same pairs and the same
    held-out units, so differences between sets are not pair noise.
    rows_a, rows_b : the two rows of each identity pair; for timing,
        rows_a holds the single rows and rows_b is None.
    roles : held-out rows' unit codes (0..n_test_units-1), one array per
        unit role, for the cluster bootstrap.
    """
    kind: str
    mode: str
    k: int
    rows_a: np.ndarray
    rows_b: np.ndarray | None
    y: np.ndarray
    tr: np.ndarray
    roles: list
    n_test_units: int

    def features(self, frame: Frame, cols: list[int]) -> np.ndarray:
        if self.rows_b is not None:
            return P.pair_features(frame, self.rows_a, self.rows_b, cols)
        return frame.X[np.ix_(self.rows_a, cols)]

    def fit(self, frame: Frame, cols: list[int], scorer: Scorer, seed: int,
            threads: int) -> np.ndarray:
        """Held-out scores of a fresh scorer fitted on ``cols``."""
        X = self.features(frame, cols)
        return fit_score(scorer, seed, threads, X[self.tr], self.y[self.tr],
                         X[~self.tr])


def draw(frame: Frame, kind: str, mode: str, k: int, seed: int, lo: float,
         hi: float, n_pairs: int, max_per_unit: int) -> Draw:
    """The pairs and unit split of seed ``k``; deterministic in
    (seed, k), so audit() and rank_columns() draw the same pairs."""
    rng = np.random.default_rng(seed + k)
    train = P.split_units(frame, k, frame.repeats,
                          rng if frame.random else None)
    if kind == "identity":
        ra, rb, y, tr = P.identity_pairs(frame, lo, hi, n_pairs,
                                         max_per_unit, rng, train, mode)
        roles_all = [frame.unit[ra], frame.unit[rb]]
    else:
        ra, y = P.timing_pairs(frame, lo, hi, n_pairs, max_per_unit, rng,
                               placebo=(kind == "placebo"))
        rb = None
        tr = train[frame.unit[ra]]
        roles_all = [frame.unit[ra]]
    te = ~tr
    if tr.sum() < 20 or te.sum() < 20:
        raise ValueError(
            f"the {kind} audit drew {int(tr.sum())} training and "
            f"{int(te.sum())} held-out rows; too few units have two rows "
            f"this offset apart")
    codes, inv = np.unique(np.concatenate([r[te] for r in roles_all]),
                           return_inverse=True)
    return Draw(kind, mode, k, ra, rb, y, tr,
                np.split(inv, len(roles_all)), int(len(codes)))


def pick_seed(seed: int | None, random: bool = False) -> int:
    """The seed to use: the one given; otherwise 0, or a fresh one when
    ``random``. The value is recorded in every result so a run can be
    repeated by passing it back."""
    if seed is not None:
        return int(seed)
    if random:
        return int(np.random.SeedSequence().entropy % 2 ** 31)
    return 0


def _one_seed(frame: Frame, kind: str, mode: str, k: int, seed: int,
              sets: dict, lo: float, hi: float, n_pairs: int,
              max_per_unit: int, n_boot: int, scorer: Scorer,
              threads: int) -> dict:
    """One pair draw, one unit split, one fit per feature set."""
    d = draw(frame, kind, mode, k, seed, lo, hi, n_pairs, max_per_unit)
    brng = np.random.default_rng(seed + 7 + 1000 * k)
    yte = d.y[~d.tr]
    out = {"kind": kind, "impostors": mode, "k": k,
           "n_train": int(d.tr.sum()), "n_test": int((~d.tr).sum()),
           "n_test_units": d.n_test_units, "sets": {}}
    for name, cols in sets.items():
        s = d.fit(frame, cols, scorer, seed, threads)
        out["sets"][name] = {
            "auc": auc(yte, s),
            "boot": bootstrap_auc(yte, s, d.roles, d.n_test_units, n_boot,
                                  brng)}
    return out


@dataclass
class AuditResult:
    """What the audits found.

    identity, timing, placebo : one row per feature set, with the mean
        AUC over repeats, a 95 percent interval (unit-cluster bootstrap
        pooled over repeats, so it reflects both which pairs were drawn
        and which units were held out) and the range over repeats. None
        for an audit that was not run.
    scorer : the classifier that produced the numbers.
    info : what was audited (unit counts, offset, pair counts, repeats).
    """
    scorer: dict
    info: dict
    identity: pd.DataFrame | None = None
    timing: pd.DataFrame | None = None
    placebo: pd.DataFrame | None = None
    tables: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = {"scorer": self.scorer, "info": self.info}
        for kind in KINDS:
            t = getattr(self, kind)
            if t is not None:
                d[kind] = t.to_dict(orient="records")
        return d

    def to_json(self, path=None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w") as f:
                f.write(s + "\n")
        return s

    def __repr__(self) -> str:
        lines = [f"scorer: {self.scorer['name']}"]
        for kind in KINDS:
            t = getattr(self, kind)
            if t is not None:
                lines += ["", f"{kind} (AUC; 0.5 = no signal)",
                          t.round(3).to_string(index=False)]
        return "\n".join(lines)


def _table(runs: list[dict], kind: str) -> pd.DataFrame | None:
    runs = [r for r in runs if r["kind"] == kind]
    if not runs:
        return None
    rows = []
    for mode in dict.fromkeys(r["impostors"] for r in runs):
        sub = [r for r in runs if r["impostors"] == mode]
        for name in sub[0]["sets"]:
            aucs = [r["sets"][name]["auc"] for r in sub]
            boot = np.concatenate([r["sets"][name]["boot"] for r in sub])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row = {"features": name}
            if kind == "identity":
                row["impostors"] = mode
            row.update(auc=float(np.mean(aucs)), lo=float(lo), hi=float(hi),
                       lowest_run=float(np.min(aucs)),
                       highest_run=float(np.max(aucs)),
                       test_rows=int(np.mean([r["n_test"] for r in sub])))
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_sets(features) -> dict:
    if isinstance(features, dict):
        sets = {str(k): list(v) for k, v in features.items()}
    else:
        sets = {"all": list(features)}
    if not sets or any(not v for v in sets.values()):
        raise ValueError("every feature set needs at least one column")
    return sets


def audit(panel: Panel, features, offset, *, scorer=None,
          which=("identity", "timing"), impostors="random",
          n_pairs: int = 20000,
          max_pairs_per_unit: int = 20, repeats: int = 4,
          bootstrap: int = 2000, random: bool = False,
          seed: int | None = None,
          n_jobs: int | None = None) -> AuditResult:
    """Audit a panel for identity and timing signal.

    Parameters
    ----------
    panel : a ``Panel``.
    features : a list of columns, or a dict of named column lists to
        audit side by side on the same pairs (for example with and
        without a clock column, or before and after imputation).
    offset : (min, max) gap between the two rows of a pair, in the time
        column's scale; strings such as ("6h", "24h") for datetimes.
    scorer : None for the default, 'mojolearn', 'catboost', an estimator
        with fit and predict_proba, or a function from a seed to one.
    which : any of 'identity', 'timing', 'placebo'.
    impostors : 'random', 'nearest', or both in a tuple. See
        ``whowhen.pairs.identity_pairs``.
    n_pairs, max_pairs_per_unit : how many pairs to draw, and how many
        one unit may contribute.
    repeats : how many times to repeat. Units fall into repeats + 1
        fixed groups (by a checksum of the unit identifier) and each
        repeat holds out a different group and draws its own pairs: 1
        holds out a half, 2 a third each, 4 a fifth each. More repeats
        test more of the units and leave more to train on.
    random : False by default: the held-out groups are fixed and the
        same call gives the same numbers on every run. True draws the
        held-out units and the pairs at random instead, differently on
        each run.
    seed : leave unset. A random run records the seed it used in the
        result; pass that value here to repeat that exact run.
    """
    sets = _feature_sets(features)
    which = (which,) if isinstance(which, str) else tuple(which)
    bad = [w for w in which if w not in KINDS]
    if bad:
        raise ValueError(f"unknown audit {bad}; choose from {KINDS}")
    modes = (impostors,) if isinstance(impostors, str) else tuple(impostors)
    lo, hi = panel.offset(offset[0]), panel.offset(offset[1])
    if not 0 < lo <= hi:
        raise ValueError("offset needs 0 < min <= max")
    seed = pick_seed(seed, random)
    sc = resolve(scorer)
    columns = list(dict.fromkeys(c for v in sets.values() for c in v))
    frame = panel.frame(columns)
    frame.repeats, frame.random = repeats, random
    if frame.n_units < 10:
        raise ValueError(f"only {frame.n_units} units without an event; "
                         f"the audits need many units")
    col_sets = {name: frame.cols(v) for name, v in sets.items()}
    tasks = [(kind, mode, k) for kind in which
             for mode in (modes if kind == "identity" else ("random",))
             for k in range(repeats)]
    workers = n_workers(n_jobs, len(tasks))
    threads = max(1, cores() // workers)
    runs = Parallel(n_jobs=workers)(
        delayed(_one_seed)(frame, kind, mode, k, seed, col_sets, lo, hi,
                           n_pairs, max_pairs_per_unit, bootstrap, sc,
                           threads)
        for kind, mode, k in tasks)
    info = {"n_units": frame.n_units_all,
            "n_units_without_event": frame.n_units,
            "n_rows_audited": int(len(frame.unit)),
            "offset": [str(offset[0]), str(offset[1])],
            "repeats": repeats, "bootstrap": bootstrap, "seed": seed, "random": random,
            "feature_sets": sets, "workers": workers}
    return AuditResult(scorer=sc.describe(), info=info,
                       **{kind: _table(runs, kind) for kind in which})
