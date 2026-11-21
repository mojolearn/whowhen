"""Record-wise vs subject-wise validation: does training on other rows of
the test units inflate the score?

Each replicate scores the same test rows twice, with the same number of
training rows of each class:

* disjoint: training holds no row of any test unit;
* overlap: some training rows are swapped for other rows of the test
  units (rows that are not themselves test rows).

The two fits differ only in whether training shares units with the test
rows. The gap in AUC (overlap minus disjoint), over many repeats with
fresh unit splits, is the result. A gap whose interval excludes zero
means this classifier, on these columns, scores higher when training
shares units with the test rows. It shows the evaluation is sensitive
to unit overlap; it does not measure what another model gains from it.

With a forecast horizon, two rules keep labels honest. A row of a unit
with no event is used only if the unit was still observed one horizon
later, since otherwise "no event within the horizon" was never seen.
And under a time split, a training row from before the cut is used
only if its horizon has passed by the cut, since otherwise its label
was not yet known at the time of the prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._stats import auc
from .audit import cores, n_workers, pick_seed
from .pairs import unit_groups
from .panel import Panel
from .scorers import Scorer, fit_score, resolve

TIME_SPLITS = (None, "calendar", "chronological")


def overlap_masks(unit: np.ndarray, time: np.ndarray, rng: np.random.Generator,
                  test_units: np.ndarray, time_split: str | None = None,
                  settle: float = 0.0):
    """(test, leak, base) boolean row masks for one repeat.

    ``test_units`` flags the held-out units. Without a time split, test
    rows are a random half of each test unit's rows and leak rows are
    the other half. With ``time_split="calendar"``, the cut leaves the
    same share of rows after it as the share of units held out: test
    rows are test-unit rows after the cut, and leak and base rows are at
    or before it. Base rows are rows of units that are
    not test units. Leak and base never contain a test row; base never
    contains a row of a test unit.

    ``settle`` is how long after a row its label becomes known: the
    forecast horizon, or 0 when the label is the row's own flag. Under
    a time split, a row from before a cut is trained on only if its
    label is known by that cut. With "chronological", a unit's cut is
    the time of the last row in its earlier half.
    """
    tu = test_units[unit]
    if time_split is None:
        half = rng.random(len(unit)) < 0.5
        return tu & half, tu & ~half, ~tu
    if time_split == "calendar":
        cut = float(np.quantile(time, 1.0 - test_units.mean()))
        early = time + settle <= cut
        return tu & (time > cut), tu & early, ~tu & early
    if time_split == "chronological":
        # rows are sorted by (unit, time): the later half of each test
        # unit's rows is tested, the earlier half may be trained on
        n = len(unit)
        starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
        counts = np.diff(np.r_[starts, n])
        pos = np.arange(n) - np.repeat(starts, counts)
        late = pos >= np.repeat(counts, counts) / 2
        cut = np.full(int(unit.max()) + 1, -np.inf)
        np.maximum.at(cut, unit[~late], time[~late])
        settled = time + settle <= cut[unit]
        return tu & late, tu & ~late & settled, ~tu
    raise ValueError(f"time_split must be one of {TIME_SPLITS}; "
                     f"got {time_split!r}")


def forecast_labels(unit: np.ndarray, time: np.ndarray, y: np.ndarray,
                    horizon: float):
    """(keep, label, rows without follow-up) for a fixed-horizon task.

    A unit's event time is the time of its first flagged row. Rows at or
    after it are dropped; a kept row is positive when the event comes
    within ``horizon`` after it. A row of a unit with no event is
    negative if the unit was still observed one horizon later, taking
    the unit's last row as the end of its follow-up. Otherwise the row
    is dropped: nobody saw whether the event came within the horizon.
    The third value counts the rows dropped for that reason.
    """
    n_units = int(unit.max()) + 1 if len(unit) else 0
    onset = np.full(n_units, np.inf)
    ev = np.flatnonzero(y == 1)
    np.minimum.at(onset, unit[ev], time[ev])
    first = np.full(n_units, np.inf)
    np.minimum.at(first, unit, time)
    had = np.isfinite(onset)
    if had.any() and (onset[had] <= first[had]).mean() > 0.5:
        raise ValueError(
            "most units with an event are flagged from their first row, so "
            "the event column says whether a unit had the event, not when; "
            "a horizon needs the flag to start on the row where the event "
            "happens")
    last = np.full(n_units, -np.inf)
    np.maximum.at(last, unit, time)
    t_on = onset[unit]
    before = time < t_on
    followed = np.isfinite(t_on) | (time + horizon <= last[unit])
    keep = before & followed
    label = (keep & (t_on - time <= horizon)).astype(np.int8)
    if not label[keep].any():
        raise ValueError("no row falls within the horizon before an event")
    return keep, label[keep], int((before & ~followed).sum())


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


def held_out(groups: np.ndarray, b: int, random: bool,
             rng: np.random.Generator) -> np.ndarray:
    """Per unit: is it held out in repeat ``b``? Fixed group ``b``, or a
    random draw of the same share."""
    if random:
        return rng.random(len(groups)) < 1.0 / (int(groups.max()) + 1)
    return groups == b


def _replicate(X, y, unit, time, b: int, seed: int, groups, random: bool,
               time_split, settle: float, scorer: Scorer,
               threads: int) -> dict | None:
    """One unit split, one matched pair of fits. None if a class is
    absent; {"no_leak": True} if no other row of a test unit may be
    trained on."""
    rng = np.random.default_rng([seed, b])
    te, leak, base = overlap_masks(unit, time, rng,
                                   held_out(groups, b, random, rng),
                                   time_split, settle)
    if not leak.any():
        return {"no_leak": True}
    ov, dj, share = matched_training(y, leak, base, rng)
    tei = np.flatnonzero(te)
    if not (_two_classes(y[tei]) and _two_classes(y[dj])):
        return None
    fit_seed = seed + b
    s_ov = fit_score(scorer, fit_seed, threads, X[ov], y[ov], X[tei])
    s_dj = fit_score(scorer, fit_seed, threads, X[dj], y[dj], X[tei])
    a_ov, a_dj = auc(y[tei], s_ov), auc(y[tei], s_dj)
    return {"replicate": b, "auc_record_wise": a_ov, "auc_subject_wise": a_dj,
            "gap": a_ov - a_dj, "n_test": int(len(tei)),
            "n_train": int(len(dj)), "n_test_units": int(len(np.unique(unit[tei]))),
            "leak_share_0": float(share[0]), "leak_share_1": float(share[1])}


@dataclass
class OverlapResult:
    """What record-wise vs subject-wise validation found.

    gap : mean over repeats of overlap AUC minus disjoint AUC.
    lo, hi : 2.5 and 97.5 percentiles of the per-replicate gap.
    auc_record_wise, auc_subject_wise : mean AUC of each arm.
    repeats : repeats that produced a gap.
    scorer : the classifier that produced the numbers.
    per_replicate : one row per replicate.
    info : what was checked (feature set, split, counts).
    """
    gap: float
    lo: float
    hi: float
    auc_record_wise: float
    auc_subject_wise: float
    repeats: int
    scorer: dict
    per_replicate: pd.DataFrame
    info: dict

    def to_dict(self) -> dict:
        return {"gap": self.gap, "lo": self.lo, "hi": self.hi,
                "auc_record_wise": self.auc_record_wise,
                "auc_subject_wise": self.auc_subject_wise,
                "repeats": self.repeats, "scorer": self.scorer,
                "info": self.info,
                "per_replicate": self.per_replicate.to_dict(orient="records")}

    def __repr__(self) -> str:
        split = {None: "a random half of each test unit's rows is tested",
                 "chronological": "the later half of each test unit's rows "
                                  "is tested",
                 "calendar": "rows after a cut in time are tested, earlier "
                             "rows are trained on",
                 }[self.info.get("time_split")]
        return (f"record-wise vs subject-wise validation ({self.repeats} "
                f"repeats, split: {split}, scorer: {self.scorer['name']})\n"
                f"  AUC record-wise  {self.auc_record_wise:.3f}\n"
                f"  AUC subject-wise {self.auc_subject_wise:.3f}\n"
                f"  gap              {self.gap:+.3f}  "
                f"[{self.lo:+.3f}, {self.hi:+.3f}]"
                + follow_up_note(self.info)
                + ("\n  training rows from before the cut are used only "
                   "if their horizon has passed by the cut"
                   if self.info.get("labels_settled_by_cut") else ""))


def follow_up_note(info: dict) -> str:
    n = info.get("rows_without_follow_up")
    return (f"\n  {n:,} rows left out: their unit's record ends before "
            f"the horizon does" if n else "")



def overlap_check(panel: Panel, features, *, model=None,
                  repeats: int = 4,
                  time_split: str | None = None,
                  horizon=None, random: bool = False, seed: int | None = None,
                  n_jobs: int | None = None) -> OverlapResult:
    """Measure how much sharing units between training and test inflates AUC.

    Parameters
    ----------
    panel : a ``Panel`` with an event column; the per-row event flag is
        the label.
    features : list of feature columns (one feature set).
    model : None for the library default, 'mojolearn', 'catboost', an
        estimator with fit and predict_proba, or a function from a seed
        to one.
    repeats : one number, N. Units fall into N + 1 fixed groups and the
        check runs N times, each holding out a different group as the
        test units. More repeats give a steadier range but smaller test
        sets.
    random : False by default (same numbers every run). True draws the
        test units at random instead.
    horizon : None to use the event flag of each row as the label, or an
        offset in the time column's scale to forecast instead: the label
        becomes "the unit's event comes within this long after the row",
        and rows at or after the event are dropped. The event column
        must then flag the row where the event happens.
    time_split : None; "calendar" to take test rows from after a cut
        time and training rows from before it, in both arms; or
        "chronological" to test the later half of each test unit's rows
        and let the overlap arm train on the earlier half, which asks
        whether a unit's own past helps predict its own future.
    n_jobs : parallel workers over repeats; every core by default, or
        the WHOWHEN_JOBS environment variable.
    """
    if not panel.event:
        raise ValueError("record-wise vs subject-wise validation needs a panel with an event "
                         "column; pass event= to Panel")
    if time_split not in TIME_SPLITS:
        raise ValueError(f"time_split must be one of {TIME_SPLITS}; "
                         f"got {time_split!r}")
    if panel.time is None and (time_split or horizon is not None):
        raise ValueError("a time split or a horizon needs a time column")
    features = list(features)
    if not features:
        raise ValueError("the feature set needs at least one column")
    seed = pick_seed(seed, random)
    sc = resolve(model)
    frame = panel.frame(features, keep_event_units=True)
    if frame.n_units < 10:
        raise ValueError(f"only {frame.n_units} units; record-wise vs subject-wise validation "
                         f"needs many units")
    X, y, unit, time = frame.X, frame.y, frame.unit, frame.time
    settle, unfollowed = 0.0, None
    if horizon is not None:
        settle = panel.offset(horizon)
        keep, y, unfollowed = forecast_labels(unit, time, y, settle)
        X, unit, time = X[keep], unit[keep], time[keep]
    groups = unit_groups(frame, repeats + 1)
    workers = n_workers(n_jobs, repeats)
    threads = max(1, cores() // workers)
    runs = Parallel(n_jobs=workers)(
        delayed(_replicate)(X, y, unit, time, b, seed,
                            groups, random, time_split, settle, sc, threads)
        for b in range(repeats))
    runs = [r for r in runs if r is not None]
    if runs and all("no_leak" in r for r in runs):
        raise ValueError("no earlier row of a test unit has its label "
                         "known by the cut; the horizon is too long for "
                         "this split")
    runs = [r for r in runs if "no_leak" not in r]
    if not runs:
        raise ValueError(
            "no repeat had both classes in its test rows and in the "
            "training rows whose labels are known by the cut"
            if settle and time_split else
            "no replicate had both classes in its test and "
            "training rows; the panel has too few events")
    per = pd.DataFrame(runs)
    lo, hi = np.percentile(per["gap"], [2.5, 97.5])
    info = {"features": features, "n_units": frame.n_units,
            "n_rows": int(len(y)), "event_rate": float(y.mean()),
            "random": random, "time_split": time_split,
            "horizon": None if horizon is None else str(horizon),
            "rows_without_follow_up": unfollowed,
            "labels_settled_by_cut": bool(settle and time_split),
            "repeats_requested": repeats, "seed": seed,
            "workers": workers}
    return OverlapResult(gap=float(per["gap"].mean()), lo=float(lo),
                         hi=float(hi),
                         auc_record_wise=float(per["auc_record_wise"].mean()),
                         auc_subject_wise=float(per["auc_subject_wise"].mean()),
                         repeats=len(per), scorer=sc.describe(),
                         per_replicate=per, info=info)
