"""Invariants and a planted-signal check for whowhen.validation.

1. Both arms score one shared test set, and their training sets hold the
   same number of rows of each class.
2. The disjoint arm never contains a row of a test unit; no training set
   contains a test row; the overlap arm's swapped-in rows are rows of
   test units; calendar arms never train past their cut.
3. When labels follow a per-unit level that a model can memorise, the
   gap's interval excludes zero; when labels are independent per row,
   it includes zero.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from whowhen.panel import Panel
from whowhen.validation import matched_training, record_wise_validation, split_masks

FEATS = ["level", "s0", "s1", "s2", "s3"]


def synth(n_units=300, rows=40, seed=0, per_unit_labels=True):
    """Labels from a per-unit latent level, or independent per row.

    ``level`` shows the latent level through heavy row noise; ``s0..s3``
    are a per-unit signature that carries no label information by itself
    but lets a model recognise the unit.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 2.0, n_units)
    sig = rng.normal(0, 3.0, (n_units, 4))
    u = np.repeat(np.arange(n_units), rows)
    if per_unit_labels:
        p = 1 / (1 + np.exp(-(latent[u] - 1.0)))
    else:
        p = np.full(len(u), 0.3)
    df = pd.DataFrame({
        "unit": np.array([f"u{i:03d}" for i in range(n_units)])[u],
        "time": np.tile(pd.date_range("2025-01-01", periods=rows,
                                      freq="D").to_numpy(), n_units),
        "event": (rng.random(len(u)) < p).astype(int),
        "level": latent[u] + rng.normal(0, 4.0, len(u)),
    })
    for j in range(4):
        df[f"s{j}"] = sig[u, j] + rng.normal(0, 0.5, len(u))
    return Panel(df, unit="unit", time="time", event="event")


@pytest.mark.parametrize("split", [None, "calendar"])
def test_shared_test_equal_counts_and_disjointness(split):
    f = synth(seed=1).frame(FEATS, keep_event_units=True)
    rng = np.random.default_rng(1)
    te, leak, base = split_masks(f.unit, f.time, rng,
                                   rng.random(f.n_units) < 0.2, split)
    assert te.sum() > 0 and leak.sum() > 0 and base.sum() > 0
    assert not (te & leak).any() and not (te & base).any()
    ov, dj, _ = matched_training(f.y, leak, base, rng)
    for c in (0, 1):
        assert (f.y[ov] == c).sum() == (f.y[dj] == c).sum()
    test_units = set(f.unit[te])
    assert not test_units & set(f.unit[dj])
    assert not set(np.flatnonzero(te)) & set(ov)
    swapped = np.setdiff1d(ov, dj)
    assert len(swapped) > 0
    assert set(f.unit[swapped]) <= test_units
    assert not te[swapped].any()
    if split == "calendar":
        cut = np.quantile(f.time, 0.8)
        assert f.time[np.concatenate([ov, dj])].max() <= cut
        assert f.time[te].min() > cut


def test_needs_event_column():
    p = synth()
    bare = Panel(p.df.drop(columns="event"), unit="unit", time="time")
    with pytest.raises(ValueError, match="event column"):
        record_wise_validation(bare, FEATS, model=HistGradientBoostingClassifier())


KW = dict(model=HistGradientBoostingClassifier(max_iter=40), repeats=20,
          n_jobs=2)


def test_memorisable_units_inflate_score():
    r = record_wise_validation(synth(seed=2), FEATS, **KW)
    print(r)
    assert r.repeats == 20 and len(r.per_replicate) == 20
    assert r.scorer["name"] == "sklearn.HistGradientBoostingClassifier"
    # the gap is positive on average and in nearly every repeat; the
    # 2.5 percentile of twenty repeats is one repeat, too thin to test
    assert r.gap > 0.01, r
    assert (r.per_replicate["gap"] > 0).mean() >= 0.85, r
    assert r.to_dict()["gap"] == r.gap


def test_independent_labels_no_inflation():
    r = record_wise_validation(synth(seed=3, per_unit_labels=False), FEATS, **KW)
    print(r)
    assert r.lo <= 0 <= r.hi, r


def test_calendar_split_runs():
    r = record_wise_validation(synth(seed=4), FEATS, time_split="calendar",
                      **{**KW, "repeats": 4})
    assert r.info["time_split"] == "calendar" and r.repeats == 4
