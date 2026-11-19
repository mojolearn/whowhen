"""Forecast horizon, chronological control, clock baseline, one call."""
import numpy as np
import pandas as pd
import pytest

import whowhen as ww
from whowhen.audit import audit
from whowhen.clock import clock_baseline
from whowhen.panel import Panel
from whowhen.overlap import forecast_labels, overlap_masks


def event_table(n_units=300, n_t=40, seed=0, clock_driven=True):
    """Events come late in a unit's life (so a clock predicts them) and
    more often for units with a high level, which ``level`` shows noisily
    in the 12 steps before the event."""
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_units), n_t)
    t = np.tile(np.arange(n_t), n_units)
    onset = np.where(rng.random(n_units) < 0.5,
                     rng.integers(25, n_t, n_units) if clock_driven
                     else rng.integers(5, n_t, n_units), 10 ** 6)
    near = (onset[u] - t > 0) & (onset[u] - t <= 12)
    df = pd.DataFrame({"unit": u, "time": t,
                       "level": near * 1.5 + rng.normal(0, 1.0, len(u)),
                       "noise": rng.normal(0, 1, len(u)),
                       "event": (t == onset[u]).astype(int)})
    return df[t <= onset[u]]


def test_forecast_labels():
    unit = np.array([0, 0, 0, 0, 1, 1, 1])
    time = np.array([0., 1, 2, 3, 0, 1, 2])
    y = np.array([0, 0, 0, 1, 0, 0, 0])
    keep, lab = forecast_labels(unit, time, y, 2.0)
    assert keep.tolist() == [1, 1, 1, 0, 1, 1, 1]
    assert lab.tolist() == [0, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="not when"):
        forecast_labels(unit, time, np.array([1, 1, 1, 1, 0, 0, 0]), 2.0)


def test_chronological_masks():
    unit = np.repeat(np.arange(50), 10)
    time = np.tile(np.arange(10.0), 50)
    rng = np.random.default_rng(0)
    te, leak, base = overlap_masks(unit, time, rng, rng.random(50) < 0.2,
                                   time_split="chronological")
    assert te.any() and not (te & leak).any() and not (te & base).any()
    assert set(unit[te]) == set(unit[leak])
    assert not set(unit[base]) & set(unit[te])
    assert time[te].min() > time[leak].max()


def test_clock_baseline_sees_the_clock(fast_scorer):
    p = Panel(event_table(), "unit", "time", event="event")
    r = clock_baseline(p, ["level", "noise"], horizon=12, model=fast_scorer,
                       repeats=6)
    t = r.table.set_index("input")["auc"]
    assert t["clock"] > 0.65
    assert t["features + clock"] > t["clock"]
    assert r.gain > 0


def test_report_runs_everything(fast_scorer):
    r = ww.report(event_table(), unit="unit", time="time", event="event",
                  horizon=12, repeats=2, n_pairs=1500, _scorer=fast_scorer)
    assert r.info["features"] == ["level", "noise"]
    assert r.info["offset"] and not r.skipped, r.skipped
    for part in ("audit", "columns", "overlap", "chronological", "clock"):
        assert getattr(r, part) is not None
    assert set(r.audit.identity["impostors"]) == {"random", "nearest"}
    assert "clock baseline" in repr(r) and "identity audit" in repr(r)
    assert set(r.to_dict()) >= {"audit", "columns", "overlap", "clock"}


def test_report_without_event(fast_scorer):
    df = event_table().drop(columns="event")
    r = ww.report(df, unit="unit", time="time", repeats=1, n_pairs=800,
                  _scorer=fast_scorer)
    assert r.audit is not None and r.columns is not None
    assert set(r.skipped) == {"overlap", "chronological", "clock"}
    assert "no event column" in repr(r)


def test_report_without_time(fast_scorer):
    df = event_table().drop(columns="time")
    r = ww.report(df, unit="unit", event="event", repeats=2,
                  _scorer=fast_scorer)
    assert r.overlap is not None
    assert set(r.skipped) == {"audit", "columns", "chronological", "clock"}
    with pytest.raises(ValueError, match="nothing to check"):
        ww.report(df.drop(columns="event"), unit="unit")


def test_random_run_records_its_seed(fast_scorer):
    df = event_table(n_units=120).drop(columns="event")
    a = ww.report(df, unit="unit", time="time", repeats=1, n_pairs=300,
                  random=True, _scorer=fast_scorer)
    assert a.info["random"] and isinstance(a.info["seed"], int)
    assert f"seed {a.info['seed']}" in repr(a)


def test_only_report_is_public():
    assert ww.__all__ == ["report", "Report", "__version__"]


def test_catboost_fallback_runs():
    pytest.importorskip("catboost")
    from whowhen.scorers import fit_score, resolve
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.2] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    sc = resolve("catboost")
    s = fit_score(sc, 0, 1, X[:400], y[:400], X[400:])
    assert sc.name == "catboost.CatBoostClassifier"
    assert ((s > 0.5) == (y[400:] == 1)).mean() > 0.7
