"""Forecast horizon, chronological control, clock baseline, one call."""
import numpy as np
import pandas as pd
import pytest

import whowhen as ww
from whowhen.overlap import forecast_labels, overlap_masks


def event_panel(n_units=300, n_t=40, seed=0, clock_driven=True):
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
    return ww.Panel(df[t <= onset[u]], "unit", "time", event="event")


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
    r = ww.clock_baseline(event_panel(), ["level", "noise"], horizon=12,
                          model=fast_scorer, repeats=6, n_jobs=2)
    t = r.table.set_index("input")["auc"]
    assert t["clock"] > 0.65
    assert t["features + clock"] > t["clock"]
    assert r.gain_lo > 0


def test_report_runs_everything_it_can(fast_scorer):
    r = ww.report(event_panel(), horizon=12, scorer=fast_scorer,
                  repeats=2, n_pairs=1500, n_jobs=2)
    assert r.info["features"] == ["level", "noise"]
    assert r.info["offset_chosen"]
    assert not r.skipped, r.skipped
    for part in ("audit", "columns", "overlap", "chronological", "clock"):
        assert getattr(r, part) is not None
    assert "clock baseline" in repr(r) and "identity audit" in repr(r)
    assert set(r.to_dict()) >= {"audit", "columns", "overlap", "clock"}


def test_report_without_event_says_what_it_skipped(fast_scorer):
    from whowhen import demo
    p = ww.Panel(demo.synthesize(n_units=150, n_days=60), "unit", "time")
    r = ww.report(p, scorer=fast_scorer, repeats=2, n_pairs=800,
                  skip=("columns",), n_jobs=2)
    assert r.audit is not None and r.columns is None
    assert set(r.skipped) == {"overlap", "chronological", "clock"}
    assert "no event column" in repr(r)


def test_random_run_records_its_seed_and_can_be_repeated(fast_scorer):
    from whowhen import demo
    p = ww.Panel(demo.synthesize(n_units=120, n_days=50), "unit", "time")
    r = ww.audit(p, demo.SIG, offset=("10D", "30D"), which="timing",
                 n_pairs=300, repeats=1, bootstrap=20, random=True, n_jobs=1,
                 scorer=fast_scorer)
    assert isinstance(r.info["seed"], int) and r.info["random"]
    again = ww.audit(p, demo.SIG, offset=("10D", "30D"), which="timing",
                     n_pairs=300, repeats=1, bootstrap=20, random=True,
                     seed=r.info["seed"], n_jobs=1, scorer=fast_scorer)
    assert again.timing.equals(r.timing)
