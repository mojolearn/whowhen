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
    keep, lab, unfollowed = forecast_labels(unit, time, y, 2.0)
    # unit 1 has no event and is last seen at time 2, so only its row at
    # time 0 was followed for the whole horizon
    assert keep.tolist() == [1, 1, 1, 0, 1, 0, 0]
    assert lab.tolist() == [0, 1, 1, 0]
    assert unfollowed == 2
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
    r = ww.report(event_table(clock_driven=False), unit="unit", time="time",
                  event="event", horizon=12, repeats=2, n_pairs=1500,
                  importance=True, _scorer=fast_scorer)
    assert r.info["features"] == ["level", "noise"]
    assert r.info["offset"] and not r.skipped, r.skipped
    for part in ("audit", "gaps", "columns", "overlap", "chronological",
                 "time_cut", "clock"):
        assert getattr(r, part) is not None
    assert set(r.gaps["audit"]) == {"identity", "timing"}
    assert len(r.gaps) == 4 and r.gaps["gap"].str.startswith("short").sum() == 2
    assert set(r.audit.identity["impostors"]) == {"random", "nearest"}
    assert r.audit.placebo is not None and "timing placebo" in repr(r)
    assert "clock baseline" in repr(r) and "identity audit" in repr(r)
    assert set(r.to_dict()) >= {"audit", "gaps", "columns", "overlap",
                                "time_cut", "clock"}


def test_report_without_event(fast_scorer):
    df = event_table().drop(columns="event")
    r = ww.report(df, unit="unit", time="time", repeats=1, n_pairs=800,
                  _scorer=fast_scorer)
    assert r.audit is not None and r.columns is None
    assert set(r.skipped) == {"columns", "overlap", "chronological",
                              "time_cut", "clock"}
    assert "off by default" in r.skipped["columns"]
    assert "no event column" in repr(r)


def test_report_on_chosen_columns(fast_scorer):
    df = event_table().drop(columns="event")
    r = ww.report(df, unit="unit", time="time", features=["level"],
                  importance=True, repeats=1, n_pairs=600,
                  _scorer=fast_scorer)
    assert r.info["features"] == ["level"]
    assert list(r.columns["identity"]["column"]) == ["level"]
    # a list ranks only the named columns
    r = ww.report(df, unit="unit", time="time", importance=["noise"],
                  repeats=1, n_pairs=600, _scorer=fast_scorer)
    t = r.columns["identity"]
    assert list(t["column"]) == ["noise"]
    with pytest.raises(ValueError, match="not checked"):
        ww.report(df, unit="unit", time="time", importance=["nope"])
    r = ww.report(df, unit="unit", time="time", repeats=1, n_pairs=600,
                  features={"a": ["level"], "b": ["level", "noise"]},
                  _scorer=fast_scorer)
    assert set(r.audit.timing["features"]) == {"a", "b"}
    assert r.info["features"] == ["level", "noise"]


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


def ageing_table(shared_start, n=300, m=30, seed=0):
    """Each unit has a column that counts its own rows. With
    ``shared_start`` every unit starts at time 0, so that count is the
    same for all units at a given time."""
    rng = np.random.default_rng(seed)
    start = np.zeros(n, int) if shared_start else rng.integers(0, 200, n)
    u = np.repeat(np.arange(n), m)
    age = np.tile(np.arange(m), n)
    return pd.DataFrame({"unit": u, "time": start[u] + age,
                         "age": age + rng.normal(0, 0.5, len(u))})


def test_placebo_separates_own_change_from_shared_change(fast_scorer):
    def run(shared):
        r = audit(Panel(ageing_table(shared), "unit", "time"), ["age"],
                  (1, 10), which=("timing", "placebo"), repeats=2,
                  n_pairs=3000, scorer=fast_scorer)
        return r.timing["auc"][0], r.placebo["auc"][0]
    timing, placebo = run(shared=False)
    assert timing > 0.58 and placebo < 0.56
    timing, placebo = run(shared=True)
    assert timing > 0.58 and placebo > 0.58


def test_whole_report_is_the_same_every_time(fast_scorer):
    """Every part of the report, run twice, gives identical numbers."""
    def run():
        return ww.report(event_table(n_units=200, clock_driven=False),
                         unit="unit", time="time", event="event", horizon=12,
                         repeats=2, n_pairs=800, importance=True,
                         _scorer=fast_scorer).to_dict()
    a, b = run(), run()
    assert not a["skipped"], a["skipped"]
    assert set(a) >= {"audit", "gaps", "columns", "overlap", "chronological",
                      "time_cut", "clock"}
    assert "placebo" in a["audit"]
    assert a == b
    assert set(a["headline"]) == {
        "identity", "identity_nearest", "timing", "placebo", "overlap_gap",
        "chronological_gap", "time_cut_gap", "clock", "clock_gain"}


def test_fail_if_conditions():
    from whowhen.cli import check, parse_condition
    assert parse_condition("overlap_gap > 0.02") == ("overlap_gap", ">", 0.02)
    assert parse_condition("clock_gain<=-0.5") == ("clock_gain", "<=", -0.5)
    for bad in ("overlap_gap", "nonsense > 1", "identity == 0.5"):
        with pytest.raises(ValueError):
            parse_condition(bad)
    numbers = {"overlap_gap": 0.031, "identity": 0.97}
    held, missing = check(["overlap_gap > 0.02", "identity > 0.99",
                           "clock_gain < 0.01"], numbers)
    assert len(held) == 1 and held[0].startswith("overlap_gap is 0.031")
    assert missing == ["clock_gain: this check did not run"]


def test_report_names_the_software_that_ran(fast_scorer):
    from whowhen.scorers import _doi
    df = event_table().drop(columns="event")
    r = ww.report(df, unit="unit", time="time", repeats=1, n_pairs=600,
                  _scorer=fast_scorer)
    assert r.to_dict()["info"]["whowhen"] == {
        "version": ww.__version__, "doi": ww.__doi__}
    assert f"whowhen {ww.__version__} (doi:{ww.__doi__})" in repr(r)
    assert _doi("a-package-that-is-not-installed") == {}


def test_default_scorer_is_named_with_its_version(default_scorer_runs):
    from whowhen.scorers import resolve
    sc = resolve(None)
    assert sc.params["version"]
    if sc.name.startswith("mojolearn"):
        assert sc.params["doi"].startswith("10.")


@pytest.mark.parametrize("split", ["calendar", "chronological"])
def test_time_splits_train_only_on_settled_labels(split):
    unit = np.repeat(np.arange(60), 20)
    time = np.tile(np.arange(20.0), 60)
    rng = np.random.default_rng(0)
    tu = rng.random(60) < 0.3
    te, leak, base = overlap_masks(unit, time, rng, tu, split, settle=4.0)
    loose = overlap_masks(unit, time, np.random.default_rng(0), tu, split)
    assert te.tolist() == loose[0].tolist()          # the test rows stay
    assert leak.any() and leak.sum() < loose[1].sum()
    first_test = np.full(60, np.inf)
    np.minimum.at(first_test, unit[te], time[te])
    # every earlier row of a test unit that is trained on has its label
    # known before that unit's first test row
    assert (time[leak] + 4.0 < first_test[unit[leak]]).all()
    if split == "calendar":
        assert (time[base] + 4.0 < time[te].min()).all()


def recognisable_units(n_units=400, n_t=40, seed=0):
    """Units a model can recognise from four columns that say nothing
    about the event. Half the units have an event, early in their
    record; the rest are observed to the end."""
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_units), n_t)
    t = np.tile(np.arange(n_t), n_units)
    onset = np.where(rng.random(n_units) < 0.5,
                     rng.integers(6, 12, n_units), 10 ** 6)
    sig = rng.normal(0, 3.0, (n_units, 4))
    df = pd.DataFrame({"unit": u, "time": t,
                       "event": (t == onset[u]).astype(int)})
    for j in range(4):
        df[f"s{j}"] = sig[u, j] + rng.normal(0, 0.3, len(u))
    return df[t <= onset[u]]


def test_unsettled_labels_would_inflate_the_chronological_gap():
    """A training row whose horizon runs past the cut carries the
    answer for the test rows of the same unit. Planted here: units are
    recognisable and the columns say nothing else about the event."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from whowhen.overlap import _replicate
    from whowhen.pairs import unit_groups
    from whowhen.scorers import resolve
    cols = ["s0", "s1", "s2", "s3"]
    f = Panel(recognisable_units(), "unit", "time", event="event").frame(
        cols, keep_event_units=True)
    keep, y, _ = forecast_labels(f.unit, f.time, f.y, 6.0)
    X, unit, time = f.X[keep], f.unit[keep], f.time[keep]
    sc = resolve(HistGradientBoostingClassifier(max_iter=80))
    groups = unit_groups(f, 5)

    def gap(settle):
        runs = [_replicate(X, y, unit, time, b, 0, groups, False,
                           "chronological", settle, sc, 1) for b in range(4)]
        return float(np.mean([r["gap"] for r in runs]))
    unsettled, settled = gap(0.0), gap(6.0)
    print(unsettled, settled)
    # the settled gap is not zero: a unit seen to survive its early rows
    # is known, legitimately, to have had no event by then
    assert unsettled > 0.2
    assert settled < unsettled - 0.1


def test_each_named_set_gets_its_own_event_checks(fast_scorer):
    r = ww.report(event_table(clock_driven=False), unit="unit", time="time",
                  event="event",
                  features={"signal": ["level"], "junk": ["noise"]},
                  horizon=12, repeats=2, n_pairs=800, _scorer=fast_scorer)
    assert set(r.skipped) == {"columns"}, r.skipped
    for part in ("overlap", "chronological", "time_cut", "clock"):
        got = getattr(r, part)
        assert list(got) == ["signal", "junk"]
        assert got["signal"].info["features"] == ["level"]
        assert got["junk"].info["features"] == ["noise"]
        assert set(r.to_dict()[part]) == {"signal", "junk"}
    assert "columns: signal" in repr(r) and "columns: junk" in repr(r)
    gain = {k: v.gain for k, v in r.clock.items()}
    assert gain["signal"] > gain["junk"]
    assert r.headline()["clock_gain"] == min(gain.values())
    assert r.overlap["signal"].info["rows_without_follow_up"] > 0
    assert "rows left out" in repr(r)
    assert "only if their horizon has passed" in repr(r)


def test_time_cut_says_when_no_label_is_known_by_the_cut(fast_scorer):
    # every event here comes late, so no row before the cut has a
    # positive label that was already known at the cut
    r = ww.report(event_table(), unit="unit", time="time", event="event",
                  features=["level"], horizon=12, repeats=2, n_pairs=600,
                  _scorer=fast_scorer)
    assert "known by the cut" in r.skipped["time_cut"]


def untimed_table(n_units=300, rows=12, seed=0):
    """Rows with no order: ``trait`` is a unit's own level seen noisily,
    ``noise`` is unrelated to the unit. The event follows the trait."""
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(n_units), rows)
    trait = rng.normal(0, 1, n_units)
    df = pd.DataFrame({"unit": u,
                       "trait": trait[u] + rng.normal(0, 0.3, len(u)),
                       "noise": rng.normal(0, 1, len(u)),
                       "event": (rng.random(len(u))
                                 < 1 / (1 + np.exp(-2 * trait[u]))).astype(int)})
    return df.sample(frac=1.0, random_state=seed)


def test_pairs_without_time_respect_unit_and_split():
    from whowhen import pairs
    panel = Panel(untimed_table().drop(columns="event"), unit="unit")
    f = panel.frame(["trait", "noise"])
    assert f.time is None
    train = pairs.split_units(f, 0, 4)
    for mode in ("random", "nearest"):
        ra, rb, lab, tr = pairs.identity_pairs(
            f, 0.0, 0.0, 800, 2, np.random.default_rng(0), train, mode)
        assert (ra != rb).all() and (lab == 1).sum() and (lab == 0).sum()
        assert ((f.unit[ra] == f.unit[rb]) == (lab == 1)).all()
        assert (train[f.unit[ra]] == tr).all()
        assert (train[f.unit[rb]] == tr).all()


def test_report_without_time(fast_scorer):
    df = untimed_table()
    r = ww.report(df.drop(columns="event"), unit="unit",
                  features={"trait": ["trait"], "noise": ["noise"]},
                  importance=True, repeats=2, n_pairs=3000, _scorer=fast_scorer)
    t = r.audit.identity
    auc = lambda name, mode="random": float(
        t[(t["features"] == name) & (t["impostors"] == mode)]["auc"].iloc[0])
    assert auc("trait") > 0.75
    # against units with nearly the same trait, the trait says little
    assert auc("trait", "nearest") < auc("trait") - 0.15
    assert abs(auc("noise") - 0.5) < 0.06
    assert abs(auc("noise", "nearest") - 0.5) < 0.06
    assert r.audit.timing is None and r.audit.placebo is None
    assert list(r.columns) == ["identity"]
    assert r.info["offset"] is None and r.info["time"] is None
    assert {k for k, v in r.skipped.items()
            if v.startswith("no time column")} == {
        "gaps", "chronological", "time_cut", "clock"}
    assert "no time column: a pair is" in repr(r)
    assert set(r.headline()) == {"identity", "identity_nearest"}
    # with an event column the overlap check runs too
    r = ww.report(df, unit="unit", event="event", features=["trait"],
                  repeats=2, n_pairs=1500, _scorer=fast_scorer)
    assert r.overlap is not None and "overlap_gap" in r.headline()
    assert r.chronological is None and r.clock is None


def test_horizon_without_time_is_an_error():
    with pytest.raises(ValueError, match="horizon needs a time column"):
        ww.report(untimed_table(), unit="unit", event="event", horizon=3)
