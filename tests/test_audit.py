"""Each audit reads its own planted signal and nothing else."""
import numpy as np
import pandas as pd
import pytest

from whowhen.audit import audit
from whowhen.panel import Panel
from whowhen import demo, pairs
from whowhen._stats import bootstrap_auc
from whowhen.scorers import resolve

KW = dict(n_pairs=2000, max_pairs_per_unit=4, repeats=2, bootstrap=200,
          n_jobs=2)


@pytest.fixture(scope="module")
def panel():
    return Panel(demo.synthesize(n_units=400, n_days=100),
                    unit="unit", time="time")


def test_each_audit_reads_only_its_own_signal(panel, fast_scorer):
    r = audit(panel, demo.SETS, offset=("30D", "90D"),
                 scorer=fast_scorer, **KW)
    assert all(ok for *_, ok in demo.verdicts(r)), r
    assert r.scorer["name"] == "sklearn.HistGradientBoostingClassifier"
    assert r.scorer["params"]["max_iter"] == 60


def test_default_scorer_is_recorded(panel, default_scorer_runs):
    r = audit(panel, {"signature": demo.SIG, "noise": demo.NOISE},
                 offset=("30D", "90D"), which="identity", n_pairs=1500,
                 max_pairs_per_unit=4, repeats=1, bootstrap=100, n_jobs=1)
    assert r.scorer == resolve(None).describe()
    t = r.identity.set_index("features")
    assert t.loc["signature", "auc"] > 0.6
    assert abs(t.loc["noise", "auc"] - 0.5) < 0.06
    assert r.timing is None


def test_scorer_factory_and_names(panel):
    from sklearn.linear_model import LogisticRegression
    sc = resolve(lambda seed: LogisticRegression(random_state=seed))
    assert sc.name == "sklearn.LogisticRegression"
    with pytest.raises(ValueError):
        resolve("xgboost")
    with pytest.raises(TypeError):
        resolve(object())


def test_pairs_respect_unit_offset_and_split(panel):
    f = panel.frame(demo.SIG)
    lo, hi = panel.offset("30D"), panel.offset("90D")
    rng = np.random.default_rng(0)
    rows, y = pairs.timing_pairs(f, lo, hi, 500, 2, rng)
    early, late = rows[y == 0], rows[y == 1]
    assert len(early) and len(late)
    train = pairs.split_units(f, 0, 4)
    ra, rb, lab, tr = pairs.identity_pairs(f, lo, hi, 800, 2, rng, train)
    gap = f.time[rb] - f.time[ra]
    assert ((gap >= lo) & (gap <= hi)).all()
    same = f.unit[ra] == f.unit[rb]
    assert (same == (lab == 1)).all()
    # both units of every pair sit on the pair's side of the split
    assert (train[f.unit[ra]] == tr).all() and (train[f.unit[rb]] == tr).all()
    # a different-unit pair shares its anchor and both times with a
    # same-unit pair
    key = lambda m: set(zip(ra[m], f.time[rb[m]]))
    assert key(lab == 0) <= key(lab == 1)


def test_placebo_pairs_cross_units(panel):
    f = panel.frame(demo.CLOCKS)
    rng = np.random.default_rng(1)
    rows, y = pairs.timing_pairs(f, panel.offset("30D"), panel.offset("90D"),
                                 300, 1, rng, placebo=True)
    assert len(rows) and set(np.unique(y)) == {0, 1}


def test_event_units_are_left_out():
    df = demo.synthesize(n_units=60, n_days=40)
    df["event"] = ((df["unit"] < "u0020") & (df["time"] == df["time"].max())
                   ).astype(int)
    p = Panel(df, unit="unit", time="time", event="event")
    f = p.frame(demo.SIG)
    assert f.n_units == 40 and f.n_units_all == 60
    assert not set(f.unit_labels) & {f"u{i:04d}" for i in range(20)}


def test_numeric_time_and_offset_errors():
    df = demo.synthesize(n_units=30, n_days=20)
    with pytest.raises(TypeError):
        Panel(df, "unit", "time").offset(5)
    df["time"] = df.groupby("unit").cumcount()
    assert Panel(df, "unit", "time").offset(5) == 5.0


def test_bootstrap_matches_direct_resampling():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n_units, n = 40, 400
    unit = rng.integers(0, n_units, n)
    y = rng.integers(0, 2, n)
    s = np.round(rng.normal(size=n) + 0.5 * y, 1)      # ties on purpose
    got = bootstrap_auc(y, s, [unit], n_units, 50, np.random.default_rng(5))
    brng = np.random.default_rng(5)
    want = []
    for _ in range(50):
        m = np.bincount(brng.integers(0, n_units, n_units), minlength=n_units)
        want.append(roc_auc_score(y, s, sample_weight=m[unit]))
    assert np.allclose(got, want)


def test_carry_forward_raises_identity(fast_scorer):
    """A carried-forward column is easier to re-identify than the raw one."""
    rng = np.random.default_rng(3)
    n_units, n_t = 300, 60
    u = np.repeat(np.arange(n_units), n_t)
    df = pd.DataFrame({"unit": u, "time": np.tile(np.arange(n_t), n_units)})
    level = rng.normal(0, 1, n_units)[u]
    raw = level + rng.normal(0, 3, len(df))
    raw[rng.random(len(df)) < 0.9] = np.nan
    df["raw"] = raw
    df["locf"] = df.groupby("unit")["raw"].ffill()
    r = audit(Panel(df, "unit", "time"), {"raw": ["raw"], "locf": ["locf"]},
              offset=(1, 6), which="identity", scorer=fast_scorer, **KW)
    t = r.identity.set_index("features")["auc"]
    assert t["locf"] > t["raw"] + 0.05


def test_nearest_impostors_separate_type_from_individual(fast_scorer):
    """Planted types, no individual signal: random impostors read the
    type, nearest impostors read nothing. Add an individual signal and
    nearest impostors read it."""
    rng = np.random.default_rng(4)
    n_units, n_t = 400, 60
    u = np.repeat(np.arange(n_units), n_t)
    n = len(u)
    df = pd.DataFrame({"unit": u, "time": np.tile(np.arange(n_t), n_units)})
    kind = rng.integers(0, 8, n_units)
    for j in range(3):
        df[f"type{j}"] = (rng.normal(0, 3, 8)[kind])[u] + rng.normal(0, .3, n)
    df["self"] = rng.normal(0, 1, n_units)[u] + rng.normal(0, .3, n)
    p = Panel(df, "unit", "time")
    types = [f"type{j}" for j in range(3)]
    r = audit(p, {"types": types, "types_and_self": types + ["self"]},
                 offset=(5, 30), which="identity",
                 impostors=("random", "nearest"), scorer=fast_scorer, **KW)
    t = r.identity.set_index(["features", "impostors"])
    assert t.loc[("types", "random"), "auc"] > 0.75
    assert abs(t.loc[("types", "nearest"), "auc"] - 0.5) < 0.06
    assert t.loc[("types_and_self", "nearest"), "lo"] > 0.5


def test_folds_are_fixed_and_cover_every_unit(panel):
    f = panel.frame(demo.SIG)
    for n, share in ((1, 1 / 2), (2, 1 / 3), (4, 1 / 5)):
        held = [~pairs.split_units(f, k, n) for k in range(n)]
        assert (np.sum(held, axis=0) <= 1).all()       # never held out twice
        assert all(abs(h.mean() - share) < 0.08 for h in held)
    shuffled = Panel(panel.df.sample(frac=1, random_state=1), "unit",
                        "time").frame(demo.SIG)
    a = dict(zip(f.unit_labels, pairs.unit_groups(f, 5)))
    b = dict(zip(shuffled.unit_labels, pairs.unit_groups(shuffled, 5)))
    assert a == b


def test_same_call_gives_the_same_numbers(panel, fast_scorer):
    kw = dict(offset=("30D", "90D"), which="identity", n_pairs=600,
              max_pairs_per_unit=2, repeats=2, bootstrap=50, n_jobs=2,
              scorer=fast_scorer)
    a = audit(panel, demo.SIG, **kw).identity
    b = audit(panel, demo.SIG, **kw).identity
    assert a.equals(b)
