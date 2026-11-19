"""rank_columns puts the planted columns on top of their own audit."""
import numpy as np
import pytest

import whowhen as ww
from whowhen import demo
from whowhen.audit import draw
from whowhen.columns import rank_columns

KW = dict(n_pairs=1500, max_pairs_per_unit=4, repeats=2, n_jobs=2)
COLS = demo.CLOCKS + demo.SIG + demo.NOISE


@pytest.fixture(scope="module")
def panel():
    return ww.Panel(demo.synthesize(n_units=300, n_days=80),
                    unit="unit", time="time")


def test_planted_columns_rank_first(panel, fast_scorer):
    r = rank_columns(panel, COLS, offset=("30D", "70D"), scorer=fast_scorer,
                     **KW)
    ident, timing = r["identity"], r["timing"]
    for t in (ident, timing):
        assert list(t.columns) == ["column", "alone", "drop", "all"]
        assert sorted(t["column"]) == sorted(COLS)
        assert (np.diff(t["alone"]) <= 0).all()
        assert t["all"].nunique() == 1
        assert t.attrs["scorer"]["name"] == \
            "sklearn.HistGradientBoostingClassifier"
    assert set(ident["column"][:3]) == set(demo.SIG), ident
    assert timing["column"][0] == "clock", timing
    for t in (ident, timing):
        drops = t.set_index("column")["drop"]
        assert (drops[demo.NOISE].abs() < 0.03).all(), t


def test_draw_is_shared_and_matches_audit_seed(panel):
    f = panel.frame(COLS)
    lo, hi = panel.offset("30D"), panel.offset("70D")
    a = draw(f, "identity", "random", 1, 0, lo, hi, 500, 2)
    b = draw(f, "identity", "random", 1, 0, lo, hi, 500, 2)
    assert (a.rows_a == b.rows_a).all() and (a.rows_b == b.rows_b).all()
    assert (a.tr == b.tr).all()
    assert a.features(f, [0, 1]).shape == (len(a.y), 6)
    t = draw(f, "timing", "random", 0, 0, lo, hi, 500, 2)
    assert t.rows_b is None and t.features(f, [0]).shape == (len(t.y), 1)


def test_single_column_and_bad_audit(panel, fast_scorer):
    r = rank_columns(panel, ["sig0"], offset=("30D", "70D"),
                     which="identity", scorer=fast_scorer, **KW)
    t = r["identity"]
    assert list(r) == ["identity"]
    assert t.loc[0, "alone"] == t.loc[0, "all"]
    assert t.loc[0, "drop"] == pytest.approx(t.loc[0, "all"] - 0.5)
    with pytest.raises(ValueError):
        rank_columns(panel, ["sig0"], offset=("30D", "70D"), which="when",
                     scorer=fast_scorer, **KW)
