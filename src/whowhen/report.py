"""Everything in one call.

    print(ww.report(df, unit="patient_id", time="hour", event="sepsis"))

runs every check the panel supports and says which it skipped and why.
Leave ``time`` out for a table whose rows are not ordered in time:

    print(ww.report(df, unit="patient_id", event="tumour"))
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .audit import AuditResult, _feature_sets, audit, pick_seed
from .clock import ClockResult, clock_baseline
from .columns import rank_columns
from .overlap import OverlapResult, overlap_check
from .panel import Panel
from .scorers import resolve

PARTS = ("pair_tests", "by_gap", "importance", "record_wise", "temporal_split",
         "out_of_time", "time_only")


TITLES = {"reidentification": "re-identification test",
          "time_order": "time-order test"}


@dataclass
class Report:
    """Results of ``report``. A part that did not run is None and its
    reason is in ``skipped``. The four checks that need an event column
    (record_wise, temporal_split, out_of_time, time_only) hold one result per
    feature set, keyed by the set's name ("all" when the columns were
    not given as named sets)."""
    scorer: dict
    info: dict
    pair_tests: AuditResult | None = None
    by_gap: pd.DataFrame | None = None
    importance: dict | None = None
    record_wise: dict[str, OverlapResult] | None = None
    temporal_split: dict[str, OverlapResult] | None = None
    out_of_time: dict[str, OverlapResult] | None = None
    time_only: dict[str, ClockResult] | None = None
    skipped: dict = field(default_factory=dict)

    def headline(self) -> dict:
        """One number per check, for scripts. With several feature sets
        each number is the one furthest from a clean result: the highest
        AUC, the highest gap, and the lowest gain over time alone. A check that
        did not run has no entry."""
        h = {}
        if self.pair_tests is not None:
            t = self.pair_tests.reidentification
            if t is not None:
                for mode, name in (("random", "reidentification"),
                                   ("nearest_neighbour", "reidentification_nearest_neighbour")):
                    sub = t[t["controls"] == mode]
                    if len(sub):
                        h[name] = float(sub["auc"].max())
            for kind in ("time_order", "placebo"):
                t = getattr(self.pair_tests, kind)
                if t is not None:
                    h[kind] = float(t["auc"].max())
        for part, name in (("record_wise", "record_wise_gap"),
                           ("temporal_split", "temporal_split_gap"),
                           ("out_of_time", "out_of_time_gap")):
            r = getattr(self, part)
            if r:
                h[name] = max(float(x.gap) for x in r.values())
        if self.time_only:
            h["time_only"] = max(
                float(x.table.set_index("input")["auc"]["time"])
                for x in self.time_only.values())
            h["time_only_gain"] = min(float(x.gain) for x in self.time_only.values())
        return h

    def to_dict(self) -> dict:
        d = {"scorer": self.scorer, "info": self.info,
             "skipped": self.skipped, "headline": self.headline()}
        for part in ("pair_tests", "record_wise", "temporal_split", "out_of_time",
                     "time_only"):
            r = getattr(self, part)
            if r is not None:
                d[part] = (r.to_dict() if part == "pair_tests" else
                           {k: x.to_dict() for k, x in r.items()})
        if self.by_gap is not None:
            d["by_gap"] = self.by_gap.to_dict(orient="records")
        if self.importance is not None:
            d["importance"] = {k: t.to_dict(orient="records")
                            for k, t in self.importance.items()}
        return d

    def __repr__(self) -> str:
        i = self.info
        out = [f"whowhen report: {i['n_units']:,} units, "
               f"{len(i['features'])} columns checked, {i['repeats']} "
               f"repeats{' (random, seed ' + str(i['seed']) + ')' if i['random'] else ''}"]
        if i["offset"]:
            out.append(f"rows of a pair are {i['offset'][0]} to "
                       f"{i['offset'][1]} apart (chosen from the data)")
        if i.get("time", "") is None:
            out.append("no time column: a pair is two rows of one unit, or "
                       "one row each from two units")
        w = i.get("whowhen")
        if w:
            out.append(f"whowhen {w['version']} (doi:{w['doi']})")
        p = self.scorer.get("params", {})
        out.append(f"scorer: {self.scorer['name']}"
                   + (f" {p['version']}" if "version" in p else "")
                   + (f" (doi:{p['doi']})" if "doi" in p else ""))
        if self.pair_tests is not None:
            for kind, title in TITLES.items():
                t = getattr(self.pair_tests, kind)
                if t is not None:
                    out += ["", f"{title} (AUC; 0.5 = no signal)",
                            t.round(3).to_string(index=False)]
            t = self.pair_tests.placebo
            if t is not None:
                out += ["", "placebo test: the time-order test with the "
                        "later row taken from a different unit",
                        t.round(3).to_string(index=False),
                        "near 0.5: the time-order signal is each unit's "
                        "own. near the time-order test: it is change shared "
                        "by all units."]
        if self.by_gap is not None:
            out += ["", "results by time gap between the two rows",
                    self.by_gap.round(3).to_string(index=False)]
        if self.importance is not None:
            for kind, t in self.importance.items():
                out += ["", f"column importance, {TITLES[kind]}",
                        t.head(10).round(3).to_string(index=False)]
        for part in ("record_wise", "temporal_split", "out_of_time", "time_only"):
            r = getattr(self, part)
            for name, x in (r or {}).items():
                out += [""] + ([f"columns: {name}"] if len(r) > 1 else [])
                out.append(repr(x))
        if self.skipped:
            out += ["", "not run:"]
            out += [f"  {k}: {v}" for k, v in self.skipped.items()]
        return "\n".join(out)


NO_TIME = "no time column was given"
NEEDS_TIME = ("by_gap", "temporal_split", "out_of_time", "time_only")


def report(df: pd.DataFrame, unit: str, time: str | None = None,
           event: str | None = None, *, features=None, importance=False,
           horizon=None,
           repeats: int = 4,
           random: bool = False, n_pairs: int = 20000,
           _scorer=None) -> Report:
    """Run every check the table supports and return them together.

    Parameters
    ----------
    df : one row per observation.
    unit : the column naming the unit (patient, machine, loan, vessel).
    time : optional column giving when the row was observed. Leave it
        out for a table whose rows are not ordered in time. The
        re-identification test, the column importance and record-wise
        vs subject-wise validation still run; the time-order test and
        the checks that split by time do not.
    event : optional column, nonzero where the event happens. Without
        it record-wise vs subject-wise validation, the temporal split,
        out-of-time validation and the time-only baseline are left out.
    features : optional. The columns to check: a list, or a dict of
        named lists to check side by side (for example
        {"raw": [...], "filled": [...]}, or with and without a time
        column). The pair tests score each named set on the same pairs, and
        each check that needs an event column runs once per set.
        Default: every numeric column other than unit, time and event.
    importance : False by default. True ranks every checked column by
        single-column AUC and drop-column importance; a list ranks only
        those columns. Each ranked column costs two model fits per test
        and repeat, so this is the slowest part on a wide table.
    horizon : optional. With it the event checks forecast: "does the
        event come within this long after the row?", in the time
        column's scale (a string such as '12h' for datetimes). The
        event column must then flag the row where the event happens. A
        unit's last row is taken as the end of its follow-up: a row of a
        unit with no event is used only if the unit was still observed
        one horizon later, and a time split trains only on rows whose
        horizon has passed by the cut.
    repeats : one number, N: every check runs N times. Units fall into
        N + 1 fixed groups and each run holds out a different group: 1
        holds out a half, 2 a third each, 4 (the default) a fifth each.
    random : False by default, so the same call gives the same numbers
        every time. True draws held-out units and pairs at random,
        differently on each run, and records the seed it used.
    n_pairs : the most row pairs each pair-test run is trained and
        tested on. Smaller tables use as many as they have.

    The gap between the two rows of a pair is chosen from the data and stated in the
    report. A check that cannot run is listed under "not run" with the
    reason.
    """
    if time is None and horizon is not None:
        raise ValueError("a horizon needs a time column")
    panel = Panel(df, unit=unit, time=time, event=event)
    if features is None:
        features = [c for c in panel.df.columns
                    if c not in (unit, time, event)
                    and pd.api.types.is_numeric_dtype(panel.df[c])]
    flat = (list(dict.fromkeys(c for v in features.values() for c in v))
            if isinstance(features, dict) else list(features))
    if not flat:
        raise ValueError("there are no columns to check")
    if importance is not True and importance is not False:
        if isinstance(importance, str):
            raise TypeError("importance is True, False, or a list of "
                            "column names")
        importance = list(importance)
        unknown = [c for c in importance if c not in flat]
        if unknown or not importance:
            raise ValueError(f"importance names columns that are not "
                             f"checked: {unknown}")
    sets = _feature_sets(features)
    seed = pick_seed(None, random)
    sc = resolve(_scorer)
    from . import __doi__, __version__
    rep = Report(scorer=sc.describe(), info={
        "whowhen": {"version": __version__, "doi": __doi__},
        "n_units": int(panel.df[unit].nunique()), "features": flat,
        "time": time,
        "offset": None, "horizon": None if horizon is None else str(horizon),
        "repeats": repeats, "random": random, "seed": seed})
    common = dict(seed=seed, random=random, repeats=repeats)
    event_kw = dict(model=sc, horizon=horizon, **common)
    offset = no_pairs = None
    if time is not None:
        try:
            offset = panel.default_offset()
            rep.info["offset"] = [str(offset[0]), str(offset[1])]
        except ValueError as e:
            no_pairs = str(e)
    def gaps():
        """The pair tests on the shorter and the longer half of the gap range."""
        lo, hi = offset
        if not lo < hi:
            raise ValueError("every pair of rows is the same gap apart")
        mid = lo + (hi - lo) / 2
        out = []
        for name, (a, b) in (("short", (lo, mid)), ("long", (mid, hi))):
            r = audit(panel, features, (a, b), scorer=sc, n_pairs=n_pairs,
                      **common)
            for kind in ("reidentification", "time_order"):
                t = getattr(r, kind).drop(columns="controls",
                                          errors="ignore")
                t.insert(0, "test", kind)
                t.insert(1, "gap", f"{name}: {a} to {b}")
                out.append(t)
        return pd.concat(out, ignore_index=True).sort_values(
            ["test", "features"], kind="stable", ignore_index=True)

    def per_set(check, **kw):
        """One result per named feature set. The sets run one after
        another: each check already spreads its repeats over every core."""
        return {name: check(panel, cols, **kw, **event_kw)
                for name, cols in sets.items()}

    steps = {
        "by_gap": gaps,
        "out_of_time": lambda: per_set(overlap_check, time_split="calendar"),
        "pair_tests": lambda: audit(panel, features, offset, scorer=sc,
                               n_pairs=n_pairs,
                               which=(("reidentification", "time_order", "placebo")
                                      if time else ("reidentification",)),
                               controls=("random", "nearest_neighbour"), **common),
        "importance": lambda: rank_columns(panel, flat, offset,
                                        rank=None if importance is True
                                        else importance, scorer=sc,
                                        n_pairs=n_pairs, **common),
        "record_wise": lambda: per_set(overlap_check),
        "temporal_split": lambda: per_set(overlap_check,
                                         time_split="chronological"),
        "time_only": lambda: per_set(clock_baseline),
    }
    for part in PARTS:
        if part == "importance" and importance is False:
            rep.skipped[part] = ("off by default; importance=True ranks "
                                 "every column, or give a list")
        elif time is None and part in NEEDS_TIME:
            rep.skipped[part] = NO_TIME
        elif part in ("pair_tests", "by_gap", "importance") and no_pairs:
            rep.skipped[part] = no_pairs
        elif (part in ("record_wise", "temporal_split", "out_of_time", "time_only")
              and not event):
            rep.skipped[part] = "no event column was given"
        elif (part in ("record_wise", "temporal_split", "out_of_time", "time_only")
              and not (panel.df[event].fillna(0) != 0).any()):
            rep.skipped[part] = f"no row of {event!r} marks an event"
        else:
            try:
                setattr(rep, part, steps[part]())
            except ValueError as e:
                rep.skipped[part] = str(e)
    return rep
