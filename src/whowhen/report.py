"""Everything in one call.

    print(ww.report(df, unit="patient_id", time="hour", event="sepsis"))

runs every check the panel supports and says which it skipped and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .audit import AuditResult, audit, pick_seed
from .clock import ClockResult, clock_baseline
from .columns import rank_columns
from .overlap import OverlapResult, overlap_check
from .panel import Panel
from .scorers import resolve

PARTS = ("audit", "columns", "overlap", "chronological", "clock")


@dataclass
class Report:
    """Results of ``report``. A part that did not run is None and its
    reason is in ``skipped``."""
    scorer: dict
    info: dict
    audit: AuditResult | None = None
    columns: dict | None = None
    overlap: OverlapResult | None = None
    chronological: OverlapResult | None = None
    clock: ClockResult | None = None
    skipped: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"scorer": self.scorer, "info": self.info,
             "skipped": self.skipped}
        for part in ("audit", "overlap", "chronological", "clock"):
            r = getattr(self, part)
            if r is not None:
                d[part] = r.to_dict()
        if self.columns is not None:
            d["columns"] = {k: t.to_dict(orient="records")
                            for k, t in self.columns.items()}
        return d

    def __repr__(self) -> str:
        i = self.info
        out = [f"whowhen report: {i['n_units']:,} units, "
               f"{len(i['features'])} columns audited, {i['repeats']} "
               f"repeats{' (random, seed ' + str(i['seed']) + ')' if i['random'] else ''}"]
        if i["offset"]:
            out.append(f"rows of a pair are {i['offset'][0]} to "
                       f"{i['offset'][1]} apart (chosen from the data)")
        out.append(f"scorer: {self.scorer['name']}")
        if self.audit is not None:
            for kind in ("identity", "timing"):
                t = getattr(self.audit, kind)
                if t is not None:
                    out += ["", f"{kind} audit (AUC; 0.5 = no signal)",
                            t.round(3).to_string(index=False)]
        if self.columns is not None:
            for kind, t in self.columns.items():
                out += ["", f"columns by {kind} signal",
                        t.head(10).round(3).to_string(index=False)]
        for part in ("overlap", "chronological", "clock"):
            r = getattr(self, part)
            if r is not None:
                out += ["", repr(r)]
        if self.skipped:
            out += ["", "not run:"]
            out += [f"  {k}: {v}" for k, v in self.skipped.items()]
        return "\n".join(out)


def report(df: pd.DataFrame, unit: str, time: str | None = None,
           event: str | None = None, *, horizon=None, repeats: int = 4,
           random: bool = False, n_pairs: int = 20000,
           _scorer=None) -> Report:
    """Run every check the table supports and return them together.

    Parameters
    ----------
    df : one row per observation.
    unit : the column naming the unit (patient, machine, loan, vessel).
    time : the column giving when the row was observed. Without it only
        the overlap check can run.
    event : optional column, nonzero where the event happens. Without
        it the overlap check, the chronological control and the clock
        baseline are left out.
    horizon : optional. With it the event checks forecast: "does the
        event come within this long after the row?", in the time
        column's scale (a string such as '12h' for datetimes). The
        event column must then flag the row where the event happens.
    repeats : one number, N: every check runs N times. Units fall into
        N + 1 fixed groups and each run holds out a different group: 1
        holds out a half, 2 a third each, 4 (the default) a fifth each.
    random : False by default, so the same call gives the same numbers
        every time. True draws held-out units and pairs at random,
        differently on each run, and records the seed it used.
    n_pairs : the most row pairs each audit run is trained and tested
        on. Smaller tables use as many as they have.

    Every numeric column other than these is audited. The gap between
    the two rows of a pair is chosen from the data and stated in the
    report. A check that cannot run is listed under "not run" with the
    reason.
    """
    if time is None and event is None:
        raise ValueError("give a time column, an event column, or both; "
                         "with only a unit column there is nothing to check")
    no_time = time is None
    if no_time:
        time = "__row__"
        df = df.assign(**{time: df.groupby(unit, sort=False).cumcount()})
    panel = Panel(df, unit=unit, time=time, event=event)
    features = [c for c in panel.df.columns if c not in (unit, time, event)
                and pd.api.types.is_numeric_dtype(panel.df[c])]
    if not features:
        raise ValueError("the table has no numeric columns to audit")
    seed = pick_seed(None, random)
    sc = resolve(_scorer)
    rep = Report(scorer=sc.describe(), info={
        "n_units": int(panel.df[unit].nunique()), "features": features,
        "offset": None, "horizon": None if horizon is None else str(horizon),
        "repeats": repeats, "random": random, "seed": seed})
    common = dict(seed=seed, random=random, repeats=repeats)
    event_kw = dict(model=sc, horizon=horizon, **common)
    offset = None
    if not no_time:
        try:
            offset = panel.default_offset()
            rep.info["offset"] = [str(offset[0]), str(offset[1])]
        except ValueError as e:
            no_pairs = str(e)
    steps = {
        "audit": lambda: audit(panel, features, offset, scorer=sc,
                               n_pairs=n_pairs,
                               impostors=("random", "nearest"), **common),
        "columns": lambda: rank_columns(panel, features, offset, scorer=sc,
                                        n_pairs=n_pairs, **common),
        "overlap": lambda: overlap_check(panel, features, **event_kw),
        "chronological": lambda: overlap_check(
            panel, features, time_split="chronological", **event_kw),
        "clock": lambda: clock_baseline(panel, features, **event_kw),
    }
    for part in PARTS:
        if part != "overlap" and no_time:
            rep.skipped[part] = "no time column was given"
        elif part in ("audit", "columns") and offset is None:
            rep.skipped[part] = no_pairs
        elif part in ("overlap", "chronological", "clock") and not event:
            rep.skipped[part] = "no event column was given"
        elif horizon is not None and no_time:
            rep.skipped[part] = "a horizon needs a time column"
        else:
            try:
                setattr(rep, part, steps[part]())
            except ValueError as e:
                rep.skipped[part] = str(e)
    return rep
