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

PARTS = ("audit", "gaps", "columns", "overlap", "chronological", "time_cut",
         "clock")


@dataclass
class Report:
    """Results of ``report``. A part that did not run is None and its
    reason is in ``skipped``."""
    scorer: dict
    info: dict
    audit: AuditResult | None = None
    gaps: pd.DataFrame | None = None
    columns: dict | None = None
    overlap: OverlapResult | None = None
    chronological: OverlapResult | None = None
    time_cut: OverlapResult | None = None
    clock: ClockResult | None = None
    skipped: dict = field(default_factory=dict)

    def headline(self) -> dict:
        """One number per check, for scripts. With several feature sets
        an audit's number is the highest across them. A check that did
        not run has no entry."""
        h = {}
        if self.audit is not None:
            t = self.audit.identity
            if t is not None:
                for mode, name in (("random", "identity"),
                                   ("nearest", "identity_nearest")):
                    sub = t[t["impostors"] == mode]
                    if len(sub):
                        h[name] = float(sub["auc"].max())
            for kind in ("timing", "placebo"):
                t = getattr(self.audit, kind)
                if t is not None:
                    h[kind] = float(t["auc"].max())
        for part, name in (("overlap", "overlap_gap"),
                           ("chronological", "chronological_gap"),
                           ("time_cut", "time_cut_gap")):
            r = getattr(self, part)
            if r is not None:
                h[name] = float(r.gap)
        if self.clock is not None:
            t = self.clock.table.set_index("input")["auc"]
            h["clock"] = float(t["clock"])
            h["clock_gain"] = float(self.clock.gain)
        return h

    def to_dict(self) -> dict:
        d = {"scorer": self.scorer, "info": self.info,
             "skipped": self.skipped, "headline": self.headline()}
        for part in ("audit", "overlap", "chronological", "time_cut",
                     "clock"):
            r = getattr(self, part)
            if r is not None:
                d[part] = r.to_dict()
        if self.gaps is not None:
            d["gaps"] = self.gaps.to_dict(orient="records")
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
        w = i.get("whowhen")
        if w:
            out.append(f"whowhen {w['version']} (doi:{w['doi']})")
        p = self.scorer.get("params", {})
        out.append(f"scorer: {self.scorer['name']}"
                   + (f" {p['version']}" if "version" in p else "")
                   + (f" (doi:{p['doi']})" if "doi" in p else ""))
        if self.audit is not None:
            for kind in ("identity", "timing"):
                t = getattr(self.audit, kind)
                if t is not None:
                    out += ["", f"{kind} audit (AUC; 0.5 = no signal)",
                            t.round(3).to_string(index=False)]
            t = self.audit.placebo
            if t is not None:
                out += ["", "timing placebo (the later row is taken from a "
                        "different unit)",
                        t.round(3).to_string(index=False),
                        "near 0.5: the timing signal is each unit's own. "
                        "near the timing audit: it is change shared by all "
                        "units."]
        if self.gaps is not None:
            out += ["", "the audits by how far apart the two rows are",
                    self.gaps.round(3).to_string(index=False)]
        if self.columns is not None:
            for kind, t in self.columns.items():
                out += ["", f"columns by {kind} signal",
                        t.head(10).round(3).to_string(index=False)]
        for part in ("overlap", "chronological", "time_cut", "clock"):
            r = getattr(self, part)
            if r is not None:
                out += ["", repr(r)]
        if self.skipped:
            out += ["", "not run:"]
            out += [f"  {k}: {v}" for k, v in self.skipped.items()]
        return "\n".join(out)


def report(df: pd.DataFrame, unit: str, time: str,
           event: str | None = None, *, features=None, horizon=None,
           repeats: int = 4,
           random: bool = False, n_pairs: int = 20000,
           _scorer=None) -> Report:
    """Run every check the table supports and return them together.

    Parameters
    ----------
    df : one row per observation.
    unit : the column naming the unit (patient, machine, loan, vessel).
    time : the column giving when the row was observed.
    event : optional column, nonzero where the event happens. Without
        it the overlap check, the chronological control and the clock
        baseline are left out.
    features : optional. The columns to check: a list, or a dict of
        named lists to audit side by side on the same pairs (for
        example {"raw": [...], "filled": [...]}, or with and without a
        clock column). Default: every numeric column other than unit,
        time and event.
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

    The gap between the two rows of a pair is chosen from the data and stated in the
    report. A check that cannot run is listed under "not run" with the
    reason.
    """
    panel = Panel(df, unit=unit, time=time, event=event)
    if features is None:
        features = [c for c in panel.df.columns
                    if c not in (unit, time, event)
                    and pd.api.types.is_numeric_dtype(panel.df[c])]
    flat = (list(dict.fromkeys(c for v in features.values() for c in v))
            if isinstance(features, dict) else list(features))
    if not flat:
        raise ValueError("there are no columns to check")
    seed = pick_seed(None, random)
    sc = resolve(_scorer)
    from . import __doi__, __version__
    rep = Report(scorer=sc.describe(), info={
        "whowhen": {"version": __version__, "doi": __doi__},
        "n_units": int(panel.df[unit].nunique()), "features": flat,
        "offset": None, "horizon": None if horizon is None else str(horizon),
        "repeats": repeats, "random": random, "seed": seed})
    common = dict(seed=seed, random=random, repeats=repeats)
    event_kw = dict(model=sc, horizon=horizon, **common)
    offset = None
    try:
        offset = panel.default_offset()
        rep.info["offset"] = [str(offset[0]), str(offset[1])]
    except ValueError as e:
        no_pairs = str(e)
    def gaps():
        """The audits on the shorter and the longer half of the gap range."""
        lo, hi = offset
        if not lo < hi:
            raise ValueError("every pair of rows is the same gap apart")
        mid = lo + (hi - lo) / 2
        out = []
        for name, (a, b) in (("short", (lo, mid)), ("long", (mid, hi))):
            r = audit(panel, features, (a, b), scorer=sc, n_pairs=n_pairs,
                      **common)
            for kind in ("identity", "timing"):
                t = getattr(r, kind).drop(columns="impostors",
                                          errors="ignore")
                t.insert(0, "audit", kind)
                t.insert(1, "gap", f"{name}: {a} to {b}")
                out.append(t)
        return pd.concat(out, ignore_index=True).sort_values(
            ["audit", "features"], kind="stable", ignore_index=True)

    steps = {
        "gaps": gaps,
        "time_cut": lambda: overlap_check(
            panel, flat, time_split="calendar", **event_kw),
        "audit": lambda: audit(panel, features, offset, scorer=sc,
                               n_pairs=n_pairs,
                               which=("identity", "timing", "placebo"),
                               impostors=("random", "nearest"), **common),
        "columns": lambda: rank_columns(panel, flat, offset, scorer=sc,
                                        n_pairs=n_pairs, **common),
        "overlap": lambda: overlap_check(panel, flat, **event_kw),
        "chronological": lambda: overlap_check(
            panel, flat, time_split="chronological", **event_kw),
        "clock": lambda: clock_baseline(panel, flat, **event_kw),
    }
    for part in PARTS:
        if part in ("audit", "gaps", "columns") and offset is None:
            rep.skipped[part] = no_pairs
        elif (part in ("overlap", "chronological", "time_cut", "clock")
              and not event):
            rep.skipped[part] = "no event column was given"
        elif (part in ("overlap", "chronological", "time_cut", "clock")
              and not (panel.df[event].fillna(0) != 0).any()):
            rep.skipped[part] = f"no row of {event!r} marks an event"
        else:
            try:
                setattr(rep, part, steps[part]())
            except ValueError as e:
                rep.skipped[part] = str(e)
    return rep
