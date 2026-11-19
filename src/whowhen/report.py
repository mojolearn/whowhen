"""Everything in one call.

    print(ww.report(panel))

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
               f"{len(i['features'])} feature columns, pair offset "
               f"{i['offset'][0]} to {i['offset'][1]}"
               f"{' (chosen from the data)' if i['offset_chosen'] else ''}",
               f"scorer: {self.scorer['name']}"]
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


def report(panel: Panel, features=None, offset=None, *, horizon=None,
           scorer=None, match: str | None = None, repeats: int = 4,
           skip=(), n_pairs: int = 20000,
           random: bool = False) -> Report:
    """Run every check the panel supports and return them together.

    Only the panel is required. What is missing decides what runs:

    features : default, every numeric column other than the unit, time
        and event columns.
    offset : default, chosen from the data (see Panel.default_offset)
        and stated in the report.
    event column on the panel : without one, the overlap check, the
        chronological control and the clock baseline are skipped.
    horizon : with one, the event checks forecast ("event within this
        long after the row"); without one, they use each row's flag.
    repeats : one number, N. Units fall into N + 1 fixed groups and the
        checks run N times, each holding out a
        different group: 1 holds out a half, 2 a third each, 4 (the
        default) a fifth each.
    random : False by default, so the same call gives the same numbers
        every time. True draws held-out units and pairs at random,
        differently on each run; the seed used is recorded in the
        report and can be passed back as ``seed`` to repeat that run.
    skip : names from ('audit', 'columns', 'overlap',
        'chronological', 'clock') to leave out.
    """
    seed = pick_seed(None, random)
    parts = [p for p in PARTS if p not in skip]
    unknown = [p for p in skip if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown part {unknown}; choose from {PARTS}")
    if features is None:
        keys = {panel.unit, panel.time, panel.event, match}
        features = [c for c in panel.df.columns if c not in keys
                    and pd.api.types.is_numeric_dtype(panel.df[c])]
    chosen = offset is None
    if chosen:
        offset = panel.default_offset()
    sc = resolve(scorer)
    flat = (list(dict.fromkeys(c for v in features.values() for c in v))
            if isinstance(features, dict) else list(features))
    rep = Report(scorer=sc.describe(), info={
        "n_units": int(panel.df[panel.unit].nunique()), "features": flat,
        "offset": [str(offset[0]), str(offset[1])], "offset_chosen": chosen,
        "horizon": None if horizon is None else str(horizon),
        "repeats": repeats, "random": random, "seed": seed})
    common = dict(seed=seed, random=random)
    event = dict(model=sc, horizon=horizon, repeats=repeats, **common)
    steps = {
        "audit": lambda: audit(panel, features, offset, scorer=sc,
                               match=match, n_pairs=n_pairs, repeats=repeats,
                               impostors=("random", "nearest"), **common),
        "columns": lambda: rank_columns(panel, flat, offset, scorer=sc,
                                        match=match, n_pairs=n_pairs,
                                        repeats=repeats, **common),
        "overlap": lambda: overlap_check(panel, flat, **event),
        "chronological": lambda: overlap_check(
            panel, flat, time_split="chronological", **event),
        "clock": lambda: clock_baseline(panel, flat, **event),
    }
    for part in PARTS:
        if part not in parts:
            continue
        if part in ("overlap", "chronological", "clock") and not panel.event:
            rep.skipped[part] = "the panel has no event column"
            continue
        try:
            setattr(rep, part, steps[part]())
        except ValueError as e:
            rep.skipped[part] = str(e)
    return rep
