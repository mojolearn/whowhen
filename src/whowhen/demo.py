"""Both audits on a synthetic panel where the right answer is known.

What is planted:

  clock, drift   move with time. Readable by the timing audit only.
  sig0..sig2     a constant per unit plus noise. Readable by the
                 identity audit only: flat in time, so they cannot order
                 two rows of one unit.
  noise0..noise2 independent draws. Readable by neither.

Each audit must read its own columns well above 0.5 and the other
columns at 0.5. A feature set that misses is a defect in the library,
not a finding.

Every unit is observed on the same time grid. Real panels stagger
entry, which makes age partly identifying at a matched time; the demo
leaves that out so its verdicts are crisp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import AuditResult, audit
from .panel import Panel

SIG = [f"sig{i}" for i in range(3)]
NOISE = [f"noise{i}" for i in range(3)]
CLOCKS = ["clock", "drift"]
SETS = {"all": CLOCKS + SIG + NOISE, "clocks": CLOCKS,
        "signature": SIG, "noise": NOISE}
# feature set -> does the audit have something to read?
EXPECT = {"identity": {"all": True, "clocks": False, "signature": True,
                       "noise": False},
          "timing": {"all": True, "clocks": True, "signature": False,
                     "noise": False}}
SIGNAL_MIN = 0.60


def synthesize(n_units: int = 700, n_days: int = 120,
               seed: int = 20260821) -> pd.DataFrame:
    """A panel with time and identity planted on disjoint columns."""
    rng = np.random.default_rng(seed)
    n = n_units * n_days
    t = np.tile(np.arange(n_days, dtype=float), n_units)
    u = np.repeat(np.arange(n_units), n_days)
    df = pd.DataFrame({
        "unit": np.char.add("u", np.char.zfill(u.astype(str), 4)),
        "time": np.tile(pd.date_range("2025-01-01", periods=n_days,
                                      freq="D"), n_units),
        "clock": t + rng.normal(0.0, 30.0, n),
        "drift": 0.02 * t + rng.normal(0.0, 1.0, n)})
    offsets = rng.normal(0.0, 2.0, size=(n_units, len(SIG)))
    for j, c in enumerate(SIG):
        df[c] = offsets[u, j] + rng.normal(0.0, 1.0, n)
    for c in NOISE:
        df[c] = rng.normal(0.0, 1.0, n)
    return df


def verdicts(result: AuditResult) -> list[tuple]:
    """(audit, feature set, AUC, half-width, expected signal, ok).

    A no-signal set passes when 0.5 lies within the interval's
    half-width of the mean; a signal set passes when it does not and
    the mean clears SIGNAL_MIN.
    """
    out = []
    for kind, expect in EXPECT.items():
        table = getattr(result, kind)
        if table is None:
            continue
        for row in table.itertuples():
            half = (row.hi - row.lo) / 2
            signal = expect[row.features]
            ok = (row.auc - half > 0.5 and row.auc >= SIGNAL_MIN if signal
                  else abs(row.auc - 0.5) <= half)
            out.append((kind, row.features, row.auc, half, signal, ok))
    return out


def run_demo(repeats: int = 3, quiet: bool = False, _scorer=None) -> bool:
    panel = Panel(synthesize(), unit="unit", time="time")
    result = audit(panel, SETS, offset=("30D", "90D"), scorer=_scorer,
                   n_pairs=3000, max_pairs_per_unit=4, repeats=repeats,
                   bootstrap=500)
    rows = verdicts(result)
    if not quiet:
        print(f"scorer: {result.scorer['name']}\n")
        print(f"{'audit':10s}{'features':12s}{'AUC':>7s}{'+/-':>7s}"
              f"  planted    verdict")
        for kind, name, a, half, signal, ok in rows:
            print(f"{kind:10s}{name:12s}{a:7.3f}{half:7.3f}  "
                  f"{'signal' if signal else 'none':9s}  "
                  f"{'ok' if ok else 'MISSED'}")
        print("\nall checks passed" if all(r[-1] for r in rows)
              else "\nat least one check MISSED")
    return all(r[-1] for r in rows)

