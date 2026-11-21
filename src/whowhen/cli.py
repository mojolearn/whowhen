"""Command line.

    whowhen rows.parquet --unit patient_id --time hour --event sepsis
    whowhen samples.csv --unit patient_id --event tumour
    whowhen demo

Leave --time out for a table whose rows are not ordered in time.

To use a report in a script, name what counts as a failure:

    whowhen rows.parquet --unit u --time t --event e \\
        --fail-if "overlap_gap > 0.02" --fail-if "clock_gain < 0.01"

The command then exits with 1 if any condition holds, and with 2 if a
condition names a check that did not run. The names are identity,
identity_nearest, timing, placebo, overlap_gap, chronological_gap,
time_cut_gap, clock and clock_gain.
"""
from __future__ import annotations

import argparse
import operator
import re
import sys
from pathlib import Path

import pandas as pd

from .report import report


NAMES = ("identity", "identity_nearest", "timing", "placebo", "overlap_gap",
         "chronological_gap", "time_cut_gap", "clock", "clock_gain")
OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
       "<=": operator.le}


def parse_condition(text: str):
    """('overlap_gap', '>', 0.02) from 'overlap_gap > 0.02'."""
    m = re.fullmatch(r"\s*(\w+)\s*(>=|<=|>|<)\s*([-+]?[\d.]+(?:e-?\d+)?)\s*",
                     text)
    if not m or m.group(1) not in NAMES:
        raise ValueError(
            f"cannot read the condition {text!r}; write NAME > NUMBER or "
            f"NAME < NUMBER, with NAME one of: {', '.join(NAMES)}")
    return m.group(1), m.group(2), float(m.group(3))


def check(conditions: list[str], numbers: dict) -> tuple[list[str], list[str]]:
    """(conditions that hold, conditions whose check did not run)."""
    held, missing = [], []
    for text in conditions:
        name, op, value = parse_condition(text)
        if name not in numbers:
            missing.append(f"{name}: this check did not run")
        elif OPS[op](numbers[name], value):
            held.append(f"{name} is {numbers[name]:.3f}, and "
                        f"{name} {op} {value:g} counts as a failure")
    return held, missing


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="whowhen", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    argv = sys.argv[1:]
    if argv and argv[0] == "demo":
        demo = argparse.ArgumentParser(prog="whowhen demo",
                                       description="synthetic table with known answers")
        demo.add_argument("--repeats", type=int, default=3)
        args = demo.parse_args(argv[1:])
        from .demo import run_demo
        raise SystemExit(0 if run_demo(args.repeats) else 1)
    r = ap
    r.add_argument("table", type=Path, help="parquet or CSV")
    r.add_argument("--unit", required=True)
    r.add_argument("--time", default=None,
                   help="leave out for a table not ordered in time")
    r.add_argument("--event", default=None)
    r.add_argument("--horizon", default=None)
    r.add_argument("--repeats", type=int, default=4)
    r.add_argument("--random", action="store_true")
    r.add_argument("--out", type=Path, default=None, help="write JSON here")
    r.add_argument("--fail-if", action="append", default=[],
                   metavar='"NAME > NUMBER"',
                   help="exit with 1 if this holds; may be given more "
                        "than once. NAME is one of: " + ", ".join(NAMES))
    args = ap.parse_args(argv)
    for text in args.fail_if:        # reject a bad condition before the run
        try:
            parse_condition(text)
        except ValueError as e:
            ap.error(str(e))
    df = (pd.read_parquet(args.table)
          if args.table.suffix in (".parquet", ".pq")
          else pd.read_csv(args.table))
    horizon = args.horizon
    if horizon is not None:
        try:
            horizon = float(horizon)
        except ValueError:
            pass
    rep = report(df, unit=args.unit, time=args.time, event=args.event,
                 horizon=horizon, repeats=args.repeats, random=args.random)
    print(rep)
    if args.out:
        import json
        args.out.write_text(json.dumps(rep.to_dict(), indent=2, default=str)
                            + "\n")
    held, missing = check(args.fail_if, rep.headline())
    for line in held:
        print("FAILED:", line, file=sys.stderr)
    for line in missing:
        print("CANNOT CHECK:", line, file=sys.stderr)
    if missing:
        raise SystemExit(2)
    if held:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
