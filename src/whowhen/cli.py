"""Command line.

    whowhen report rows.parquet --unit patient_id --time hour --event sepsis
    whowhen demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .report import report


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="whowhen", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="check a table")
    r.add_argument("table", type=Path, help="parquet or CSV")
    r.add_argument("--unit", required=True)
    r.add_argument("--time", default=None)
    r.add_argument("--event", default=None)
    r.add_argument("--horizon", default=None)
    r.add_argument("--repeats", type=int, default=4)
    r.add_argument("--random", action="store_true")
    r.add_argument("--out", type=Path, default=None, help="write JSON here")
    d = sub.add_parser("demo", help="synthetic table with known answers")
    d.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "demo":
        from .demo import run_demo
        raise SystemExit(0 if run_demo(args.repeats) else 1)
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


if __name__ == "__main__":  # pragma: no cover
    main()
