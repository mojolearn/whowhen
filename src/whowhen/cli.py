"""Command line: a table file and a JSON declaration in, JSON out.

    whowhen audit --panel rows.parquet --config audit.json --out r.json
    whowhen demo

The declaration:

    {"unit": "serial_number", "time": "date", "event": "failure",
     "features": {"all": ["a", "b", "age"], "no_clock": ["a", "b"]},
     "offset": ["30D", "90D"],
     "which": ["identity", "timing"], "impostors": "random",
     "match": null, "n_pairs": 20000, "seeds": 5, "scorer": null}

Only unit, time, features and offset are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import audit
from .panel import Panel

PANEL_KEYS = ("unit", "time", "event")


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="whowhen", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a table")
    a.add_argument("--panel", type=Path, required=True,
                   help="parquet or CSV")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--out", type=Path, default=None)
    a.add_argument("--jobs", type=int, default=None)
    d = sub.add_parser("demo", help="synthetic panel with known answers")
    d.add_argument("--scorer", default=None)
    d.add_argument("--seeds", type=int, default=3)
    d.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "demo":
        from .demo import run_demo
        raise SystemExit(0 if run_demo(args.scorer, args.seeds, args.jobs)
                         else 1)
    cfg = json.loads(args.config.read_text())
    feats = cfg["features"]
    sets = feats if isinstance(feats, dict) else {"all": feats}
    cols = [cfg[k] for k in PANEL_KEYS if cfg.get(k)]
    cols += [c for v in sets.values() for c in v]
    if cfg.get("match"):
        cols.append(cfg["match"])
    df = _read(args.panel, list(dict.fromkeys(cols)))
    panel = Panel(df, **{k: cfg.get(k) for k in PANEL_KEYS})
    kw = {k: cfg[k] for k in ("which", "impostors", "match", "n_pairs",
                              "max_pairs_per_unit", "seeds", "bootstrap",
                              "seed", "scorer") if k in cfg}
    result = audit(panel, feats, tuple(cfg["offset"]), n_jobs=args.jobs, **kw)
    print(result)
    if args.out:
        result.to_json(args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
