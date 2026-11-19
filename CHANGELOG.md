# Changelog

## 0.1.0 (unreleased)

First release.

- `report(df, unit, time, event)`: every check the table supports, in one call.
- Identity and timing audits, with random and nearest-unit impostors.
- Per-column ranking.
- Overlap check, chronological control and clock baseline for tables with an event column, as row labels or as a forecast (`horizon`).
- `repeats=N` runs every check N times on fixed groups of units; `random=True` draws them instead.
- mojolearn as the classifier, CatBoost where mojolearn has no wheel.
- Synthetic demo and command line (`whowhen report`, `whowhen demo`).
