# Changelog

## 0.2.0 (unreleased)

- Run `whowhen TABLE --unit ... --time ...` directly. This replaces the `whowhen report ...` command. `whowhen demo` and the Python interface are unchanged.

## 0.1.2

- The report records the whowhen version and DOI, and the classifier's DOI beside its version.

## 0.1.1

Same code as 0.1.0. Adds the DOI to the README and the citation file.

## 0.1.0

First release.

- `report(df, unit, time, event)`: every check the table supports, in one call.
- Identity and timing audits, with random and nearest-unit impostors.
- Timing placebo: the timing audit with the later row taken from a different unit.
- Both audits by short and long gap between the two rows.
- Per-column ranking.
- Overlap check, chronological control, time-cut split and clock baseline for tables with an event column, as row labels or as a forecast (`horizon`).
- `repeats=N` runs every check N times on fixed groups of units; `random=True` draws them instead.
- mojolearn as the classifier, CatBoost where mojolearn has no wheel.
- `--fail-if "NAME > NUMBER"` on the command line, for scripts.
- Synthetic demo and command line (`whowhen report`, `whowhen demo`).
