# Changelog

## 0.2.0 (unreleased)

- Run `whowhen TABLE --unit ... --time ...` directly. This replaces the `whowhen report ...` command. `whowhen demo` and the Python interface are unchanged.

Results change for calls that use `horizon` or named feature sets.

- With `horizon`, a row of a unit with no event is used only if the unit was still observed one horizon later. Such rows were all counted as negative before. The report says how many rows are left out.
- With `horizon`, the chronological and time-cut splits train only on rows whose horizon has passed by the cut, so no label is used before it could have been known.
- With named feature sets, the overlap check, chronological control, time-cut split and clock baseline run once per set. They ran once on all the columns together before. In the report and its JSON these four results are keyed by the set's name (`all` when no names were given).
- For scripts, `overlap_gap`, `chronological_gap` and `time_cut_gap` are the highest across sets and `clock_gain` is the lowest.
- The README no longer says the overlap check tests whether your model uses a signal. It measures what unit overlap is worth to whowhen's classifier.

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
