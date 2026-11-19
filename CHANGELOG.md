# Changelog

## 0.1.0 (unreleased)

First release.

- `Panel`: repeated observations per unit, with unit, time and optional event columns.
- `audit`: identity, timing and placebo checks.
- `compare`: audits several named column sets on the same pairs, in one table.
- Nearest-unit impostors for the identity check (`impostors="nearest"`).
- `rank_columns`.
- `overlap_check`: does sharing units between training and test rows change the score?
- `overlap_check(horizon=...)`: the same check on a forecasting task.
- `overlap_check(time_split="chronological")`: does a unit's own past help predict its own future?
- `clock_baseline`: how much of the task elapsed time alone predicts.
- `report`: every check the panel supports, in one call.
- Pluggable scorer, with mojolearn as the default and LightGBM as the fallback.
- Synthetic demo (`whowhen-demo`).
- Command line (`whowhen`).
