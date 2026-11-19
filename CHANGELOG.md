# Changelog

## 0.1.0 (unreleased)

First release.

- `Panel`: repeated observations per unit, with unit, time and optional event columns.
- `audit`: identity, timing and placebo checks.
- `compare`: audits several named column sets on the same pairs, in one table.
- Nearest-unit impostors for the identity check (`impostors="nearest"`).
- `rank_columns`.
- `overlap_check`: does sharing units between training and test rows change the score?
- Pluggable scorer, with mojolearn as the default and LightGBM as the fallback.
- Synthetic demo (`whowhen-demo`).
- Command line (`whowhen`).
