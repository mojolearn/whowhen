# whowhen

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24406266.svg)](https://doi.org/10.5281/zenodo.24406266)

Check a table of repeated observations per unit (patients, machines,
loans, users, vessels) for two signals a model can exploit without
predicting anything:

- **who**: can a row be matched to the unit it came from?
- **when**: can two rows of one unit be put in time order?

If the answer to either is yes, a model evaluated on your data may be
scoring well by recognizing units or by reading elapsed time.

## Install

```
pip install whowhen
```

## Use

From the terminal, give a CSV file and the names of its columns:

```bash
whowhen rows.csv --unit patient_id --time hour --event sepsis
```

If the rows are not ordered in time (several samples per patient,
several measurements per batch), leave the time column out:

```bash
whowhen samples.csv --unit patient_id --event tumour
```

In Python, pass a pandas DataFrame (your data table):

```python
import pandas as pd
import whowhen as ww

df = pd.read_parquet("stays.parquet")
print(ww.report(df, unit="patient_id", time="hour", event="sepsis"))
print(ww.report(samples, unit="patient_id", event="tumour"))   # no time order
```

Both interfaces run the same checks.

| Setting | Required | Default | What it is |
|---|---|---|---|
| `df` | yes | | the table, one row per observation |
| `unit` | yes | | the column naming the unit |
| `time` | no | none | the column giving when the row was observed; leave it out if the rows are not ordered in time |
| `event` | no | none | a column that is nonzero where the event happens |
| `features` | no | every numeric column | the columns to check: a list, or a dict of named lists to compare side by side |
| `importance` | no | `False` | column importance: `True` for every checked column, or a list of columns. It costs two model fits per ranked column, per test and repeat, so it is the slowest part on a wide table |
| `horizon` | no | none | for forecasting tasks: how far ahead the event is predicted, in the time column's scale (`12`, or `"12h"` for datetimes) |
| `repeats` | no | 4 | how many times every check is run |
| `random` | no | `False` | draw held-out units at random instead of the fixed groups |
| `n_pairs` | no | 20000 | the most row pairs each run uses; smaller tables use what they have |

What runs depends on what you give:

| You give | You get |
|---|---|
| `unit` only | the re-identification test |
| `unit` and `event` | plus record-wise vs subject-wise validation |
| `unit` and `time` | the re-identification and time-order tests, the placebo test, results by time gap |
| plus `event` | record-wise vs subject-wise validation, the within-unit temporal split, out-of-time validation, the time-only baseline |
| plus `horizon` | the event checks run as a forecast: "does the event come within this long after the row?" |
| `importance` | column importance for the re-identification and time-order tests |

A check that cannot run is listed at the end of the report with the
reason.

To compare choices made before modeling, such as raw values against a
carried-forward or imputed version, put both versions in the table and
name them. Every check then runs once per named set, the pair tests on
the same pairs and the event checks on the same splits:

```python
ww.report(df, unit="patient_id", time="hour",
          features={"raw": raw_cols, "carried_forward": filled_cols})
```

To see the library work before pointing it at your own data, run it on
a synthetic table where the right answers are known:

```
whowhen demo
```

## What the report contains

Every check trains a gradient-boosted classifier and reports the AUC on
units it did not train on. An AUC of 0.5 means the classifier found
nothing; 1.0 means it was always right.

**Re-identification test and time-order test.** Both classify pairs of
rows from units with no recorded event, built so that the two classes
look the same unless the signal is there. The gap between the two rows
of a pair is chosen from the data and stated in the report.

| Test | The pair | Label | AUC 0.5 means |
|---|---|---|---|
| re-identification | two rows a set time apart | same unit, or two different units seen at the same two times | rows carry nothing that says which unit produced them |
| time order | two rows of one unit a set time apart | which row is the later one | a unit's rows carry nothing that says when they were recorded |

Units are split into training and held-out sides before any pair is
drawn. The interval comes from a bootstrap over held-out units, pooled
over repeats.

The re-identification test is reported with two kinds of control (the
different unit in a pair):

- `random`: any other unit;
- `nearest_neighbour`: one of the 5 units nearest the first, by
  squared Euclidean distance between standardized per-unit column
  means. This matches away differences shared by a kind of unit (a tug
  against a tanker), and also each unit's own average level.

**Placebo test.** The time-order test again, with the later row taken
from a different unit observed at that time. If this score is near 0.5,
the time-order signal is each unit's own history. If it is near the
time-order score, the signal is change shared by all units, such as
calendar drift or a change in how every unit is recorded.

**Results by time gap.** Both tests again, once with the two rows of a
pair in the shorter half of the gap range and once in the longer half.
The range runs from the median time between a unit's consecutive rows
to half the median time a unit is observed. If re-identification stays
high at the long gap, units stay recognizable as time passes.

**Column importance.** Off unless you set `importance`. For each
column: the single-column AUC (that column alone) and the drop-column
importance (how much the AUC falls when that column is removed), on the
same pairs.

**Record-wise vs subject-wise validation.** Needs `event`. Here subject
means unit. It scores the same test rows twice, with the same number of
training rows per class: once with the test units held out of training
(subject-wise), once with other rows of those units swapped in
(record-wise, as a split of rows at random would do). The gap between
the two scores is how much sharing units between training and test is
worth to this classifier on your task. whowhen makes these splits from
your one table; it does not use your own split.

**Within-unit temporal split.** The same comparison, testing the later
half of each test unit's rows and letting training see the earlier
half: does a unit's own past help predict its own future?

**Out-of-time validation.** The same comparison with a single cut in
the time column: test rows come after it, training rows before it.
When the time column is a shared calendar, this is the usual split by
date, and the gap shows how much unit overlap such a split leaves in
place.

**Time-only baseline.** Many event tasks can be scored well above
chance from one number: how long the unit has been observed. This fits
the same classifier on elapsed time alone, on your columns alone, and
on both, with units held out, and reports what the columns add once
elapsed time is known.

## What the numbers do not mean

A pair-test AUC above 0.5 says the signal is present in the columns.
It does not say your model uses it.

Record-wise vs subject-wise validation does not say so either. It
measures what unit overlap is worth to whowhen's classifier, on your
columns and your outcome, under the split it names. A gap there shows
the evaluation is sensitive to unit overlap. To learn what your own
model gains, train it both ways.

**An AUC near 0.5 does not certify an evaluation.** It says this
classifier, on these columns, found nothing. Leakage through the
label, through preprocessing fitted on test rows, or through columns
not in the table is outside what these checks see.

## When it applies

- many units (hundreds or more), each observed more than once;
- for the re-identification test, several units observed at the same
  time values, since the different-unit pair is matched on time;
- without a time column, a pair is two rows of one unit or one row each
  from two units, with no matching on time. If the rows do have an
  order, give the time column: unmatched pairs can also be told apart
  by anything that changed between the times the units were observed;
- with `horizon`, an event column flagged on the row where the event
  happens, not on every row of the unit.

With `horizon`, a unit's last row is taken as the end of its follow-up.
A row of a unit with no event counts as "no event within the horizon"
only if the unit was still observed one horizon later; other such rows
are left out, and the report says how many. Under the temporal split
and out-of-time validation, a row from before the cut is trained on
only if its horizon has passed by the cut, so no label is used before
it could have been known. If your table ends a unit's rows before its
follow-up truly ends, extend the rows or shorten the horizon.

It does not apply to tables with one row per example and no repeated
unit.

## Repeats and repeatability

`repeats=N` is one number: every check runs N times. Units fall into
N + 1 fixed groups, by a hash of the unit identifier, and each run
holds out a different group:

| `repeats` | Each run tests |
|---|---|
| 1 | one half of the units |
| 2 | one third |
| 3 | one quarter |
| 4 (default) | one fifth |

More repeats give a steadier answer, but each run then tests a smaller
share of the units, so very high values suit only large tables.

By default the same call on the same table gives the same numbers
every time. `random=True` draws the held-out units and the pairs at
random instead, differently on each run, and the report states the
seed that run used.

## The classifier

The checks use [mojolearn](https://github.com/mojolearn/mojolearn)'s
`GradientBoosting` with oblivious (symmetric) trees. mojolearn ships
wheels for macOS on Apple silicon and Linux on x86-64. On other
platforms, including Windows, CatBoost is installed and used instead,
with the same settings. The report names the whowhen version and DOI,
and the classifier that ran with its version and, where the classifier
declares one, its DOI.

Independent runs are spread across all cores. Set the `WHOWHEN_JOBS`
environment variable to use fewer.

## Command line

```
whowhen rows.parquet --unit patient_id --time hour --event sepsis
whowhen rows.parquet --unit patient_id --time hour --importance
whowhen demo
```

CSV and Parquet files are supported. `--importance` alone ranks every
column; `--importance a b` ranks only those. Use `whowhen --help` for
report options or `whowhen demo --help` for demo options.

To use a report in a script or a scheduled job, say what counts as a
failure:

```
whowhen rows.parquet --unit patient_id --time hour --event sepsis \
    --fail-if "record_wise_gap > 0.02" --fail-if "time_only_gain < 0.01"
```

The command exits with 1 if any condition holds, and with 2 if a
condition names a check that did not run. You choose the conditions;
there are no built-in thresholds.

| Name | The number |
|---|---|
| `reidentification`, `reidentification_nearest_neighbour` | re-identification AUC against random controls, and against nearest-neighbour controls |
| `time_order`, `placebo` | time-order AUC, and the placebo test |
| `record_wise_gap`, `temporal_split_gap`, `out_of_time_gap` | record-wise AUC minus subject-wise AUC under each split |
| `time_only` | AUC of elapsed time alone |
| `time_only_gain` | what the columns add once elapsed time is known |

With several named feature sets, each number is the one furthest from
a clean result: the highest AUC, the highest gap, the lowest gain. A
high re-identification AUC is normal for many tables and is not by
itself a failure; the gaps say whether it matters for your task.

## Citing

The method is described in:

> Andrew Hendel. Auditing Identity and Timing Signals in Operational
> Logs. 2026. Under review.

The library is archived on Zenodo. This DOI always points to the newest
version: [10.5281/zenodo.24406266](https://doi.org/10.5281/zenodo.24406266).

See `CITATION.cff`.

## License

Apache License 2.0. See `LICENSE`.
