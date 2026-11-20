# whowhen

Check a table of repeated observations per unit (patients, machines,
loans, users, vessels) for two signals a model can exploit without
predicting anything:

- **who**: can a row be matched to the unit it came from?
- **when**: can two rows of one unit be put in time order?

If the answer to either is yes, a model evaluated on your data may be
scoring well by recognizing units or reading the clock.

## Install

```
pip install git+https://github.com/mojolearn/whowhen
```

## Use

```python
import pandas as pd
import whowhen as ww

df = pd.read_parquet("stays.parquet")
print(ww.report(df, unit="patient_id", time="hour", event="sepsis"))
```

That is the whole interface.

| Setting | Required | Default | What it is |
|---|---|---|---|
| `df` | yes | | the table, one row per observation |
| `unit` | yes | | the column naming the unit |
| `time` | yes | | the column giving when the row was observed |
| `event` | no | none | a column that is nonzero where the event happens |
| `features` | no | every numeric column | the columns to check: a list, or a dict of named lists to compare side by side |
| `horizon` | no | none | for forecasting tasks: how far ahead the event is predicted, in the time column's scale (`12`, or `"12h"` for datetimes) |
| `repeats` | no | 4 | how many times every check is run |
| `random` | no | `False` | draw held-out units at random instead of the fixed groups |
| `n_pairs` | no | 20000 | the most row pairs each run uses; smaller tables use what they have |

What runs depends on what you give:

| You give | You get |
|---|---|
| `unit` and `time` | the identity and timing audits, the timing placebo, the individual-or-type test, the audits by short and long gap, the per-column ranking |
| plus `event` | the overlap check, the chronological control, the time-cut split, the clock baseline |
| plus `horizon` | those three run as a forecast: "does the event come within this long after the row?" |

A check that cannot run is listed at the end of the report with the
reason.

To compare choices made before modeling, such as raw values against a
carried-forward or imputed version, put both versions in the table and
name them:

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

**Identity audit and timing audit.** Both train a classifier on pairs
of rows from units with no recorded event, built so that the two
classes look the same unless the signal is there. The gap between the
two rows of a pair is chosen from the data and stated in the report.

| Audit | The pair | Label | AUC 0.5 means |
|---|---|---|---|
| identity | two rows a set time apart | same unit, or two different units seen at the same two times | rows carry nothing that says which unit produced them |
| timing | two rows of one unit a set time apart | which row is the later one | a unit's rows carry nothing that says when they were recorded |

Units are split into training and held-out sides before any pair is
drawn, so the AUC is measured on units the classifier never saw. The
interval comes from resampling held-out units, pooled over repeats.

**Timing placebo.** The timing audit again, with the later row taken
from a different unit observed at that time. If this score is near 0.5,
the timing signal is each unit's own history. If it is near the timing
audit's score, the signal is change shared by all units, such as
calendar drift or a change in how every unit is recorded.

**Individual or type.** The identity audit is reported twice. With
`random` impostors the different unit is any other unit, so a high
score can mean "these two rows are from different kinds of unit" (a
tug and a tanker). With `nearest` impostors the different unit is one
of the units most similar to the first, by per-unit column averages.
If the score stays high, individual units are recognizable. If it
falls toward 0.5, the columns identify the kind of unit, not the
individual.

**Short and long gaps.** Both audits again, once with the two rows of
a pair in the shorter half of the gap range and once in the longer
half. If identity stays high at the long gap, units stay recognizable
as time passes.

**Columns.** For each column: the audit using that column alone, and
how much the full score drops when that column is removed.

**Overlap check.** Needs `event`. It scores the same test rows twice
with the same number of training rows per class: once with the test
units held out of training, once with other rows of those units
swapped in. The gap between the two scores is how much sharing units
between training and test is worth on your task.

**Chronological control.** The same check, testing the later half of
each test unit's rows and letting training see the earlier half: does
a unit's own past help predict its own future?

**Time-cut split.** The same check with a single cut in the time
column: test rows come after it, training rows before it. When the
time column is a shared calendar, this is the usual split by date, and
the gap shows how much unit overlap such a split leaves in place.

**Clock baseline.** Many event tasks can be scored well above chance
from one number: how long the unit has been observed. This fits the
same classifier on the clock alone, on your columns alone, and on
both, with units held out, and reports what the columns add once the
clock is known.

## What the numbers do not mean

An audit AUC above 0.5 says the signal is present in the columns. It
does not say your model uses it; the overlap check tests that.

**An AUC near 0.5 does not certify an evaluation.** It says this
classifier, on these columns, found nothing. Leakage through the
label, through preprocessing fitted on test rows, or through columns
not in the table is outside what these checks see.

## When it applies

- many units (hundreds or more), each observed more than once;
- for the identity audit, several units observed at the same time
  values, since the different-unit pair is matched on time;
- with `horizon`, an event column flagged on the row where the event
  happens, not on every row of the unit.

It does not apply to tables with one row per example and no repeated
unit.

## Repeats and repeatability

`repeats=N` is one number: every check runs N times. Units fall into
N + 1 fixed groups, by a checksum of the unit identifier, and each run
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
with the same settings. The report names the classifier that ran.

Independent runs are spread across all cores. Set the `WHOWHEN_JOBS`
environment variable to use fewer.

## Command line

```
whowhen report rows.parquet --unit patient_id --time hour --event sepsis
whowhen demo
```

## Citing

The method is described in:

> Andrew Hendel. Auditing Identity and Timing Signals in Operational
> Logs. 2026. Under review.

See `CITATION.cff`.

## License

Apache License 2.0. See `LICENSE`.
