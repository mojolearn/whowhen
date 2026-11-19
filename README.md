# whowhen

Audit a dataset of repeated observations per unit (patients, machines,
loans, users, vessels) for two signals a model can exploit without
predicting anything:

- **who**: can a row be matched to the unit it came from?
- **when**: can two rows of one unit be put in time order?

If the answer to either is yes, a model evaluated on your data may be
scoring well by recognizing units or reading the clock. `whowhen`
measures both, shows which columns carry them, and checks whether
sharing units between training and test rows changes your score.

## Install

```
pip install git+https://github.com/mojolearn/whowhen
```

## Five-minute example

You need one table with a column naming the unit and a column giving
the time of each row.

```python
import pandas as pd
import whowhen as ww

df = pd.read_parquet("stays.parquet")
panel = ww.Panel(df, unit="patient_id", time="hour", event="sepsis")

cols = ["heart_rate", "lactate", "creatinine", "hours_since_admission"]
result = ww.audit(
    panel,
    features={"all": cols, "no_clock": cols[:3]},
    offset=(6, 24),          # the two rows of a pair are 6 to 24 hours apart
)
print(result)
```

`result.identity` and `result.timing` are tables with one row per
feature set: the AUC, a 95 percent interval, and the range over
repeats. `result.scorer` names the classifier that produced them.

To see the library work before pointing it at your own data, run it on
a synthetic panel where the right answers are known:

```
whowhen-demo
```

## What the numbers mean

Both audits train a classifier on pairs of rows from units with no
recorded event, built so that the two classes look the same unless the
signal is there.

| Audit | The pair | Label | AUC 0.5 means |
|---|---|---|---|
| identity | two rows a set time apart | same unit, or two different units seen at the same two times | rows carry nothing that says which unit produced them |
| timing | two rows of one unit a set time apart | which row is the later one | a unit's rows carry nothing that says when they were recorded |

Units are split into training and held-out sides before any pair is
drawn, so the AUC is measured on units the classifier never saw. The
interval comes from resampling held-out units, pooled over repeats.

An AUC above 0.5 says the signal is present in those columns. It does
not say your model uses it, and it does not say your evaluation is
inflated. `overlap_check` below tests that directly.

**An AUC near 0.5 does not certify an evaluation.** It says this
classifier, on these columns, at this offset, found nothing. Leakage
through the label, through preprocessing fitted on test rows, or
through columns you did not pass in is outside what these audits see.

## When it applies

Your data needs:

- many units (hundreds or more), each observed more than once;
- a time for every row;
- units whose rows are at least the minimum offset apart;
- for the identity audit, several units observed at the same time
  values, since the different-unit pair is matched on time.

It does not apply to tables with one row per example and no repeated
unit.

## Which columns carry the signal

```python
tables = ww.rank_columns(panel, cols, offset=(6, 24))
tables["identity"]     # one row per column, strongest first
tables["timing"]
```

For each column: the audit using that column alone, and how much the
full score drops when that column is removed.

## Individual or type?

By default the different-unit row in an identity pair comes from any
other unit. A high score can then mean "these two rows are from
different kinds of unit" (a tug and a tanker) instead of "this is the
same individual".

```python
ww.audit(panel, cols, offset=("1D", "7D"),
         impostors=("random", "nearest"))
```

With `nearest`, the other unit is drawn from the units most similar to
the anchor's unit (by per-unit feature averages). If the score stays
high, individual units are recognizable. If it falls toward 0.5, the
columns identify the kind of unit, not the individual. If you already
have a type column, `match="vessel_type"` restricts every pair to one
type.

## Comparing choices made before modeling

Filling gaps (carry-forward, imputation) can make units easier to
recognize. `whowhen` does not impute. Build each version as its own
columns and compare them on the same pairs:

```python
ww.compare(panel,
           variants={"raw": raw_cols, "carried_forward": locf_cols},
           offset=(6, 24))
```

## Does unit overlap change my score?

```python
r = ww.overlap_check(panel, features=cols, replicates=200)
print(r)
```

This needs an event column. It scores the same test rows twice with
the same number of training rows per class: once with the test units
held out of training, once with other rows of those units swapped in.
The gap between the two scores, with its range over replicates, is how
much sharing units is worth on your task.

## The scorer

The classifier is pluggable. Anything with `fit` and `predict_proba`
works:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
ww.audit(panel, cols, offset=(6, 24),
         scorer=HistGradientBoostingClassifier())
```

The default is [mojolearn](https://github.com/mojolearn/mojolearn)'s
`GradientBoosting` with oblivious (symmetric) trees. mojolearn ships
wheels for macOS on Apple silicon and Linux on x86-64; on other
platforms, including Windows, the default is LightGBM, installed
automatically. Pass `scorer="lightgbm"` or `scorer="mojolearn"` to
choose. Every result records the scorer and its parameters, and scores
from different scorers should not be compared with each other.

## Command line

```
whowhen audit --panel rows.parquet --config audit.json --out result.json
whowhen demo
```

`whowhen audit --help` shows the declaration format.

## Parallelism

Repeats, audits, columns and replicates are independent and run across
all cores. Set `n_jobs=` or the `WHOWHEN_JOBS` environment variable to
use fewer.

## Citing

The method is described in:

> Andrew Hendel. Auditing Identity and Timing Signals in Operational
> Logs. 2026. Under review.

See `CITATION.cff`.

## License

MIT.
