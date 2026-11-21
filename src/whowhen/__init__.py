"""whowhen: check a table of repeated observations for re-identification and time-order signal.

    import whowhen as ww
    print(ww.report(df, unit="patient_id", time="hour", event="sepsis"))

who   can a row be matched to the unit it came from?
when  can two rows of one unit be put in time order?

Both are measured as the AUC of a classifier on pairs built so that
0.5 means no signal. See the README for what the numbers do and do not
mean.
"""
__version__ = "0.3.1"
__doi__ = "10.5281/zenodo.24406266"   # every version of whowhen

from .report import Report, report  # noqa: E402
__all__ = ["report", "Report", "__version__"]
