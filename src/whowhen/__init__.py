"""whowhen: audit panel data for identity and timing signal.

    import whowhen as ww
    panel = ww.Panel(df, unit="patient_id", time="hour", event="sepsis")
    print(ww.audit(panel, features=cols, offset=(6, 24)))

who   can a row be matched to the unit it came from?
when  can two rows of one unit be put in time order?

Both are measured as the AUC of a classifier on pairs built so that
0.5 means no signal. See the README for what the numbers do and do not
mean.
"""
from .audit import AuditResult, audit, compare
from .columns import rank_columns
from .overlap import OverlapResult, overlap_check
from .panel import Panel
from .scorers import Scorer, resolve as resolve_scorer

__version__ = "0.1.0"
__all__ = ["Panel", "audit", "compare", "rank_columns", "overlap_check",
           "AuditResult", "OverlapResult", "Scorer",
           "resolve_scorer", "__version__"]
