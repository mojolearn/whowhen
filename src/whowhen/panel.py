"""The input: one table with a unit column and a time column."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class Panel:
    """Repeated observations per unit.

    Parameters
    ----------
    df : DataFrame with one row per (unit, time) observation.
    unit : column naming the unit (patient, machine, loan, user, vessel).
    time : column giving when the row was observed. Datetime or numeric.
    event : optional column; a unit with any nonzero value is left out of
        the audits, so every audited row comes from a unit with no
        recorded event. The overlap check uses it as the label.
    """

    def __init__(self, df: pd.DataFrame, unit: str, time: str,
                 event: str | None = None):
        for c in (unit, time, event):
            if c is not None and c not in df.columns:
                raise KeyError(f"column {c!r} is not in the table")
        t = df[time]
        if not (pd.api.types.is_datetime64_any_dtype(t)
                or pd.api.types.is_numeric_dtype(t)):
            df = df.assign(**{time: pd.to_datetime(t)})
        n_before = len(df)
        df = (df.drop_duplicates([unit, time])
                .sort_values([unit, time], kind="stable")
                .reset_index(drop=True))
        self.df = df
        self.unit, self.time, self.event = unit, time, event
        self.n_duplicates_dropped = n_before - len(df)
        self.datetime = pd.api.types.is_datetime64_any_dtype(df[time])

    def __repr__(self) -> str:
        return (f"Panel({len(self.df):,} rows, "
                f"{self.df[self.unit].nunique():,} units, "
                f"unit={self.unit!r}, time={self.time!r}, "
                f"event={self.event!r})")

    def offset(self, value) -> float:
        """An offset in the time column's own scale."""
        if self.datetime:
            if isinstance(value, (int, float)):
                raise TypeError(
                    f"time column {self.time!r} is a datetime, so offsets "
                    f"need a unit, for example '6h' or '30D'; got {value!r}")
            return float(pd.Timedelta(value).value)
        return float(value)

    def default_offset(self):
        """A pair offset chosen from the data, for when none is given:
        from the typical gap between a unit's consecutive rows up to half
        the typical span of a unit."""
        d = self.df
        step = d[self.time].diff()[d[self.unit].eq(d[self.unit].shift())]
        g = d.groupby(self.unit, sort=False)[self.time]
        span = (g.max() - g.min()).median() / 2
        lo = step.median()
        if pd.isna(lo) or pd.isna(span):
            raise ValueError("units have one row each; nothing to pair")
        return lo, max(lo, span)

    def frame(self, columns: list[str], match: str | None = None,
              keep_event_units: bool = False) -> "Frame":
        """Plain arrays for the audits. Rows stay sorted by (unit, time)."""
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise KeyError(f"feature columns not in the table: {missing}")
        df = self.df
        n_units_all = df[self.unit].nunique()
        if self.event and not keep_event_units:
            bad = df.loc[df[self.event].fillna(0) != 0, self.unit].unique()
            df = df[~df[self.unit].isin(bad)]
        unit, labels = pd.factorize(df[self.unit], sort=False)
        t = df[self.time]
        time = (t.astype("datetime64[ns]").astype("int64").to_numpy(float)
                if self.datetime else t.to_numpy(float))
        X = np.empty((len(df), len(columns)), dtype=np.float32)
        for j, c in enumerate(columns):
            col = df[c]
            if not pd.api.types.is_numeric_dtype(col):
                codes = pd.factorize(col)[0].astype(np.float32)
                codes[codes < 0] = np.nan
                X[:, j] = codes
            else:
                X[:, j] = col.to_numpy(np.float32, na_value=np.nan)
        cohort = None
        if match:
            first = df.groupby(self.unit, sort=False)[match].first()
            cohort = pd.factorize(first.reindex(labels))[0].astype(np.int64)
        y = None
        if self.event:
            y = (df[self.event].fillna(0).to_numpy() != 0).astype(np.int8)
        starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1], True])
        return Frame(unit=unit.astype(np.int64), time=time, X=X,
                     columns=list(columns), starts=starts,
                     unit_labels=np.asarray(labels), cohort=cohort, y=y,
                     n_units_all=int(n_units_all))


@dataclass
class Frame:
    unit: np.ndarray          # int code per row; rows of a unit contiguous
    time: np.ndarray          # float per row, ascending within a unit
    X: np.ndarray             # float32, rows x columns
    columns: list
    starts: np.ndarray        # unit u owns rows starts[u]:starts[u + 1]
    unit_labels: np.ndarray
    cohort: np.ndarray | None  # per-unit code of the match column
    y: np.ndarray | None       # per-row event flag
    n_units_all: int

    @property
    def n_units(self) -> int:
        return len(self.starts) - 1

    def cols(self, names: list[str]) -> list[int]:
        pos = {c: i for i, c in enumerate(self.columns)}
        return [pos[c] for c in names]
