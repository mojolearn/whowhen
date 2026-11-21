"""The input: one table with a unit column and, usually, a time column."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP_MIN_UNITS = 10   # a group column has at least this many units per value


class Panel:
    """Repeated observations per unit.

    Parameters
    ----------
    df : DataFrame with one row per (unit, time) observation.
    unit : column naming the unit (patient, machine, loan, user, vessel).
    time : column giving when the row was observed. Datetime or numeric.
        Left out for a table whose rows are not ordered in time; the
        checks that need an order then cannot run.
    event : optional column, nonzero where the event happens. The
        time-order test leaves out units with any event. The
        re-identification test keeps them and matches each control on
        event status. Record-wise vs subject-wise validation uses it as
        the label.
    """

    def __init__(self, df: pd.DataFrame, unit: str, time: str | None = None,
                 event: str | None = None):
        for c in (unit, time, event):
            if c is not None and c not in df.columns:
                raise KeyError(f"column {c!r} is not in the table")
        n_before = len(df)
        if time is None:
            df = df.sort_values(unit, kind="stable").reset_index(drop=True)
        else:
            t = df[time]
            if not (pd.api.types.is_datetime64_any_dtype(t)
                    or pd.api.types.is_numeric_dtype(t)):
                df = df.assign(**{time: pd.to_datetime(t)})
            df = (df.drop_duplicates([unit, time])
                    .sort_values([unit, time], kind="stable")
                    .reset_index(drop=True))
        self.df = df
        self.unit, self.time, self.event = unit, time, event
        self.n_duplicates_dropped = n_before - len(df)
        self.datetime = (time is not None
                         and pd.api.types.is_datetime64_any_dtype(df[time]))
        self._groups = None

    def group_columns(self) -> list[str]:
        """Columns that record a group a unit belongs to (site, batch,
        scanner, model): the same value on every row of every unit, at
        least two values, and on average at least GROUP_MIN_UNITS units
        per value. Re-identification controls are matched on them."""
        if self._groups is None:
            d, u = self.df, self.unit
            n_units = d[u].nunique()
            cand = [c for c in d.columns if c not in (u, self.time,
                                                      self.event)]
            out = []
            if cand:
                per_unit = d.groupby(u, sort=False)[cand].nunique(dropna=False)
                for c in cand:
                    n = d[c].nunique(dropna=False)
                    if (per_unit[c].max() == 1 and n >= 2
                            and n * GROUP_MIN_UNITS <= n_units):
                        out.append(c)
            self._groups = out
        return self._groups

    def matched_on(self) -> list[str]:
        """What re-identification controls are matched on, in words."""
        return ((["time"] if self.time else [])
                + (["event status"] if self.event else [])
                + self.group_columns())

    def __repr__(self) -> str:
        return (f"Panel({len(self.df):,} rows, "
                f"{self.df[self.unit].nunique():,} units, "
                f"unit={self.unit!r}, time={self.time!r}, "
                f"event={self.event!r})")

    def offset(self, value) -> float:
        """An offset in the time column's own scale."""
        if self.time is None:
            raise ValueError("an offset or a horizon needs a time column")
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

    def frame(self, columns: list[str],
              keep_event_units: bool = False,
              match: bool = False) -> "Frame":
        """Plain arrays for the checks. Rows stay sorted by (unit, time),
        or by unit alone when there is no time column."""
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise KeyError(f"feature columns not in the table: {missing}")
        df = self.df
        n_units_all = df[self.unit].nunique()
        if self.event and not keep_event_units:
            bad = df.loc[df[self.event].fillna(0) != 0, self.unit].unique()
            df = df[~df[self.unit].isin(bad)]
        unit, labels = pd.factorize(df[self.unit], sort=False)
        time = None
        if self.time is not None:
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
        y = None
        if self.event:
            y = (df[self.event].fillna(0).to_numpy() != 0).astype(np.int8)
        starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1], True])
        stratum = None
        if match:
            keys = self.group_columns()
            per = df.groupby(self.unit, sort=False)
            per = (per[keys].first() if keys
                   else pd.DataFrame(index=per.size().index))
            if self.event:
                per["__event__"] = (df[self.event].fillna(0).ne(0)
                                    .groupby(df[self.unit], sort=False).any())
            per = per.reindex(labels)
            stratum = (per.groupby(list(per.columns), sort=False,
                                   dropna=False).ngroup().to_numpy(np.int64)
                       if len(per.columns) else
                       np.zeros(len(labels), np.int64))
        return Frame(unit=unit.astype(np.int64), time=time, X=X,
                     columns=list(columns), starts=starts,
                     unit_labels=np.asarray(labels), y=y,
                     n_units_all=int(n_units_all), stratum=stratum)


@dataclass
class Frame:
    unit: np.ndarray          # int code per row; rows of a unit contiguous
    time: np.ndarray | None   # float per row, ascending within a unit;
                              # None when the table has no time column
    X: np.ndarray             # float32, rows x columns
    columns: list
    starts: np.ndarray        # unit u owns rows starts[u]:starts[u + 1]
    unit_labels: np.ndarray
    y: np.ndarray | None       # per-row event flag
    n_units_all: int
    stratum: np.ndarray | None = None   # per unit: units a control may
                                        # come from share this value
    repeats: int = 3          # set by the caller; fixes the unit groups
    random: bool = False      # held-out units drawn at random instead

    @property
    def n_units(self) -> int:
        return len(self.starts) - 1

    def cols(self, names: list[str]) -> list[int]:
        pos = {c: i for i, c in enumerate(self.columns)}
        return [pos[c] for c in names]
