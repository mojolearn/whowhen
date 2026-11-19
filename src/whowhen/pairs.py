"""Pair construction for the two audits.

Both audits draw pairs of rows from units with no recorded event.

timing    two rows of one unit, a set offset apart; the label says which
          is the later one. If a unit's rows carry no information about
          their order, no classifier can beat AUC 0.5.
identity  two rows a set offset apart, either from one unit (label 1) or
          from two units observed at the same two times (label 0). If
          rows carry no information about which unit produced them, no
          classifier can beat AUC 0.5.

All functions take and return row positions into a ``Frame``.
"""
from __future__ import annotations

import numpy as np

from .panel import Frame

NEAREST_POOL = 256   # candidates examined per anchor for nearest impostors
NEAREST_K = 5        # the impostor is drawn from this many nearest units


def split_units(frame: Frame, rng: np.random.Generator,
                train_frac: float = 0.8) -> np.ndarray:
    """Boolean per unit: True = training side. Units never straddle."""
    return rng.random(frame.n_units) < train_frac


def _within_unit(frame: Frame, units: np.ndarray, lo: float, hi: float,
                 n_pairs: int, max_per_unit: int,
                 rng: np.random.Generator) -> np.ndarray:
    """(first row, second row) of one unit, lo <= time gap <= hi."""
    span = frame.time[frame.starts[units + 1] - 1] - frame.time[frame.starts[units]]
    eligible = units[span >= lo].copy()
    rng.shuffle(eligible)
    out = []
    for _ in range(max_per_unit):
        for u in eligible:
            if len(out) >= n_pairs:
                break
            s, e = frame.starts[u], frame.starts[u + 1]
            ts = frame.time[s:e]
            i = int(rng.integers(0, e - s))
            j0 = int(np.searchsorted(ts, ts[i] + lo, "left"))
            j1 = int(np.searchsorted(ts, ts[i] + hi, "right"))
            j0 = max(j0, i + 1)
            if j1 <= j0:
                continue
            out.append((s + i, s + int(rng.integers(j0, j1))))
        if len(out) >= n_pairs:
            break
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.asarray(out, dtype=np.int64), axis=0)


class _Pool:
    """Rows of one side indexed by (cohort, time), for impostor draws."""

    def __init__(self, frame: Frame, rows: np.ndarray):
        tcode = np.unique(frame.time, return_inverse=True)[1]
        n_t = int(tcode.max()) + 1 if len(tcode) else 1
        self.n_t = n_t
        self.tcode = tcode
        c = frame.cohort[frame.unit[rows]] if frame.cohort is not None else 0
        key = c * n_t + tcode[rows]
        order = np.argsort(key, kind="stable")
        self.key, self.rows = key[order], rows[order]
        self.frame = frame

    def at(self, row: int, unit: int) -> np.ndarray:
        """Rows observed at ``row``'s time, in ``unit``'s cohort."""
        c = self.frame.cohort[unit] if self.frame.cohort is not None else 0
        k = c * self.n_t + self.tcode[row]
        i0 = np.searchsorted(self.key, k, "left")
        i1 = np.searchsorted(self.key, k, "right")
        return self.rows[i0:i1]


def unit_profiles(frame: Frame, units: np.ndarray) -> np.ndarray:
    """Per-unit feature means, standardized across ``units``. Rows of
    units outside ``units`` are left NaN."""
    P = np.full((frame.n_units, frame.X.shape[1]), np.nan)
    for u in units:
        block = frame.X[frame.starts[u]:frame.starts[u + 1]]
        seen = ~np.isnan(block)
        n = seen.sum(0)
        P[u] = np.where(n > 0, np.where(seen, block, 0).sum(0)
                        / np.maximum(n, 1), np.nan)
    mu = np.nanmean(P[units], axis=0) if len(units) else 0.0
    sd = np.nanstd(P[units], axis=0) if len(units) else 1.0
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    P = (P - np.where(np.isfinite(mu), mu, 0.0)) / sd
    return np.nan_to_num(P, nan=0.0)


def timing_pairs(frame: Frame, lo: float, hi: float, n_pairs: int,
                 max_per_unit: int, rng: np.random.Generator,
                 placebo: bool = False):
    """Rows and labels for the timing audit (label 1 = the later row).

    With ``placebo`` the later row comes from a different unit observed
    at the same time, so the pair carries no within-unit information and
    what remains is change shared by the whole population.
    """
    pairs = _within_unit(frame, np.arange(frame.n_units), lo, hi, n_pairs,
                         max_per_unit, rng)
    if placebo and len(pairs):
        pool = _Pool(frame, np.arange(len(frame.unit)))
        keep = []
        for a, b in pairs:
            u = frame.unit[a]
            cand = pool.at(b, u)
            cand = cand[frame.unit[cand] != u]
            if len(cand):
                keep.append((a, cand[rng.integers(0, len(cand))]))
        pairs = np.asarray(keep, dtype=np.int64).reshape(-1, 2)
    rows = np.r_[pairs[:, 0], pairs[:, 1]]
    labels = np.r_[np.zeros(len(pairs), np.int8), np.ones(len(pairs), np.int8)]
    _, first = np.unique(np.c_[rows, labels], axis=0, return_index=True)
    first.sort()
    return rows[first], labels[first]


def identity_pairs(frame: Frame, lo: float, hi: float, n_pairs: int,
                   max_per_unit: int, rng: np.random.Generator,
                   train: np.ndarray, impostors: str = "random"):
    """Pairs for the identity audit.

    ``train`` is the unit split, drawn before any pair: anchors and
    impostors come from one side, so every matched pair reaches the
    scorer whole. Returns (row_a, row_b, label, is_train).

    impostors="random"   the other unit is any unit observed at the
                         matched time (in the anchor's cohort, when a
                         match column is declared).
    impostors="nearest"  the other unit is one of the units most like
                         the anchor's unit, by standardized per-unit
                         feature means computed within the side. Such
                         pairs cannot be told apart by the kind of unit,
                         only by the individual.
    """
    if impostors not in ("random", "nearest"):
        raise ValueError("impostors is 'random' or 'nearest'")
    n_train = int(round(n_pairs * train.mean()))
    out = []
    for side, n in ((True, n_train), (False, n_pairs - n_train)):
        units = np.flatnonzero(train == side)
        same = _within_unit(frame, units, lo, hi, n, max_per_unit, rng)
        in_side = np.flatnonzero(train[frame.unit] == side)
        pool = _Pool(frame, in_side)
        prof = unit_profiles(frame, units) if impostors == "nearest" else None
        for a, b in same:
            u = frame.unit[a]
            out.append((a, b, 1, side))
            cand = pool.at(b, u)
            cand = cand[frame.unit[cand] != u]
            if not len(cand):
                continue
            if prof is None:
                j = cand[rng.integers(0, len(cand))]
            else:
                if len(cand) > NEAREST_POOL:
                    cand = rng.choice(cand, NEAREST_POOL, replace=False)
                d = ((prof[frame.unit[cand]] - prof[u]) ** 2).sum(1)
                near = cand[np.argsort(d, kind="stable")[:NEAREST_K]]
                j = near[rng.integers(0, len(near))]
            out.append((a, j, 0, side))
    arr = np.asarray(out, dtype=np.int64).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(np.int8), arr[:, 3] == 1


def pair_features(frame: Frame, row_a, row_b, cols: list[int]) -> np.ndarray:
    """What the identity scorer sees: both rows and their absolute gap."""
    xa = frame.X[np.ix_(row_a, cols)]
    xb = frame.X[np.ix_(row_b, cols)]
    return np.hstack([xa, xb, np.abs(xa - xb)])
