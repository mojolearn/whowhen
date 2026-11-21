"""Pair construction for the two pair tests.

time order: two rows of one unit with no recorded event, a set offset
    apart; the label says which is the later one. If a unit's rows carry
    no information about their order, no classifier can beat AUC 0.5.
re-identification: two rows of one unit a set offset apart (label 1),
    or the first of them with a row of a matched control unit (label 0).
    If rows carry no information about which unit produced them, no
    classifier can beat AUC 0.5.

Without a time column only the re-identification test can run, and its
controls are matched on event status and group columns only.

All functions take and return row positions into a ``Frame``.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .panel import Frame



def unit_groups(frame: Frame, n_groups: int) -> np.ndarray:
    """A fixed group, 0 to n_groups - 1, for every unit.

    The group comes from a hash of the unit's own identifier, so it
    does not depend on a seed, on row order, or on which other units
    are in the table, and it is unrelated to anything the identifier's
    order might track (time of entry, site). The hash is BLAKE2, not a
    CRC: a CRC is linear, and identifiers that share a structure (a
    check digit, for example) can all land in one group.
    """
    def group(u) -> int:
        h = hashlib.blake2b(str(u).encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") % n_groups
    return np.fromiter((group(u) for u in frame.unit_labels), np.int64,
                       len(frame.unit_labels))


def split_units(frame: Frame, k: int, repeats: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Boolean per unit: True = training side.

    With ``repeats`` repeats the units fall into ``repeats + 1`` fixed
    groups and repeat ``k`` holds out group ``k``: one repeat holds out
    a half, two hold out a third each, four hold out a fifth each. No
    unit is held out twice, and units never straddle the split.

    With ``rng`` the held-out units are instead a random draw of the
    same share, fresh for each repeat.
    """
    if rng is not None:
        return rng.random(frame.n_units) >= 1.0 / (repeats + 1)
    return unit_groups(frame, repeats + 1) != k


def _within_unit(frame: Frame, units: np.ndarray, lo: float, hi: float,
                 n_pairs: int, max_per_unit: int,
                 rng: np.random.Generator) -> np.ndarray:
    """(first row, second row) of one unit, lo <= time gap <= hi.
    Without a time column: any two rows of one unit, in either order."""
    timed = frame.time is not None
    if timed:
        span = (frame.time[frame.starts[units + 1] - 1]
                - frame.time[frame.starts[units]])
        eligible = units[span >= lo].copy()
    else:
        eligible = units[frame.starts[units + 1] - frame.starts[units]
                         >= 2].copy()
    rng.shuffle(eligible)
    out = []
    for _ in range(max_per_unit):
        for u in eligible:
            if len(out) >= n_pairs:
                break
            s, e = frame.starts[u], frame.starts[u + 1]
            if not timed:
                i, j = rng.choice(e - s, 2, replace=False)
                out.append((s + int(i), s + int(j)))
                continue
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
    """Rows of one side, indexed by matching group and time, for control
    draws."""

    def __init__(self, frame: Frame, rows: np.ndarray):
        self.frame = frame
        g = (frame.stratum[frame.unit[rows]] if frame.stratum is not None
             else np.zeros(len(rows), np.int64))
        t = (frame.time[rows] if frame.time is not None
             else np.zeros(len(rows)))
        order = np.lexsort((t, g))
        self.rows, self.g, self.t = rows[order], g[order], t[order]

    def pick(self, row: int, unit: int, rng: np.random.Generator) -> int:
        """A random row of another unit in ``unit``'s matching group,
        observed at ``row``'s time (any time, without a time column);
        -1 if there is none."""
        f = self.frame
        g = f.stratum[unit] if f.stratum is not None else 0
        i0 = np.searchsorted(self.g, g, "left")
        i1 = np.searchsorted(self.g, g, "right")
        if f.time is not None:
            t = f.time[row]
            i0, i1 = (i0 + np.searchsorted(self.t[i0:i1], t, "left"),
                      i0 + np.searchsorted(self.t[i0:i1], t, "right"))
        if i1 <= i0:
            return -1
        for _ in range(8):            # most draws land on another unit
            j = self.rows[rng.integers(i0, i1)]
            if f.unit[j] != unit:
                return int(j)
        cand = self.rows[i0:i1]
        cand = cand[f.unit[cand] != unit]
        return int(cand[rng.integers(0, len(cand))]) if len(cand) else -1


def time_order_pairs(frame: Frame, lo: float, hi: float, n_pairs: int,
                 max_per_unit: int, rng: np.random.Generator,
                 placebo: bool = False):
    """Rows and labels for the time-order test (label 1 = the later row).

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
            j = pool.pick(b, frame.unit[a], rng)
            if j >= 0:
                keep.append((a, j))
        pairs = np.asarray(keep, dtype=np.int64).reshape(-1, 2)
    rows = np.r_[pairs[:, 0], pairs[:, 1]]
    labels = np.r_[np.zeros(len(pairs), np.int8), np.ones(len(pairs), np.int8)]
    _, first = np.unique(np.c_[rows, labels], axis=0, return_index=True)
    first.sort()
    return rows[first], labels[first]


def reidentification_pairs(frame: Frame, lo: float, hi: float,
                           n_pairs: int, max_per_unit: int,
                           rng: np.random.Generator, train: np.ndarray):
    """Pairs for the re-identification test: two rows of one unit
    (label 1), and the first of them with a row of a control unit
    (label 0).

    The control is a random unit matched to the first on everything the
    table records about a unit other than who it is: observed at the
    second row's time (when there is a time column) and in the same
    matching group (``frame.stratum``: event status and group columns).
    A pair whose unit has no such control is left out whole, so both
    labels come from the same anchors.

    ``train`` is the unit split, drawn before any pair: anchors and
    controls come from one side, so every pair reaches the scorer whole.
    Returns (row_a, row_b, label, is_train).
    """
    n_train = int(round(n_pairs * train.mean()))
    out = []
    for side, n in ((True, n_train), (False, n_pairs - n_train)):
        units = np.flatnonzero(train == side)
        same = _within_unit(frame, units, lo, hi, n, max_per_unit, rng)
        pool = _Pool(frame, np.flatnonzero(train[frame.unit] == side))
        for a, b in same:
            j = pool.pick(b, frame.unit[a], rng)
            if j >= 0:
                out.append((a, b, 1, side))
                out.append((a, j, 0, side))
    arr = np.asarray(out, dtype=np.int64).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(np.int8), arr[:, 3] == 1


def pair_features(frame: Frame, row_a, row_b, cols: list[int]) -> np.ndarray:
    """What the re-identification scorer sees: both rows and their absolute
    difference."""
    xa = frame.X[np.ix_(row_a, cols)]
    xb = frame.X[np.ix_(row_b, cols)]
    return np.hstack([xa, xb, np.abs(xa - xb)])
