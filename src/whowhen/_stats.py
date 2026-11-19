"""AUC and cluster-bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        raise ValueError(
            "the held-out pairs contain one class only; the panel has too "
            "few eligible units for this offset range")
    return float(roc_auc_score(y, scores))


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, roles: list[np.ndarray],
                  n_units: int, n_boot: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Unit-cluster bootstrap of the AUC.

    Units are resampled with replacement. Each row's weight is the
    product, over ``roles``, of the multiplicity of the unit in that
    role: one role for single-record rows, two for pairs, so a unit
    reused across many pairs widens the interval as it should.
    Replicates are computed in batches from one sort of the scores.
    """
    order = np.argsort(scores, kind="stable")
    ys = y[order].astype(float)
    ss = scores[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    roles = [r[order] for r in roles]
    out = []
    batch = max(1, min(n_boot, int(4e6 // max(len(ys), 1))))
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        m = np.stack([np.bincount(rng.integers(0, n_units, n_units),
                                  minlength=n_units) for _ in range(b)])
        w = np.ones((b, len(ys)))
        for r in roles:
            w *= m[:, r]
        gp = np.add.reduceat(w * ys, starts, axis=1)
        gn = np.add.reduceat(w * (1.0 - ys), starts, axis=1)
        below = np.cumsum(gn, axis=1) - gn
        den = gp.sum(1) * gn.sum(1)
        ok = den > 0
        out.append(((gp * (below + 0.5 * gn)).sum(1)[ok]) / den[ok])
        done += b
    return np.concatenate(out)
