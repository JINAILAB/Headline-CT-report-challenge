from __future__ import annotations

from typing import Optional

import numpy as np


def auroc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    """rank 기반 AUROC(동점은 평균 rank). 한쪽 클래스만 있으면 None."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    rk = np.arange(1, len(p) + 1, dtype=float)
    j = 0
    while j < len(p):                                    # 동점 구간을 평균 rank로
        k = j
        while k + 1 < len(p) and sp[k + 1] == sp[j]:
            k += 1
        rk[j:k + 1] = (rk[j] + rk[k]) / 2
        j = k + 1
    ranks = np.empty(len(p))
    ranks[order] = rk
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return None
    best = 0.0
    for t in np.quantile(p, np.linspace(0.01, 0.99, 99)):
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        best = max(best, 2 * tp / (2 * tp + fp + fn + 1e-9))
    return best
