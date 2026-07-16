"""perturb-poker-2 — PT2Bag: perturbation-bag bot detector.

Family: perturbation ensembles. Instead of a few heterogeneous members, this
model bags MANY perturbed replicas of a strong core learner (different seeds,
feature subsamples and row subsamples), then blends replica groups by rank.
Perturbation bagging trades single-model sharpness for stability across the
daily distribution shift of live validator traffic.
Trim mode (False): with >=4 replicas, the per-chunk best and worst replica
ranks are dropped before averaging (outlier-robust).
"""
import numpy as np


class PT2Bag:
    def __init__(self, groups, cols_base, cols_wide, trim=False):
        # groups: list of (name, [fitted estimators], view_key, weight); view_key: base|wide
        self.groups = list(groups)
        self.cols_base = list(cols_base)
        self.cols_wide = list(cols_wide)
        self.trim = bool(trim)

    @staticmethod
    def _rank01(s):
        s = np.asarray(s, dtype=float)
        if s.size <= 1:
            return np.zeros_like(s)
        return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)

    def _group_score(self, ests, X):
        P = np.vstack([e.predict_proba(X)[:, 1] for e in ests])
        if self.trim and P.shape[0] >= 4:
            R = np.vstack([self._rank01(p) for p in P])
            R.sort(axis=0)
            return R[1:-1].mean(axis=0)
        return self._rank01(P.mean(axis=0))

    def score(self, Xbase, Xwide):
        views = {"base": np.asarray(Xbase, float), "wide": np.asarray(Xwide, float)}
        out = np.zeros(views["base"].shape[0], dtype=float)
        total = 0.0
        for name, ests, view_key, w in self.groups:
            out += float(w) * self._group_score(ests, views[view_key])
            total += float(w)
        return out / max(total, 1e-9)
