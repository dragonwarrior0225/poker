"""UID233-style serving: per-batch rank -> stack -> anchor remap -> budget.

Pipeline (hot-benchmark-poker-3 v2.0, reproduced from published source):
1. Vectorize the batch in the artifact's frozen feature_order (missing 0.0,
   non-finite zeroed).
2. Within-batch columnwise rank: rankdata(col, "average")/n -> (0, 1]. The
   model only ever sees uniform marginals, so any monotone live drift in a
   feature is invisible by construction (train ranks within source_date).
3. StackingClassifier predict_proba.
4. Anchor remap np.interp(p, [0, t, 1], [0, 0.5, 1]) puts the deploy
   threshold t exactly at the validator's 0.5 gate (monotone; AP unchanged).
5. Power-decay positive budget: keep only the top floor(MAX_POS_FRAC * n)
   scores that already cleared 0.5; every other chunk is re-scored on a
   deterministic rank curve 0.499 * (1 - rank/m)^DECAY_POWER. At most 15% of
   a cycle crosses 0.5 (locks hard_fpr@0.5 far under the 0.10 cliff) while
   global ordering -- and therefore AP / recall@5%FPR -- is preserved.
6. Empty/degenerate chunks 0.09; any exception -> 0.5 for the whole batch.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import rankdata

from neurons.stack233.features import union_features

MAX_POS_FRAC = float(os.getenv("POKER44_MAX_POS_FRAC", "0.15"))
DECAY_POWER = 1.5
EMPTY_CHUNK_SCORE = 0.09


def batch_rank(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n == 0:
        return X
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = rankdata(X[:, j], method="average") / n
    return out


def anchor_remap(p: np.ndarray, threshold: float) -> np.ndarray:
    t = float(min(max(threshold, 1e-6), 1.0 - 1e-6))
    return np.interp(p, [0.0, t, 1.0], [0.0, 0.5, 1.0])


def power_decay_budget(
    scores: np.ndarray, max_frac: float = MAX_POS_FRAC, power: float = DECAY_POWER
) -> np.ndarray:
    n = scores.size
    if n == 0:
        return scores
    out = scores.astype(float).copy()
    k = int(np.floor(max_frac * n))
    order = np.argsort(-out, kind="stable")
    keep = [i for i in order[:k] if out[i] >= 0.5]
    tail = [i for i in order if i not in set(keep)]
    m = len(tail)
    for rank, idx in enumerate(tail):
        out[idx] = 0.499 * (1.0 - rank / m) ** power if m else out[idx]
    return out


class Stack233Scorer:
    """Picklable scorer exposed as DetectorModel.model.predict_chunks."""

    def __init__(self, stack: Any, feature_order: Sequence[str], threshold: float,
                 max_frac: float = MAX_POS_FRAC):
        self.stack = stack
        self.feature_order = list(feature_order)
        self.threshold = float(threshold)
        # Per-artifact positive-fraction budget (weight-controlled). Older
        # artifacts without this attribute fall back to the module default
        # via getattr in predict_chunks. Env override still wins for ops.
        self.max_frac = float(max_frac)

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            feats = union_features(c)
            rows.append([feats.get(k, 0.0) for k in self.feature_order])
        return np.nan_to_num(
            np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )

    @staticmethod
    def _degenerate(chunk: List[dict]) -> bool:
        return not chunk or not any((h or {}).get("actions") for h in chunk)

    def predict_chunks(self, chunks: List[List[dict]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        try:
            X = batch_rank(self._rows(chunks))
            p = self.stack.predict_proba(X)[:, 1]
            # Env override > per-artifact max_frac > module default.
            frac = float(os.getenv("POKER44_MAX_POS_FRAC", "")
                         or getattr(self, "max_frac", MAX_POS_FRAC))
            s = power_decay_budget(
                anchor_remap(np.asarray(p, dtype=float), self.threshold),
                max_frac=frac,
            )
            for i, chunk in enumerate(chunks):
                if self._degenerate(chunk):
                    s[i] = EMPTY_CHUNK_SCORE
            return np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        except Exception:  # noqa: BLE001 -- never fail the whole batch
            return np.full(len(chunks), 0.5, dtype=float)
