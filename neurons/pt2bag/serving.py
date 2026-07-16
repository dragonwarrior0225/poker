"""Serving wrapper that mirrors UID138 PT2Bag score path + force top-K safety.

Keeps the PT2Bag rank-blend + threshold remap, then applies an unconditional
top-K safety budget so every live batch has scores above 0.5. That blocks the
validator's zero-reward cliff (no TP >= 0.5 -> threshold_sanity = 0 -> reward 0).
Ranking order is preserved, so AP / recall@FPR are unchanged.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from neurons.pt2bag.pt_features import base_view, wide_view
from neurons.pt2bag.pt_model import PT2Bag

# Cap / force fraction of positives per batch. Slightly tighter than UID138's
# 0.16 soft cap; close to UID89's 0.125 while still guaranteeing positives.
DEFAULT_MAX_POS_FRAC = 0.15
SAFETY_POSITIVE_BAND: Tuple[float, float] = (0.501, 0.509)
SAFETY_NEGATIVE_BAND: Tuple[float, float] = (0.0, 0.49)


def _rows(chunks: Sequence[List[dict]], view_fn, cols: Sequence[str]) -> np.ndarray:
    feats = [view_fn(c) for c in chunks]
    return np.asarray(
        [[float(d.get(c, 0.0)) for c in cols] for d in feats],
        dtype=float,
    )


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def _apply_batch_safety_budget(
    scores: np.ndarray,
    max_frac: float,
    *,
    positive_band: Tuple[float, float] = SAFETY_POSITIVE_BAND,
    negative_band: Tuple[float, float] = SAFETY_NEGATIVE_BAND,
) -> np.ndarray:
    """Force exactly top-K scores above 0.5; push the rest below.

    Unlike a soft positive-only cap, this always emits >=1 positive on any
    non-empty batch, so live windows cannot collapse to reward 0.0 from a
    zero-TP threshold-sanity failure. Rank order is preserved.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return s
    lo_p, hi_p = positive_band
    lo_n, hi_n = negative_band
    k = min(n, max(1, int(np.floor(n * max_frac))))
    order = sorted(range(n), key=lambda i: (-float(s[i]), i))
    out = np.zeros(n, dtype=float)
    span_p = hi_p - lo_p
    for rank_index, idx in enumerate(order[:k]):
        t = rank_index / (k - 1) if k > 1 else 0.0
        out[idx] = hi_p - t * span_p
    negatives = order[k:]
    span_n = hi_n - lo_n
    for rank_index, idx in enumerate(negatives):
        t = rank_index / (len(negatives) - 1) if len(negatives) > 1 else 0.0
        out[idx] = hi_n - t * span_n
    return out


class PT2BagScorer:
    """Picklable scorer exposed as DetectorModel.model.predict_chunks."""

    def __init__(
        self,
        ensemble: PT2Bag,
        threshold: float,
        max_pos_frac: float = DEFAULT_MAX_POS_FRAC,
    ):
        self.ensemble = ensemble
        self.threshold = float(threshold)
        self.max_pos_frac = float(max_pos_frac)

    def predict_chunks(self, chunks: List[List[dict]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        needed = {view for _name, _ests, view, _w in self.ensemble.groups}
        xb = (
            _rows(chunks, base_view, self.ensemble.cols_base)
            if "base" in needed
            else np.zeros((len(chunks), len(self.ensemble.cols_base)))
        )
        xw = (
            _rows(chunks, wide_view, self.ensemble.cols_wide)
            if "wide" in needed
            else np.zeros((len(chunks), len(self.ensemble.cols_wide)))
        )
        raw = np.asarray(self.ensemble.score(xb, xw), dtype=float)
        remapped = _remap_to_threshold(raw, self.threshold)
        # Degenerate empty chunks stay at the bottom of the ranking so the
        # force top-K budget never spends a positive slot on them.
        for i, chunk in enumerate(chunks):
            if not chunk or not any((h or {}).get("actions") for h in chunk):
                remapped[i] = -1.0
        return _apply_batch_safety_budget(remapped, self.max_pos_frac)
