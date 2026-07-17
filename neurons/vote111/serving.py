"""UID111 (poker111-vote) style serving: soft-vote ET+RF+HGB + a strictly
monotone rank/logit decision layer.

Learner: ExtraTrees(n=700, max_depth=9) + RandomForest(n=700, max_depth=9)
+ HistGradientBoosting(max_iter=700, lr=0.03, max_depth=9), soft-vote
weights 0.45 / 0.25 / 0.30 (replicated from UID89 / UID111's published
recipe; trained on our own benchmark rows with our own seeds -- no external
model blob is ever loaded).

Decision layer: rank01 the vote, map to logit space, anchor the batch's
Q-quantile logit to a fixed training reference (with a margin), then force
exactly ceil(FLOOR * n) chunks above 0.5 and (optionally) push the rest
below 0.5 as a block. The map from vote-rank -> served score is strictly
monotone, so AP / recall@FPR are set purely by the model's ordering, while
FLOOR guarantees at least one positive call per batch -- this blocks the
validator's zero-TP threshold-sanity failure (reward -> 0.0) even if the
raw vote collapses under live distribution shift, and it is self-calibrating
per batch (anchored on the *live* batch's own quantile) rather than a fixed
global threshold learned only from the benchmark.

IMPORTANT: inference does NOT sanitize. Live chunks arrive already
sanitized by the validator (prepare_hand_for_miner runs validator-side).
Only offline training sanitizes raw benchmark hands (train == serve).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from neurons.vote111.features import FEATURE_NAMES, chunk_features

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

# Internal separation epsilon in logit space for the floor/cap block shift.
_T_HI = 0.0004
_T_LO = -0.0004


def _rank01(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _rows(chunks: Sequence[List[dict]]) -> np.ndarray:
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    # Parity with UID111 v9: non-finite feature cells are zeroed, never
    # propagated into the vote.
    return np.nan_to_num(
        np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )


def _pin_single_thread(est: Any) -> None:
    """n_jobs only covers joblib-parallel members (ET/RF); HistGradientBoosting
    is OpenMP-parallel and must be clamped via threadpool_limits at call time."""
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:  # noqa: BLE001
            pass


class VoteScorer:
    """Picklable scorer exposed as DetectorModel.model.predict_chunks."""

    def __init__(
        self,
        members: List[Tuple[str, Any, float]],
        *,
        q: float,
        margin: float,
        temp: float,
        floor: float,
        cap: bool,
        eps: float,
        train_ref_logit: float,
    ):
        self.members = list(members)
        for _name, model, _weight in self.members:
            _pin_single_thread(model)
        self.q = float(q)
        self.margin = float(margin)
        self.temp = float(temp)
        self.floor = float(floor)
        self.cap = bool(cap)
        self.eps = float(eps)
        self.train_ref_logit = float(train_ref_logit)

    def _vote(self, X: np.ndarray) -> np.ndarray:
        def _run() -> np.ndarray:
            p = np.zeros(len(X), dtype=float)
            wsum = 0.0
            for _name, model, weight in self.members:
                p += weight * model.predict_proba(X)[:, 1]
                wsum += weight
            return p / max(wsum, 1e-12)

        if threadpool_limits is None:
            return _run()
        with threadpool_limits(limits=1):
            return _run()

    def _decision(self, v: np.ndarray) -> np.ndarray:
        tref = self.train_ref_logit - self.margin
        z = _logit(v, self.eps)
        if z.size == 0:
            return np.zeros(0, dtype=float)
        anchor = float(np.quantile(z, self.q))
        t = (z - anchor + tref) / self.temp
        order = np.argsort(-z, kind="mergesort")
        k = max(1, int(np.ceil(self.floor * len(t))))
        top, rest = order[:k], order[k:]
        d = _T_HI - t[top].min()
        if d > 0.0:
            t[top] = t[top] + d
        if self.cap and rest.size:
            d = t[rest].max() - _T_LO
            if d > 0.0:
                t[rest] = t[rest] - d
        return 1.0 / (1.0 + np.exp(-t))

    def predict_chunks(self, chunks: List[List[dict]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        try:
            X = _rows(chunks)
            raw = self._vote(X)
            # Degenerate empty chunks are forced to the bottom of the ranking
            # BEFORE rank01/decision so the floor budget never spends a
            # positive slot on them.
            for i, chunk in enumerate(chunks):
                if not chunk or not any((h or {}).get("actions") for h in chunk):
                    raw[i] = -1.0
            scores = self._decision(_rank01(raw))
        except Exception:  # noqa: BLE001 -- never fail the whole batch
            return np.full(len(chunks), 0.5, dtype=float)
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
