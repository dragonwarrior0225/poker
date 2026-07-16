#!/usr/bin/env python3
"""Train UID111-style vote detector and write neurons/models/detector.joblib.

Mirrors poker111-vote: ExtraTrees(700,d9) + RandomForest(700,d9) +
HistGradientBoosting(700,lr.03,d9) soft-vote (weights 0.45/0.25/0.30) over
180 scale-invariant behavioral features (neurons/vote111/features.py),
followed by a strictly-monotone rank/logit decision layer (Q/MARGIN/TEMP/
FLOOR/CAP) that guarantees a floor of positive calls per served batch.

Trained directly on real sanitized benchmark groups (train == serve
semantics; no live-size augmentation -- matches UID111's published n_train).

Usage:
  .venv/bin/python scripts/miner/train_vote111.py benchmark_cache
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.detector import MODEL_PATH  # noqa: E402
from neurons.vote111.features import FEATURE_NAMES, chunk_features  # noqa: E402
from neurons.vote111.serving import VoteScorer, _logit, _rank01  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 45077
NJ = 4
WF = 3
TARGET_FPR = 0.045
MAX_WF_FPR = 0.06

# Decision-layer hyperparameters, matching UID111's published artifact.
Q = 0.7
MARGIN = 3.0
TEMP = 1.0
FLOOR = 0.10
CAP = True
EPS = 1e-4


def sanitize(hands):
    out = []
    for h in hands:
        try:
            out.append(prepare_hand_for_miner(h))
        except Exception:  # noqa: BLE001
            out.append(h)
    return out


def load_examples(cache_dir: Path):
    chunks, labels, dates = [], [], []
    for path in sorted(cache_dir.glob("*.json")):
        date = path.stem
        for record in json.loads(path.read_text()):
            for group, label in zip(record["chunks"], record["groundTruth"]):
                chunks.append(sanitize(group))
                labels.append(int(label))
                dates.append(date)
    return chunks, np.asarray(labels, dtype=int), np.asarray(dates)


def mat(chunks) -> np.ndarray:
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def build_members():
    return [
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=700, max_depth=9, n_jobs=NJ, random_state=SEED
            ),
            0.45,
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=700, max_depth=9, n_jobs=NJ, random_state=SEED + 1
            ),
            0.25,
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=700, learning_rate=0.03, max_depth=9, random_state=SEED + 2
            ),
            0.30,
        ),
    ]


def fit_members(X: np.ndarray, y: np.ndarray):
    members = build_members()
    fitted = []
    for name, model, weight in members:
        model.fit(X, y)
        fitted.append((name, model, weight))
    return fitted


def vote(fitted, X: np.ndarray) -> np.ndarray:
    p = np.zeros(len(X), dtype=float)
    wsum = 0.0
    for _name, model, weight in fitted:
        p += weight * model.predict_proba(X)[:, 1]
        wsum += weight
    return p / max(wsum, 1e-12)


def decision(v: np.ndarray, train_ref_logit: float) -> np.ndarray:
    tref = train_ref_logit - MARGIN
    z = _logit(v, EPS)
    if z.size == 0:
        return np.zeros(0, dtype=float)
    anchor = float(np.quantile(z, Q))
    t = (z - anchor + tref) / TEMP
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(FLOOR * len(t))))
    top, rest = order[:k], order[k:]
    d = 0.0004 - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if CAP and rest.size:
        d = t[rest].max() - (-0.0004)
        if d > 0.0:
            t[rest] = t[rest] - d
    return 1.0 / (1.0 + np.exp(-t))


def main() -> int:
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    if not cache_dir.is_dir():
        print(f"cache dir missing: {cache_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    if not chunks:
        print("no training data found", file=sys.stderr)
        return 2

    X = mat(chunks)
    ud = sorted(set(dates.tolist()))
    print(
        f"vote111: {len(y)} real groups | features={X.shape[1]} | {len(ud)} dates "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # Walk-forward OOF: fit on strictly-past dates, score the held-out date.
    oof = np.full(len(y), np.nan)
    for td in ud[-WF:]:
        tr = dates < td
        te = dates == td
        if tr.sum() < 60 or len(set(y[tr].tolist())) < 2 or not te.any():
            continue
        fitted = fit_members(X[tr], y[tr])
        oof[te] = vote(fitted, X[te])
        print(f"  wf {td} ({time.time() - t0:.0f}s)", flush=True)

    m = ~np.isnan(oof)
    if not m.any():
        print("walk-forward produced no scores; refusing to deploy", file=sys.stderr)
        return 3

    cv_ap = float(average_precision_score(y[m], oof[m]))

    # Reference anchor: the Q-quantile of the OOF pool's rank/logit, used as
    # the fixed training reference the live per-batch quantile is aligned to.
    z_oof = _logit(_rank01(oof[m]), EPS)
    anchor_ref = float(np.quantile(z_oof, Q))
    train_ref_logit = anchor_ref + MARGIN

    # Evaluate like live serve: per validator-window (one benchmark date =
    # one served batch), apply the SAME batch-adaptive decision layer.
    wf_dates = sorted({dates[i] for i in np.flatnonzero(m)})
    per_date = []
    served_all = np.full(len(y), np.nan)
    for td in wf_dates:
        dm = m & (dates == td)
        served = decision(_rank01(oof[dm]), train_ref_logit)
        served_all[dm] = served
        rew_d, res_d = reward(served, y[dm])
        per_date.append(
            {
                "date": td,
                "reward": float(rew_d),
                "ap": float(res_d["ap_score"]),
                "hard_fpr": float(res_d["hard_fpr"]),
                "sanity": float(res_d["threshold_sanity_quality"]),
                "n_pos": int(np.sum(served >= 0.5)),
                "n": int(dm.sum()),
            }
        )
        print(
            f"  wf-serve {td}: reward={rew_d:.4f} ap={res_d['ap_score']:.4f} "
            f"hard_fpr={res_d['hard_fpr']:.4f} sanity={res_d['threshold_sanity_quality']:.3f} "
            f"pos={int(np.sum(served >= 0.5))}/{int(dm.sum())}",
            flush=True,
        )

    rew = float(np.mean([row["reward"] for row in per_date]))
    res = reward(served_all[m], y[m])[1]
    zero_sanity = [row["date"] for row in per_date if row["sanity"] <= 0.0]
    print(
        f"WALK-FORWARD[{WF}d]: cv_ap={cv_ap:.4f} mean_reward={rew:.4f} "
        f"recall@fpr={res['bot_recall']:.3f} fpr={res['fpr']:.4f} "
        f"hard_fpr={res['hard_fpr']:.4f} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    if zero_sanity:
        print(f"guard: zero threshold-sanity on {zero_sanity}; refusing deploy", file=sys.stderr)
        return 5

    if float(res["hard_fpr"]) >= MAX_WF_FPR:
        print(
            f"guard: hard_fpr {res['hard_fpr']:.4f} >= {MAX_WF_FPR}; refusing deploy",
            file=sys.stderr,
        )
        return 4

    # Final refit on all real groups, then fix the reference anchor from the
    # full-fit in-sample rank/logit at Q (same role as OOF above).
    fitted = fit_members(X, y)
    full_scores = vote(fitted, X)
    z_full = _logit(_rank01(full_scores), EPS)
    anchor_ref_full = float(np.quantile(z_full, Q))
    train_ref_logit_final = anchor_ref_full + MARGIN

    scorer = VoteScorer(
        fitted,
        q=Q,
        margin=MARGIN,
        temp=TEMP,
        floor=FLOOR,
        cap=CAP,
        eps=EPS,
        train_ref_logit=train_ref_logit_final,
    )

    meta = {
        "algorithm": "VoteRankLogit",
        "model_name": "poker111-vote",
        "feature_version": "v5-sani-c2",
        "framework": "sklearn-ensemble",
        "recipe": "ET700d9 (.45) + RF700d9 (.25) + HGB700lr.03d9 (.30) soft-vote",
        "provenance": (
            "recipe reproduced from UID111 (poker111-vote) / UID89's published "
            "artifact metadata; trained on our own benchmark groups with our "
            "own seeds -- no external model blob loaded or copied."
        ),
        "decision_layer": {
            "kind": "rank_logit_quantile_anchor_floor_cap_v1",
            "q": Q,
            "margin": MARGIN,
            "temp": TEMP,
            "floor": FLOOR,
            "cap": CAP,
            "eps": EPS,
            "train_ref_logit": train_ref_logit_final,
        },
        "safety_budget": "quantile_anchor_floor_v1",
        "max_pos_frac": FLOOR,
        "seed": SEED,
        "cv_ap": cv_ap,
        "cv_reward": float(rew),
        "cv_recall": float(res["bot_recall"]),
        "cv_fpr": float(res["fpr"]),
        "cv_hard_fpr": float(res["hard_fpr"]),
        "validation": f"walk-forward over the last {WF} dates (per-date quantile-anchor serve)",
        "wf_per_date": per_date,
        "reward_formula": "official poker44.score.scoring.reward",
        "n_train_real": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_dates": int(len(ud)),
        "train_dates": ud,
        "trained_through": ud[-1],
        "regime_start": ud[0],
        "benchmark_releases": ud,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_source": "UID111 poker111-vote / VoteRankLogit",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".joblib.tmp")
    # threshold kept for backward-compat with generic DetectorModel metadata
    # readers; the real decision boundary is the per-batch quantile anchor.
    joblib.dump({"model": scorer, "threshold": 0.5, "metadata": meta}, tmp)
    tmp.replace(MODEL_PATH)
    print(
        f"saved {MODEL_PATH} | cv_ap={cv_ap:.4f} cv_reward={rew:.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
