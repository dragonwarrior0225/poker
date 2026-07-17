#!/usr/bin/env python3
"""Train the UID233-style stacked detector -> neurons/models/detector233.joblib.

Method reproduced from hot-benchmark-poker-3 v2.0's published source (trained
on our own benchmark snapshots with our own seeds; no external weights):
- 543-feature union (293 ph_ schema + 250 v2_), sanitize-then-featurize
  (prepare_hand_for_miner; train == serve).
- Features rank-transformed WITHIN each source_date at train time, mirroring
  the per-request batch rank at serve time.
- StackingClassifier: ExtraTrees(500, balanced_subsample, sqrt) +
  [QuantileTransformer(160) -> PCA(48) -> MLP(64, alpha=1.2)] -> LogisticRegression
  meta learned on cv=3 out-of-fold base probabilities.
- Deploy threshold = quantile(1 - target_fpr) of TRAINING HUMANS' nested-OOF
  scores, target_fpr scanned over {0.02, 0.03, 0.04, 0.05}, picked by the
  official validator reward on walk-forward serve simulation.
- Serving: anchor remap (threshold -> 0.5) + 15% power-decay budget
  (neurons/stack233/serving.py).

Usage: .venv/bin/python scripts/miner/train_stack233.py benchmark_cache
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.stack233.features import union_features  # noqa: E402
from neurons.stack233.serving import (  # noqa: E402
    Stack233Scorer,
    anchor_remap,
    batch_rank,
    power_decay_budget,
)
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 45233
WF = 3
TARGET_FPR_GRID = (0.02, 0.03, 0.04, 0.05)
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"


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


def featurize(chunks):
    dicts = [union_features(c) for c in chunks]
    order = sorted(dicts[0].keys())
    X = np.asarray(
        [[d.get(k, 0.0) for k in order] for d in dicts], dtype=float
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), order


def rank_within_dates(X, dates):
    out = np.empty_like(X, dtype=float)
    for d in np.unique(dates):
        m = dates == d
        out[m] = batch_rank(X[m])
    return out


def build_stack(seed: int):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    return StackingClassifier(
        estimators=[
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=4,
                ),
            ),
            (
                "net",
                make_pipeline(
                    QuantileTransformer(
                        n_quantiles=160, output_distribution="uniform",
                        random_state=seed + 1,
                    ),
                    PCA(n_components=48, random_state=seed + 2),
                    MLPClassifier(
                        hidden_layer_sizes=(64,), alpha=1.2, max_iter=400,
                        early_stopping=True, random_state=seed + 3,
                    ),
                ),
            ),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,
    )


def nested_oof(Xr, y, dates, train_mask, seed: int):
    """Inner GroupKFold OOF probabilities over the training dates."""
    from sklearn.model_selection import GroupKFold, cross_val_predict

    idx = np.flatnonzero(train_mask)
    groups = dates[idx]
    n_splits = min(5, len(np.unique(groups)))
    stack = build_stack(seed)
    oof = cross_val_predict(
        stack,
        Xr[idx],
        y[idx],
        groups=groups,
        cv=GroupKFold(n_splits=n_splits),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    full = np.full(len(y), np.nan)
    full[idx] = oof
    return full


def serve_sim(scorer_threshold, stack, order, chunks_te, y_te):
    scorer = Stack233Scorer(stack, order, scorer_threshold)
    s = np.asarray(scorer.predict_chunks(chunks_te))
    r, met = reward(s, y_te)
    return r, met, s


def main() -> int:
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    print(
        f"stack233: {len(y)} groups | features={X.shape[1]} | {len(ud)} dates "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ---- walk-forward: strictly-prior training, official-reward serve sim ----
    wf_rows = []
    thr_votes = {f: [] for f in TARGET_FPR_GRID}
    for td in ud[-WF:]:
        tr = dates < td
        te_idx = np.flatnonzero(dates == td)
        if tr.sum() < 100 or len(set(y[tr].tolist())) < 2 or not te_idx.size:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human_oof = oof[tr & (y == 0) & ~np.isnan(oof)]
        stack = build_stack(SEED)
        stack.fit(Xr[tr], y[tr])
        chunks_te = [chunks[i] for i in te_idx]
        best = None
        for f in TARGET_FPR_GRID:
            thr = float(np.quantile(human_oof, 1.0 - f))
            r, met, _ = serve_sim(thr, stack, order, chunks_te, y[te_idx])
            thr_votes[f].append(r)
            if best is None or r > best[1]:
                best = (f, r, met, thr)
        f, r, met, thr = best
        wf_rows.append(r)
        print(
            f"  wf {td}: best target_fpr={f} thr={thr:.4f} reward={r:.4f} "
            f"ap={met['ap_score']:.4f} hard_fpr={met['hard_fpr']:.4f} "
            f"sanity={met['human_safety_penalty']:.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    if not wf_rows:
        print("no walk-forward folds -- aborting", file=sys.stderr)
        return 2
    best_fpr = max(TARGET_FPR_GRID, key=lambda f: np.mean(thr_votes[f]) if thr_votes[f] else -1)
    print(
        f"WALK-FORWARD[{len(wf_rows)}d]: mean_reward={np.mean(wf_rows):.4f} "
        f"| chosen target_fpr={best_fpr} "
        f"(grid means: { {f: round(float(np.mean(v)), 4) for f, v in thr_votes.items() if v} })",
        flush=True,
    )

    # ---- final: nested OOF on ALL dates -> threshold; refit stack on all ----
    all_mask = np.ones(len(y), dtype=bool)
    oof = nested_oof(Xr, y, dates, all_mask, SEED)
    human_oof = oof[(y == 0) & ~np.isnan(oof)]
    threshold = float(np.quantile(human_oof, 1.0 - best_fpr))
    stack = build_stack(SEED)
    stack.fit(Xr, y)
    print(f"final threshold (human q{1 - best_fpr:.3f}) = {threshold:.4f}", flush=True)

    scorer = Stack233Scorer(stack, order, threshold)
    artifact = {
        "model": scorer,
        "threshold": threshold,
        "metadata": {
            "algorithm": "Stack233",
            "model_name": "poker44-stack233",
            "feature_version": "d8-union-543",
            "framework": "sklearn-stacking",
            "recipe": (
                "ET500(balanced_subsample,sqrt) + QT160->PCA48->MLP64(a1.2) "
                "-> LR meta (cv=3 OOF), per-batch rank inputs"
            ),
            "provenance": (
                "method reproduced from UID233 (hot-benchmark-poker-3 v2.0) "
                "published source; trained on our own benchmark snapshots "
                "with our own seeds -- no external model blob loaded or copied."
            ),
            "manifest_notes": (
                "UID233-style stacked generalization: 543-feature union "
                "(schema + sanitization-invariant v2 families, bucket-grid "
                "quantized sizing), per-request columnwise rank transform "
                "(train ranks within source_date), ExtraTrees + "
                "QuantileTransformer->PCA->MLP stacked through a logistic "
                "meta-learner on out-of-fold probabilities. Serving: anchor "
                "remap of the human-quantile deploy threshold onto 0.5, then "
                "a 15% power-decay positive budget (rank-preserving; at most "
                "15% of a cycle crosses 0.5). Trained by "
                "scripts/miner/train_stack233.py."
            ),
            "decision_layer": "anchor_remap + power_decay_budget(0.15, 1.5)",
            "target_fpr": best_fpr,
            "seed": SEED,
            "wf_mean_reward": float(np.mean(wf_rows)),
            "wf_rewards": [float(r) for r in wf_rows],
            "n_train": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_dates": len(ud),
            "trained_through": ud[-1],
            "train_dates": ud,
            "reward_formula": "official poker44.score.scoring.reward",
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method_source": "UID233 hot-benchmark-poker-3 / Stack233",
        },
    }
    joblib.dump(artifact, OUT_PATH)
    print(
        f"saved {OUT_PATH} | wf_mean_reward={np.mean(wf_rows):.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
