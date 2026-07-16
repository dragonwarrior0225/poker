#!/usr/bin/env python3
"""Train UID138-style PT2Bag detector and write neurons/models/detector.joblib.

Mirrors perturb-poker-2 v5.2: sanitize train==serve, live-size pool/subset
augmentation, walk-forward OOF, human-quantile deploy threshold, then fit full
PT2Bag (6× XGB bag + 2× PCA→MLP) wrapped in PT2BagScorer.

Usage:
  .venv/bin/python scripts/miner/train_pt2.py benchmark_cache
"""
from __future__ import annotations

import json
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.detector import MODEL_PATH  # noqa: E402
from neurons.pt2bag.pt_features import base_view, wide_view  # noqa: E402
from neurons.pt2bag.pt_model import PT2Bag  # noqa: E402
from neurons.pt2bag.serving import (  # noqa: E402
    DEFAULT_MAX_POS_FRAC,
    PT2BagScorer,
    _apply_batch_safety_budget,
    _remap_to_threshold,
)
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 3200
TARGET_FPR = 0.045
WF = 3
NJ = 4
POOL_RANGE = (86, 102)
POOL_PER_DATE = 2
SUBSET_RANGE = (15, 21)
SUBSET_PER_DATE = 1
TRIM = False
# Soft guard vs prior artifact (skip if missing / force via env).
MAX_WF_FPR = 0.06


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


def augment(san, y, dates):
    rng = random.Random(SEED)
    aug_chunks, aug_y, aug_dates = [], [], []
    by_key: dict = {}
    for i, (d, lab) in enumerate(zip(dates, y)):
        by_key.setdefault((d, int(lab)), []).append(i)
    for (d, lab), idxs in sorted(by_key.items()):
        if len(idxs) < 2:
            continue
        for _ in range(POOL_PER_DATE):
            target = rng.randint(*POOL_RANGE)
            pool, used = [], 0
            for i in rng.sample(idxs, len(idxs)):
                pool.extend(san[i])
                used += 1
                if len(pool) >= target:
                    break
            if used >= 2 and len(pool) >= POOL_RANGE[0]:
                aug_chunks.append(pool[:target])
                aug_y.append(lab)
                aug_dates.append(d)
        for _ in range(SUBSET_PER_DATE):
            i = rng.choice(idxs)
            take = min(rng.randint(*SUBSET_RANGE), len(san[i]))
            if take >= 8 and len(san[i]) > take:
                start = rng.randint(0, len(san[i]) - take)
                aug_chunks.append(san[i][start : start + take])
                aug_y.append(lab)
                aug_dates.append(d)
    return aug_chunks, np.asarray(aug_y, dtype=int), np.asarray(aug_dates)


def mat(chunks, view_fn, cols=None):
    X = pd.DataFrame([view_fn(c) for c in chunks]).fillna(0.0)
    if cols is None:
        cols = sorted(X.columns)
    return X.reindex(columns=cols, fill_value=0.0).values.astype(float), list(cols)


def rk(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (
        s.size - 1
    )


def fpr_target_threshold(p_neg: np.ndarray, target_fpr: float) -> float:
    if len(p_neg) == 0:
        return 0.5
    q = 1.0 - float(target_fpr)
    return float(np.quantile(p_neg, min(max(q, 0.0), 1.0)))


def build_groups():
    bag = [
        xgb.XGBClassifier(
            n_estimators=500 + 30 * i,
            learning_rate=0.03,
            max_depth=4 + (i % 3),
            subsample=0.7 + 0.04 * i,
            colsample_bytree=0.6 + 0.05 * i,
            tree_method="hist",
            n_jobs=NJ,
            random_state=SEED + i,
            eval_metric="logloss",
        )
        for i in range(6)
    ]
    mlps = [
        Pipeline(
            [
                ("s", StandardScaler()),
                ("p", PCA(50, random_state=SEED)),
                (
                    "m",
                    MLPClassifier(
                        (64,),
                        alpha=2.0,
                        max_iter=700,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=SEED + 40 + i,
                    ),
                ),
            ]
        )
        for i in range(2)
    ]
    return [
        ("xgb_bag", bag, "base", 0.72),
        ("mlp_wide", mlps, "wide", 0.28),
    ]


def group_score(ests, X):
    P = np.vstack([e.predict_proba(X)[:, 1] for e in ests])
    if TRIM and P.shape[0] >= 4:
        R = np.vstack([rk(p) for p in P])
        R.sort(axis=0)
        return R[1:-1].mean(axis=0)
    return rk(P.mean(axis=0))


def fit_groups(views, yv, mask):
    fitted = []
    for name, ests, view_key, w in build_groups():
        fitted.append(
            (
                name,
                [e.fit(views[view_key][mask], yv[mask]) for e in ests],
                view_key,
                w,
            )
        )
    return fitted


def blend_groups(fitted, views, mask):
    out, total = None, 0.0
    for _name, ests, view_key, w in fitted:
        g = group_score(ests, views[view_key][mask])
        out = g * w if out is None else out + g * w
        total += float(w)
    return out / total


def main() -> int:
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    if not cache_dir.is_dir():
        print(f"cache dir missing: {cache_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    san, y, dates = load_examples(cache_dir)
    if not san:
        print("no training data found", file=sys.stderr)
        return 2

    aug_chunks, aug_y, aug_dates = augment(san, y, dates)
    all_chunks = san + aug_chunks
    ally = np.concatenate([y, aug_y]) if len(aug_y) else y
    alldates = np.concatenate([dates, aug_dates]) if len(aug_dates) else dates
    is_real = np.zeros(len(all_chunks), dtype=bool)
    is_real[: len(san)] = True

    BASE, cols_base = mat(all_chunks, base_view)
    WIDE, cols_wide = mat(all_chunks, wide_view)
    VIEWS = {"base": BASE, "wide": WIDE}
    ud = sorted(set(dates.tolist()))
    print(
        f"pt2bag: {len(y)} real + {len(aug_y)} aug | "
        f"base{BASE.shape[1]} wide{WIDE.shape[1]} | {len(ud)} dates "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    oof = np.full(len(y), np.nan)
    for td in ud[-WF:]:
        tr = alldates < td
        te_real = dates == td
        te = np.zeros(len(all_chunks), dtype=bool)
        te[: len(san)] = te_real
        if tr.sum() < 60 or len(set(ally[tr].tolist())) < 2 or not te.any():
            continue
        fitted = fit_groups(VIEWS, ally, tr)
        oof[te_real] = blend_groups(fitted, VIEWS, te)
        print(f"  wf {td} ({time.time() - t0:.0f}s)", flush=True)

    m = ~np.isnan(oof)
    if not m.any():
        print("walk-forward produced no scores; refusing to deploy", file=sys.stderr)
        return 3

    cv_ap = float(average_precision_score(y[m], oof[m]))
    deploy_t = fpr_target_threshold(oof[m][y[m] == 0], TARGET_FPR)

    # Evaluate like live serve: remap, then force top-K safety per validator window
    # (each benchmark date = one scored batch), not one pooled pseudo-batch.
    wf_dates = sorted({dates[i] for i in np.flatnonzero(m)})
    per_date = []
    served_all = np.full(len(y), np.nan)
    for td in wf_dates:
        dm = m & (dates == td)
        raw = oof[dm]
        served = _apply_batch_safety_budget(
            _remap_to_threshold(raw, deploy_t), DEFAULT_MAX_POS_FRAC
        )
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
    n_pos = int(np.sum(served_all[m] >= 0.5))
    zero_sanity = [row["date"] for row in per_date if row["sanity"] <= 0.0]
    print(
        f"WALK-FORWARD[{WF}d]: cv_ap={cv_ap:.4f} mean_reward={rew:.4f} "
        f"recall@fpr={res['bot_recall']:.3f} fpr={res['fpr']:.4f} "
        f"hard_fpr={res['hard_fpr']:.4f} t={deploy_t:.4f} "
        f"force_top_k_pos={n_pos}/{int(m.sum())} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    if zero_sanity:
        print(
            f"guard: zero threshold-sanity on {zero_sanity}; refusing deploy",
            file=sys.stderr,
        )
        return 5

    if float(res["hard_fpr"]) >= MAX_WF_FPR:
        print(
            f"guard: hard_fpr {res['hard_fpr']:.4f} >= {MAX_WF_FPR}; refusing deploy",
            file=sys.stderr,
        )
        return 4

    fitted = fit_groups(VIEWS, ally, np.ones(len(all_chunks), dtype=bool))
    ens = PT2Bag(fitted, cols_base, cols_wide, trim=TRIM)
    scorer = PT2BagScorer(ens, threshold=deploy_t, max_pos_frac=DEFAULT_MAX_POS_FRAC)

    meta = {
        "algorithm": "PT2Bag",
        "model_name": "perturb-poker-2",
        "feature_version": "pt.v1",
        "framework": "xgboost+sklearn",
        "trained_on": "sanitized (prepare_hand_for_miner; train == serve)",
        "deploy_threshold": float(deploy_t),
        "threshold": float(deploy_t),
        "target_fpr": TARGET_FPR,
        "max_pos_frac": DEFAULT_MAX_POS_FRAC,
        "safety_budget": "force_topk_v1",
        "seed": SEED,
        "holdout_reward": float(rew),
        "cv_ap": cv_ap,
        "cv_reward": float(rew),
        "cv_recall": float(res["bot_recall"]),
        "cv_fpr": float(res["fpr"]),
        "cv_hard_fpr": float(res["hard_fpr"]),
        "validation": f"walk-forward over the last {WF} dates (per-date force top-K serve)",
        "wf_per_date": per_date,
        "reward_formula": "official poker44.score.scoring.reward",
        "n_train_real": int(len(y)),
        "n_train_aug": int(len(aug_y)),
        "augmentation": {
            "pool_range": list(POOL_RANGE),
            "pool_per_date": POOL_PER_DATE,
            "subset_range": list(SUBSET_RANGE),
            "subset_per_date": SUBSET_PER_DATE,
        },
        "trim": bool(TRIM),
        "groups": [
            (name, len(ests), view_key, w) for name, ests, view_key, w in build_groups()
        ],
        "n_features_base": int(BASE.shape[1]),
        "n_features_wide": int(WIDE.shape[1]),
        "n_dates": int(len(ud)),
        "train_dates": ud,
        "trained_through": ud[-1],
        "regime_start": ud[0],
        "benchmark_releases": ud,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_source": "UID138 perturb-poker-2 / PT2Bag",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump({"model": scorer, "threshold": float(deploy_t), "metadata": meta}, tmp)
    tmp.replace(MODEL_PATH)
    print(
        f"saved {MODEL_PATH} | cv_ap={cv_ap:.4f} cv_reward={rew:.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
