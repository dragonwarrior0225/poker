#!/usr/bin/env python3
"""Ranker upgrade for uid38's Stack233: add a LightGBM base learner to the
stack (ET + PCA/MLP + LGBM -> LR meta), keeping the Stack233 serving contract
and staying deployable (<100MB, no 3x bag). Validated at TRUE live geometry
(pooled 86-102-hand units, official reward, walk-forward) vs the current
2-base Stack233. Deploys only if it beats the incumbent on the mean.

Usage: .venv/bin/python scripts/miner/upgrade_stack233_ranker.py benchmark_cache
"""
from __future__ import annotations

import glob
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "miner"))

from neurons.stack233.serving import Stack233Scorer  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from train_stack233 import (  # noqa: E402
    SEED, TARGET_FPR_GRID, build_stack, featurize, load_examples, nested_oof,
    rank_within_dates, sanitize, union_features,
)

warnings.filterwarnings("ignore")
WF = 4
POOL_RANGE = (86, 102)
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"


def build_stack_plus(seed: int):
    """Stack233 + a LightGBM base learner (the deployable ranker upgrade)."""
    from lightgbm import LGBMClassifier
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    return StackingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(
                n_estimators=500, class_weight="balanced_subsample",
                max_features="sqrt", random_state=seed, n_jobs=4)),
            ("net", make_pipeline(
                QuantileTransformer(n_quantiles=160, output_distribution="uniform",
                                    random_state=seed + 1),
                PCA(n_components=48, random_state=seed + 2),
                MLPClassifier(hidden_layer_sizes=(64,), alpha=1.2, max_iter=400,
                              early_stopping=True, random_state=seed + 3))),
            ("lgb", LGBMClassifier(
                n_estimators=800, learning_rate=0.03, num_leaves=63,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=seed + 4, n_jobs=4, verbose=-1)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3, stack_method="predict_proba", n_jobs=1,
    )


def live_units(cache_dir, date, rng):
    raw = json.loads((cache_dir / f"{date}.json").read_text())
    by = {0: [], 1: []}
    for record in raw:
        for g, lab in zip(record["chunks"], record["groundTruth"]):
            by[int(lab)].append(sanitize(g))
    units, labs = [], []
    for lab, groups in by.items():
        rng.shuffle(groups); i = 0
        while i < len(groups) - 1:
            unit = list(groups[i]); i += 1
            target = int(rng.integers(*POOL_RANGE))
            while len(unit) < target and i < len(groups):
                unit.extend(groups[i]); i += 1
            units.append(unit[:POOL_RANGE[1]]); labs.append(lab)
    return units, np.asarray(labs)


def serve_reward(stack, order, thr, units, labs):
    sc = Stack233Scorer(stack, order, thr)
    s = np.asarray(sc.predict_chunks(units))
    return reward(s, labs)[0]


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    tfpr = TARGET_FPR_GRID[0]
    print(f"ranker upgrade: {len(y)} groups | {len(ud)} dates | WF={WF}", flush=True)

    base_r, plus_r = [], []
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human = oof[tr & (y == 0) & ~np.isnan(oof)]
        thr = float(np.quantile(human, 1.0 - tfpr))
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        sb = build_stack(SEED); sb.fit(Xr[tr], y[tr])
        sp = build_stack_plus(SEED); sp.fit(Xr[tr], y[tr])
        rb = serve_reward(sb, order, thr, units, labs)
        rp = serve_reward(sp, order, thr, units, labs)
        base_r.append(rb); plus_r.append(rp)
        print(f"  wf {td}: base={rb:.4f}  plus(+lgb)={rp:.4f}  "
              f"{'WIN' if rp>rb else ''} ({time.time()-t0:.0f}s)", flush=True)

    bm, pm = float(np.mean(base_r)), float(np.mean(plus_r))
    regress = sum(1 for b, p in zip(base_r, plus_r) if p < b - 1e-9)
    print(f"\nMEAN live-geo: base={bm:.4f}  plus(+lgb)={pm:.4f}  "
          f"(delta {pm-bm:+.4f}, fold regressions {regress})", flush=True)

    if pm <= bm + 1e-4 or regress > 1:
        print("Ranker upgrade does NOT clearly beat base -> keeping current "
              "Stack233. Nothing rewritten.", flush=True)
        return 0

    # winner: refit plus on all data, threshold from full nested OOF, save
    oof = nested_oof(Xr, y, dates, np.ones(len(y), bool), SEED)
    human = oof[(y == 0) & ~np.isnan(oof)]
    thr = float(np.quantile(human, 1.0 - tfpr))
    stack = build_stack_plus(SEED); stack.fit(Xr, y)
    scorer = Stack233Scorer(stack, order, thr, max_frac=0.15)
    prev = joblib.load(OUT_PATH); md = dict(prev.get("metadata", {}))
    md.update({
        "algorithm": "Stack233+lgb", "ranker_upgrade": "et+pca-mlp+lgb",
        "upgrade_wf_mean": pm, "upgrade_baseline": bm,
        "upgrade_folds_base": [float(r) for r in base_r],
        "upgrade_folds_plus": [float(r) for r in plus_r],
        "trained_through": ud[-1], "threshold": thr,
        "upgraded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    joblib.dump({"model": scorer, "threshold": thr, "metadata": md}, OUT_PATH)
    sz = OUT_PATH.stat().st_size / 1048576
    print(f"saved {OUT_PATH} | plus mean {pm:.4f} vs base {bm:.4f} | size {sz:.1f} MB "
          f"({'OK to push' if sz < 100 else 'TOO BIG for github'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
