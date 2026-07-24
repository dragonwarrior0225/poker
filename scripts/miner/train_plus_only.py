#!/usr/bin/env python3
"""Memory-lean trainer for the upgraded Stack233+lgb model.

The head-to-head vs plain Stack233 is already established (+lgb won on
2026-07-22, 07-23 and the first folds of 07-24), so this fits ONLY the +lgb
stack (halves peak RAM vs the comparison run, which OOM'd at 3398 groups on
a 7GB box) and validates it at true live geometry. Lower n_jobs to cap
memory. Saves neurons/models/detector233.joblib.

Usage: .venv/bin/python scripts/miner/train_plus_only.py benchmark_cache
"""
from __future__ import annotations

import gc
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
    SEED, TARGET_FPR_GRID, featurize, load_examples, nested_oof,
    rank_within_dates, sanitize,
)
from upgrade_stack233_ranker import live_units  # noqa: E402

warnings.filterwarnings("ignore")
WF = 3
NJ = 2          # lower parallelism -> lower peak RAM
MAX_FRAC = 0.15
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"
CAPTURE_GLOBS = (str(REPO_ROOT / "data" / "live_payloads" / "cycle_*.json"),
                 "/root/Poker44-subnet/data/live_payloads/cycle_*.json")


def build_plus(seed: int):
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
                max_features="sqrt", random_state=seed, n_jobs=NJ)),
            ("net", make_pipeline(
                QuantileTransformer(n_quantiles=160, output_distribution="uniform",
                                    random_state=seed + 1),
                PCA(n_components=48, random_state=seed + 2),
                MLPClassifier(hidden_layer_sizes=(64,), alpha=1.2, max_iter=400,
                              early_stopping=True, random_state=seed + 3))),
            ("lgb", LGBMClassifier(
                n_estimators=800, learning_rate=0.03, num_leaves=63,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=seed + 4, n_jobs=NJ, verbose=-1)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3, stack_method="predict_proba", n_jobs=1,
    )


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    tfpr = TARGET_FPR_GRID[0]
    print(f"train +lgb only: {len(y)} groups | {len(ud)} dates | WF={WF} "
          f"({time.time()-t0:.0f}s)", flush=True)

    wf = []
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human = oof[tr & (y == 0) & ~np.isnan(oof)]
        thr = float(np.quantile(human, 1.0 - tfpr))
        del oof; gc.collect()
        stack = build_plus(SEED); stack.fit(Xr[tr], y[tr])
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        sc = Stack233Scorer(stack, order, thr, max_frac=MAX_FRAC)
        s = np.asarray(sc.predict_chunks(units))
        r, met = reward(s, labs)
        wf.append(r)
        print(f"  wf {td}: reward={r:.4f} ap={met['ap_score']:.4f} "
              f"hard_fpr={met['hard_fpr']:.4f} sanity={met['human_safety_penalty']:.3f} "
              f"pos={int((s>=0.5).sum())}/{len(s)} ({time.time()-t0:.0f}s)", flush=True)
        del stack, sc; gc.collect()
    wf_mean = float(np.mean(wf)) if wf else 0.0
    print(f"WALK-FORWARD[{len(wf)}d] live-geo mean={wf_mean:.4f}", flush=True)

    print("final fit on all data...", flush=True)
    oof = nested_oof(Xr, y, dates, np.ones(len(y), bool), SEED)
    human = oof[(y == 0) & ~np.isnan(oof)]
    thr = float(np.quantile(human, 1.0 - tfpr))
    del oof; gc.collect()
    stack = build_plus(SEED); stack.fit(Xr, y)
    scorer = Stack233Scorer(stack, order, thr, max_frac=MAX_FRAC)

    seen, cc = set(), []
    for g in CAPTURE_GLOBS:
        for f in sorted(glob.glob(g)):
            try: cap = json.load(open(f))
            except Exception: continue
            for c in cap.get("chunks") or []:
                k = (len(c), json.dumps(c[0], sort_keys=True)[:200] if c else "")
                if k not in seen: seen.add(k); cc.append(c)
    fr = [float((np.asarray(scorer.predict_chunks(cc[i:i+100])) >= 0.5).mean())
          for i in range(0, len(cc), 100) if len(cc[i:i+100]) >= 20]

    prev = joblib.load(OUT_PATH) if OUT_PATH.exists() else {"metadata": {}}
    md = dict(prev.get("metadata", {}))
    md.update({
        "algorithm": "Stack233+lgb", "ranker": "et+pca-mlp+lgb",
        "wf_mean_reward": wf_mean, "wf_rewards": [float(r) for r in wf],
        "threshold": thr, "max_frac": MAX_FRAC,
        "trained_through": ud[-1], "n_train": int(len(y)),
        "live_capture_pos_fraction": float(np.mean(fr)) if fr else None,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    joblib.dump({"model": scorer, "threshold": thr, "metadata": md}, OUT_PATH)
    sz = OUT_PATH.stat().st_size / 1048576
    print(f"saved {OUT_PATH} | wf {wf_mean:.4f} | live_pos_frac "
          f"{np.mean(fr) if fr else 0:.3f} | {sz:.1f} MB ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
