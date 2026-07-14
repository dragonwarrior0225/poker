"""Train the v4 drift-robust detector (rank-armored, live-size aware).

Design (evidence: live capture 2026-07-14 + competitor convergence):
- Features: extract_features_v4 = KS-stable v3 subset + scale-free extras
  (bucket-grid sizing, pot fractions, replay redundancy). Live generator runs
  a different scale regime (pots ~14x smaller, stacks pinned 100bb), so only
  scale-free behavior survives the benchmark->live transfer.
- Rank channel: every training batch (one source date) is augmented with
  within-batch percentile ranks per column, exactly like V4RankEnsemble does
  across one live request's segments. Monotone drift cannot move ranks.
- Rows per date: real projected groups (~30-40 hands) + pooled same-label
  units (86-102 hands, mimicking live chunks) + contiguous subsets (15-21
  hands, mimicking odd segments).
- Validation: walk-forward on the last 3 dates; reward simulated at live
  geometry (pooled units -> segmented serve -> top-15% budget) under the
  CURRENT validator formula.

Usage: .venv/bin/python scripts/miner/train_detector_v2.py benchmark_cache
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.detector import (  # noqa: E402
    DetectorModel,
    V4RankEnsemble,
    extract_features_v4,
)
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 44
POOL_RANGE = (86, 102)
SUBSET_RANGE = (15, 21)
POOLS_PER_LABEL = 3
TARGET_FPR = 0.045
WF_DATES = 3


def project(group):
    return [prepare_hand_for_miner(h) for h in group]


def load_rows(cache_dir: Path):
    """Per date: real groups + pooled live-size units + subset views."""
    rng = np.random.default_rng(SEED)
    X, y, dates, kinds = [], [], [], []
    for path in sorted(cache_dir.glob("*.json")):
        date = path.stem
        by_label = {0: [], 1: []}
        for record in json.loads(path.read_text()):
            for group, label in zip(record["chunks"], record["groundTruth"]):
                g = project(group)
                by_label[int(label)].append(g)
                X.append(extract_features_v4(g))
                y.append(int(label))
                dates.append(date)
                kinds.append("real")
        for label, groups in by_label.items():
            if len(groups) < 2:
                continue
            for _ in range(POOLS_PER_LABEL):
                order = rng.permutation(len(groups))
                unit, target = [], int(rng.integers(*POOL_RANGE))
                for gi in order:
                    unit.extend(groups[gi])
                    if len(unit) >= target:
                        break
                X.append(extract_features_v4(unit[:target]))
                y.append(label)
                dates.append(date)
                kinds.append("pooled")
            for _ in range(2):
                g = groups[int(rng.integers(len(groups)))]
                ln = int(rng.integers(*SUBSET_RANGE))
                if len(g) > ln:
                    start = int(rng.integers(0, len(g) - ln))
                    X.append(extract_features_v4(g[start : start + ln]))
                    y.append(label)
                    dates.append(date)
                    kinds.append("subset")
    return (
        np.vstack(X),
        np.asarray(y),
        np.asarray(dates),
        np.asarray(kinds),
    )


def rank_within_dates(X, dates):
    """Train-time analog of V4RankEnsemble.rank_augment (per-date batches)."""
    out = np.empty_like(X)
    for d in np.unique(dates):
        m = dates == d
        out[m] = V4RankEnsemble.rank_augment(X[m])[:, X.shape[1] :]
    return np.hstack([X, out])


def candidates():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from xgboost import XGBClassifier

    c = {}
    for i in range(3):
        c[f"xgb_{i}"] = XGBClassifier(
            n_estimators=500 + 60 * i, learning_rate=0.03, max_depth=4 + i % 3,
            subsample=0.7 + 0.05 * i, colsample_bytree=0.6 + 0.06 * i,
            tree_method="hist", random_state=SEED + i, eval_metric="logloss",
        )
    for i, (leaves, lr) in enumerate([(31, 0.05), (63, 0.03)]):
        c[f"lgb_{i}"] = LGBMClassifier(
            n_estimators=800, learning_rate=lr, num_leaves=leaves,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=1.0, random_state=SEED + i, n_jobs=-1, verbose=-1,
        )
    c["cat"] = CatBoostClassifier(
        iterations=700, learning_rate=0.04, depth=6, verbose=0,
        random_seed=SEED, allow_writing_files=False,
    )
    c["et"] = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=3,
        random_state=SEED, n_jobs=-1,
    )
    return c


def rank_objective(scores, labels):
    from sklearn.metrics import average_precision_score

    ap = average_precision_score(labels, scores) if labels.any() else 0.0
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    rec = tp / max((labels == 1).sum(), 1)
    fpr = fp / max((labels == 0).sum(), 1)
    ok = fpr <= 0.05
    r5 = float(rec[ok].max()) if ok.any() else 0.0
    return 0.35 * ap + 0.30 * r5


def oof_scores(Xr, y, dates, models, train_mask_extra=None):
    """Leave-one-date-out OOF for weight/threshold selection (last 10 dates)."""
    uniq = np.unique(dates)
    eval_dates = uniq[-10:]
    oof = {k: np.full(len(y), np.nan) for k in models}
    for d in eval_dates:
        te = dates == d
        tr = ~te
        if train_mask_extra is not None:
            tr &= train_mask_extra
        for k, proto in models.items():
            import copy

            m = copy.deepcopy(proto)
            m.fit(Xr[tr], y[tr])
            oof[k][te] = m.predict_proba(Xr[te])[:, 1]
    mask = ~np.isnan(next(iter(oof.values())))
    return oof, mask


def greedy_weights(oof, mask, y, kinds):
    """Greedy forward selection on real+pooled OOF rows, 0.1 weight steps."""
    score_rows = mask & np.isin(kinds, ("real", "pooled"))
    yv = y[score_rows]
    mats = {k: v[score_rows] for k, v in oof.items()}
    weights = {}
    current = np.zeros(yv.shape[0])
    best_obj = -1.0
    for _ in range(10):
        best_k, best_new = None, None
        for k, s in mats.items():
            cand = current + 0.1 * s
            obj = rank_objective(cand / (sum(weights.values()) + 0.1), yv)
            if obj > best_obj + 1e-6:
                best_obj, best_k, best_new = obj, k, cand
        if best_k is None:
            break
        current = best_new
        weights[best_k] = weights.get(best_k, 0.0) + 0.1
    return weights, best_obj


def main():
    cache_dir = Path(sys.argv[1])
    print("loading rows (real + pooled live-size + subsets)...", flush=True)
    X, y, dates, kinds = load_rows(cache_dir)
    Xr = rank_within_dates(X, dates)
    print(f"rows={len(y)} (real={np.sum(kinds=='real')}, pooled={np.sum(kinds=='pooled')}, "
          f"subset={np.sum(kinds=='subset')}) features={Xr.shape[1]} dates={len(np.unique(dates))}",
          flush=True)

    models = candidates()
    print("computing leave-one-date-out OOF (last 10 dates)...", flush=True)
    oof, mask = oof_scores(Xr, y, dates, models)
    weights, obj = greedy_weights(oof, mask, y, kinds)
    print(f"greedy ensemble rank_obj={obj:.4f} weights={weights}", flush=True)

    blended = np.zeros(len(y))
    tw = sum(weights.values())
    for k, w in weights.items():
        blended += w * np.nan_to_num(oof[k], nan=0.0)
    blended /= tw
    human_scores = blended[mask & (y == 0) & np.isin(kinds, ("real", "pooled"))]
    threshold = float(np.quantile(human_scores, 1 - TARGET_FPR))
    threshold = min(max(threshold, 1e-4), 1 - 1e-4)
    print(f"threshold (human q{1-TARGET_FPR:.3f}) = {threshold:.4f}", flush=True)

    # ---- walk-forward at live geometry (uses full refits per fold) ----
    print("\nwalk-forward at live geometry (last %d dates):" % WF_DATES, flush=True)
    uniq = np.unique(dates)
    wf_rewards = []
    for d in uniq[-WF_DATES:]:
        tr = dates < d
        members = []
        import copy

        for k, w in weights.items():
            m = copy.deepcopy(models[k])
            m.fit(Xr[tr], y[tr])
            members.append((m, w))
        ens = V4RankEnsemble(members)
        det = DetectorModel.__new__(DetectorModel)
        det.model, det.threshold, det.metadata = ens, threshold, {}
        # simulate live cycles from this date's real groups
        rng = np.random.default_rng(7)
        raw = json.loads((cache_dir / f"{d}.json").read_text())
        by_label = {0: [], 1: []}
        for record in raw:
            for group, label in zip(record["chunks"], record["groundTruth"]):
                by_label[int(label)].append(project(group))
        units, labs = [], []
        for label, groups in by_label.items():
            rng.shuffle(groups)
            i = 0
            while i < len(groups) - 1:
                unit = list(groups[i]); i += 1
                target = int(rng.integers(*POOL_RANGE))
                while len(unit) < target and i < len(groups):
                    unit.extend(groups[i]); i += 1
                units.append(unit[:POOL_RANGE[1]]); labs.append(label)
        labs = np.asarray(labs)
        scores = np.asarray(det.score_chunks(units))
        r, met = reward(scores, labs)
        wf_rewards.append(r)
        print(f"  {d}: live-sim reward={r:.4f} ap={met['ap_score']:.4f} "
              f"recall={met['bot_recall']:.4f} hard_fpr={met['hard_fpr']:.4f} "
              f"n_units={len(units)}", flush=True)
    print(f"  MEAN live-sim reward = {np.mean(wf_rewards):.4f}", flush=True)

    # ---- final refit on all data ----
    print("\nrefitting on all data for deployment...", flush=True)
    import copy

    members = []
    for k, w in weights.items():
        m = copy.deepcopy(models[k])
        m.fit(Xr, y)
        members.append((m, w))
    ens = V4RankEnsemble(members)

    import joblib

    artifact = {
        "model": ens,
        "threshold": threshold,
        "metadata": {
            "algorithm": "v4-rank-ensemble",
            "feature_version": 4,
            "ensemble_weights": weights,
            "oof_rank_objective": float(obj),
            "walk_forward_live_sim_rewards": [float(r) for r in wf_rewards],
            "trained_through": str(uniq[-1]),
            "target_fpr": TARGET_FPR,
        },
    }
    out = REPO_ROOT / "neurons" / "models" / "detector.joblib"
    joblib.dump(artifact, out)
    print(f"artifact saved to {out}", flush=True)


if __name__ == "__main__":
    main()
