"""Walk-forward (temporal) evaluation harness.

For each evaluation date d, train on strictly-earlier dates and score d.
This simulates the live validator: the model has never seen date d's data.
Compares training-set policies to test the distribution-shift hypothesis.

Usage: python scripts/miner/walk_forward_eval.py <cache_dir>
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.detector import extract_features, hand_feature_matrix  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from train_detector import (  # noqa: E402
    best_threshold,
    evaluate,
    load_dataset,
)

warnings.filterwarnings("ignore")
SEED = 44


def fit_fixed(Xg, y, Xh, yh):
    """A single fast, representative model pair (group + hand) for diagnostics."""
    from lightgbm import LGBMClassifier

    g = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    g.fit(Xg, y)
    h = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    h.fit(Xh, yh)
    return g, h


def stack_hands(hand_mats, labels):
    Xh = np.vstack([m for m in hand_mats if m.shape[0]])
    yh = np.concatenate(
        [np.full(m.shape[0], lab) for m, lab in zip(hand_mats, labels) if m.shape[0]]
    )
    idx = np.concatenate(
        [np.full(m.shape[0], i) for i, m in enumerate(hand_mats) if m.shape[0]]
    )
    return Xh, yh, idx


def group_scores(g, h, Xg, hand_mats):
    blended = 0.5 * g.predict_proba(Xg)[:, 1]
    lengths = [m.shape[0] for m in hand_mats]
    stacked = np.vstack([m for m in hand_mats if m.shape[0]])
    hp = h.predict_proba(stacked)[:, 1]
    pos = 0
    hand_agg = np.zeros(len(hand_mats))
    for i, ln in enumerate(lengths):
        p = hp[pos:pos + ln]
        hand_agg[i] = p.mean() if p.size else 0.5
        pos += ln
    return blended + 0.5 * hand_agg


def oof_scores(Xg, y, dates, hand_mats, n_splits=4):
    """Date-grouped out-of-fold scores on the training window (deploy-faithful)."""
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in cv.split(Xg, y, dates):
        Xh, yh, _ = stack_hands([hand_mats[i] for i in tr], y[tr])
        g, h = fit_fixed(Xg[tr], y[tr], Xh, yh)
        oof[te] = group_scores(g, h, Xg[te], [hand_mats[i] for i in te])
    return oof


def threshold_with_margin(scores, labels, fpr_target):
    """best_threshold but tie-broken toward a chosen hard-FPR safety margin."""
    candidates = []
    for t in np.unique(np.round(scores, 4)):
        t = float(min(max(t, 1e-6), 1 - 1e-6))
        remapped = np.where(
            scores >= t, 0.5 + 0.5 * (scores - t) / (1 - t), 0.5 * scores / t
        )
        r, m = reward(remapped, labels)
        candidates.append((t, r, m["hard_fpr"]))
    best_r = max(r for _, r, _ in candidates)
    tied = [(t, fpr) for t, r, fpr in candidates if r >= best_r - 1e-9]
    return min(tied, key=lambda tf: abs(tf[1] - fpr_target))[0]


def run(cache_dir, train_policy, eval_dates, fpr_target=0.04):
    Xg, y, dates, hand_mats = load_dataset(cache_dir)
    rewards, ceil_rewards = [], []
    for d in eval_dates:
        if train_policy == "all_prior":
            tr = dates < d
        elif train_policy == "big_prior":
            tr = (dates < d) & (dates >= "2026-07-06")
        elif train_policy == "small_only":
            # train ONLY on the old regime; test on novel big-regime dates.
            # Closest available analog to live "novel bot" generalization.
            tr = dates < "2026-07-06"
        else:
            raise ValueError(train_policy)
        te = dates == d
        if tr.sum() == 0 or te.sum() == 0:
            continue
        tr_i = np.flatnonzero(tr)
        te_i = np.flatnonzero(te)
        # deploy-faithful threshold: pick on date-grouped OOF of training window
        oof = oof_scores(Xg[tr_i], y[tr_i], dates[tr_i],
                         [hand_mats[i] for i in tr_i])
        thr = threshold_with_margin(oof, y[tr_i], fpr_target)
        Xh, yh, _ = stack_hands([hand_mats[i] for i in tr_i], y[tr_i])
        g, h = fit_fixed(Xg[tr_i], y[tr_i], Xh, yh)
        s_te = group_scores(g, h, Xg[te_i], [hand_mats[i] for i in te_i])
        r, m = evaluate(s_te, y[te_i], thr)
        # rank ceiling: reward if the safety gate were fully satisfied
        ceil = 0.35 * m["ap_score"] + 0.30 * m["bot_recall"] + 0.35
        rewards.append(r)
        ceil_rewards.append(ceil)
        print(f"  {d}: reward={r:.4f} (ceil={ceil:.4f}) ap={m['ap_score']:.4f} "
              f"recall={m['bot_recall']:.4f} hard_fpr={m['hard_fpr']:.4f}")
    print(f"  MEAN reward = {np.mean(rewards):.4f} | "
          f"rank-ceiling = {np.mean(ceil_rewards):.4f}\n")
    return float(np.mean(rewards))


def main():
    cache_dir = Path(sys.argv[1])
    all_dates = sorted(p.stem for p in cache_dir.glob("*.json"))
    eval_dates = [d for d in all_dates if d >= "2026-07-08"]
    print(f"walk-forward on MINER-VISIBLE payloads (live-faithful): {eval_dates}\n")
    print("=== train on all prior dates, score each novel date ===")
    run(cache_dir, "all_prior", eval_dates, fpr_target=0.04)


if __name__ == "__main__":
    main()
