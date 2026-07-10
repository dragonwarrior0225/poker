"""Train the Poker44 bot detector from cached training-benchmark releases.

Usage:
    python scripts/miner/train_detector.py <cache_dir> [--holdout YYYY-MM-DD]

<cache_dir> holds one JSON file per benchmark sourceDate (the raw chunk
records from /api/v1/benchmark/chunks).

Pipeline:
1. Build group-level features and hand-level features (group label broadcast
   to hands, ~30x more training rows for the hand models).
2. Compare a wide algorithm zoo (logistic, SVM, MLP, naive bayes, random
   forest, extra trees, HistGBM, XGBoost, LightGBM, CatBoost, plus hand-level
   GBMs aggregated per group) with date-grouped stratified CV.
3. Greedy Caruana ensemble selection on out-of-fold predictions, maximizing
   the ranked part of the validator reward (0.35*AP + 0.30*recall@FPR<=5%).
4. Pick the decision threshold that maximizes the full validator reward on
   OOF predictions, tie-broken toward a ~4% hard FPR safety margin.
5. Report metrics on the held-out newest date and a second-date stability
   check, then refit the selected ensemble on ALL data for deployment.
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from neurons.detector import (  # noqa: E402
    FEATURE_VERSION,
    MODEL_PATH,
    EnsembleScorer,
    extract_features,
    feature_names,
    hand_feature_matrix,
)
from poker44.score.scoring import _recall_at_fpr, reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 44
HAND_AGGS = ("mean", 0.75, 0.9)
N_SEARCH_PER_FAMILY = 18


# ---------------------------------------------------------------- data


def to_miner_visible(group):
    """Project a raw benchmark group through the exact canonicalizer the
    validator applies before sending hands to miners (windowed actions,
    coarsened amounts, aliased seats, suppressed identity fields).

    Training on the raw benchmark payload while the validator scores the
    miner-visible projection is a train/serve skew that inflates hard_fpr
    and collapses reward. We train on the projection so the feature
    distribution matches what the model actually sees live.
    """
    return [prepare_hand_for_miner(h) for h in group]


def load_dataset(cache_dir: Path):
    """Group features, labels, dates, plus per-group hand feature matrices.

    Every group is projected to the validator's miner-visible payload first
    so training and serving see the same feature distribution.
    """
    Xg, y, dates, hand_mats = [], [], [], []
    for path in sorted(cache_dir.glob("*.json")):
        date = path.stem
        for record in json.loads(path.read_text()):
            for group, label in zip(record["chunks"], record["groundTruth"]):
                group = to_miner_visible(group)
                Xg.append(extract_features(group))
                hand_mats.append(hand_feature_matrix(group))
                y.append(int(label))
                dates.append(date)
    return np.vstack(Xg), np.asarray(y), np.asarray(dates), hand_mats


def stack_hands(hand_mats, labels):
    """Hand-level matrix with group labels broadcast + group index per row."""
    Xh = np.vstack(hand_mats)
    yh = np.concatenate(
        [np.full(m.shape[0], lab) for m, lab in zip(hand_mats, labels)]
    )
    gidx = np.concatenate(
        [np.full(m.shape[0], i) for i, m in enumerate(hand_mats)]
    )
    return Xh, yh, gidx


def aggregate_hand_probs(probs, gidx, n_groups, agg):
    out = np.full(n_groups, 0.5)
    for g in range(n_groups):
        p = probs[gidx == g]
        if p.size:
            out[g] = p.mean() if agg == "mean" else np.quantile(p, float(agg))
    return out


# ---------------------------------------------------------------- models


def group_candidates():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logreg_c1": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)
        ),
        "logreg_c01": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)
        ),
        "svc_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=2.0, gamma="scale", probability=True, random_state=SEED),
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                max_iter=800,
                early_stopping=True,
                random_state=SEED,
            ),
        ),
        "gaussian_nb": make_pipeline(StandardScaler(), GaussianNB()),
        "random_forest": RandomForestClassifier(
            n_estimators=700, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
        "rf_deep": RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features=0.3,
            n_jobs=-1,
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
        "hist_gb_a": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),
        "hist_gb_b": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            random_state=SEED,
        ),
        "xgboost_a": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "xgboost_b": XGBClassifier(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "lightgbm_a": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "lightgbm_b": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=10,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def sampled_candidates():
    """Randomized hyperparameter search zoo, deterministic under SEED."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    rng = random.Random(SEED)
    cands = {}
    for i in range(N_SEARCH_PER_FAMILY):
        cands[f"xgb_s{i}"] = XGBClassifier(
            n_estimators=rng.choice([300, 500, 800, 1200]),
            learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08, 0.12]),
            max_depth=rng.choice([3, 4, 5, 6, 8]),
            subsample=round(rng.uniform(0.6, 1.0), 2),
            colsample_bytree=round(rng.uniform(0.5, 1.0), 2),
            reg_lambda=rng.choice([0.5, 1.0, 2.0, 5.0]),
            min_child_weight=rng.choice([1, 3, 5]),
            eval_metric="logloss",
            random_state=SEED + i,
            n_jobs=-1,
        )
    for i in range(N_SEARCH_PER_FAMILY):
        cands[f"lgb_s{i}"] = LGBMClassifier(
            n_estimators=rng.choice([300, 500, 800, 1200]),
            learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08, 0.12]),
            num_leaves=rng.choice([7, 15, 31, 63]),
            min_child_samples=rng.choice([5, 10, 20, 40]),
            subsample=round(rng.uniform(0.6, 1.0), 2),
            subsample_freq=1,
            colsample_bytree=round(rng.uniform(0.5, 1.0), 2),
            reg_lambda=rng.choice([0.0, 0.5, 1.0, 5.0]),
            random_state=SEED + i,
            n_jobs=-1,
            verbose=-1,
        )
    for i in range(N_SEARCH_PER_FAMILY):
        cands[f"cat_s{i}"] = CatBoostClassifier(
            iterations=rng.choice([400, 800, 1200]),
            learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08]),
            depth=rng.choice([4, 5, 6, 8]),
            l2_leaf_reg=rng.choice([1.0, 3.0, 10.0]),
            random_strength=rng.choice([0.5, 1.0, 2.0]),
            random_seed=SEED + i,
            verbose=0,
            allow_writing_files=False,
        )
    for i in range(N_SEARCH_PER_FAMILY):
        cands[f"hgb_s{i}"] = HistGradientBoostingClassifier(
            max_iter=rng.choice([200, 400, 700]),
            learning_rate=rng.choice([0.03, 0.06, 0.1, 0.15]),
            max_leaf_nodes=rng.choice([7, 15, 31, 63]),
            min_samples_leaf=rng.choice([5, 10, 20, 40]),
            l2_regularization=rng.choice([0.0, 0.5, 1.0, 5.0]),
            random_state=SEED + i,
        )
    for i in range(6):
        cands[f"rf_s{i}"] = RandomForestClassifier(
            n_estimators=rng.choice([500, 1000]),
            min_samples_leaf=rng.choice([1, 2, 4]),
            max_features=rng.choice(["sqrt", 0.3, 0.5]),
            n_jobs=-1,
            random_state=SEED + i,
        )
    for i in range(6):
        cands[f"et_s{i}"] = ExtraTreesClassifier(
            n_estimators=rng.choice([500, 1000]),
            min_samples_leaf=rng.choice([1, 2, 4]),
            max_features=rng.choice(["sqrt", 0.3, 0.5]),
            n_jobs=-1,
            random_state=SEED + i,
        )
    return cands


def all_group_candidates():
    merged = dict(group_candidates())
    merged.update(sampled_candidates())
    return merged


def hand_candidates():
    from lightgbm import LGBMClassifier

    return {
        "hand_hist_gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED
        ),
        "hand_lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "hand_lgb_deep": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------- scoring


def rank_objective(scores, labels):
    """Ranked share of the validator reward (threshold handled separately)."""
    ap = average_precision_score(labels, scores)
    recall, _ = _recall_at_fpr(scores, labels, max_fpr=0.05)
    return 0.35 * ap + 0.30 * recall


def best_threshold(scores, labels):
    """Threshold maximizing the validator reward on remapped scores.

    Reward ties are broken toward ~4% hard FPR: margin against live drift
    under the 10% threshold-sanity boundary.
    """
    candidates = []
    for t in np.unique(np.round(scores, 4)):
        t = float(min(max(t, 1e-6), 1 - 1e-6))
        remapped = np.where(
            scores >= t, 0.5 + 0.5 * (scores - t) / (1 - t), 0.5 * scores / t
        )
        r, metrics = reward(remapped, labels)
        candidates.append((t, r, metrics["hard_fpr"]))
    best_r = max(r for _, r, _ in candidates)
    tied = [(t, fpr) for t, r, fpr in candidates if r >= best_r - 1e-9]
    best_t = min(tied, key=lambda tf: abs(tf[1] - 0.04))[0]
    return best_t, best_r


def evaluate(scores, labels, threshold):
    t = min(max(threshold, 1e-6), 1 - 1e-6)
    remapped = np.where(
        scores >= t, 0.5 + 0.5 * (scores - t) / (1 - t), 0.5 * scores / t
    )
    return reward(remapped, labels)


# ---------------------------------------------------------------- OOF + greedy


def compute_oof(Xg, y, dates, hand_mats, n_splits=5):
    """Out-of-fold predictions for every candidate (group + hand level)."""
    Xh, yh, gidx = stack_hands(hand_mats, y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    names = list(all_group_candidates()) + [
        f"{n}@{a}" for n in hand_candidates() for a in HAND_AGGS
    ]
    oof = {name: np.zeros(len(y)) for name in names}

    for fold, (tr, te) in enumerate(cv.split(Xg, y, dates)):
        for name, proto in all_group_candidates().items():
            model = clone(proto)
            model.fit(Xg[tr], y[tr])
            oof[name][te] = model.predict_proba(Xg[te])[:, 1]

        tr_rows = np.isin(gidx, tr)
        te_rows = np.isin(gidx, te)
        for name, proto in hand_candidates().items():
            model = clone(proto)
            model.fit(Xh[tr_rows], yh[tr_rows])
            probs = model.predict_proba(Xh[te_rows])[:, 1]
            sub_gidx = gidx[te_rows]
            for agg in HAND_AGGS:
                for g in te:
                    p = probs[sub_gidx == g]
                    if p.size:
                        oof[f"{name}@{agg}"][g] = (
                            p.mean() if agg == "mean" else np.quantile(p, float(agg))
                        )
        print(f"  fold {fold + 1}/{n_splits} done", flush=True)
    return oof


def greedy_ensemble(oof, y, max_iters=60):
    """Caruana forward selection with replacement on the rank objective."""
    names = list(oof)
    bag: list[str] = []
    blended = np.zeros(len(y))
    best_obj = -1.0
    for _ in range(max_iters):
        gains = []
        for name in names:
            trial = (blended * len(bag) + oof[name]) / (len(bag) + 1)
            gains.append((rank_objective(trial, y), name))
        obj, pick = max(gains)
        if obj <= best_obj + 1e-6:
            break
        best_obj = obj
        bag.append(pick)
        blended = (blended * (len(bag) - 1) + oof[pick]) / len(bag)
    weights = {name: bag.count(name) / len(bag) for name in set(bag)}
    return weights, blended, best_obj


def refine_weights(oof, y, greedy_weights, top_n=12):
    """SLSQP weight refinement over greedy support + top singles."""
    from scipy.optimize import minimize

    singles = sorted(oof, key=lambda n: -rank_objective(oof[n], y))[:top_n]
    support = sorted(set(greedy_weights) | set(singles))
    M = np.column_stack([oof[n] for n in support])
    w0 = np.array([greedy_weights.get(n, 0.0) for n in support])
    if w0.sum() <= 0:
        w0 = np.full(len(support), 1.0 / len(support))
    w0 = w0 / w0.sum()

    def neg_obj(w):
        return -rank_objective(M @ w, y)

    res = minimize(
        neg_obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(support),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 300},
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    weights = {n: float(v) for n, v in zip(support, w) if v > 1e-4}
    total = sum(weights.values())
    weights = {n: v / total for n, v in weights.items()}
    blended = sum(oof[n] * v for n, v in weights.items())
    return weights, blended, rank_objective(blended, y)


# ---------------------------------------------------------------- fit/predict


def fit_scorer(weights, Xg, y, hand_mats):
    """Fit the selected members on the given data, return an EnsembleScorer."""
    group_protos = all_group_candidates()
    hand_protos = hand_candidates()
    Xh, yh, _ = stack_hands(hand_mats, y)

    group_members, hand_members = [], []
    fitted_hand = {}
    for name, w in weights.items():
        if "@" in name:
            base, agg = name.split("@")
            if base not in fitted_hand:
                model = clone(hand_protos[base])
                model.fit(Xh, yh)
                fitted_hand[base] = model
            agg_val = agg if agg == "mean" else float(agg)
            hand_members.append((fitted_hand[base], agg_val, w))
        else:
            model = clone(group_protos[name])
            model.fit(Xg, y)
            group_members.append((model, w))
    return EnsembleScorer(group_members, hand_members)


def predict_groups(scorer, Xg, hand_mats):
    """EnsembleScorer prediction from precomputed features (training path)."""
    n = Xg.shape[0]
    total_w = sum(w for _, w in scorer.group_members) + sum(
        w for _, _, w in scorer.hand_members
    )
    blended = np.zeros(n)
    for model, w in scorer.group_members:
        blended += w * model.predict_proba(Xg)[:, 1]
    if scorer.hand_members:
        lengths = [m.shape[0] for m in hand_mats]
        stacked = np.vstack([m for m in hand_mats if m.shape[0]])
        for model, agg, w in scorer.hand_members:
            probs = model.predict_proba(stacked)[:, 1]
            pos = 0
            for i, ln in enumerate(lengths):
                p = probs[pos : pos + ln]
                blended[i] += w * (
                    (p.mean() if agg == "mean" else np.quantile(p, float(agg)))
                    if p.size
                    else 0.5
                )
                pos += ln
    return blended / max(total_w, 1e-12)


# ---------------------------------------------------------------- main


def main():
    cache_dir = Path(sys.argv[1])
    dates_available = sorted(p.stem for p in cache_dir.glob("*.json"))
    holdout_date = (
        sys.argv[sys.argv.index("--holdout") + 1]
        if "--holdout" in sys.argv
        else dates_available[-1]
    )
    stability_date = dates_available[-2]
    print(
        f"{len(dates_available)} release dates | holdout: {holdout_date} | "
        f"stability check: {stability_date}"
    )

    Xg, y, dates, hand_mats = load_dataset(cache_dir)
    print(
        f"{len(y)} groups ({int(y.sum())} bot) | {Xg.shape[1]} group features | "
        f"{sum(m.shape[0] for m in hand_mats)} hands"
    )

    train_mask = dates != holdout_date
    tr_idx = np.flatnonzero(train_mask)
    Xg_tr, y_tr, d_tr = Xg[tr_idx], y[tr_idx], dates[tr_idx]
    hm_tr = [hand_mats[i] for i in tr_idx]
    ho_idx = np.flatnonzero(~train_mask)

    print("\n--- computing out-of-fold predictions (date-grouped 5-fold CV) ---")
    oof = compute_oof(Xg_tr, y_tr, d_tr, hm_tr)

    print(f"\n--- candidate leaderboard (OOF, top 25 of {len(oof)}) ---")
    board = sorted(
        (
            (
                rank_objective(p, y_tr),
                average_precision_score(y_tr, p),
                roc_auc_score(y_tr, p),
                name,
            )
            for name, p in oof.items()
        ),
        reverse=True,
    )
    for obj, ap, auc, name in board[:25]:
        print(f"{name:24s} rank_obj={obj:.4f} AP={ap:.4f} AUC={auc:.4f}")

    g_weights, g_blend, g_obj = greedy_ensemble(oof, y_tr)
    print(f"\ngreedy ensemble rank_obj={g_obj:.4f} weights={g_weights}")
    r_weights, r_blend, r_obj = refine_weights(oof, y_tr, g_weights)
    print(f"refined ensemble rank_obj={r_obj:.4f} weights={ {k: round(v, 3) for k, v in r_weights.items()} }")
    if r_obj > g_obj:
        weights, oof_blend, obj = r_weights, r_blend, r_obj
        print("using refined weights")
    else:
        weights, oof_blend, obj = g_weights, g_blend, g_obj
        print("using greedy weights")

    threshold, oof_reward_val = best_threshold(oof_blend, y_tr)
    print(f"OOF reward={oof_reward_val:.4f} at threshold={threshold:.4f}")

    # Honest evaluation: fit on training dates only, score the two newest.
    scorer = fit_scorer(weights, Xg_tr, y_tr, hm_tr)
    p_ho = predict_groups(scorer, Xg[ho_idx], [hand_mats[i] for i in ho_idx])
    ho_reward, ho_metrics = evaluate(p_ho, y[ho_idx], threshold)
    print(f"\n--- holdout ({holdout_date}) ---")
    print(f"REWARD = {ho_reward:.4f}")
    for key in ("ap_score", "bot_recall", "fpr", "human_safety_penalty",
                "hard_bot_recall", "hard_fpr"):
        print(f"  {key} = {ho_metrics[key]:.4f}")

    stab_mask = (dates != stability_date) & (dates != holdout_date)
    st_idx = np.flatnonzero(stab_mask)
    sc_idx = np.flatnonzero(dates == stability_date)
    scorer_stab = fit_scorer(
        weights, Xg[st_idx], y[st_idx], [hand_mats[i] for i in st_idx]
    )
    p_st = predict_groups(scorer_stab, Xg[sc_idx], [hand_mats[i] for i in sc_idx])
    st_reward, st_metrics = evaluate(p_st, y[sc_idx], threshold)
    print(f"\n--- stability check ({stability_date}) ---")
    print(
        f"REWARD = {st_reward:.4f} (ap={st_metrics['ap_score']:.4f} "
        f"hard_fpr={st_metrics['hard_fpr']:.4f})"
    )

    # Deployment artifact: refit the same recipe on ALL data (live batches
    # come from future dates; the newest release is the most relevant signal).
    print("\nrefitting selected ensemble on all data for deployment...")
    final_scorer = fit_scorer(weights, Xg, y, hand_mats)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_scorer,
            "threshold": threshold,
            "metadata": {
                "algorithm": "greedy-ensemble",
                "ensemble_weights": weights,
                "feature_version": FEATURE_VERSION,
                "feature_names": feature_names(),
                "train_dates": dates_available,
                "holdout_date": holdout_date,
                "oof_rank_objective": obj,
                "oof_reward": oof_reward_val,
                "holdout_reward": ho_reward,
                "stability_reward": st_reward,
                "candidate_leaderboard": [
                    {"name": name, "rank_obj": o, "ap": ap, "auc": auc}
                    for o, ap, auc, name in board
                ],
            },
        },
        MODEL_PATH,
    )
    print(f"artifact saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
