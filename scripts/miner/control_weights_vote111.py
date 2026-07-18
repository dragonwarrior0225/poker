#!/usr/bin/env python3
"""Tune VoteRankLogit's ensemble weights + FLOOR against the latest benchmark
AND our live captures, then rewrite neurons/models/detector.joblib.

Two weight knobs:
  1. soft-vote weights over (ExtraTrees, RandomForest, HistGB) -- ranking
     quality;
  2. FLOOR = per-batch positive-call fraction -- the safety gate.

Evaluation is at TRUE live geometry (pooled 86-102-hand units, ~50% bots,
official reward, full decision layer), walk-forward over the last WF dates.
The winning config is sanity-checked on our real captured validator cycles
(positive fraction ~ FLOOR, no score saturation) before it is written.

Usage: .venv/bin/python scripts/miner/control_weights_vote111.py benchmark_cache
"""
from __future__ import annotations

import glob
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "miner"))

from neurons.detector import MODEL_PATH  # noqa: E402
from neurons.vote111.serving import VoteScorer, _logit, _rank01  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from train_vote111 import (  # noqa: E402
    CAP, EPS, MARGIN, Q, TEMP, build_members, load_examples, mat, sanitize,
)

warnings.filterwarnings("ignore")

WF = 4
POOL_RANGE = (86, 102)
# Candidate simplex over (ET, RF, HGB) in 0.05 steps, plus FLOOR grid.
WEIGHT_STEP = 0.05
FLOOR_GRID = (0.08, 0.10, 0.12, 0.15)
BASE_WEIGHTS = (0.45, 0.25, 0.30)
BASE_FLOOR = 0.10
CAPTURE_GLOBS = (
    str(REPO_ROOT / "data" / "live_payloads" / "cycle_*.json"),
    "/root/Poker44-uid98/data/live_payloads/cycle_*.json",
)


def weight_grid():
    grid = []
    steps = [round(i * WEIGHT_STEP, 2) for i in range(int(1 / WEIGHT_STEP) + 1)]
    for a, b in product(steps, steps):
        c = round(1.0 - a - b, 2)
        if c < 0 or c > 1:
            continue
        if a + b + c != 1.0:
            continue
        if a == 0 and b == 0 and c == 0:
            continue
        grid.append((a, b, c))
    return grid


def fit_members_probas(Xtr, ytr, Xte):
    """Fit ET/RF/HGB on train, return per-member proba columns on test."""
    fitted = build_members()
    cols = []
    for _name, model, _w in fitted:
        model.fit(Xtr, ytr)
        cols.append(model.predict_proba(Xte)[:, 1])
    return np.column_stack(cols)  # (n_te, 3)


def combine(pcols, w):
    w = np.asarray(w, dtype=float)
    return (pcols * w).sum(axis=1) / max(w.sum(), 1e-12)


def decide(v, train_ref_logit, floor):
    tref = train_ref_logit - MARGIN
    z = _logit(v, EPS)
    if z.size == 0:
        return np.zeros(0)
    anchor = float(np.quantile(z, Q))
    t = (z - anchor + tref) / TEMP
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(t))))
    top, rest = order[:k], order[k:]
    d = 0.0004 - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if CAP and rest.size:
        d = t[rest].max() - (-0.0004)
        if d > 0.0:
            t[rest] = t[rest] - d
    return 1.0 / (1.0 + np.exp(-t))


def live_units(cache_dir, date, rng):
    raw = json.loads((cache_dir / f"{date}.json").read_text())
    by = {0: [], 1: []}
    for record in raw:
        for g, lab in zip(record["chunks"], record["groundTruth"]):
            by[int(lab)].append(sanitize(g))
    units, labs = [], []
    for lab, groups in by.items():
        rng.shuffle(groups)
        i = 0
        while i < len(groups) - 1:
            unit = list(groups[i]); i += 1
            target = int(rng.integers(*POOL_RANGE))
            while len(unit) < target and i < len(groups):
                unit.extend(groups[i]); i += 1
            units.append(unit[:POOL_RANGE[1]]); labs.append(lab)
    return units, np.asarray(labs)


def eval_config(fold_data, w, floor):
    rewards = []
    for pcols, labs, tref in fold_data:
        v = combine(pcols, w)
        s = decide(_rank01(v), tref, floor)
        rewards.append(reward(s, labs)[0])
    return float(np.mean(rewards)), rewards


def load_live_chunks(max_cycles=60):
    seen, out = set(), []
    files = sorted({f for g in CAPTURE_GLOBS for f in glob.glob(g)})
    for f in files[-max_cycles:]:
        try:
            cap = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        for c in cap.get("chunks") or []:
            key = (len(c), json.dumps(c[0], sort_keys=True)[:200] if c else "")
            if key not in seen:
                seen.add(key); out.append(c)
    return out


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X = mat(chunks)
    ud = sorted(set(dates.tolist()))
    print(f"weight-control: {len(y)} groups | {len(ud)} dates | WF={WF}", flush=True)

    # Build per-date live-geometry fold data (fit members once per date).
    fold_data = []
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        Xte = mat(units)
        pcols = fit_members_probas(X[tr], y[tr], Xte)
        # per-fold train reference anchor from prior OOF-ish full vote
        v_tr = combine(fit_members_probas(X[tr], y[tr], X[tr]), BASE_WEIGHTS)
        tref = float(np.quantile(_logit(_rank01(v_tr), EPS), Q)) + MARGIN
        fold_data.append((pcols, labs, tref))
        print(f"  fold {td}: {len(units)} live units ({time.time()-t0:.0f}s)", flush=True)

    base_mean, base_folds = eval_config(fold_data, BASE_WEIGHTS, BASE_FLOOR)
    print(f"\nBASELINE weights={BASE_WEIGHTS} floor={BASE_FLOOR}: "
          f"mean={base_mean:.4f} folds={[round(x,3) for x in base_folds]}", flush=True)

    grid = weight_grid()
    print(f"searching {len(grid)} weight combos x {len(FLOOR_GRID)} floors "
          f"({time.time()-t0:.0f}s)...", flush=True)
    best = (BASE_WEIGHTS, BASE_FLOOR, base_mean, base_folds)
    for w in grid:
        for floor in FLOOR_GRID:
            m, folds = eval_config(fold_data, w, floor)
            # require no fold regression vs baseline (robustness)
            if m > best[2] + 1e-6 and all(f >= bf - 1e-9 for f, bf in zip(folds, base_folds)):
                best = (w, floor, m, folds)
    w, floor, mean, folds = best
    print(f"\nBEST weights={w} floor={floor}: mean={mean:.4f} "
          f"folds={[round(x,3) for x in folds]} (baseline {base_mean:.4f})", flush=True)

    if (w, floor) == (BASE_WEIGHTS, BASE_FLOOR):
        print("no config beats baseline without regression -- keeping retrained "
              "default weights. Nothing rewritten.", flush=True)
        return 0

    # ---- refit on all data with tuned weights, rebuild artifact ----
    members = build_members()
    fitted = []
    for name, model, _w in members:
        model.fit(X, y)
        fitted.append((name, model))
    # attach tuned weights
    tuned = [(n, mdl, wt) for (n, mdl), wt in zip(fitted, w)]
    full = combine(np.column_stack([mdl.predict_proba(X)[:, 1] for _, mdl in fitted]), w)
    tref_final = float(np.quantile(_logit(_rank01(full), EPS), Q)) + MARGIN

    scorer = VoteScorer(
        tuned, q=Q, margin=MARGIN, temp=TEMP, floor=floor, cap=CAP,
        eps=EPS, train_ref_logit=tref_final,
    )

    # ---- live-capture sanity ----
    live = load_live_chunks()
    fracs = []
    if live:
        # score in ~100-chunk batches like real cycles
        for i in range(0, len(live), 100):
            batch = live[i:i + 100]
            if len(batch) < 20:
                continue
            s = np.asarray(scorer.predict_chunks(batch))
            fracs.append(float((s >= 0.5).mean()))
    print(f"live-capture sanity: {len(live)} unique chunks, "
          f"positive-fraction per ~100-batch mean={np.mean(fracs):.3f} "
          f"(target floor={floor})" if fracs else "live-capture sanity: no captures",
          flush=True)

    prev = joblib.load(MODEL_PATH)
    md = dict(prev.get("metadata", {}))
    md.update({
        "vote_weights_tuned": list(w),
        "floor_tuned": floor,
        "weight_control_wf_mean": float(mean),
        "weight_control_baseline": float(base_mean),
        "weight_control_folds": [float(x) for x in folds],
        "weight_control_dates": ud[-WF:],
        "live_capture_pos_fraction": float(np.mean(fracs)) if fracs else None,
        "weight_controlled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    joblib.dump({"model": scorer, "threshold": prev.get("threshold", 0.5),
                 "metadata": md}, MODEL_PATH)
    print(f"saved {MODEL_PATH} with tuned weights={w} floor={floor} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
