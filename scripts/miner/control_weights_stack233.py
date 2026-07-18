#!/usr/bin/env python3
"""Tune Stack233's positive-fraction gate (MAX_POS_FRAC) against the latest
benchmark AND our live captures, then bake it into detector233.joblib.

Stack233's meta weights are LEARNED (logistic stack over OOF base probs), so
the tunable "weight" is the serving gate: the power-decay positive budget
fraction. It is swept at true live geometry (pooled 86-102-hand units,
official reward, walk-forward last WF dates, no-fold-regression gate) and the
winner is sanity-checked on our real captured validator cycles.

Usage: .venv/bin/python scripts/miner/control_weights_stack233.py benchmark_cache
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

from neurons.stack233.serving import (  # noqa: E402
    anchor_remap, batch_rank, power_decay_budget,
)
from poker44.score.scoring import reward  # noqa: E402
from train_stack233 import (  # noqa: E402
    SEED, TARGET_FPR_GRID, build_stack, featurize, load_examples,
    nested_oof, rank_within_dates, sanitize, union_features,
)

warnings.filterwarnings("ignore")

WF = 4
POOL_RANGE = (86, 102)
FRAC_GRID = (0.08, 0.10, 0.12, 0.15, 0.20)
BASE_FRAC = 0.15


def fit_stack(Xr, y, seed):
    s = build_stack(seed)
    s.fit(Xr, y)
    return s


def project(group):
    return sanitize(group)
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"
CAPTURE_GLOBS = (
    str(REPO_ROOT / "data" / "live_payloads" / "cycle_*.json"),
    "/root/Poker44-subnet/data/live_payloads/cycle_*.json",
)


def live_units(cache_dir, date, rng):
    raw = json.loads((cache_dir / f"{date}.json").read_text())
    by = {0: [], 1: []}
    for record in raw:
        for g, lab in zip(record["chunks"], record["groundTruth"]):
            by[int(lab)].append(project(g))
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


def pregate_scores(stack, order, units):
    rows = np.nan_to_num(np.asarray(
        [[union_features(u).get(k, 0.0) for k in order] for u in units], dtype=float))
    Xr = batch_rank(rows)
    p = stack.predict_proba(Xr)[:, 1]
    return anchor_remap(np.asarray(p, dtype=float), stack_threshold[0])


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


stack_threshold = [0.5]  # set per fold


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    print(f"stack233 weight-control: {len(y)} groups | {len(ud)} dates | WF={WF}", flush=True)

    fold_scores = []  # (pregate_scores, labels)
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human_oof = oof[tr & (y == 0) & ~np.isnan(oof)]
        thr = float(np.quantile(human_oof, 1.0 - TARGET_FPR_GRID[0]))
        stack = fit_stack(Xr[tr], y[tr], SEED)
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        stack_threshold[0] = thr
        pg = pregate_scores(stack, order, units)
        fold_scores.append((pg, labs))
        print(f"  fold {td}: {len(units)} units thr={thr:.3f} ({time.time()-t0:.0f}s)", flush=True)

    def eval_frac(frac):
        rs = []
        for pg, labs in fold_scores:
            s = power_decay_budget(pg.copy(), max_frac=frac)
            rs.append(reward(s, labs)[0])
        return float(np.mean(rs)), rs

    base_mean, base_folds = eval_frac(BASE_FRAC)
    print(f"\nBASELINE max_frac={BASE_FRAC}: mean={base_mean:.4f} "
          f"folds={[round(x,3) for x in base_folds]}", flush=True)
    best = (BASE_FRAC, base_mean, base_folds)
    for frac in FRAC_GRID:
        m, folds = eval_frac(frac)
        print(f"  max_frac={frac}: mean={m:.4f} folds={[round(x,3) for x in folds]}", flush=True)
        if m > best[1] + 1e-6 and all(f >= bf - 1e-9 for f, bf in zip(folds, base_folds)):
            best = (frac, m, folds)
    frac, mean, folds = best
    print(f"\nBEST max_frac={frac}: mean={mean:.4f} (baseline {base_mean:.4f})", flush=True)

    # bake chosen frac into the already-retrained artifact
    art = joblib.load(OUT_PATH)
    scorer = art["model"]
    scorer.max_frac = float(frac)
    live = load_live_chunks()
    fracs = []
    for i in range(0, len(live), 100):
        b = live[i:i + 100]
        if len(b) < 20:
            continue
        s = np.asarray(scorer.predict_chunks(b))
        fracs.append(float((s >= 0.5).mean()))
    md = dict(art.get("metadata", {}))
    md.update({
        "max_frac_tuned": float(frac),
        "weight_control_wf_mean": float(mean),
        "weight_control_baseline": float(base_mean),
        "weight_control_folds": [float(x) for x in folds],
        "weight_control_dates": ud[-WF:],
        "live_capture_pos_fraction": float(np.mean(fracs)) if fracs else None,
        "weight_controlled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    joblib.dump({"model": scorer, "threshold": art.get("threshold", 0.5),
                 "metadata": md}, OUT_PATH)
    print(f"live-capture sanity: {len(live)} unique chunks pos-fraction "
          f"mean={np.mean(fracs):.3f} (target {frac})" if fracs else "no captures", flush=True)
    print(f"saved {OUT_PATH} with max_frac={frac} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
