#!/usr/bin/env python3
"""Thorough weight re-control for the Stack233 model: sweep BOTH the
operating-point threshold (target_fpr -> anchor-remap point) AND the
positive-fraction gate (max_frac), at true live geometry, and bake the best
combo into detector233.joblib. Per-date refit; each fit reused across the
whole (target_fpr x max_frac) grid.
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

from neurons.stack233.serving import anchor_remap, batch_rank, power_decay_budget  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from train_stack233 import (  # noqa: E402
    SEED, build_stack, featurize, load_examples, nested_oof, rank_within_dates,
    sanitize, union_features,
)

warnings.filterwarnings("ignore")
WF = 4
POOL_RANGE = (86, 102)
FPR_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.08)
FRAC_GRID = (0.08, 0.10, 0.12, 0.15, 0.20)
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"
CAPTURE_GLOBS = (
    str(REPO_ROOT / "data" / "live_payloads" / "cycle_*.json"),
    "/root/Poker44-uid98/data/live_payloads/cycle_*.json",
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


def load_live_chunks(max_cycles=80):
    seen, out = set(), []
    for f in sorted({f for g in CAPTURE_GLOBS for f in glob.glob(g)})[-max_cycles:]:
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
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    print(f"recontrol stack233: {len(y)} groups | {len(ud)} dates | "
          f"{len(FPR_GRID)}x{len(FRAC_GRID)} grid", flush=True)

    # per-date: fit once, cache pre-remap batch-ranked stack proba on units + human OOF quantiles
    folds = []
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human_oof = oof[tr & (y == 0) & ~np.isnan(oof)]
        stack = build_stack(SEED); stack.fit(Xr[tr], y[tr])
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        rows = np.nan_to_num(np.asarray(
            [[union_features(u).get(k, 0.0) for k in order] for u in units], float))
        p = stack.predict_proba(batch_rank(rows))[:, 1]
        thr = {f: float(np.quantile(human_oof, 1.0 - f)) for f in FPR_GRID}
        folds.append((p, labs, thr))
        print(f"  fold {td}: {len(units)} units ({time.time()-t0:.0f}s)", flush=True)

    grid = {}
    for f_fpr in FPR_GRID:
        for frac in FRAC_GRID:
            rs = []
            for p, labs, thr in folds:
                s = power_decay_budget(anchor_remap(p, thr[f_fpr]), max_frac=frac)
                rs.append(reward(s, labs)[0])
            grid[(f_fpr, frac)] = (float(np.mean(rs)), [round(x, 3) for x in rs])

    base = grid[(0.02, 0.15)][0]  # current config
    best_key = max(grid, key=lambda k: grid[k][0])
    best_mean, best_folds = grid[best_key]
    print(f"\ncurrent (fpr0.02, frac0.15): {base:.4f}", flush=True)
    # show best per target_fpr
    for f_fpr in FPR_GRID:
        row = {frac: round(grid[(f_fpr, frac)][0], 4) for frac in FRAC_GRID}
        print(f"  target_fpr={f_fpr}: {row}", flush=True)
    print(f"\nBEST: target_fpr={best_key[0]} max_frac={best_key[1]} "
          f"mean={best_mean:.4f} folds={best_folds} "
          f"({'+' if best_mean>base else ''}{best_mean-base:.4f} vs current)", flush=True)

    if best_mean <= base + 1e-6:
        print("No combo beats current config -> keeping fpr0.02/frac0.15. "
              "Nothing rewritten.", flush=True)
        return 0

    # re-derive threshold on ALL data at the winning target_fpr; bake frac
    oof = nested_oof(Xr, y, dates, np.ones(len(y), bool), SEED)
    human_oof = oof[(y == 0) & ~np.isnan(oof)]
    new_thr = float(np.quantile(human_oof, 1.0 - best_key[0]))
    art = joblib.load(OUT_PATH)
    scorer = art["model"]
    scorer.threshold = new_thr
    scorer.max_frac = float(best_key[1])
    live = load_live_chunks()
    fr = [float((np.asarray(scorer.predict_chunks(live[i:i+100])) >= 0.5).mean())
          for i in range(0, len(live), 100) if len(live[i:i+100]) >= 20]
    md = dict(art.get("metadata", {}))
    md.update({
        "recontrol_target_fpr": best_key[0], "recontrol_max_frac": best_key[1],
        "recontrol_threshold": new_thr, "recontrol_wf_mean": best_mean,
        "recontrol_baseline": base,
        "live_capture_pos_fraction": float(np.mean(fr)) if fr else None,
        "recontrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    joblib.dump({"model": scorer, "threshold": new_thr, "metadata": md}, OUT_PATH)
    print(f"saved {OUT_PATH}: target_fpr={best_key[0]} max_frac={best_key[1]} "
          f"thr={new_thr:.4f} live_pos_frac={np.mean(fr) if fr else 0:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
