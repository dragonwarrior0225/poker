#!/usr/bin/env python3
"""Gate sweep for the upgraded Stack233+lgb model (whose natural positive
rate ~= the 0.15 cap, so the gate now BINDS and is worth tuning). Per-date
refit of the +lgb stack, evaluate max_frac at true live geometry + capture
sanity; bake the best into detector233.joblib.
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
    SEED, TARGET_FPR_GRID, featurize, load_examples, nested_oof,
    rank_within_dates, sanitize, union_features,
)
from upgrade_stack233_ranker import build_stack_plus, live_units  # noqa: E402

warnings.filterwarnings("ignore")
WF = 4
FRAC_GRID = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
OUT_PATH = REPO_ROOT / "neurons" / "models" / "detector233.joblib"
CAPTURE_GLOBS = (str(REPO_ROOT / "data" / "live_payloads" / "cycle_*.json"),
                 "/root/Poker44-subnet/data/live_payloads/cycle_*.json")


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
    t0 = time.time()
    chunks, y, dates = load_examples(cache_dir)
    X, order = featurize(chunks)
    Xr = rank_within_dates(X, dates)
    ud = sorted(set(dates.tolist()))
    tfpr = TARGET_FPR_GRID[0]
    print(f"gate sweep (+lgb): {len(y)} groups | {len(ud)} dates", flush=True)

    folds = []
    for td in ud[-WF:]:
        tr = dates < td
        if tr.sum() < 100:
            continue
        oof = nested_oof(Xr, y, dates, tr, SEED)
        human = oof[tr & (y == 0) & ~np.isnan(oof)]
        thr = float(np.quantile(human, 1.0 - tfpr))
        stack = build_stack_plus(SEED); stack.fit(Xr[tr], y[tr])
        units, labs = live_units(cache_dir, td, np.random.default_rng(7))
        rows = np.nan_to_num(np.asarray(
            [[union_features(u).get(k, 0.0) for k in order] for u in units], float))
        p = anchor_remap(stack.predict_proba(batch_rank(rows))[:, 1], thr)
        folds.append((p, labs))
        print(f"  fold {td}: {len(units)} units ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    for frac in FRAC_GRID:
        rs = [reward(power_decay_budget(p.copy(), max_frac=frac), labs)[0] for p, labs in folds]
        res[frac] = (float(np.mean(rs)), [round(x, 3) for x in rs])
    base = res[0.15][0]
    for frac in FRAC_GRID:
        print(f"  max_frac={frac}: mean={res[frac][0]:.4f} folds={res[frac][1]}", flush=True)
    best = max(FRAC_GRID, key=lambda f: (res[f][0], -abs(f - 0.15)))
    print(f"\nBEST max_frac={best} mean={res[best][0]:.4f} "
          f"({res[best][0]-base:+.4f} vs current 0.15)", flush=True)

    art = joblib.load(OUT_PATH); m = art["model"]
    m.max_frac = float(best)
    seen, cc = set(), []
    for g in CAPTURE_GLOBS:
        for f in sorted(glob.glob(g)):
            try: cap = json.load(open(f))
            except Exception: continue
            for c in cap.get("chunks") or []:
                k = (len(c), json.dumps(c[0], sort_keys=True)[:200] if c else "")
                if k not in seen: seen.add(k); cc.append(c)
    fr = [float((np.asarray(m.predict_chunks(cc[i:i+100])) >= 0.5).mean())
          for i in range(0, len(cc), 100) if len(cc[i:i+100]) >= 20]
    md = dict(art.get("metadata", {}))
    md.update({"gate_recontrol_max_frac": best, "gate_recontrol_mean": res[best][0],
               "gate_recontrol_baseline": base,
               "live_capture_pos_fraction": float(np.mean(fr)) if fr else None})
    joblib.dump({"model": m, "threshold": art.get("threshold", 0.5), "metadata": md}, OUT_PATH)
    print(f"saved max_frac={best} live_pos_frac={np.mean(fr) if fr else 0:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
