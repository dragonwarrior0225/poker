"""features_v2: hero-free, sanitization-invariant chunk features.

WHY this exists: the validator serves each hand through
`poker44.validator.payload_view.prepare_hand_for_miner`, which re-aliases seats
(so absolute hero_seat / button position is meaningless), forces button_seat=0,
coarsens every bet amount into 16 fixed bb-buckets with hash noise, and keeps
only a random 5-8 action window per hand (so the hero frequently has ZERO visible
actions). Our old features.py keyed everything off `actor_seat == hero_seat` and
raw amounts/positions, so on the live feed it collapsed to near-constant zeros
=> live AP 0.42 vs 0.82 on raw benchmark.

features_v2 follows the approach the top miners use:
  * every statistic is computed over ALL actions in the hand (hero only enters as
    a *share*), so it degrades gracefully when the hero is absent;
  * bet sizes are quantized to the validator's exact bb-bucket grid, cancelling
    the injected bucket noise;
  * cross-hand "signature" regularity features detect bots that replay near
    identical action / sizing sequences (chunk-size & hero invariant);
  * per-hand scalars are aggregated to the chunk with order-stats
    (mean/std/min/max/q10/q50/q90), so 30-hand and 85-hand chunks look alike.

Train == serve: featurize SANITIZED hands at train time (run each hand through
prepare_hand_for_miner) so the model sees the same distribution it serves.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

# the validator's exact visible bb-bucket grid (payload_view._VISIBLE_BB_BUCKETS)
_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0,
           56.0, 84.0, 126.0)
_MEANINGFUL = ("check", "call", "bet", "raise", "fold")
_AGGR = ("bet", "raise")
_PASSIVE = ("check", "call")
_STREETS = ("preflop", "flop", "turn", "river")
# per-hand scalar feature keys aggregated across the chunk
_AGG_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _f(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except (TypeError, ValueError):
        return d


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _bucket_index(bb: float) -> int:
    return min(range(len(_BUCKETS)), key=lambda i: abs(_BUCKETS[i] - bb))


def _entropy(counts: List[float]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in ps)
    return ent / math.log(len(ps))  # normalized to [0,1]


def _max_run_share(seq: List[Any]) -> float:
    if not seq:
        return 0.0
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best / len(seq)


def _switch_rate(seq: List[Any]) -> float:
    if len(seq) < 2:
        return 0.0
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) / (len(seq) - 1)


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _hand_view(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Per-hand summary + the signature tuples used for cross-hand regularity."""
    meta = hand.get("metadata") or {}
    hero = meta.get("hero_seat")
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    a_types, actors, a_streets, buckets, roles = [], [], [], [], []
    amts, pots_before, pots_after = [], [], []
    for a in actions:
        at = str(a.get("action_type") or "").strip().lower()
        if not at:
            continue
        a_types.append(at)
        seat = a.get("actor_seat")
        actors.append(seat)
        roles.append("H" if (hero is not None and seat == hero) else "o")
        a_streets.append(str(a.get("street") or "").strip().lower())
        bb = _f(a.get("normalized_amount_bb"))
        buckets.append(_bucket_index(bb))
        if at in _AGGR and bb > 0:
            amts.append(bb)
        pots_before.append(_f(a.get("pot_before")))
        pots_after.append(_f(a.get("pot_after")))

    n = len(a_types)
    meaningful = [t for t in a_types if t in _MEANINGFUL]
    nm = max(len(meaningful), 1)
    cnt = Counter(a_types)

    feat: Dict[str, float] = {}
    feat["n_actions"] = float(n)
    for t in _MEANINGFUL:
        feat[f"share_{t}"] = _div(cnt.get(t, 0), nm)
    feat["share_blind"] = _div(cnt.get("small_blind", 0) + cnt.get("big_blind", 0), nm)
    feat["share_aggr"] = _div(sum(cnt.get(t, 0) for t in _AGGR), nm)
    feat["share_passive"] = _div(sum(cnt.get(t, 0) for t in _PASSIVE), nm)
    feat["aggr_ratio"] = _div(sum(cnt.get(t, 0) for t in _AGGR),
                              max(sum(cnt.get(t, 0) for t in _PASSIVE), 1))
    feat["share_preflop"] = _div(sum(1 for s in a_streets if s == "preflop"), max(n, 1))
    feat["share_postflop"] = _div(sum(1 for s in a_streets if s in ("flop", "turn", "river")), max(n, 1))

    feat["action_entropy"] = _entropy([cnt.get(t, 0) for t in set(a_types)])
    feat["actor_entropy"] = _entropy(list(Counter(actors).values()))
    feat["street_entropy"] = _entropy(list(Counter(a_streets).values()))
    feat["action_switch_rate"] = _switch_rate(a_types)
    feat["actor_switch_rate"] = _switch_rate(actors)
    feat["action_run_max_share"] = _max_run_share(a_types)
    feat["actor_run_max_share"] = _max_run_share(actors)

    feat["hero_action_share"] = _div(sum(1 for r in roles if r == "H"), max(n, 1))
    feat["n_distinct_actors"] = float(len(set(actors)))
    feat["n_streets"] = float(len(set(s for s in a_streets if s)) or len(streets))

    # bet sizing in bb (quantize-aware: amts are already coarse on live)
    if amts:
        s = sorted(amts)
        feat["amt_mean"] = sum(amts) / len(amts)
        feat["amt_std"] = (sum((x - feat["amt_mean"]) ** 2 for x in amts) / len(amts)) ** 0.5
        feat["amt_max"] = s[-1]
        feat["amt_q90"] = _quantile(s, 0.9)
        feat["amt_min"] = s[0]
    else:
        feat["amt_mean"] = feat["amt_std"] = feat["amt_max"] = feat["amt_q90"] = feat["amt_min"] = 0.0
    feat["nonzero_amt_share"] = _div(len(amts), max(n, 1))
    feat["bucket_entropy"] = _entropy(list(Counter(buckets).values()))

    # pot dynamics
    if pots_after:
        feat["pot_after_mean"] = sum(pots_after) / len(pots_after)
        feat["pot_before_mean"] = sum(pots_before) / len(pots_before)
        deltas = [pots_after[i] - pots_before[i] for i in range(len(pots_after))]
        feat["pot_delta_mean"] = sum(deltas) / len(deltas)
        feat["pot_growth"] = _div(pots_after[-1], pots_before[0]) if pots_before and pots_before[0] else 0.0
        feat["pot_monotonic_rate"] = _div(sum(1 for d in deltas if d >= 0), max(len(deltas), 1))
    else:
        feat["pot_after_mean"] = feat["pot_before_mean"] = feat["pot_delta_mean"] = 0.0
        feat["pot_growth"] = feat["pot_monotonic_rate"] = 0.0

    sig = {
        "action_sig": tuple(a_types),
        "role_sig": tuple(roles),
        "street_sig": tuple(a_streets),
        "bucket_sig": tuple(buckets),
    }
    return {"feat": feat, "sig": sig}


def _aggregate(prefix: str, series: List[float], out: Dict[str, float]) -> None:
    if not series:
        for st in _AGG_STATS:
            out[f"{prefix}_{st}"] = 0.0
        out[f"{prefix}_cv"] = 0.0
        return
    s = sorted(series)
    mean = sum(series) / len(series)
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = (sum((x - mean) ** 2 for x in series) / len(series)) ** 0.5
    out[f"{prefix}_min"] = s[0]
    out[f"{prefix}_max"] = s[-1]
    out[f"{prefix}_q10"] = _quantile(s, 0.10)
    out[f"{prefix}_q50"] = _quantile(s, 0.50)
    out[f"{prefix}_q90"] = _quantile(s, 0.90)
    out[f"{prefix}_cv"] = _div(out[f"{prefix}_std"], abs(mean))


def extract_features_v2(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    hands = hands or []
    out["hand_count"] = float(len(hands))
    if not hands:
        return out

    views = [_hand_view(h) for h in hands]
    # aggregate every per-hand scalar to chunk-level order-stats
    keys = list(views[0]["feat"].keys())
    for k in keys:
        _aggregate(k, [v["feat"].get(k, 0.0) for v in views], out)

    # cross-hand signature regularity (the bot "replays the same hand" tell)
    n = len(views)
    for name in ("action_sig", "role_sig", "street_sig", "bucket_sig"):
        sigs = [v["sig"][name] for v in views]
        c = Counter(sigs)
        out[f"{name}_top_share"] = _div(max(c.values()), n)
        out[f"{name}_unique_share"] = _div(len(c), n)
    # rate of "extreme" hands across the chunk
    out["high_aggr_hand_rate"] = _div(sum(1 for v in views if v["feat"]["share_aggr"] > 0.5), n)
    out["low_entropy_hand_rate"] = _div(sum(1 for v in views if v["feat"]["action_entropy"] < 0.3), n)
    out["zero_hero_action_rate"] = _div(sum(1 for v in views if v["feat"]["hero_action_share"] == 0.0), n)
    return out


def feature_names_v2(hands: List[Dict[str, Any]]) -> List[str]:
    return sorted(extract_features_v2(hands).keys())


if __name__ == "__main__":
    demo = [{}] * 3
    feats = extract_features_v2(demo)
    print(f"n_features={len(feats)} (on an empty 3-hand chunk)")
