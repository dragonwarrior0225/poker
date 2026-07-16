"""v5 features (UID111-style): behavioral + entropy + duplication-signature
features over a chunk (list of sanitized hands). Distribution-robust:
bb-normalized amounts, shares, entropy, run-length and signature features
(relative quantities that transfer live), deliberately excluding absolute-
level columns (raw bet/pot/stack magnitudes, hero/button identity shares)
that go out-of-distribution on the live feed and collapse predict_proba
spread.

Method reproduced from UID111 (poker111-vote)'s published inference recipe
(MIT-licensed reference implementation); reimplemented here, trained on our
own benchmark rows with our own seeds.
"""
from __future__ import annotations

import math
from collections import Counter

BB = 0.02  # visible big blind in the sanitized payload


def _f(v, d=0.0):
    try:
        return d if v is None else float(v)
    except (TypeError, ValueError):
        return d


def _i(v, d=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _div(a, b):
    return a / b if b else 0.0


def _entropy(xs):
    if not xs:
        return 0.0
    c = Counter(xs)
    if len(c) <= 1:
        return 0.0
    tot = float(sum(c.values()))
    e = -sum((n / tot) * math.log(n / tot + 1e-12) for n in c.values())
    return _div(e, math.log(len(c)))


def _quant(xs, q):
    if not xs:
        return 0.0
    s = sorted(float(x) for x in xs)
    if len(s) == 1:
        return s[0]
    pos = min(max(q, 0.0), 1.0) * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] if lo == hi else s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def _mean(xs):
    return _div(sum(xs), len(xs))


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(max(0.0, _mean([(x - m) ** 2 for x in xs])))


def _run_max_share(xs):
    if not xs:
        return 0.0
    longest = cur = 1
    for a, b in zip(xs, xs[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    return _div(longest, len(xs))


def _amt_bucket(v):
    if v <= 0.0:
        return "z"
    for thr, tag in ((0.5, "xs"), (1.0, "s"), (2.0, "m"), (5.0, "l")):
        if v <= thr:
            return tag
    return "xl"


def hand_features(hand):
    """Per-hand behavioral signals. Includes raw-magnitude fields (amt_*,
    pot_*, stack_*, hero/button share) that FEATURE_NAMES below excludes --
    keeping them here lets chunk_features aggregate a superset and the model
    input columns are selected purely by name filtering."""
    meta = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []
    max_seats = max(1, _i(meta.get("max_seats"), 6))
    hero = _i(meta.get("hero_seat"), 0)
    button = _i(meta.get("button_seat"), 0)

    atypes, actors, snames, amt, pb, pa = [], [], [], [], [], []
    raise_to = call_to = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        atypes.append(str(a.get("action_type") or "").lower().strip())
        s = _i(a.get("actor_seat"), 0)
        if s > 0:
            actors.append(s)
        snames.append(str(a.get("street") or "").lower().strip())
        amt.append(max(0.0, _f(a.get("normalized_amount_bb"))))
        pb.append(max(0.0, _div(_f(a.get("pot_before")), BB)))
        pa.append(max(0.0, _div(_f(a.get("pot_after")), BB)))
        raise_to += int(a.get("raise_to") is not None)
        call_to += int(a.get("call_to") is not None)
    stacks = [_div(_f(p.get("starting_stack")), BB) for p in players if isinstance(p, dict)]

    c = Counter(atypes)
    nact = max(1.0, float(len(actions)))
    meaningful = max(1, sum(c.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
    aggr = c.get("bet", 0) + c.get("raise", 0)
    passive = c.get("call", 0) + c.get("check", 0)
    pre = sum(1 for s in snames if s == "preflop")
    post = sum(1 for s in snames if s not in ("", "preflop"))
    pdelta = [max(0.0, x - y) for x, y in zip(pa, pb)]
    mono = sum(1 for x, y in zip(pa, pa[1:]) if y + 1e-9 >= x)

    return {
        "player_count": float(len(players)),
        "seat_util": _div(len(players), max_seats),
        "action_count": float(len(actions)),
        "street_count": float(len(streets)),
        "call_sh": _div(c.get("call", 0), meaningful),
        "check_sh": _div(c.get("check", 0), meaningful),
        "fold_sh": _div(c.get("fold", 0), meaningful),
        "bet_sh": _div(c.get("bet", 0), meaningful),
        "raise_sh": _div(c.get("raise", 0), meaningful),
        "aggr_sh": _div(aggr, nact),
        "passive_sh": _div(passive, nact),
        "preflop_sh": _div(pre, nact),
        "postflop_sh": _div(post, nact),
        "action_entropy": _entropy(atypes),
        "actor_entropy": _entropy(actors),
        "street_entropy": _entropy(snames),
        "unique_actor_sh": _div(len(set(actors)), max(1.0, len(players))),
        "actor_switch_rate": _div(
            sum(1 for x, y in zip(actors, actors[1:]) if x != y), max(len(actors) - 1, 1)
        ),
        "actor_run_max": _run_max_share(actors),
        "action_run_max": _run_max_share(atypes),
        "amt_mean": _mean(amt),
        "amt_std": _std(amt),
        "amt_q90": _quant(amt, 0.9),
        "nonzero_amt_sh": _div(sum(1 for v in amt if v > 0), nact),
        "pot_delta_mean": _mean(pdelta),
        "pot_growth": (max(pa) - min(pb)) if pa and pb else 0.0,
        "pot_monotonic": _div(mono, max(len(pa) - 1, 1)),
        "raise_to_sh": _div(raise_to, nact),
        "call_to_sh": _div(call_to, nact),
        "stack_std": _std(stacks),
        "stack_iqr": _quant(stacks, 0.75) - _quant(stacks, 0.25),
        "hero_action_sh": _div(sum(1 for s in actors if s == hero and hero > 0), nact),
        "button_action_sh": _div(sum(1 for s in actors if s == button and button > 0), nact),
    }


PERHAND = sorted(hand_features({"metadata": {}, "players": [], "streets": [], "actions": []}).keys())
_AGG = ("mean", "std", "min", "max", "q10", "q50", "q90")


def chunk_features(chunk):
    if not chunk:
        return {"hand_count": 0.0}
    out = {"hand_count": float(len(chunk))}
    rows = [hand_features(h) for h in chunk]
    for name in PERHAND:
        xs = [r[name] for r in rows]
        out[f"{name}_mean"] = _mean(xs)
        out[f"{name}_std"] = _std(xs)
        out[f"{name}_min"] = min(xs) if xs else 0.0
        out[f"{name}_max"] = max(xs) if xs else 0.0
        out[f"{name}_q10"] = _quant(xs, 0.1)
        out[f"{name}_q50"] = _quant(xs, 0.5)
        out[f"{name}_q90"] = _quant(xs, 0.9)

    asig, ksig, ssig, bsig = [], [], [], []
    low_ent = high_actor_ent = long_hand = 0
    for h, r in zip(chunk, rows):
        acts = h.get("actions") or []
        asig.append(tuple(str((a or {}).get("action_type") or "").lower() for a in acts))
        ksig.append(
            tuple(
                _i((a or {}).get("actor_seat"), 0)
                for a in acts
                if _i((a or {}).get("actor_seat"), 0) > 0
            )
        )
        ssig.append(tuple(str((a or {}).get("street") or "").lower() for a in acts))
        bsig.append(
            tuple(
                _amt_bucket(max(0.0, _f((a or {}).get("normalized_amount_bb"))))
                for a in acts
            )
        )
        low_ent += int(r["action_entropy"] <= 0.35)
        high_actor_ent += int(r["actor_entropy"] >= 0.75)
        long_hand += int(r["action_count"] >= 12.0)
    n = float(len(chunk))
    for tag, sig in (("action", asig), ("actor", ksig), ("street", ssig), ("amtbucket", bsig)):
        out[f"sig_{tag}_top_share"] = _div(max(Counter(sig).values()), n)
        out[f"sig_{tag}_unique_share"] = _div(len(set(sig)), n)
    out["low_action_entropy_rate"] = _div(low_ent, n)
    out["high_actor_entropy_rate"] = _div(high_actor_ent, n)
    out["long_action_hand_rate"] = _div(long_hand, n)
    return out


# v5 (sanitization-robust): the full chunk-feature vocabulary MINUS the
# fragile identity / raw-magnitude aggregates (hero/button action share, raw
# bet magnitudes amt_mean/std/q90, raw pot magnitudes pot_delta_mean/
# pot_growth, raw stack magnitudes stack_std/stack_iqr). Those columns go
# out-of-distribution on a validator-sanitized live feed and collapse the
# raw predict_proba spread. This is the exact column set/order the model is
# trained and served on -- do not reorder or regenerate ad hoc.
FEATURE_NAMES = ["action_count_max", "action_count_mean", "action_count_min", "action_count_q10", "action_count_q50", "action_count_q90", "action_count_std", "action_entropy_max", "action_entropy_mean", "action_entropy_min", "action_entropy_q10", "action_entropy_q50", "action_entropy_q90", "action_entropy_std", "action_run_max_max", "action_run_max_mean", "action_run_max_min", "action_run_max_q10", "action_run_max_q50", "action_run_max_q90", "action_run_max_std", "actor_entropy_max", "actor_entropy_mean", "actor_entropy_min", "actor_entropy_q10", "actor_entropy_q50", "actor_entropy_q90", "actor_entropy_std", "actor_run_max_max", "actor_run_max_mean", "actor_run_max_min", "actor_run_max_q10", "actor_run_max_q50", "actor_run_max_q90", "actor_run_max_std", "actor_switch_rate_max", "actor_switch_rate_mean", "actor_switch_rate_min", "actor_switch_rate_q10", "actor_switch_rate_q50", "actor_switch_rate_q90", "actor_switch_rate_std", "aggr_sh_max", "aggr_sh_mean", "aggr_sh_min", "aggr_sh_q10", "aggr_sh_q50", "aggr_sh_q90", "aggr_sh_std", "bet_sh_max", "bet_sh_mean", "bet_sh_min", "bet_sh_q10", "bet_sh_q50", "bet_sh_q90", "bet_sh_std", "call_sh_max", "call_sh_mean", "call_sh_min", "call_sh_q10", "call_sh_q50", "call_sh_q90", "call_sh_std", "call_to_sh_max", "call_to_sh_mean", "call_to_sh_min", "call_to_sh_q10", "call_to_sh_q50", "call_to_sh_q90", "call_to_sh_std", "check_sh_max", "check_sh_mean", "check_sh_min", "check_sh_q10", "check_sh_q50", "check_sh_q90", "check_sh_std", "fold_sh_max", "fold_sh_mean", "fold_sh_min", "fold_sh_q10", "fold_sh_q50", "fold_sh_q90", "fold_sh_std", "hand_count", "high_actor_entropy_rate", "long_action_hand_rate", "low_action_entropy_rate", "nonzero_amt_sh_max", "nonzero_amt_sh_mean", "nonzero_amt_sh_min", "nonzero_amt_sh_q10", "nonzero_amt_sh_q50", "nonzero_amt_sh_q90", "nonzero_amt_sh_std", "passive_sh_max", "passive_sh_mean", "passive_sh_min", "passive_sh_q10", "passive_sh_q50", "passive_sh_q90", "passive_sh_std", "player_count_max", "player_count_mean", "player_count_min", "player_count_q10", "player_count_q50", "player_count_q90", "player_count_std", "postflop_sh_max", "postflop_sh_mean", "postflop_sh_min", "postflop_sh_q10", "postflop_sh_q50", "postflop_sh_q90", "postflop_sh_std", "pot_monotonic_max", "pot_monotonic_mean", "pot_monotonic_min", "pot_monotonic_q10", "pot_monotonic_q50", "pot_monotonic_q90", "pot_monotonic_std", "preflop_sh_max", "preflop_sh_mean", "preflop_sh_min", "preflop_sh_q10", "preflop_sh_q50", "preflop_sh_q90", "preflop_sh_std", "raise_sh_max", "raise_sh_mean", "raise_sh_min", "raise_sh_q10", "raise_sh_q50", "raise_sh_q90", "raise_sh_std", "raise_to_sh_max", "raise_to_sh_mean", "raise_to_sh_min", "raise_to_sh_q10", "raise_to_sh_q50", "raise_to_sh_q90", "raise_to_sh_std", "seat_util_max", "seat_util_mean", "seat_util_min", "seat_util_q10", "seat_util_q50", "seat_util_q90", "seat_util_std", "sig_action_top_share", "sig_action_unique_share", "sig_actor_top_share", "sig_actor_unique_share", "sig_amtbucket_top_share", "sig_amtbucket_unique_share", "sig_street_top_share", "sig_street_unique_share", "street_count_max", "street_count_mean", "street_count_min", "street_count_q10", "street_count_q50", "street_count_q90", "street_count_std", "street_entropy_max", "street_entropy_mean", "street_entropy_min", "street_entropy_q10", "street_entropy_q50", "street_entropy_q90", "street_entropy_std", "unique_actor_sh_max", "unique_actor_sh_mean", "unique_actor_sh_min", "unique_actor_sh_q10", "unique_actor_sh_q50", "unique_actor_sh_q90", "unique_actor_sh_std"]
