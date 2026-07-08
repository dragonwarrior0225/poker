"""Feature extraction and trained-model inference for Poker44 bot detection.

A chunk group is a list of sanitized hand payloads for one tracked player
(``metadata.hero_seat`` marks that player in each hand). Two feature levels:

- per-hand behavioral features (``hand_features``) used both directly by a
  hand-level model (trained on every hand, group label broadcast to hands)
  and aggregated into the group vector;
- per-group features (``extract_features``): mean/std aggregates plus
  group-only consistency signals — pooled bet-size distribution, sizing-grid
  regularity, half-vs-half stability, serial behavior flips. Bots behave
  uniformly across hands; humans drift.

``EnsembleScorer`` is the trained artifact: a weighted blend of group-level
classifiers and hand-level classifiers (aggregated per group). Probabilities
are monotonically remapped at inference so the trained decision threshold
lands exactly on 0.5, where the validator's threshold-sanity gate operates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "detector.joblib"
FEATURE_VERSION = 3

_ACTION_TYPES = ("fold", "call", "check", "bet", "raise")
_STREETS = ("preflop", "flop", "turn", "river")
_STREET_ORDER = {s: i for i, s in enumerate(_STREETS)}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _stats(values: Sequence[float]) -> tuple:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.max())


def hand_features(hand: dict) -> Dict[str, float]:
    """Behavioral features for one sanitized hand, centered on the hero."""
    meta = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    hero = meta.get("hero_seat")
    bb = meta.get("bb") or 0.0

    hero_actions = []
    facing = {}  # hero action index -> was facing a bet/raise on that street
    aggressor_streets = set()
    street_counts: Dict[str, int] = {s: 0 for s in _STREETS}
    hero_street_counts: Dict[str, int] = {s: 0 for s in _STREETS}
    hero_checked_streets = set()
    check_raise = 0.0
    for a in actions:
        street = a.get("street") or "preflop"
        if street in street_counts:
            street_counts[street] += 1
        is_hero = a.get("actor_seat") == hero
        if is_hero:
            facing[len(hero_actions)] = street in aggressor_streets
            hero_actions.append(a)
            if street in hero_street_counts:
                hero_street_counts[street] += 1
            if a.get("action_type") == "check":
                hero_checked_streets.add(street)
            if (
                a.get("action_type") in ("bet", "raise")
                and street in hero_checked_streets
            ):
                check_raise = 1.0
        if a.get("action_type") in ("bet", "raise"):
            aggressor_streets.add(street)

    n_hero = len(hero_actions)
    n_table = len(actions)

    hero_counts = {t: 0 for t in _ACTION_TYPES}
    for a in hero_actions:
        t = a.get("action_type")
        if t in hero_counts:
            hero_counts[t] += 1
    table_counts = {t: 0 for t in _ACTION_TYPES}
    for a in actions:
        t = a.get("action_type")
        if t in table_counts:
            table_counts[t] += 1

    # Reactions when facing aggression.
    faced = [i for i in range(n_hero) if facing[i]]
    fold_vs_raise = sum(
        1 for i in faced if hero_actions[i].get("action_type") == "fold"
    )
    call_vs_raise = sum(
        1 for i in faced if hero_actions[i].get("action_type") == "call"
    )
    raise_vs_raise = sum(
        1 for i in faced if hero_actions[i].get("action_type") == "raise"
    )

    # Preflop archetypes.
    limp = 0.0
    three_bet = 0.0
    for i, a in enumerate(hero_actions):
        if a.get("street") != "preflop":
            continue
        if (
            a.get("action_type") == "call"
            and not facing[i]
            and (a.get("normalized_amount_bb") or 0.0) <= 2.0
        ):
            limp = 1.0
        if a.get("action_type") == "raise" and facing[i]:
            three_bet = 1.0

    hero_sizes = [
        a.get("normalized_amount_bb") or 0.0
        for a in hero_actions
        if a.get("action_type") in ("call", "bet", "raise")
        and (a.get("normalized_amount_bb") or 0.0) > 0
    ]
    hero_pot_ratios = [
        _safe_div(a.get("amount") or 0.0, a.get("pot_before") or 0.0)
        for a in hero_actions
        if a.get("action_type") in ("bet", "raise")
    ]
    pot_odds = [
        _safe_div(
            a.get("amount") or 0.0,
            (a.get("pot_before") or 0.0) + (a.get("amount") or 0.0),
        )
        for a in hero_actions
        if a.get("action_type") == "call"
    ]
    raise_to_bb = [
        _safe_div(a.get("raise_to") or 0.0, bb)
        for a in hero_actions
        if a.get("action_type") == "raise" and (a.get("raise_to") or 0.0) > 0
    ]
    size_mean, size_std, size_max = _stats(hero_sizes)
    pot_ratio_mean, pot_ratio_std, _ = _stats(hero_pot_ratios)
    pot_odds_mean, pot_odds_std, _ = _stats(pot_odds)
    raise_to_mean, _, raise_to_max = _stats(raise_to_bb)

    hero_stack_bb = 0.0
    hero_showed = 0.0
    for p in players:
        if p.get("seat") == hero:
            hero_stack_bb = _safe_div(p.get("starting_stack") or 0.0, bb)
            hero_showed = 1.0 if p.get("showed_hand") else 0.0
    stacks_bb = [_safe_div(p.get("starting_stack") or 0.0, bb) for p in players]
    stack_mean, stack_std, _ = _stats(stacks_bb)

    streets_seen = {a.get("street") for a in actions if a.get("street")}
    max_street = max((_STREET_ORDER.get(s, 0) for s in streets_seen), default=0)
    first_hero = hero_actions[0] if hero_actions else None
    first_street = (
        _STREET_ORDER.get(first_hero.get("street"), 0) if first_hero else 0.0
    )

    vpip = any(
        a.get("action_type") in ("call", "bet", "raise") for a in hero_actions
    )
    pfr = any(
        a.get("action_type") == "raise" and a.get("street") == "preflop"
        for a in hero_actions
    )
    postflop_hero = sum(
        hero_street_counts[s] for s in ("flop", "turn", "river")
    )

    button = meta.get("button_seat")
    max_seats = meta.get("max_seats") or 0
    pos_rel = 0.0
    if hero is not None and button is not None and max_seats:
        pos_rel = ((hero - button) % max_seats) / max_seats

    pot_final_bb = _safe_div(
        max((a.get("pot_after") or 0.0) for a in actions) if actions else 0.0,
        bb,
    )

    # Hero action-sequence bigrams: bots repeat exact action patterns.
    hero_seq = [
        a.get("action_type")
        for a in hero_actions
        if a.get("action_type") in _ACTION_TYPES
    ]
    bigram_counts = {
        (a, b): 0.0 for a in _ACTION_TYPES for b in _ACTION_TYPES
    }
    for a, b in zip(hero_seq, hero_seq[1:]):
        bigram_counts[(a, b)] += 1.0
    bigram_denom = max(1, len(hero_seq) - 1)

    feats = {
        "n_table_actions": float(n_table),
        "n_hero_actions": float(n_hero),
        "n_players": float(len(players)),
        "max_seats": float(max_seats),
        "hero_share": _safe_div(n_hero, n_table),
        "hero_fold_ratio": _safe_div(hero_counts["fold"], n_hero),
        "hero_call_ratio": _safe_div(hero_counts["call"], n_hero),
        "hero_check_ratio": _safe_div(hero_counts["check"], n_hero),
        "hero_bet_ratio": _safe_div(hero_counts["bet"], n_hero),
        "hero_raise_ratio": _safe_div(hero_counts["raise"], n_hero),
        "hero_aggression": _safe_div(
            hero_counts["bet"] + hero_counts["raise"],
            hero_counts["call"] + hero_counts["check"] + hero_counts["fold"],
        ),
        "table_fold_ratio": _safe_div(table_counts["fold"], n_table),
        "table_raise_ratio": _safe_div(
            table_counts["raise"] + table_counts["bet"], n_table
        ),
        "vpip": 1.0 if vpip else 0.0,
        "pfr": 1.0 if pfr else 0.0,
        "limp": limp,
        "three_bet": three_bet,
        "check_raise": check_raise,
        "faced_aggression_ratio": _safe_div(len(faced), n_hero),
        "fold_vs_raise_ratio": _safe_div(fold_vs_raise, len(faced)),
        "call_vs_raise_ratio": _safe_div(call_vs_raise, len(faced)),
        "raise_vs_raise_ratio": _safe_div(raise_vs_raise, len(faced)),
        "first_action_fold": (
            1.0
            if first_hero and first_hero.get("action_type") == "fold"
            else 0.0
        ),
        "first_action_street": float(first_street),
        "postflop_share": _safe_div(postflop_hero, n_hero),
        "hero_size_mean_bb": size_mean,
        "hero_size_std_bb": size_std,
        "hero_size_max_bb": size_max,
        "hero_size_cv": _safe_div(size_std, size_mean),
        "hero_pot_ratio_mean": pot_ratio_mean,
        "hero_pot_ratio_std": pot_ratio_std,
        "pot_odds_mean": pot_odds_mean,
        "pot_odds_std": pot_odds_std,
        "raise_to_mean_bb": raise_to_mean,
        "raise_to_max_bb": raise_to_max,
        "n_streets": float(len(streets_seen)),
        "max_street": float(max_street),
        "pos_rel": pos_rel,
        "hero_stack_bb": hero_stack_bb,
        "hero_showed": hero_showed,
        "stack_mean_bb": stack_mean,
        "stack_std_bb": stack_std,
        "pot_final_bb": pot_final_bb,
        "actions_per_street": _safe_div(n_table, len(streets_seen)),
    }
    for s in _STREETS:
        feats[f"hero_{s}_share"] = _safe_div(hero_street_counts[s], n_hero)
    feats["hero_n_distinct_actions"] = float(len(set(hero_seq)))
    for (a, b), count in bigram_counts.items():
        feats[f"bg_{a}_{b}"] = count / bigram_denom
    return feats


_HAND_KEYS: List[str] | None = None


def hand_feature_names() -> List[str]:
    global _HAND_KEYS
    if _HAND_KEYS is None:
        _HAND_KEYS = sorted(hand_features({}).keys())
    return _HAND_KEYS


def hand_feature_matrix(chunk: List[dict]) -> np.ndarray:
    keys = hand_feature_names()
    if not chunk:
        return np.zeros((0, len(keys)), dtype=float)
    rows = []
    for hand in chunk:
        feats = hand_features(hand)
        rows.append([feats[k] for k in keys])
    return np.asarray(rows, dtype=float)


_GRID_STEP = 0.5  # bots tend to bet on a fixed size grid (in big blinds)


def _pooled_size_features(sizes: np.ndarray) -> List[float]:
    if sizes.size == 0:
        return [0.0] * 12
    qs = np.quantile(sizes, [0.1, 0.25, 0.5, 0.75, 0.9])
    iqr = float(qs[3] - qs[1])
    mean = float(sizes.mean())
    std = float(sizes.std())
    rounded = np.round(sizes, 2)
    unique_ratio = float(len(np.unique(rounded)) / sizes.size)
    on_grid = float(
        np.mean(np.abs(sizes / _GRID_STEP - np.round(sizes / _GRID_STEP)) < 0.1)
    )
    uniq = np.unique(np.round(sizes, 3))
    gap_std = float(np.diff(uniq).std()) if uniq.size >= 3 else 0.0
    return [*map(float, qs), iqr, mean, std, _safe_div(std, mean), unique_ratio, on_grid, gap_std]


_POOLED_SIZE_NAMES = [
    "size_q10", "size_q25", "size_q50", "size_q75", "size_q90",
    "size_iqr", "size_mean", "size_std", "size_cv",
    "size_unique_ratio", "size_on_grid", "size_gap_std",
]

_CONSISTENCY_COLS = ("vpip", "hero_fold_ratio", "hero_size_mean_bb", "hero_aggression")


def extract_features(chunk: List[dict]) -> np.ndarray:
    """One feature vector for a chunk group (list of hands for one player)."""
    keys = hand_feature_names()
    mat = hand_feature_matrix(chunk)
    n = mat.shape[0]
    if n == 0:
        return np.zeros(len(feature_names()), dtype=float)

    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    idx = {k: i for i, k in enumerate(keys)}

    # Pooled hero action distribution across the whole group + entropy.
    weights = mat[:, idx["n_hero_actions"]]
    total = weights.sum()
    pooled = [
        float(_safe_div((mat[:, idx[f"hero_{t}_ratio"]] * weights).sum(), total))
        for t in _ACTION_TYPES
    ]
    probs = [p for p in pooled if p > 0]
    entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0

    # Pooled sizing distribution across all hands in the group.
    sizes = []
    for hand in chunk:
        meta = hand.get("metadata") or {}
        hero = meta.get("hero_seat")
        for a in hand.get("actions") or []:
            if (
                a.get("actor_seat") == hero
                and a.get("action_type") in ("call", "bet", "raise")
                and (a.get("normalized_amount_bb") or 0.0) > 0
            ):
                sizes.append(a.get("normalized_amount_bb"))
    size_feats = _pooled_size_features(np.asarray(sizes, dtype=float))

    # Stability: first half vs second half of the (ordered) hand sequence,
    # plus serial flip rate of participation. Bots stay stable; humans drift.
    half = n // 2
    halves = []
    for col in _CONSISTENCY_COLS:
        c = mat[:, idx[col]]
        halves.append(
            abs(float(c[:half].mean()) - float(c[half:].mean())) if half else 0.0
        )
    vpip_seq = mat[:, idx["vpip"]]
    flips = (
        float(np.mean(vpip_seq[1:] != vpip_seq[:-1])) if n > 1 else 0.0
    )

    extras = [float(n), entropy, float(_safe_div(total, n)), flips]
    return np.concatenate(
        [means, stds, np.asarray(pooled + size_feats + halves + extras)]
    )


def feature_names() -> List[str]:
    keys = hand_feature_names()
    return (
        [f"{k}_mean" for k in keys]
        + [f"{k}_std" for k in keys]
        + [f"pooled_{t}_ratio" for t in _ACTION_TYPES]
        + _POOLED_SIZE_NAMES
        + [f"half_diff_{c}" for c in _CONSISTENCY_COLS]
        + ["n_hands", "pooled_action_entropy", "hero_actions_per_hand", "vpip_flip_rate"]
    )


class EnsembleScorer:
    """Weighted blend of group-level and hand-level classifiers.

    group_members: [(fitted sklearn-like model over group features, weight)]
    hand_members:  [(fitted sklearn-like model over hand features, agg, weight)]
                   where agg is "mean" or a float quantile in (0, 1).
    """

    def __init__(self, group_members, hand_members):
        self.group_members = group_members
        self.hand_members = hand_members

    @staticmethod
    def _aggregate(hand_probs: np.ndarray, agg) -> float:
        if hand_probs.size == 0:
            return 0.5
        if agg == "mean":
            return float(hand_probs.mean())
        return float(np.quantile(hand_probs, float(agg)))

    def predict_chunks(self, chunks: List[List[dict]]) -> np.ndarray:
        n = len(chunks)
        total_w = sum(w for _, w in self.group_members) + sum(
            w for _, _, w in self.hand_members
        )
        blended = np.zeros(n, dtype=float)

        if self.group_members:
            Xg = np.vstack([extract_features(c) for c in chunks])
            for model, w in self.group_members:
                blended += w * model.predict_proba(Xg)[:, 1]

        if self.hand_members:
            mats = [hand_feature_matrix(c) for c in chunks]
            lengths = [m.shape[0] for m in mats]
            stacked = (
                np.vstack([m for m in mats if m.shape[0]])
                if any(lengths)
                else np.zeros((0, len(hand_feature_names())))
            )
            for model, agg, w in self.hand_members:
                probs = (
                    model.predict_proba(stacked)[:, 1]
                    if stacked.shape[0]
                    else np.zeros(0)
                )
                pos = 0
                for i, ln in enumerate(lengths):
                    blended[i] += w * self._aggregate(probs[pos : pos + ln], agg)
                    pos += ln

        return blended / max(total_w, 1e-12)


class DetectorModel:
    """Loads the trained artifact and scores chunk groups.

    The artifact stores the fitted scorer and the decision threshold chosen
    during training; predictions are monotonically remapped so that threshold
    lands exactly on 0.5, where the validator's threshold-sanity gate
    operates.
    """

    def __init__(self, path: Path = MODEL_PATH):
        import joblib

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.metadata = artifact.get("metadata", {})

    def _remap(self, p: np.ndarray) -> np.ndarray:
        t = min(max(self.threshold, 1e-6), 1 - 1e-6)
        return np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)

    @staticmethod
    def _degenerate(chunk: List[dict]) -> bool:
        return not chunk or not any((h or {}).get("actions") for h in chunk)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        if not chunks:
            return []
        if hasattr(self.model, "predict_chunks"):
            raw = self.model.predict_chunks(chunks)
        else:
            X = np.vstack([extract_features(c) for c in chunks])
            raw = self.model.predict_proba(X)[:, 1]
        scores = np.clip(self._remap(raw), 0.0, 1.0)
        # A chunk with no behavioral evidence must not cross the validator's
        # 0.5 gate (>= 0.5 counts as a bot flag → human-FPR risk).
        for i, chunk in enumerate(chunks):
            if self._degenerate(chunk):
                scores[i] = 0.3
        return [float(round(s, 6)) for s in scores]
