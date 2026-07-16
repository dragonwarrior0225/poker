"""Feature views for PT2Bag (train == serve)."""

from __future__ import annotations

from typing import Any, Dict, List

from neurons.pt2bag.features_v2 import extract_features_v2
from neurons.pt2bag.schema_features import chunk_features


def base_view(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    hands = chunk or []
    feats = chunk_features(hands)
    n = float(len(hands))
    feats["hand_count"] = n
    feats["hand_count_inv"] = 1.0 / (1.0 + n)
    return feats


def wide_view(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    feats = dict(extract_features_v2(chunk or []))
    feats.update(base_view(chunk))
    return feats
