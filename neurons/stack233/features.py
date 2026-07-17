"""UID233-style 543-feature union: 293 "ph_" schema + 250 "v2_" features.

Replicates hot-benchmark-poker-3's d8_features union (method reproduced from
the operator's published source; trained on our own data with our own seeds):
- ph_* : the phasberg/schema family (neurons.pt2bag.schema_features), with
  ph_hand_count overridden to len(chunk);
- v2_* : the sanitization-invariant family (neurons.pt2bag.features_v2).
Prefixing keeps overlapping names (hand_count, signature shares) as separate
columns, matching the published 543-column feature_order contract. Missing
keys default to 0.0 at vectorize time.
"""
from __future__ import annotations

from typing import Any, Dict, List

from neurons.pt2bag.features_v2 import extract_features_v2
from neurons.pt2bag.schema_features import chunk_features


def union_features(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    hands = chunk or []
    out: Dict[str, float] = {}
    ph = chunk_features(hands)
    ph["hand_count"] = float(len(hands))
    for k, v in ph.items():
        out[f"ph_{k}"] = float(v)
    for k, v in extract_features_v2(hands).items():
        out[f"v2_{k}"] = float(v)
    return out
