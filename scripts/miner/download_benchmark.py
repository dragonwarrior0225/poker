"""Download and cache all Poker44 benchmark releases."""
import json
import pathlib
import sys

import requests

BASE = "https://api.poker44.net/api/v1/benchmark"
CACHE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark_cache")
CACHE.mkdir(exist_ok=True)

data = requests.get(f"{BASE}/releases", params={"limit": 100}, timeout=60).json()["data"]
releases = data["releases"] if isinstance(data, dict) else data
dates = [r["sourceDate"] for r in releases]
print(f"{len(dates)} releases: {dates[-1]} .. {dates[0]}")

for date in dates:
    out = CACHE / f"{date}.json"
    if out.exists():
        continue
    records, cursor = [], None
    while True:
        params = {"sourceDate": date, "limit": 24}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/chunks", params=params, timeout=120).json()["data"]
        records.extend(resp["chunks"])
        cursor = resp.get("nextCursor")
        if not cursor:
            break
    out.write_text(json.dumps(records))
    n_groups = sum(len(r["chunks"]) for r in records)
    n_hands = sum(len(g) for r in records for g in r["chunks"])
    print(f"{date}: {len(records)} records, {n_groups} groups, {n_hands} hands", flush=True)

print("done")
