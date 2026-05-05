<div align="center">
  <h1>🂡 <strong>Poker44</strong> — Poker Bot Detection Subnet</h1>
  <img src="poker44/assets/logopoker44.png" alt="Poker44 logo" style="width:320px;">
  <p>
    <a href="docs/validator.md">🔐 Validator Guide</a> &bull;
    <a href="docs/miner.md">🛠️ Miner Guide</a> &bull;
    <a href="docs/training-benchmark.md">📦 Training Benchmark</a> &bull;
    <a href="https://poker44.net">🌐 Platform</a>
  </p>
</div>

---

## Official Links

- X: https://x.com/poker44subnet
- Web: https://poker44.net
- Whitepaper: https://poker44.net/Poker44_Whitepaper.pdf

---

## What is Poker44?

Poker44 is a Bittensor subnet focused on one problem: detecting bots in online poker with
objective, reproducible evaluation.

Validators query miners with poker-behavior chunk payloads, score predictions, and publish
weights on-chain. Miners compete by returning robust bot-risk predictions that generalize to
evolving live-table behavior.

Poker44 is security infrastructure, not a poker room.

---

## Current Production Model

The current production direction is:

- live benchmark tables run on Poker44 platform infrastructure;
- those tables include both human and bot seats;
- hands are persisted to central platform SQL;
- `poker44-platform-backend` builds evaluation batches from those benchmark-table hands;
- validators do **not** run their own tables;
- validators fetch the active canonical batch set through the central eval API;
- validators send those batches to miners, compute rewards, and set weights.

On top of that, the current production path also carries the public
observability layer needed for daily competition:

- signed validator runtime snapshots;
- signed metagraph-backed network snapshots;
- public miners/network dashboard surfaces on `poker44-platform-*`;
- a daily competition view built on the canonical eval feed and the latest
  signed subnet snapshot.

The validator production path is now the central `provider_runtime` model.

---

## What Miners Receive Today

Miners receive `DetectionSynapse(chunks=...)`.

Current semantics:

- `chunks` is a list of chunks;
- each chunk is a list of hand payloads;
- validators expect one `risk_score` per chunk;
- each chunk may contain one or many hands.

This means:

- the overall validator request can contain both human-labeled and bot-labeled chunks;
- each individual chunk is homogeneous, so the hands inside a chunk are all human or all bot;
- miners should treat each chunk as one scoring unit, regardless of how many hands it contains.

The competition framing should be understood as:

- daily epoch as the public competition unit;
- continuous evaluation on canonical live batches during that epoch;
- public provisional leaderboard derived from the signed subnet snapshot;
- target settlement model: winner-take-all.

In the current runtime, validators read the canonical competition weight
vector from the backend. Once the backend has settled at least one daily
winner, that latest settled winner becomes the active on-chain competitive
allocation for the current period: `97%` is burned to `uid 0`, and the
remaining `3%` follows the backend-provided winner vector. Before the first
settlement exists, the backend returns its explicit fallback vector (typically
`uid 0`, which keeps the burn at `100%`).

See:

- [Miner Guide](docs/miner.md)
- [Validator Guide](docs/validator.md)
- [Training Benchmark](docs/training-benchmark.md)

---

## Data Model Boundary

Production validators now target:

- live hands from Poker44 benchmark tables;
- SQL-persisted events and hand results;
- centralized batch generation through `/internal/eval/*`.

The repo may still include reference tooling for miner development, but production evaluation is
driven by the central platform runtime and should not be inferred from local helper artifacts.

---

## Open-Source Miner Standard

Poker44 supports a lightweight `model_manifest` attached to miner responses.

This does not change validator scoring or on-chain `set_weights`. It adds:

- traceability
- training-data disclosure
- transparency metadata
- evaluation observability

Recommended manifest fields include:

- repo URL
- repo commit or tag
- model name and version
- framework
- license
- training-data statement
- data-handling attestation

---

## Quick Start

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then follow:

- [Validator setup](docs/validator.md)
- [Miner setup](docs/miner.md)
- [Training benchmark](docs/training-benchmark.md)

Validated current production-like validator profile:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `POKER44_CHUNK_COUNT=80`
- `POKER44_REWARD_WINDOW=40`
- `POKER44_POLL_INTERVAL_SECONDS=300`
- `--neuron.timeout 60`

---

## Repository Links

- Validator docs: [`docs/validator.md`](docs/validator.md)
- Miner docs: [`docs/miner.md`](docs/miner.md)
- Training benchmark docs: [`docs/training-benchmark.md`](docs/training-benchmark.md)

---

## License

MIT — see [`LICENSE`](LICENSE).
