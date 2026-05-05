# Poker44 Training Benchmark

Public training benchmark guide for Poker44 subnet `126`.

## What This Is

Poker44 exposes a public training benchmark built from chunks that have already been used in evaluation.

The purpose is straightforward:

- give miners real historical chunks for supervised training;
- keep the payload aligned with what miners actually saw at inference time;
- keep the live competitive field separated from the public training benchmark.

This benchmark is served by the central backend API.

## What Makes It Safe To Use

The benchmark follows a simple rule:

- only chunks that have already been used in evaluation are released;
- once a chunk is released through the benchmark, it is treated as training-only;
- released chunks do not return to the live evaluation field;
- the dataset is updated once per day;
- released chunks include `groundTruth`.

That means miners can train on real historical evaluation material without polluting the active competitive field.

## Live Evaluation vs Training Benchmark

These are two different things.

Live evaluation:

- validators fetch the active canonical chunk;
- validators transform it into the miner-visible payload;
- miners receive unlabeled `DetectionSynapse(chunks=...)`;
- validators keep labels on their side for scoring.

Training benchmark:

- chunks are released only after they leave live evaluation;
- the API returns the miner-visible chunk payload;
- `groundTruth` is returned separately so miners can train on it.

## API Base

- `https://api.poker44.net/api/v1/benchmark`

## Endpoints

- `GET /api/v1/benchmark`
- `GET /api/v1/benchmark/releases`
- `GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`
- `GET /api/v1/benchmark/chunks/:chunkId`

## Endpoint Semantics

### `GET /api/v1/benchmark`

Returns benchmark status metadata, including:

- current release version
- total released chunks
- total released hands
- latest released source date

### `GET /api/v1/benchmark/releases`

Returns released daily benchmark groups.

Useful fields:

- `sourceDate`
- `chunkCount`
- `handCount`
- `releasedAt`
- `releaseVersion`

This is the best first call if you want to discover what benchmark data is already available.

### `GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`

Returns released chunks for one source day.

Each item includes:

- chunk metadata
- `chunks`
- `groundTruth`
- `groundTruthLabels`

Important:

- `chunks` is the miner-visible chunk payload
- `groundTruth` is one label per chunk
- `groundTruthLabels` is the semantic form of the same labels

Current label convention:

- `0 = human`
- `1 = bot`

## How To Read The Ground Truth

The benchmark keeps labels outside the hand payload itself.

That means:

- `chunks[n]` is the miner-visible payload for one scoring unit;
- `groundTruth[n]` is the numeric label for that same scoring unit;
- `groundTruthLabels[n]` is the semantic form of the same label.

Example:

- `groundTruth[n] = 0` means the chunk is human
- `groundTruth[n] = 1` means the chunk is bot

This is intentional. It mirrors the live miner contract, where the chunk payload and the label are separate concerns.

### `GET /api/v1/benchmark/chunks/:chunkId`

Returns one released benchmark chunk by id.

Use this if you need to reproduce one specific historical chunk exactly.

## Training Shape

The critical structure is:

- `chunks: list[list[hand]]`
- `groundTruth: list[int]`

That means:

- each item in `chunks` is one scoring unit
- each entry in `groundTruth` maps one-to-one to one chunk

If a response contains:

- `chunks[0]`
- `groundTruth[0] = 1`

then the first chunk is labeled bot.

## Miner-Visible Payload

The benchmark does not expose backend-only training labels or internal identifiers inside the hand payload itself.

Instead, it exposes the miner-visible hand shape:

- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`

And it keeps labels outside the hand payload in `groundTruth`.

## How To Use It

The benchmark is meant to support practical miner iteration.

Typical uses:

- supervised training
- offline validation
- calibration
- feature engineering
- regression testing against historical eval material

The most direct workflow is:

1. call `GET /api/v1/benchmark/releases`
2. choose one or more released `sourceDate` values
3. fetch chunks for those days
4. train on `chunks + groundTruth`
5. hold out one or more release days for offline validation
6. deploy and compare against live evaluation performance

## Good Practices

Use the benchmark to speed up iteration, but keep a few things in mind:

- do not assume one release day is enough
- do not assume chunk composition will never evolve
- do not optimize only for one historical slice
- use multiple released days when possible
- validate out of sample before shipping a new miner

Optimize for the live miner contract, and use the benchmark to speed up iteration.
