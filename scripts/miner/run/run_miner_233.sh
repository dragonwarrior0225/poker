#!/bin/bash
# Second Poker44 miner (UID233-style Stack233 model) on the same repo.
# The primary miner (uid38, VoteRankLogit) is untouched: this process points
# POKER44_MODEL_PATH at the stack233 artifact and uses its own hotkey/port.
#
# Usage: HOTKEY=<hotkey-name> ./scripts/miner/run/run_miner_233.sh

set -euo pipefail

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-rhg0314}"
HOTKEY="${HOTKEY:?set HOTKEY=<second registered hotkey>}"
NETWORK="${NETWORK:-finney}"
AXON_PORT="${AXON_PORT:-8092}"
PM2_NAME="${PM2_NAME:-poker44_miner_233}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

pm2 delete "$PM2_NAME" 2>/dev/null || true

cd "$REPO_ROOT"
POKER44_MODEL_PATH="$REPO_ROOT/neurons/models/detector233.joblib" \
PYTHONPATH="$REPO_ROOT" \
pm2 start "$REPO_ROOT/neurons/miner.py" \
  --name "$PM2_NAME" \
  --interpreter "$REPO_ROOT/.venv/bin/python" -- \
  --netuid "$NETUID" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY" \
  --subtensor.network "$NETWORK" \
  --axon.port "$AXON_PORT" \
  --blacklist.force_validator_permit \
  --logging.info

pm2 save
echo "Second miner started: $PM2_NAME (hotkey=$HOTKEY, port=$AXON_PORT, model=detector233.joblib)"
