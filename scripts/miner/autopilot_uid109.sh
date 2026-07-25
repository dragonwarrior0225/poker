#!/bin/bash
# uid109 daily auto-updater (cron @ 02:00 UTC -- staggered 1h after uid98 so
# both heavy trainings don't run at once on the 7GB box).
# download latest benchmark -> retrain Stack233+lgb (OWN SEED, decorrelated
# from uid98) -> GUARD -> commit -> push origin main -> restart poker44_miner.
set -uo pipefail

export PATH="/root/.nvm/versions/node/v22.14.0/bin:/usr/local/bin:/usr/bin:/bin"
export HOME=/root
export POKER44_TRAIN_SEED=91109      # != uid98's default seed -> different ranker
REPO=/root/Poker44-subnet
PY="$REPO/.venv/bin/python"
BR=main
ART="$REPO/neurons/models/detector233.joblib"
BAK="$REPO/neurons/models/detector233.autopilot_bak.joblib"
HOTKEY=rhg0314
PORT=8091
PM2NAME=poker44_miner
LOG="$REPO/logs/autopilot_uid109.log"
FLOOR=0.70
MAX_DROP=0.10
mkdir -p "$REPO/logs"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

cd "$REPO" || { log "FATAL: cannot cd $REPO"; exit 1; }
log "=== uid109 autopilot start (branch $(git rev-parse --abbrev-ref HEAD), seed $POKER44_TRAIN_SEED) ==="

LOCK="$REPO/logs/.autopilot109.lock"
if ! mkdir "$LOCK" 2>/dev/null; then log "another run holds the lock; abort"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

OLD_WF=$("$PY" - <<PYEOF 2>/dev/null || echo 0
import warnings,joblib; warnings.filterwarnings("ignore")
try: print(round(float(joblib.load("$ART")["metadata"].get("wf_mean_reward",0)),4))
except Exception: print(0)
PYEOF
)
log "incumbent wf_mean=$OLD_WF"
cp -f "$ART" "$BAK" 2>/dev/null && log "backed up current artifact"

if "$PY" scripts/miner/download_benchmark.py benchmark_cache >>"$LOG" 2>&1; then
  log "benchmark: $(ls benchmark_cache/ | tail -1)"
else
  log "ERROR benchmark download failed; keeping current model"; exit 1
fi

if "$PY" scripts/miner/train_plus_only.py benchmark_cache >>"$LOG" 2>&1; then
  log "training finished"
else
  rc=$?; log "ERROR training failed (rc=$rc, likely OOM); restoring backup"
  cp -f "$BAK" "$ART"; exit 1
fi

read NEW_WF THRU POSFRAC SEEDV < <("$PY" - <<PYEOF 2>/dev/null
import warnings,joblib; warnings.filterwarnings("ignore")
m=joblib.load("$ART")["metadata"]
print(round(float(m.get("wf_mean_reward",0)),4), m.get("trained_through","?"),
      round(float(m.get("live_capture_pos_fraction",0) or 0),3), m.get("train_seed","?"))
PYEOF
)
NEW_WF=${NEW_WF:-0}; THRU=${THRU:-?}
log "new model: wf_mean=$NEW_WF through=$THRU live_pos_frac=$POSFRAC seed=$SEEDV"

DEPLOY=$("$PY" - <<PYEOF
n=float("$NEW_WF"); o=float("$OLD_WF")
print("yes" if (n>=$FLOOR and n>=o-$MAX_DROP) else "no")
PYEOF
)
if [ "$DEPLOY" != "yes" ]; then
  log "GUARD REJECTED (wf $NEW_WF < floor $FLOOR or < incumbent $OLD_WF-$MAX_DROP); restoring backup, no deploy"
  cp -f "$BAK" "$ART"; exit 0
fi

git add neurons/models/detector233.joblib
if git diff --cached --quiet; then log "no artifact change; nothing to deploy"; exit 0; fi
git commit -q -m "uid109 autopilot: retrain Stack233+lgb on $THRU (wf $NEW_WF, seed $SEEDV)

Automated daily update, seed $SEEDV (decorrelated from uid98). Walk-forward
live-geometry mean $NEW_WF, live-capture positive fraction $POSFRAC.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && log "committed $(git rev-parse --short HEAD)"

if git push origin "$BR" >>"$LOG" 2>&1; then
  log "pushed to origin/$BR"
else
  log "ERROR push failed; reverting commit + artifact, no restart"
  git reset --hard HEAD~1 >>"$LOG" 2>&1; cp -f "$BAK" "$ART"; exit 1
fi

# restart uid109 fresh (pm2 restart has been flaky; delete+start is reliable)
pm2 delete "$PM2NAME" >>"$LOG" 2>&1 || true
PYTHONPATH="$REPO" pm2 start "$REPO/neurons/miner.py" \
  --name "$PM2NAME" --interpreter "$PY" --cwd "$REPO" -- \
  --netuid 126 --wallet.name "$HOTKEY" --wallet.hotkey "$HOTKEY" \
  --subtensor.network finney --axon.port "$PORT" \
  --blacklist.force_validator_permit --logging.info >>"$LOG" 2>&1
pm2 save >>"$LOG" 2>&1
sleep 25
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  log "SUCCESS: uid109 restarted on $(git rev-parse --short HEAD), serving :$PORT"
else
  log "WARNING: miner not listening on :$PORT after restart; check pm2 logs"
fi
log "=== uid109 autopilot done ==="
