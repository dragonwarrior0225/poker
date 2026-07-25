#!/bin/bash
# uid98 daily auto-updater (cron @ 01:00 UTC).
# download latest benchmark -> retrain Stack233+lgb -> GUARD -> commit -> push
# -> restart miner. Rolls back the artifact and aborts if training fails, the
# model is degenerate, or the push fails (so the manifest never points at a
# commit that isn't public, and a bad night can't tank the live score).
set -uo pipefail

# --- environment (cron has a bare env) ---
export PATH="/root/.nvm/versions/node/v22.14.0/bin:/usr/local/bin:/usr/bin:/bin"
export HOME=/root
REPO=/root/Poker44-uid98
PY="$REPO/.venv/bin/python"
BR=uid98-stack233
ART="$REPO/neurons/models/detector233.joblib"
BAK="$REPO/neurons/models/detector233.autopilot_bak.joblib"
PM2=/root/.nvm/versions/node/v22.14.0/bin/pm2
HOTKEY=rmb225
LOG="$REPO/logs/autopilot.log"
FLOOR=0.70       # reject degenerate models
MAX_DROP=0.10    # reject a catastrophic drop vs the incumbent
mkdir -p "$REPO/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

cd "$REPO" || { log "FATAL: cannot cd $REPO"; exit 1; }
log "=== autopilot start (branch $(git rev-parse --abbrev-ref HEAD)) ==="

# guard against overlapping runs
LOCK="$REPO/logs/.autopilot.lock"
if ! mkdir "$LOCK" 2>/dev/null; then log "another run holds the lock; abort"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# read incumbent wf before we overwrite the artifact
OLD_WF=$("$PY" - <<PYEOF 2>/dev/null || echo 0
import warnings,joblib; warnings.filterwarnings("ignore")
try: print(round(float(joblib.load("$ART")["metadata"].get("wf_mean_reward",0)),4))
except Exception: print(0)
PYEOF
)
log "incumbent wf_mean=$OLD_WF"
cp -f "$ART" "$BAK" 2>/dev/null && log "backed up current artifact"

# 1) latest benchmark
if "$PY" scripts/miner/download_benchmark.py benchmark_cache >>"$LOG" 2>&1; then
  log "benchmark: $(ls benchmark_cache/ | tail -1)"
else
  log "ERROR benchmark download failed; keeping current model"; exit 1
fi

# 2) retrain (memory-lean) -- overwrites $ART on success
if "$PY" scripts/miner/train_plus_only.py benchmark_cache >>"$LOG" 2>&1; then
  log "training finished"
else
  rc=$?; log "ERROR training failed (rc=$rc, likely OOM); restoring backup"
  cp -f "$BAK" "$ART"; exit 1
fi

# 3) GUARD: read new wf + sanity, decide deploy
read NEW_WF THRU POSFRAC < <("$PY" - <<PYEOF 2>/dev/null
import warnings,joblib; warnings.filterwarnings("ignore")
m=joblib.load("$ART")["metadata"]
print(round(float(m.get("wf_mean_reward",0)),4), m.get("trained_through","?"),
      round(float(m.get("live_capture_pos_fraction",0) or 0),3))
PYEOF
)
NEW_WF=${NEW_WF:-0}; THRU=${THRU:-?}
log "new model: wf_mean=$NEW_WF through=$THRU live_pos_frac=$POSFRAC"

DEPLOY=$("$PY" - <<PYEOF
n=float("$NEW_WF"); o=float("$OLD_WF")
print("yes" if (n>=$FLOOR and n>=o-$MAX_DROP) else "no")
PYEOF
)
if [ "$DEPLOY" != "yes" ]; then
  log "GUARD REJECTED (wf $NEW_WF < floor $FLOOR or < incumbent $OLD_WF-$MAX_DROP); restoring backup, no deploy"
  cp -f "$BAK" "$ART"; exit 0
fi

# 4) commit
git add neurons/models/detector233.joblib
if git diff --cached --quiet; then log "no artifact change; nothing to deploy"; exit 0; fi
git commit -q -m "uid98 autopilot: retrain Stack233+lgb on $THRU (wf $NEW_WF)

Automated daily update. Walk-forward live-geometry mean $NEW_WF, live-capture
positive fraction $POSFRAC (gate 0.15). Guard passed (>= floor $FLOOR and
>= incumbent $OLD_WF - $MAX_DROP).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && log "committed $(git rev-parse --short HEAD)"

# 5) push (manifest must be public before restart)
if git push origin "$BR" >>"$LOG" 2>&1; then
  log "pushed to origin/$BR"
else
  log "ERROR push failed; reverting commit + artifact, no restart"
  git reset --hard HEAD~1 >>"$LOG" 2>&1; cp -f "$BAK" "$ART"; exit 1
fi

# 6) restart miner (fresh start; pm2 restart has been flaky here)
HOTKEY=$HOTKEY PM2=$PM2 "$REPO/scripts/miner/run/run_miner_233.sh" >>"$LOG" 2>&1
sleep 25
if ss -ltn 2>/dev/null | grep -q ':8092 '; then
  log "SUCCESS: uid98 restarted on $(git rev-parse --short HEAD), serving :8092"
else
  log "WARNING: miner not listening on 8092 after restart; check pm2 logs"
fi
log "=== autopilot done ==="
