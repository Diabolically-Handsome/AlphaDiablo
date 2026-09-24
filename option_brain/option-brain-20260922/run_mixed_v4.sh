#!/usr/bin/env bash
# WSL. Copy of run_mixed_v3b.sh for facts version 4 (round 9, PLAN-annotations-v4.md section 5): one fine-tuned broker
# on the RTX 5090 serving Skeleton King AND Butcher games at once, with the fact lines of FACTS_VERSION (default 4).
# Refuses to start (before the broker loads or anything is written) unless FACTS_VERSION equals the facts version the
# adapter was trained on, as recorded:
#   1. sft/<tag>/registration.json "facts_version" (option_sft.py writes it from round 9 on, from the relabel
#      directory's facts-report.json; the round-9 adapter is sft/d10facts4, the round-9b one sft/d11facts5 with facts
#      version 5: FACTS_VERSION=5), else
#   2. "facts_version" of the facts-report.json of the relabel directory named by that registration's "rel".
# An adapter with neither (e.g. d9facts3: relabel-d9f3's report predates the field) is refused; run it with
# run_mixed_v3.sh / run_mixed_v3b.sh, which set FACTS_VERSION=3. The two records must agree when both exist.
# Hands r3 (chain, target, post_hit, hold floor 0.5) + undo guard + chase + auto belt (training seeds only) + --facts,
# plus EXTRA_HANDS (e.g. --allow-heldout), exactly as run_mixed_v3b.sh.
# Usage: [FACTS_VERSION=4] run_mixed_v4.sh <batch> <adapter> <seed:mission:ticks:minutes[:out-batch]> ...
#        (adapter = $AD_ROOT/sft/<tag>/adapter; mission = skeleton_king | butcher)
# The broker bus lives in live/<batch>/bus; a game with an out-batch writes to live/<out-batch>/s<seed> (so the same
# seed can be played once per mission); request ids carry the out-batch tag, so they never collide.
# Copy of run_live_exec.sh's broker handling; every game starts at once (the broker is the shared bottleneck).
set -u
W=$AD_WORKSPACE/experiments/option-brain-20260922
PY_MODEL=$AD_HOME/strategy_brain_sft_20260922/runtime/venv/bin/python
PY_GAME=$AD_HOME/AlphaDiablo/.venv/bin/python
BATCH=$1; ADAPTER=$2; shift 2
export FACTS_VERSION=${FACTS_VERSION:-4}
case "$FACTS_VERSION" in 1|2|3|4|5) ;; *) echo "FACTS_VERSION must be 1, 2, 3, 4 or 5 (got '$FACTS_VERSION')"; exit 2 ;; esac
# the adapter's recorded facts version: "<version> <source>" or "ERR <why>" (the code is read into a variable first:
# a here-document inside $(...) trips older bash parsers)
read -r -d '' AFV_PY <<'PY'
import json, sys
from pathlib import Path
ROOT = Path('$AD_ROOT')
a = Path(sys.argv[1]).resolve()
run = a.parent if a.name == 'adapter' else a
weights = (run / 'adapter' / 'adapter_model.safetensors', a / 'adapter_model.safetensors')
if not any(w.exists() for w in weights):
    print(f'ERR no adapter weights under {a}')
    sys.exit()
reg_path = run / 'registration.json'
if not reg_path.exists():
    print(f'ERR no {reg_path}')
    sys.exit()
reg = json.loads(reg_path.read_text())
got = {}
if reg.get('facts_version') is not None:
    got['registration'] = int(reg['facts_version'])
rel = reg.get('rel')
rep = ROOT / rel / 'facts-report.json' if rel else None
if rep is not None and rep.exists():
    v = json.loads(rep.read_text()).get('facts_version')
    if v is not None:
        got[f'{rel}/facts-report.json'] = int(v)
if not got:
    print(f"ERR no facts version recorded for {a} ({reg_path} has none; rel {rel!r} has no facts-report.json with facts_version)")
elif len(set(got.values())) > 1:
    print(f'ERR the records disagree: {got}')
else:
    print(next(iter(got.values())), '+'.join(got))
PY
AFV=$(python3 -c "$AFV_PY" "$ADAPTER")
case "$AFV" in
  ERR*|"") echo "refused: ${AFV:-could not read the facts version of the adapter}"; exit 2 ;;
esac
AFV_N=${AFV%% *}; AFV_SRC=${AFV#* }
[ "$AFV_N" = "$FACTS_VERSION" ] || { echo "refused: FACTS_VERSION=$FACTS_VERSION but $ADAPTER was trained on facts version $AFV_N ($AFV_SRC)"; exit 2; }
echo "facts version $FACTS_VERSION matches the adapter ($AFV_SRC)"
HANDS="--executor r3 --r3-chain --r3-target --r3-chain-guard post_hit --r3-hold-floor 0.5 --undo-guard --r3-chase --facts --auto-belt ${EXTRA_HANDS:-}"
R=$AD_ROOT/live/$BATCH
[ -e $R/registration.json ] && { echo "batch exists"; exit 2; }
mkdir -p $R/bus
echo "{\"batch\":\"$BATCH\",\"adapter\":\"$ADAPTER\",\"games\":\"$*\",\"hands\":\"$HANDS\",\"facts_version\":$FACTS_VERSION,\"adapter_facts_version\":$AFV_N,\"adapter_facts_source\":\"$AFV_SRC\",\"time\":$(date +%s)}" > $R/registration.json
cd $W
$PY_MODEL -B hf_broker.py --adapter $ADAPTER --bus $R/bus > $R/broker.log 2>&1 &
BP=$!
until [ -f $R/bus/broker-ready.json ]; do
  kill -0 $BP 2>/dev/null || { echo "broker died"; tail -20 $R/broker.log; exit 3; }
  sleep 5
done
echo "broker ready $(date)"
pids=()
for g in "$@"; do
  IFS=: read -r SEED MISSION TICKS MINUTES OUTB <<< "$g"
  OUTB=${OUTB:-$BATCH}; O=$AD_ROOT/live/$OUTB
  [ "$OUTB" != "$BATCH" ] && { mkdir -p $O; [ -e $O/registration.json ] || echo "{\"batch\":\"$OUTB\",\"broker_batch\":\"$BATCH\",\"adapter\":\"$ADAPTER\",\"hands\":\"$HANDS\",\"facts_version\":$FACTS_VERSION,\"adapter_facts_version\":$AFV_N,\"time\":$(date +%s)}" > $O/registration.json; }
  [ -f $AD_ROOT/STOP ] && break
  ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      $PY_GAME -B live_runner.py --seed $SEED --mission $MISSION --ticks $TICKS --decisions 6000 --minutes $MINUTES \
      --bus $R/bus --out $O/s$SEED --tag $OUTB $HANDS > $O/s$SEED.log 2>&1; tail -1 $O/s$SEED.log | cut -c1-200 ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
touch $R/bus/STOP
wait $BP
echo "batch done $(date)"
