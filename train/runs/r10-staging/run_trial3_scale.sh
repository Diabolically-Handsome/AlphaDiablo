#!/bin/bash
# 试跑三:尺度探针(主席令 2026-08-30 深夜)。经济 v2 原样、步数翻倍
# 24,576——考问「裸奔哥的自杀是没教够还是教不好」。R12 路线甲立案拼图。
# 保险丝 86,400s(24h,仅拦真死锁;实测需求 ~9h;主席「跑完这些步数就行」)。
set -u
cd ~/AlphaDiablo/diablogym
PY=.venv/bin/python
WZ=train/runs/r9-reeducation/staging/worker.zip
WSHA=2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006
OUT=train/runs/r10-staging
echo "=== TRIAL3-SCALE train start $(date +%T) ===" | tee -a "$OUT/trial3.log"
timeout 86400 $PY train/train_ppo.py --options --algo mppo --gamma 1.0 \
  --max-steps 3000 --n-steps 64 --num-envs 32 --total-steps 24576 \
  --worker-zip "$WZ" --worker-zip-sha256 "$WSHA" \
  --manager-policy-observation-view raw-v4 --reward-economy v2 \
  --run-name r12-scale-probe --ent-coef 0.02 --lr 3e-4 --seed 22 \
  >> "$OUT/train-trial3.log" 2>&1
RC=$?
echo "=== train rc=$RC $(date +%T) ===" | tee -a "$OUT/trial3.log"
[ $RC -ne 0 ] && exit $RC
$PY train/export_manager_npz.py \
  train/runs/r12-scale-probe/model_final.zip \
  train/runs/r12-scale-probe/policy.npz >> "$OUT/trial3.log" 2>&1 || exit 1
for P in a b; do
  if [ "$P" = a ]; then SEEDS=2114000-2114127; else SEEDS=2115000-2115127; fi
  timeout 7200 $PY train/eval_assembled.py --worker "$WZ" \
    --manager-npz train/runs/r12-scale-probe/policy.npz \
    --manager-policy-observation-view raw-v4 \
    --seeds "$SEEDS" --tag "r12-scale-$P" --reward-economy v2 \
    >> "$OUT/trial3-eval-$P.log" 2>&1
  echo "=== eval $P rc=$? $(date +%T) ===" | tee -a "$OUT/trial3.log"
done
echo "=== TRIAL3 complete $(date +%T) ===" | tee -a "$OUT/trial3.log"
