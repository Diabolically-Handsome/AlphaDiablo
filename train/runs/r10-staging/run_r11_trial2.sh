#!/bin/bash
# R11 工程试跑:经济 v3(= v2 + 主席版杀怪锁 K=3),单变量对照 R10。
# 主席令 2026-08-28 深夜:「就按您说的 今晚先试跑一轮」。
# 非正式战役:复用已消耗池 2_114/2_115 对账,不触碰任何处女池。
set -u
cd ~/AlphaDiablo/diablogym
PY=.venv/bin/python
WZ=train/runs/r9-reeducation/staging/worker.zip
WSHA=2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006
OUT=train/runs/r10-staging
echo "=== R11-TRIAL train start $(date +%T) ===" | tee -a "$OUT/r11-trial2.log"
timeout 28800 $PY train/train_ppo.py --options --algo mppo --gamma 1.0 \
  --max-steps 3000 --n-steps 64 --num-envs 32 --total-steps 12288 \
  --worker-zip "$WZ" --worker-zip-sha256 "$WSHA" \
  --manager-policy-observation-view raw-v4 --reward-economy v3b \
  --run-name r11-trial2-k1 --ent-coef 0.02 --lr 3e-4 --seed 22 \
  >> "$OUT/train-r11-trial2.log" 2>&1
RC=$?
echo "=== train rc=$RC $(date +%T) ===" | tee -a "$OUT/r11-trial2.log"
[ $RC -ne 0 ] && exit $RC
$PY train/export_manager_npz.py \
  train/runs/r11-trial2-k1/model_final.zip \
  train/runs/r11-trial2-k1/policy.npz >> "$OUT/r11-trial2.log" 2>&1 || exit 1
for P in a b; do
  if [ "$P" = a ]; then SEEDS=2114000-2114127; else SEEDS=2115000-2115127; fi
  timeout 7200 $PY train/eval_assembled.py --worker "$WZ" \
    --manager-npz train/runs/r11-trial2-k1/policy.npz \
    --manager-policy-observation-view raw-v4 \
    --seeds "$SEEDS" --tag "r11-trial2-$P" --reward-economy v3b \
    >> "$OUT/r11-trial2-eval-$P.log" 2>&1
  echo "=== eval $P rc=$? $(date +%T) ===" | tee -a "$OUT/r11-trial2.log"
done
echo "=== R11-TRIAL complete $(date +%T) ===" | tee -a "$OUT/r11-trial2.log"
