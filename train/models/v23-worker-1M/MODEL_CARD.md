# v23-worker-1M — The first learnable operator brain (FARM worker, 1M-step peak checkpoint)

The central model of the v23 verdict, kept for the record although it lost: gold seeds **77.0** (median 91.4,
3/32 deaths), missing all three P1 tiers (0.82× the 93.9 of v22-H with the scripted worker), so **the
replaceability claim does not hold**. The verdict is in the v23 chapter of docs/design/DESIGN.md and in
docs/prereg/PREREG-v23.md. Why it is kept: it is the first operator brain in this repository that makes decisions
outside the scripted trajectory and locally beats its teacher (first 16 probe seeds 105.7 vs 93.9 for the script,
+12.6%, 29% divergence; it spontaneously used the potion-pickup and equip keys the teacher never pressed). It is
also the physical evidence for "the anchor drifts in a reward desert" (draft lesson 18): the training trajectory
went 500k=94.0 → **1M=105.7 (peak)** → 1.5M=59.2 → 4M=42.6 (collapse, training stopped).

| Item | Value |
|---|---|
| Architecture | MaskablePPO MlpPolicy (64,64), γ=1.0, 298-dim observation, Discrete(15) with 11/12 masked |
| Training | On-policy in WorkerWindowEnv (frozen v22-H manager), BC warm start + 200k-step freeze, ent 0.005, run `ppo-worker-v23` (2026-07-10) |
| World | v20 rules; wage = raw reward − level-change bonus (the bonus is stripped from the wage to close an arbitrage; level-change rate 0.0 throughout) |
| SHA-256 prefix | `104f72fd8368bc23` (re-saved on 2026-09-23 with a neutral path in the zip metadata; weights and optimizer state unchanged; previous SHA-256 prefix `b6e1cbdd0137feca`) |
| Companion log | sentinel.jsonl (sentinel at 500k-step granularity: dry/fresh mix, action shares, termination-reason spectrum) (raw log, not published) |

## Reproduce

```bash
.venv/bin/python train/eval_assembled.py --worker train/models/v23-worker-1M/model --seeds 9000-9031
```

The observation contract is defined by `_worker_obs` in `python/diablogym/options_env.py`; the manager must be
`train/models/v22-h-manager` (numpy forward pass, G0' bit-level reconciliation).
