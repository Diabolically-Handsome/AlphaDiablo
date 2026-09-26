# v24-worker-leg7 — The leashed worker (the first learnable operator brain to beat the scripted system on gold)

**Gold seeds 97.2 (median 93.8, 2/32 deaths) against 93.9 for the v22-H scripted system: a strong P1 win, and the
replaceability claim is confirmed.** The same case has a second verdict, P-crutch-false ("still on the crutch"):
the winner was trained with a light anchor of β=0.0156, and leg 8 with β=0 collapsed as soon as it let go
(55.6, 79% divergence). Gold divergence rate 0.66%, verdict "continuing close to the anchor".
Full case: docs/prereg/PREREG-v24.md + the v24 chapter of docs/design/DESIGN.md + gate_ledger.jsonl (all eight legs recorded; raw ledger, not published).

| Item | Value |
|---|---|
| Architecture | LeashedMaskablePPO MlpPolicy (64,64), γ=1.0, 298-dim observation, Discrete(15) with 11/12 masked |
| Training | 7 legs, ~7.01M steps in total, on-policy (frozen v22-H manager); β schedule 0.5→0.015625 (frozen after a soft trip in leg 6); CE leash to the frozen BC teacher |
| World | v20 rules; wage = raw reward − level-change bonus (level-change rate ≤0.05% throughout) |
| SHA-256 prefix | `fb9cd6f58c5e2122` (re-saved on 2026-09-23 with a neutral path in the zip metadata; weights and optimizer state unchanged; previous SHA-256 prefix `ac65d4eb91fdb678`) |
| Design note | The worker was first trained with a scripted crutch (a KL anchor to the frozen BC teacher) that was meant to be annealed away (design, 2026-07-10). The first half was confirmed; the second half was honestly falsified: the annealing end point is not zero but a feather's weight (draft lesson 19) |

## Reproduce

```bash
.venv/bin/python train/eval_assembled.py --worker train/models/v24-worker-leg7/model --seeds 9000-9031
```

The manager must be `train/models/v22-h-manager` (numpy forward pass); the observation contract is in
`python/diablogym/options_env.py::_worker_obs`; the leash is implemented in `train/leashed_ppo.py`.
