# v22-H — Policy brain v1 (hierarchical manager, the first model added to the repository)

The first model to win a pre-registered live head-to-head (P7), archived unchanged.

## Identity

| Item | Value |
|---|---|
| Architecture | MaskablePPO `MlpPolicy` (64,64), γ=1.0, gae_lambda=0.95 |
| Observation | 303 dims (295 base + time remaining / stall clock / kills on this level / steps on this level / last option one-hot (3) / last τ) |
| Actions | Discrete(3): FARM / DIVE / RESUPPLY (option level, SMDP; the inner loop is the oracle's macros, frozen verbatim) |
| Training | 3M micro-steps on an Apple M1 Max, `train_ppo.py --options`, run `ppo-hier-v22-h` (2026-07-09) |
| Engine | DevilutionX pinned at `34c4cfc2e733` (bootstrap.sh ENGINE_REF) + the diablogym bridge (v20 world rules: descend ladder + death ladder + automatic stat allocation) |
| SHA-256 | `9dcf40b061bcb40b5548f30587d726758f2ea27df745dd7fb418c44adaa81c38` (re-saved on 2026-09-23 with a neutral path in the zip metadata; weights and optimizer state unchanged; previous SHA-256 `f3b579d2b0c9b613045692435a46702d1a9e8de8fc62e155c651f565d8bd6f1a`) |

## Results (gold seeds 9000-9031, final-evaluation protocol)

Mean return **93.9** (median 103.45), deaths **2/32**, paired wins against devil arm F 24/32.
The teacher script scores 101.5 but dies 25/32 — this model trades 7.5% of the return for 92% fewer deaths.
Full game records are in `train/leaderboard-hier.md`; the verdict is in the v22 chapter of `docs/design/DESIGN.md`.

## Reproduce / load

```bash
.venv/bin/python train/evaluate_options.py train/models/v22-h-manager/model_final --options
```

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("train/models/v22-h-manager/model_final", device="cpu")
```

Note: the observation contract (the 303-dim layout) is defined by `python/diablogym/options_env.py`.
Changing `_mgr_obs` in that file invalidates this model; it can only be retrained, not made compatible.
