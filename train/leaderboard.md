# Leaderboard — deterministic evaluation, 32 fixed seeds

Protocol: argmax policy, seeds 9000-9031 (never used for training or
hyper-parameter selection), 1500 steps/episode, idle machine, engine
pinned to `ENGINE_REF` in bootstrap.sh. See train/evaluate.py.
All rows re-measured 2026-07-05 on the same build.

| run | mean kills | median | max | zero-kill | reached L2 |
|---|---|---|---|---|---|
| ppo-l1-v5-vision¹ | 7.6 | 0.0 | 45 | 19/32 | 0 |
| ppo-l1-v6-explore | 8.8 | 3.5 | 36 | 15/32 | 0 |
| ppo-l1-v8-lstm | 8.4 | 3.0 | 43 | 13/32 | 0 |
| ppo-l1-v9c-attn | 3.8 | 0.0 | 38 | 21/32 | 0 |
| ppo-l1-v10-longep | 5.5 | 0.0 | 49 | 18/32 | 0 |
| ppo-l1-v11-descend | 19.4 | 14.5 | 70 | 2/32 | 27 |

¹ v5 predates the explore macro; evaluated post-hoc on the current env
(same 286-dim observation, it simply never selects action 10). v1-v4
(smaller observation) and the retired v7 branch (407-dim, footprint
channel) cannot be re-evaluated under this protocol.

Long-episode probe (max_steps 1500 → 3000, same protocol): per-seed
kills are bit-identical for both v6 and v10 at both horizons (32/32
seeds each) — extra time buys zero additional kills; see
docs/DESIGN.md lesson 9.
