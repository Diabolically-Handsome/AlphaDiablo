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
| ppo-l1-v12-drink² | 12.3 | 10.0 | 46 | 9/32 | 26 |
| ppo-l1-v13-pickup³ | 35.2 | 29.0 | 65 | 1/32 | 25 |

¹ v5 predates the explore macro; evaluated post-hoc on the current env
(same 286-dim observation, it simply never selects action 10). v1-v4
(smaller observation) and the retired v7 branch (407-dim, footprint
channel) cannot be re-evaluated under this protocol.

² Built to cut deaths, and it did: 17/32 (v11) → 10/32 under this protocol.
But mean kills regressed and 4,715 of its 4,740 drink presses hit an empty
belt — the belt count is not in the observation, so the policy cannot learn
press discipline. Forensics in docs/DESIGN.md lesson 11.

³ New champion (obs 286→290: belt count + nearest floor heal; door-aware
pickup macro). The controlled companion to footnote ²: same drink button,
one observation change — real-drink share 0.5% → 93.4% (25 of 57 argmax
drinks below half HP, deepest at 1%). Deaths 17/32 (v11) → 12/32 while
fighting nearly twice as much. Residue: one seed (9001) migrated the idle
attractor onto the pickup key (1,448 no-op presses) — lesson 12.

Long-episode probe (max_steps 1500 → 3000, same protocol): per-seed
kills are bit-identical for both v6 and v10 at both horizons (32/32
seeds each) — extra time buys zero additional kills; see
docs/DESIGN.md lesson 9.
