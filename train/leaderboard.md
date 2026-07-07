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
| ppo-l1-v13-pickup-s14⁴ | 38.1 | 33.5 | 76 | 1/32 | 24 |
| ppo-l1-v14-gear⁵ | 28.0 | 26.0 | 67 | 1/32 | 19 |
| ppo-l1-v15-acshape⁶ | 31.3 | 30.5 | 66 | 2/32 | 19 |

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

⁴ Same config as v13, training seed 14 — the project's first same-config
repeat, run for error bars, not a new generation (the champion row remains
the pre-registered seed-13 run; picking the better of two repeats would be
a selection effect). The kill level replicates (35.2 / 38.1); the style
does not: deaths 12 vs 21/32, real-drink share 93.4% vs 45.7% (pooled
65%), and idle-spam seeds go 1 → 3 (9001 presses pickup for its entire
episode). Outcome-level claims are robust; behaviour-composition claims
carry single-run error bars.

⁵ The gear chapter's registered predictions went 0/5. The instrumented
autopsy found the mechanism: across all 32 episodes the policy pressed the
gear key 6 times and equipped nothing (0/32) — armor's payoff (a few
percent less damage, spread over hundreds of steps) is invisible to the
reward stream at this training horizon, so the button never earns value
(lesson 13: observable preconditions are necessary, not sufficient).
Kills 28.0 and reached-L2 19/32 sit below the v13 config's two-run spread;
v13 keeps the crown.

⁶ v15 = v14's config plus a bounded one-shot reward (+0.5 × ΔAC) on
armor-class gains — lesson 13's cheapest prescription. Verdict: the gear
key was pressed **once** in 48,000 evaluation steps (v14: six), equips
stayed 0/32 — a shaping bonus cannot amplify an event exploration almost
never completes (lesson 14). Registered predictions 2/4: mean ≥30 hit
(31.3), deaths ≤11 hit at an all-time low (9/32; previous best 10), both
gear predictions obliterated. Real-drink share drew 60% (fourth hand of
the style lottery: 93/46/37/60%). v13 keeps the crown.

Long-episode probe (max_steps 1500 → 3000, same protocol): per-seed
kills are bit-identical for both v6 and v10 at both horizons (32/32
seeds each) — extra time buys zero additional kills; see
docs/DESIGN.md lesson 9.
