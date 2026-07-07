# Deep-water leaderboard — 32 fixed seeds, 3000-step episodes

Protocol: argmax + action masks, seeds 9000-9031 (never used for
training or hyper-parameter selection), 3000 steps/episode, idle
machine, engine pinned to `ENGINE_REF` in bootstrap.sh, reward
world = depth-progressive descent ladder (8×N per level). NOT
comparable to train/leaderboard.md (1500-step episodes).
See train/evaluate_deep.py.

| run | depth med | depth max | ≥L2 | ≥L3 | ≥L4 | deaths | mean kills | median |
|---|---|---|---|---|---|---|---|---|
| ppo-deep-v17¹ | 3.0 | 4 | 28 | 17 | 11 | 22 | 9.6 | 7.5 |
| ppo-deep-v18-deathprice² | 2.0 | 4 | 19 | 7 | 1 | 19 | 32.1 | 32.0 |

¹ Chapter opener: v16's masked stack + depth-progressive ladder +
3000-step episodes (6M steps = episode-count parity with the old
chapter's 3M@1500). Registered predictions 2/5 — depth median 3.0 ✓ and
≥L4 11/32 ✓; deaths 22/32 (≤16 ✗), gear 0/32 with *zero* gear-key
presses (≥8 ✗), farm-then-dive never emerged: 28/28 first descents at
character level 1, median step 138 (✗). The ladder built a
stair-rusher — depth pays 8/16/24 while a kill pays ~1 and death costs
2, so rushing is the rational solve of the prices we set, not the
behavior we meant. 16 of 22 deaths hit with an empty belt (the
potion-runway failure mode DevilutionX-AI's author describes replicates
here), and the armor audition never convened: no farming → no drops →
nothing to wear. Lesson 16.

² Lesson 16's single knob applied: death now costs 8×level (was flat
2). The pendulum swung hard back — kills 9.6 → 32.1 (v16-class
fighting under depth economics), first descent 3× later (median step
460), 13/32 episodes never leave L1, depth median 3.0 → 2.0, L4
11 → 1. Registered predictions 2/5: ever-equipped ≥4 smashed (15/32,
228 gear presses — the armor audition finally convened) and dry-death
share <50% hit (4/19 vs v17's 16/22); deaths ≤14 missed (19),
farm-then-dive stayed thin (5/19 first descents at char level ≥2 —
though those five include the best episodes on the board), L4 ≥4
missed. Death anatomy inverted: v17 died running dry, v18 dies fully
stocked (15/19 with potions still in the belt) — at character level
1-2, L2-L3 monsters burst faster than any belt can heal. The
bottleneck moved again: resources → character power. The farming↔diving
auction now needs finer prices, or a bigger budget for the level-up
spiral (farm L1 → dive → farm L2 → dive) to emerge.
