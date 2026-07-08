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
| ppo-deep-v19-powergauge³ | 1.5 | 4 | 16 | 10 | 1 | 15 | 26.1 | 22.0 |
| ppo-deep-v20-statline⁴ | 1.0 | 2 | 1 | 0 | 0 | 4 | 21.0 | 13.5 |

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

³ Null result, recorded as such. The power gauge (char-level /
dungeon-level ratio into the obs, 294→295 dims) did not crack the
farm-then-dive spiral at 6M steps: first-descent character level stayed
at median 1, and descender mortality is unchanged (v18 17/19 ≈ 89% →
v19 14/16 ≈ 88%). Deaths fell 19 → 15 by composition — half the seeds
now simply never leave L1. Weak positives: descenders that do go
convert to L3 at 10/16 (v18: 7/19), deaths now cluster at L3 rather
than L2, dry deaths 2/15, and gear stays alive (9/32 equipped, 266
presses). Real-drink share drew 77% — the style lottery's sixth hand
(93/46/37/60/4/77). Procedural note: this generation launched without
registered predictions (the night's one ritual miss) — scored
descriptively only. Chapter status: three auction knobs mapped
(ladder → rush; death price → retreat; visibility → split), the spiral
has not emerged at M1-Max budgets; the design doc's workstation line
(10× steps, IL warm-start) is the standing next move.

⁴ The chapter's closing verdict. v20 repaired the stat-point black hole
(auto-spend on level-up, verified in the wild: zero unspent points) and
extended credit sight (γ 0.99→0.997, half-life 69→231 steps) — and the
policy, reading the corrected books with long eyes, declined the entire
deep-water enterprise: 31/32 episodes never leave L1, deaths crash to
4/32 (safest ever), the drink key goes fully extinct (0 presses), and
farming stays casual (21 mean kills; two-thirds of seeds don't even
reach level 2 — leveling isn't worth maxing when you don't intend to
dive). Registered predictions: P3's sentinel fired exactly as
pre-written (>16/32 stay home), and P5's falsification line is met —
the mechanism×horizon composite did not produce the spiral. The honest
reading after five configurations: at level-1 start, 3000 steps,
melee-warrior terms, descending is negative-EV *even in a mechanically
sound world* — the refusal is not a failure to learn but a correct
measurement of the task. Real Diablo agrees: nobody fights through
depth at clvl 1-4; the leveling spiral spans hours, not 3000 steps.
Registered fallback (v21, if the chapter continues): per-level
calibrated-spawn curriculum à la DevilutionX-AI; otherwise the chapter
rests here with its map complete, awaiting workstation budgets.
