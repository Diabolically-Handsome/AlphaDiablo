# DiabloGym design notes: twenty runs, seventeen lessons

> AlphaDiablo project · 2026-07 · Every training run either eliminates one concrete problem that the data has
> nailed down, or cleanly falsifies one hypothesis.
>
> **Gold standard** = mean kills of the final model under deterministic evaluation (argmax policy, no sampling
> help). Evaluation regime v2 (the only valid one since 2026-07-05): 32 fixed seeds (9000-9031), 1500 steps per
> episode, an idle machine, and the engine pinned at `ENGINE_REF` in bootstrap.sh. The old 8-seed criterion used
> during v1-v8 is retired; its numbers only serve as longitudinal trend references (see lesson 8: it lied in both
> directions).

## Overview

| Run | Change | 32-seed gold standard | 8-seed old criterion | One-line conclusion |
|---|---|---|---|---|
| v1 | Naive shaped reward | — | (0.0) | HP-loss penalty + zero-reward sanctuary → **wall-hugging collapse** |
| v2 | Tax removed + approach shaping | — | (0.2) | Attacks are easily interrupted; the shaping is farmed by "monster walks up to me" |
| v3 | Engagement macro (SMDP) | — | (2.8) | First autonomous attacks; 10 ticks wasted on targets behind walls |
| v4 | Per-hit damage reward + macro stop-loss | — | (3.0) | The reward ecology settles; the **perception ceiling** shows |
| v5 | 11×11 spatial vision | 7.6* | (6.5) | Spatial perception filled in; cross-room navigation still missing |
| **v6** | **Explore macro** | **8.8** 👑 | (15.6) | **Champion of the v6-v10 era** (45,836-parameter MLP) |
| v7-7d | Level-clear objective + footprint channel + 3 macro patches | not re-evaluable† | 6.4 (self-test at the time) | **Macro degeneration attractor**; macro engineering frozen |
| v8 | RecurrentPPO / LSTM-128 (451,596 parameters) | 8.4 | (13.2) | Ties (single-episode max 43); the floor did not move |
| v9a-c | Entity attention (701,980 parameters, **6M steps, double budget**) | 3.8 | — | Heavy defeat: unstable optimisation + the wrong remedy |
| v10 | v6 recipe + training episode length 1500→3000 | 5.5 | — | **Time-wall hypothesis falsified by a single variable** |
| **v11** | **Descend option (door/barrel-aware full-map BFS)** | **19.4** 👑 | — | **Gold standard doubled; reached level 2: 0→27/32** |
| v12 | Drink key (UseBeltItem; observation still 286 dims) | 12.3 | — | Combat deaths 17→10, but mean kills fell; bottle-blindness + hiding place (lesson 11) |
| **v13** | **Potion system made learnable: belt + ground potions in the observation (286→290) + door-aware pickup macro** | **35.2** 👑 | — | **Gold standard nearly doubled; real-drink discipline 0.5%→93.4%, combat deaths 17→12** |
| v14 | Gear chapter: AC + ground gear in the observation (290→294) + auto-equip | 28.0 | — | 0/32 equipped armour — armour's payoff is invisible to the reward stream (lesson 13) |
| v15 | Reward v3: one-shot bounded shaping for equipment AC gain (+0.5 per point) | 31.3 | — | Gear key pressed once per 48k steps — shaping cannot amplify events that exploration never reaches (lesson 14); combat deaths 9/32, an all-time low |
| v16 | Masking chapter: invalid-action mask on the gear key (MaskablePPO) | 34.5 | — | The button came back (258 presses, 16 equipped) but payoffs did not follow; zero-kill 0/32 and a single-episode 80 both set records; real drinking crashed to 3.7% — the attractor moved to keys 12/13 (lesson 15) |
| v17 | Deep-water opener: depth-progressive descend bonus (8×N) + 3000-step episodes + 6M steps | deep board: median L3, L4 11/32 | — | One knob changed the species (farmer → diver), but produced a **level-1 stair sprinter**: 0 farming, 0 armour, 22 combat deaths (16 with a dry belt) — paying for "touching depth" only buys "touching depth" (lesson 16) |
| v18 | Lesson 16's single knob: death priced in step with the ladder (dying on level N costs 8×N) | deep board: median L2, mean kills 32.1 | — | The pendulum swung hard back: the armour audition finally convened (15/32 equipped) and dry-belt deaths fell 16/22→4/19, but 13/32 no longer descend and only 1 reaches L4; the dead went from "dry belt" to "sudden death with a full belt" — the bottleneck jumped twice in two generations: sampling → resources → **character growth curve** |
| v19 | Strength gauge: char_level/dungeon_level ratio in the observation (294→295) | deep board: median 1.5, 15 combat deaths | — | Null result recorded honestly: the death rate of those who descended was unchanged (88%), and fewer deaths came from "not going" — visibility cannot create a discounted value that does not exist |
| v20 | **Bug fix** (stat-point black hole: for nineteen generations level-up points were never spent; auto-spend 3 vitality : 2 strength) + γ 0.99→0.997 | deep board: median 1.0, **4** combat deaths | — | **Lesson 17**: with the books fixed and the horizon extended, the policy answered "31/32 never descend + safest ever" — a melee dive from a level-1 start in 3000 steps is negative-EV, and the refusal is a measurement; the deep-water chapter closes with its figure, and revival paths are pre-registered (spawn-injection curriculum / workstation-scale horizons) |

\* v5 was re-evaluated after the fact: its observation is also 286 dims and its action space is Discrete(10), so in
the current environment it can never select the explore macro and is directly comparable. v1-v4 have different
observation sizes and cannot be re-evaluated.
† The v7 branch had a 407-dim observation (footprint channel) and was retired together with the frozen macro
engineering.

The single-episode maximum kill count fluctuates between generations: 45 (v5), 36 (v6), 43 (v8), 38 (v9c), 49 (v10),
70 (v11). The maximum over 32 episodes is an outlier statistic and cannot carry attributions such as "memory/long
horizons raise the ceiling" (the memoryless v5 already scored 45); for the root cause of the floor see lessons 9
and 10.

![learning curves](../assets/learning-curves.png)

## Lesson 1: Reward tax and sanctuaries (v1 → v2)

The naive design included an HP-loss penalty. On paper a kill paid 250 times the cost of taking hits, but learning
dynamics do not read the books: the cost arrives immediately, while the payoff is delayed and requires coherent
behaviour, so the value function's first impression of "a monster next to me" is negative — and "hiding in a
corner" earns exactly zero. PPO honestly converged to zero: the deterministic policy walks in one direction for
1500 steps and pins itself against a wall.

**Principle: do not penalise the intermediate costs the target behaviour must pass through; do not leave
zero-cost sanctuaries in the reward landscape.**

## Lesson 2: Attributing shaping (v2 → v3/v4)

The approach shaping only looked at "distance got smaller", and Fallen actively chase the player, so "standing
still and fishing" collected shaping points for free. The fix: only approach caused by **the agent's own
movement** scores.

**Principle: shaping rewards must be attributed to the agent's actions, not to spontaneous changes in the
environment.**

## Lesson 3: Action timing and SMDP macros (v2 → v3)

In the engine any movement command clears the pursuit target (`OnWalk` → `ClrPlrPath`). Finishing one chase needs
about 10 consecutive ticks of choosing attack, an exponentially small probability for an exploring policy. After
"attack" was upgraded to an **engagement macro** (lock a target and keep pursuing until there is an outcome), the
policy could observe the full causal chain "attack → return" for the first time.

**Principle: when the semantic granularity of atomic actions is finer than the causal granularity of the task,
align them with temporally extended macro-actions (options).**

## Lesson 4: Per-hit densification without swing farming (v3 → v4)

The reward was refined from "settle on the kill" to "settle on every hit" (the credit-assignment chain shrank from
~10 steps to 1). Two traps: paying per swing trains a swing farmer (and a one-hit kill would then earn nothing);
paying on the **damage fraction** keeps the total constant (≈ an HP potential function, policy-invariant). A
"low-HP multiplier" (1.0→1.5) adds an incentive to finish targets without opening a loophole.

**Principle: densified rewards should hang on a conserved quantity of task progress (damage fraction), not on
farmable event counts (number of swings).**

## Lesson 5: The perception ceiling (v4 → v5)

Once the reward was sound, every failure mode pointed to the same fact: the agent could not see walls — it locked
targets through walls, collected shaping through walls, and could not find doors. After adding an 11×11 two-channel
local map (walkability + monster occupancy), the gold standard doubled under the 8-seed criterion (3.0 → 6.5); a
later 32-seed re-evaluation gave v5 7.6, confirming a real gain rather than luck.

**Principle: reward design can only cash in information that exists in the observation; when failures cluster
in spatial behaviour, fix perception before tuning the reward.**

## Lesson 6: Exploration as a macro (v5 → v6)

Vision solved "should I fight the monster I can see", not "where do I look when I see none". A reactive MLP cannot
plan detours across rooms. The **explore macro** (walk to the nearest walkable frontier outside the footprint,
hand control back as soon as prey appears) turned exploration into a selectable option too, so the policy only had
to learn when to fight and when to search.

The 8-seed story was "doubled + the bimodal failure cleared" (6.5 → 15.6, zero-kill episodes 5/8 → 0/8); under the
32-seed criterion the real gain is smaller and more interesting: **median 0 → 3.5, zero-kill episodes 19/32 →
15/32**; the mean went 7.6 → 8.8 (+1.2, within the noise of "one training run + a 32-seed evaluation", so
directional evidence rather than a significant result), and the single-episode max actually fell 45 → 36. The
explore macro raises the floor and the typical episode (it rescues dead episodes), not the ceiling — consistent with
lesson 9's "the stuck episodes are dead zeros".

**Principle: rather than forcing a reactive policy to learn planning, wrap planning as an option and leave the
decision to the policy.**

## Lesson 7: Macro degeneration attractors (v7 → v7d)

The v7 chapter tried to add a "level-clear objective + footprint channel" in one go (observation widened to 407
dims), and as soon as behaviour degenerated it started patching the macros: first a "blacklist failed targets"
rule for the engagement macro (v7.1); the blacklist test then misfired on targets standing right next to the
player (fixed in v7.2); after that fix the cooldown itself was farmed by the policy as a new loophole, so it became
"cool down only after two consecutive failures" (v7.3). Three patch rounds, each closing one degeneration attractor
and breeding the next; under the 8-seed criterion used at the time, it (6.4) always stayed far behind the champion's
15.6 on the same criterion (the v7 branch had a different observation and cannot be re-evaluated under the current
protocol, so only these old same-criterion numbers can be compared). The final decision was not a fourth patch but
**freezing macro engineering**: roll back to v6's minimal macros and spend the effort on the task and the training
side.

**Principle: an option is an interface, not a policy body that can be patched forever. When you find yourself
plugging leak after leak inside a macro, the abstraction sits at the wrong layer — freeze the interface and change
the task, the observation or the training instead of repairing the macro.**

## Lesson 8: Evaluation protocol and the luck tax (regime v2)

The luck in 8 seeds is not just "high variance"; it **lies in both directions**: with 32 fixed seeds, champion v6
fell from 15.6 to 8.8 (overestimated by 77%), v8 fell from 13.2 to 8.4 (overestimated by 57%), while v5 rose from
6.5 to 7.6 (underestimated by 15%) — the v5-v6 gap shrank from a "crushing 9.1" to a "narrow 1.2", nearly reversing
the ranking. Beyond the mean, only the trio **median / zero-kill count / single-episode max** exposes a bimodal
distribution (half dead episodes + half big wins, with a mean that looks "fine").

The protocol itself is part of the result, and four things must be frozen: the **seed set** (9000-9031, never used
for tuning), **determinism** (argmax; a nice curve ≠ a model that can fight), the **engine version** (bootstrap.sh
pins the SHA; upgrading the engine means rebuilding the leaderboard) and the **machine state** (idle; engine turn
advancement reads the wall clock, so trajectories drift under full load — the median of the same model once drifted
from 3.5 to 4).

**Principle: freeze the evaluation protocol before believing any number. Protocol = seeds + steps + policy mode +
engine version + machine state; every item left unpinned adds one more lie to the numbers.**

## Lesson 9: At the current scale, task design > architecture (v8 / v9 / v10)

A three-architecture contest: the 45,836-parameter macro-action MLP (3M steps, 8.8) ≥ an LSTM-128 with ten times
the parameters (3M steps, 8.4) ≫ entity attention with fifteen times the parameters (**6M steps — double the budget
— and still only 3.8**, unstable throughout training). All the gains came from **task-structure** changes such as
reward attribution, action granularity and the exploration option; a "bigger brain" either tied or hurt — and the
attention model was also the wrong remedy: it upgraded perception, while v5→v6 had already shown that the bottleneck
was planning/exploration. (Note: each architecture had a single training run, and no training seed was set at the
time — train_ppo.py now supports `--seed` — so this contest is directional evidence, not a significant result.)

v10 delivered the final cut, falsifying the "not enough time" hypothesis with a single variable: doubling the
evaluation episode length from 1500 to 3000 left **the per-seed kill counts of v6 and v10 bit-identical under both
criteria (32/32 each)** — twice the time, zero increment. A zero-kill episode is a "dead zero": when the spawn area
has no reachable prey the agent is stuck forever; it did not run out of time, it never got started.

The comparison with AlphaZero-family systems (e.g. for xiangqi) is telling: in search-wrapped RL, planning is done
by the search and architecture upgrades convert directly into playing strength; in search-free RL the policy must
plan by itself — and upgrades to perception and memory never touch the planning bottleneck.

**Principle: before giving a policy a bigger brain, prove that the bottleneck is the brain. Here the bottleneck is
the spawn deadlock (task structure), so the next step is curriculum learning / cross-room frontier exploration, not
a bigger network.**

## Lesson 10: Capability lives in the action space (v10 → v11)

After v10 falsified the time wall, instrumenting the dead zeros dug up a truth more basic than "exploration
failure": **a closed door is identical to a wall in the walkability channel** (`IsTileWalkable(pos, false)` treats
closed doors as solid). The agent did not "fail to take detours"; it **could not see the connectivity of the
world**: of the 15/32 zero-kill episodes, 5 spawned in rooms that are fully sealed under door-free BFS (no reachable
prey) and the other 10 were argmax dynamics spinning in place.

v11 added just one option — the descend macro: each press runs a full-map 4-direction BFS (**closed doors and
intact barrels count as "operable soft walls"**), opens doors and smashes barrels along the path with CMD_OPOBJXY,
and stands on the down stairs until the engine triggers the transition; spotting prey does not interrupt it (it is
the deliberate evacuation key). Observation, reward and architecture were untouched.

Gold standard 8.8 → **19.4**, median 3.5 → 14.5, zero-kill 15/32 → **2/32**, reached level 2 0 → **27/32** (0 for
all five previous generations), deepest episode reached level 4. Per seed: 4/5 of the sealed group broke out; the
ten "hunter malfunction" seeds were **10/10 cured** — the behavioural dead loop was broken by a new high-yield
action. The cost is recorded just as honestly: four seeds that used to be rich L1 farms produced less because of
"greed for depth", and **17/32 episodes died in combat on deeper levels** (the fate of an unarmoured warrior, and
the proposal for the next chapter).

Two companion experiments nailed the conclusion down:

1. **Door-blindness oracle (architecture invariance)**: the static rule "sealed spawn ⇒ zero kills" scores
   **15/15 hits with zero false positives** on all three architectures, v6 (46k MLP), v8 (452k LSTM) and v9c (702k
   attention) — information destroyed at the perception layer cannot be recovered by any downstream brain; the
   zero-kill seeds outside the sealed rooms differ between the three (each dynamical disease has its own madness,
   again showing it is not a function of the map).
2. **Emergent repurposing**: on the deepest sealed seed (9000) v11 killed 25 monsters but never descended — the
   policy used the descend macro as a **door-opening key**: press it to open the door, let the monsters in and eat
   on the spot. A side effect of the option was reinvented by the policy as its main function, an unexpected dividend
   of wrapping planning as an interface (lesson 6).

**Principle: the observation decides what can be known, the action space decides what can be done, and the
architecture only tunes the efficiency in between. When adding capability, audit the three layers in that order —
the cheapest miracles live in the action space (one new action +120%, fifteen times the parameters −57%).**

## Lesson 11: A new action is also a new hiding place (v11 → v12)

The ugliest column of v11 was **17/32 combat deaths**, so v12 added a 13th action: drink a belt healing potion (the
same path as the UseBeltItem gamepad hotkey; the warrior starts with IDI_HEAL ×2, only those two potions, no
resupply in this version). The observation was **deliberately frozen at 286 dims** (the belt count only went into the
raw dict) so that every past model could still be re-evaluated — that protocol purism later proved to be the most
expensive decision of the round.

Reconciling the four pre-registered predictions: combat deaths ≤10 **✓ (exactly 10/32)**; mean kills 22+ ✗ (12.3);
reached level 2 ≥27 ✗ (26); single-episode max 80 ✗ (46). 1/4 — borderline on the numbers, a concession on the
mechanism:

1. **Only a handful of real potion warriors**: 9024 drank while fighting and survived with 43 kills (the same seed
   under v11 died with 36); the fiercest sip of the whole evaluation was a clear-headed argmax press at 8.6% HP
   (0.086→0.343). The design intent really did appear — on roughly 2-4 seeds.
2. **The mainstream use was empty-bottle calisthenics**: **4,715 of 4,740 presses hit an empty belt** (1,467 in a
   single 9014 episode). Bottle-blindness: the belt count is not in the observation, so the policy **in principle**
   cannot learn when *not* to press — lesson 5 holds for action preconditions too. Better still, the engine's walk
   command persists across steps and pressing 12 does not interrupt an ongoing move, so it effectively became a
   "**wait/continue**" key: seeds that do not clear monsters used it as risk-free time filler.
3. **The causality of the death improvement is impure**: 9 seeds that died under v11 were saved, but only 4 of
   them actually drank; 5 were saved purely by fighting more timidly (avoiding combat); meanwhile 3 new bodies
   appeared (9008 died after 32 kills; the 9009 sightseer was killed for the first time). Of the combat-death drop
   17→10, the potions probably account for less than half; the rest is pacifism.

**Principle: before giving a policy a new button, ask three questions — is its precondition observable
(bottle-blindness)? Is it a risk-free no-op (hiding place)? How does it interact with existing commands (the wait
key)? Door-blindness was the engine's debt to us; bottle-blindness we signed for ourselves.** The v13 bill was paid
right away: the belt count and the direction of the nearest ground potion entered the observation (286→290, lesson 5
homework), and a pickup macro went live (reusing the descend macro's door-aware BFS — the engine's native pickup
pathfinding is door-blind too, and seed 9003's potion lies behind a closed door); action masking stayed on hold
(with the belt visible, key discipline should be learnable by itself, left as a v13 test item). A four-lens
adversarial review caught two more bugs: with a full belt, pickups fall straight into the backpack = a value black
hole invisible to the drink key (now forbidden on the bridge side), and the pickup check implicitly depended on
pcurs==CURSOR_HAND (now re-pinned on every level load by SyncLoad).

## Lesson 12: Discipline is a function of observation, and hiding places are conserved (v12 → v13)

v13 results: gold standard 12.3 → **35.2** (median 10 → 29, zero-kill 9/32 → 1/32), combat deaths 17 (v11) → 12,
**a new champion**. The reward function was still untouched — the only changes were +4 observation dims and one
new macro. Two mechanistic conclusions:

1. **The same button, one observation bit apart, a world of difference in discipline**: v12 pressed the drink key
   on an empty belt 99.5% of the time; v13 pressed it on a stocked belt 93.4% of the time (57 real drinks / 4 empty
   presses), 25 of the 57 real drinks happened below half HP, and the deepest sip was at **1% HP**. The drinking
   skill was never missing; what was missing was its precondition in the observation — a controlled replication of
   lesson 5 (the perception ceiling) on action preconditions, with only one observation bit between control and
   treatment.
2. **The law of conservation of hiding places**: v12's idling attractor did not die, it moved — seed 9001 pressed
   the new pickup key 1,448 times (with zero items bagged) as its sanctuary. Block one zero-risk action and the
   risk-averse probability mass flows to the next; every new action must budget for the "attractor migration" tax.

The costs are booked as they are: reached level 2 27→25 (more fighting, less descending); pre-registered
predictions **2/4** (real-drink share >50% ✓ with 93.4%, mean kills ≥16 ✓ with 35.2; combat deaths ≤10 missed by 2,
reached level 2 ≥26 missed by 1). Emergent bonus: 9007, a long-time sealed-room resident, killed for the first time
(14 kills) — the door-opening side effect of the pickup macro let it out, the same interface dividend as v11's
"descend macro as a key"; the 9009 sightseer turned hunter (0 → 24 kills).

**Error bars added (seed-14 same-recipe repeat, a first for this project)**: mean kills 38.1 (vs 35.2) — **the
result replicated**, so the v13 effect does not depend on the seed; but **the style did not replicate**: combat
deaths 12 vs **21**, real-drink discipline 93.4% vs **45.7%** (pooled 65%), idling seeds 1 → **3** (in s14, 9001
spent all 1500 steps pressing the pickup key). The champion row keeps the pre-registered seed-13 run (choosing the
better of two repeats would be selection bias, the old sin of the 8-seed era). A new sub-lesson: **how much it wins
is reproducible; how it wins is a seed lottery** — any single claim about behavioural composition must survive a
seed repeat before it counts.

**Principle: capability = action × observability of its precondition. And as long as the reward landscape contains
a zero-risk action, some probability mass will move in — when auditing a new action, write "can it be used as a
hiding place" into the acceptance checklist (lesson 1's sanctuary, reincarnated endlessly in the action space).**

## Lesson 13: The reward stream is the last observer (v13 → v14)

v14 executed lesson 11's checklist to the full: AC and the nearest wearable item entered the observation (290→294),
the engine's AutoEquip was wired through (its default-off option forced on), the PM_GOTHIT timing window was closed,
the door-corner case was fixed, and every probe was green — **the gear key's precondition chain was flawless**.
Five pre-registered predictions went **0/5**: equip rate 0/32; in the whole argmax evaluation (48,000 steps) the
gear key was pressed only **6 times** and not a single piece of armour was worn. Mean kills 28.0 and reached level 2
19/32, below the range of the two v13 runs; the champion stays.

The root cause is not the plumbing but the **learning signal**: armour pays off as "a few HP less per hit", spread
over hundreds of steps, and that amplitude is completely drowned by the variance between episodes — within a 3M-step
credit-assignment horizon the value function **never observed statistical evidence that "wearing armour → higher
return"**, so the button never earned value. Compare v12/v13: bottle-blindness was **a precondition invisible to the
policy** (observation layer); the gear key is **a consequence invisible to the learning signal** (reward-horizon
layer). Drinking can be learned because the counterfactual of pressing key 12 at 8% HP and "not dying" within a few
dozen steps is plain to see; the counterfactual of armour takes hundreds of steps and dozens of episodes to show.

Three more entries booked honestly: (1) real-drink discipline in this run was 36.7% (a three-run evidence chain of
93%/46%/37%, reinforcing "style is a seed lottery"); (2) 30% of pickup-key presses were empty (2,801 presses) — the
attractor is still there; (3) the AC=4 row in the post-mortem table is a **measurement ghost of gear dropped at death**
(gear falls on the ground at death, so the terminal AC reading ≠ the AC while alive); the instrument definition now
notes this.

**Principle: perception bounds what can be known (lesson 5), the action set bounds what can be done (lesson 10),
and the reward horizon bounds what can be learned. A capability chain is only as strong as its least observable
link: the precondition must reach the observation, and the consequence must reach the learning signal — otherwise
the button is forever decoration. v15 candidates: bounded AC-gain shaping (AutoEquip only fills empty slots =
unfarmable), a gear-rich curriculum, or 10-30× more steps on bigger compute so that value seeps out of survival
statistics naturally.**

## Lesson 14: Shaping amplifies; it does not summon (v14 → v15)

Lesson 13 wrote three prescriptions and v15 tried the cheapest: one-shot bounded shaping on equipment AC gain
(+0.5 × ΔAC; AutoEquip only fills empty slots = unfarmable; passed a pre-launch Goodhart review; the pre-launch
probe measured a manual press at +0.998 = +1.0 − idle tax, so the payment pipeline was flawless). Four
pre-registered predictions went **2/4**: mean kills ≥30 hit (31.3), combat deaths ≤11 hit (**9/32, an all-time
low**, the previous best being v12's 10); equip rate ≥20/32 **went to zero again** (0/32), and a median time to first
armour does not exist — in the whole 48,000-step argmax evaluation the gear key was pressed **once** (v14 still had
6). With shaping on duty for a generation, the press count fell instead of rising.

Root cause: a shaping reward has to **be sampled** before it can bend the value function. This reward's trigger chain
is "gear happens to drop → happens to enter the observation → 14 is pressed → BFS walks there → it is picked up and
auto-equipped" — during exploration a product of small probabilities; while the key's **cost** (wasted ticks on
empty presses, hits taken on the way) settles on the spot every time. With frequent negatives and rare positives the
key's value estimate can only fall — however large the amount, a reward that is never sampled equals zero. Lesson 2
said shaping must be attributed to the agent's own actions; add one more rule here: **shaping must lie on
trajectories that exploration actually reaches**. Events first, rewards second: demonstrations/imitation learning,
exploration resets that force equipping, or a gear-rich environment (the level-weighted curriculum that rouming's
[DevilutionX-AI](https://github.com/rouming/DevilutionX-AI) validated at 16-level scale belongs to this family) —
make the event common first, then talk about shaping.

Side entries: (1) the other metrics look like a healthy redraw of the v14 family — mean kills 28.0→31.3 (≈ within the
noise band), reached level 2 also 19/32; the contributions of the seed lottery and the shaping cannot be separated
in a single run; (2) real-drink share 60% (42 real / 28 empty), the fourth hand of the style lottery
(93/46/37/60%); (3) the share of real drinks at low HP, 62% (26/42 below half HP), is actually the highest of the four
runs — but the record-low deaths more likely come from the fighting style than from the potions (potion use is
similar to v13: 42 vs 57 sips); a single run cannot settle the cause; (4) 9001's empty-press attractor on the pickup
key is still there (lesson 12's conservation law, confirmed for the third time).

## Lesson 15: Masking moves probability, not value (v15 → v16)

The prescription: an invalid-action mask on the gear key — when no wearable gear is in view, the key does not exist
(MaskablePPO; env.action_masks() only masks key 14, and 12/13 stay free to protect the style-lottery baseline). The
pre-launch probes recorded two priors honestly: the wild completion rate of a forced press is only 6.7%, and blind
pressing costs −7.1 mean kills and raises combat deaths 9→15 — so v16 was pre-registered as a **falsification
experiment**: with fair sampling and an explicit bonus, not pressing = correct pricing; pressing without higher
payoffs = the economics problem confirmed.

Results (gold standard 34.5/33.5/80, zero-kill 0/32, combat deaths 14, L2 18; our predictions 3/4):

**On sampling, the mask won outright.** The gear key came back from one press per 48k steps in v15 to **258
presses**, and 16/32 episodes actually wore armour (median time to first armour 172 steps) — the mechanism diagnosis
of lesson 14 was confirmed directly by intervention: it was never unwilling to press, it simply never had the chance
to learn. Zero-kill 0/32 and an 80-kill single episode both set records; the masked warrior is the strongest
non-champion ever.

**On economics, probe A's verdict stands.** First-armour/press ratio 16/258 ≈ 6%, matching the probe's 6.7% wild
completion rate exactly — most presses burn out on "cannot get there"; of the 16 armoured episodes, 7 died and
dropped the gear and 3 had it **broken by durability loss** (L1 junk gear has single-digit durability, so even
survivors fall back into poverty; a new mechanism added to the record); combat deaths 14 and L2 18 did not improve
with armour. The button is alive, but the goods are still not worth buying — within the 1500-step L1 accounting
period armour cannot be amortised (and the pressing behaviour is suspected to be kept alive by the +0.5×ΔAC subsidy;
a persistence test with the subsidy removed is left until after v17).

**The conservation law's third strike, and the cleanest.** The no-op probability mass evicted from key 14 by the mask
moved wholesale to the unmasked keys 12/13: real-drink share crashed to **3.7%** (2,212 empty drinks; 72% of 4,674
pickup-key presses were empty; about 11% of evaluation steps were spent pressing dead buttons), the fifth hand of the
style lottery 93/46/37/60/**4**. 9030 became the new attractor's spokesman: in the 133 steps where gear was visible
it pressed 133 times and never completed a single walk. Honest accounting: PPO→MaskablePPO swapped the whole
algorithm package and was registered before launch as the prime suspect for anomalies; the crash in real drinking
cannot be decoupled from it within a single run.

**Principle: a mask puts a button back on the menu; it cannot make the goods worth buying — that is the job of the
task economics. Structural hygiene can only relocate junk traffic; only real value can retire it.** The gear chapter
moved to deep water: v17 = 3000 steps + a depth-progressive descend bonus, letting armour audition for its real job
(a potion saver on L2-L4).

## Lesson 16: You buy the behaviour you price, not the behaviour you mean (v16 → v17)

Deep-water opener: depth-progressive descend bonus (N→N+1 pays 8×N), 3000-step episodes, a 6M-step budget (episode
count aligned with the old chapter). Pre-launch mine-sweeping probes: zero crashes on L2-L5 (the first headless entry
into the L5 catacomb tiles), 19/19 exact ladder payments, and a naive diving bot died on L3-L4 in 7/8 runs — the exam
was valid.

Results (first row of the deep board: median depth **3.0**, L4 **11/32**, L2 28/32; registered predictions 2/5): one
reward knob changed the whole species — the previous generation was an L1 farmer with 34.5 mean kills, this one is a
diver with 9.6 mean kills, and the old chapter's all-time depth record (L4) is now routine at 11/32. **But it is not
the "farm first, then dive" diver we wanted; it is a stair sprinter**: all 28/28 first descents happened on a
level-1 character (median step 138), 0 farming, 0 armour (the gear key was never pressed in the whole evaluation —
no farming means no drops, no drops means no armour to wear, so armour's deep-water audition never convened), 22/32
combat deaths.

Do the arithmetic it did for us and it is clear: L2→L3 pays 16, L3→L4 pays 24, a kill is worth about 1, and death
costs only 2 — sprinting for L4 at a 30% survival rate has an expected payoff of +24×0.3−2×0.7 ≈ **+5.8**, a sure
profit. Sprinting is not a failure; it is **the optimal solution to the price list we wrote**. We wanted to buy
"arrive at depth alive", but the price list said "touch depth" — the purchase order was wrong, and the supplier is
not to blame.

Death anatomy and a comparison with related work: 16/22 deaths happened on an **empty belt** (43 real drinks / 757
empty presses; one 9026 episode made 554 empty presses and died dry on L1) — the "potion runway runs out →
personality switch → ground down" pattern that DevilutionX-AI described at 16-level scale reproduced word for word on
our engine, except its agent died of timidity and ours of recklessness: the same runway, two ways of running out.

**Principle: the reward function is a price list and the policy is a perfect arbitrageur — it always solves the
letter of what you wrote, never the intent in your head. "Arrive" and "arrive alive" are two different goods, and the
prices must say which.** v18 single-knob candidate: price death in step with the ladder (e.g. −8×current level), so
the book loss of "dying at depth" outweighs the book gain of "touching depth" — then farming, potion pickup and armour
will all re-enter the auction.

**v18 pendulum data point (an extra run, single knob as above, predictions 2/5)**: the repricing worked immediately
but overshot — mean kills 9.6→32.1, the first descent three times later (median step 460), **the armour audition
finally convened** (15/32 wore armour, 228 presses; farming resumed → drops → armour to wear, the whole supply chain
revived), dry-belt deaths 16/22→4/19; the cost was 13/32 episodes never descending and L4 falling from 11 to 1.
**The death anatomy flipped**: v17's dead all had empty belts, v18's dead died suddenly with full belts (15/19 still
had potions at death) — level-1/2 characters are burst down on L2/L3 faster than any belt can save them. The
bottleneck jumped twice in two generations: sampling (fixed in v16) → resources (exposed by v17) → **the character
growth curve** (exposed by v18). Embryonic evidence: the 5 episodes whose first descent happened at level ≥2 are the
best on the board (9026: 79 kills, L3). The real currency of deep water appears to be XP/level, with potions and
armour only the interest — the next step is a choice between fine-tuning the auction prices and a larger budget so the
spiral emerges naturally (the workstation line), or testing the XP-economy hypothesis (v19 candidate).

## Lesson 17: Audit the world before debugging the policy — when the books are honest, a refusal is a measurement (v19 → v20)

After three knobs failed the same way (v19's strength gauge ended with a null result), we stopped debugging the
policy and audited the world instead, and dug up a **stat-point black hole**: the engine grants 5 stat points per level
(NextPlrLevel only accumulates _pStatPts), spending them has always been a human UI action, and for nineteen
generations this bridge never implemented it — about 83% of the growth currency was silently discarded, and the
"level → survivability" exchange chain simply did not exist. A faithful model shows that at the levels reachable in
3000 steps (≤ clvl 3), fighting a median L3 monster pack unarmoured is negative-EV, and the only winning line needs
spent stat points + armour. **Three generations of policies had not failed to learn "farm first, then dive"; they had
been pricing a broken economy correctly all along** (v17's sprint was the closed-form optimum: 8×0.99¹³⁸ ≈ two
kills). Secondary cause: γ=0.99 has a half-life of 69 steps, exponentially annihilating deferred bonuses and death
penalties alike (farm-dive discounted 0.14 vs dive-now 5.35). The two causes interlock, and fixing either alone does
nothing (quantitatively pre-registered).

v20 = the bug fix (automatic stat spending at the end of Step, 3 vitality : 2 strength, verified in the field with zero
leakage) + γ 0.997 (half-life 231 steps). A four-lens adversarial review found no bugs; a standing probe: on a
contested seed a bare level-1 character dies in 23 steps with zero kills → with spent points at clvl 3 it lasts 108
steps with 9 kills (4.7×). Results: **median depth 1.0, 31/32 episodes never descend, 4/32 combat deaths (the safest
ever), the drink key went extinct (0 presses; seventh hand of the style lottery: 93/46/37/60/4/77/0), and farming only
goes as far as "enough" (21 kills; two thirds of the seeds do not bother reaching level 2).** The P3 stay-at-home
sentinel fired exactly as pre-registered; the P5 falsification line held.

**Reading: this is not a fourth failure; it is the honest conclusion of this chapter.** With the books fixed and the
horizon extended, the policy saw the real price of deep water and turned down the whole deal — under a level-1 start,
3000 steps and melee terms, diving is still negative-EV in a mechanically sound world. The original Diablo agrees
with the verdict: no human warrior forces the deep levels at character level 1-4; the levelling spiral takes hours,
not 3000 steps. **Sometimes the policy is not good enough; this time the task was not good enough — and the policy
was the first to see it.** Principle: when the reward will not move, audit the world first (what is missing may be a
mechanism, not a signal); if the policy still refuses after the world is fixed, respect the refusal as a measurement.
The deep-water chapter closes here with its figure: five configurations, two interlocking root causes, one world fix
and one precedent for "when to trust the policy's no". Revival paths are pre-registered: a spawn-state injection
curriculum (DevilutionX-AI style, strength scaled by level) or workstation-scale long horizons.

**Correction the next day (an oracle falsification, the seed of lesson 18)**: a scripted oracle grid of 8 strategies
× 32 probe seeds × 3 horizons overturned this lesson's economic claim — a "fight while descending" script (fight
whatever can be fought on each level, go down when it cannot, accept death on L3-L4) already beats retirement at
**3000 steps**, 39.9 vs 15.9 (26/32 paired wins), despite a 94% death rate. The learner's refusal was only **the
correct choice between two gradient-reachable local peaks (sprint / retire)**; the main peak lies across the death
valley between them, and the v17→v20 pendulum never passed it. The revised principle: **a policy's "no" is evidence
about the optimisation landscape, never about the task ceiling; ceilings are measured with oracles, not inferred from
silence.** The death-valley hypothesis thus moved from inference to evidence — the conditions for starting a
hierarchical architecture (mode-level decisions) were formally met.

Four headless-engine pitfalls fixed while building the bridge:

1. `RegisterCustomEvents()` must be called by hand (upstream only registers it when creating a window); otherwise all
   level-transition events are lost;
2. the event pump must use `devilution::FetchMessage`; `demo::FetchMessage` swallows everything outside demo mode;
3. an event handler other than DisableInput must be installed, or the event pump refuses to deliver;
4. `ControlMode = KeyboardAndMouse`, or the gamepad module treats "no input" as a released stick and brakes the
   pathfinder every tick.

See also `patches/`: under upstream HeadlessMode a failed asset open returns "success + null pointer", which breaks
the town's Hellfire detection fallback chain (four places, cel/til/sol/min, patch 0001); fixed and suitable for
upstreaming. The deep-level mine clearing (patches 0002/0003) is described below under "Headless mine clearing".

## Determinism rules (engineering side, audit of 2026-07-05)

The headless engine hides three wall-clock dependencies, handled one by one:

1. **`CreatePlayer` reseeds the global RNG with `SDL_GetTicks()`** (player.cpp) — triggered every time reset creates
   a new hero. Dungeon seeds are overwritten by the bridge, so layouts are fine; quest selection is fine too (in the
   pinned engine `InitQuests` goes through `InitialiseQuestPools(DungeonSeeds[15])`, a local RNG whose seed the bridge
   already controls). But any code that reads the global stream in the window "before the first level load" eats the
   wall-clock stream. **Fix: the bridge defensively takes over the global RNG with the episode seed in `reset()`**
   (diablogym.cpp) — the 32-seed fingerprints are bit-identical before and after the fix, so under current metrics it
   is pure hygiene; its real value is turning "the global RNG belongs to the episode seed" into an invariant that
   upstream changes cannot break.
2. **Turn advancement reads real time** (`nthread_has_500ms_passed`) — bit-reproducible across processes on an idle
   machine (measured 4 runs × 32 seeds bit-identical, including different `PYTHONHASHSEED`s); under full load an
   occasional tick advances one logical turn less and the trajectory drifts by one step. **Fix: the evaluation
   protocol requires an idle machine**; training and gold-standard evaluation must not run in parallel on the same
   machine.
3. **Pickup deduplication records use wall-clock timestamps** (items.cpp, 6-second expiry) — headless runs finish an
   episode in 1 second, so records never expire; harmless for current metrics; documented for reference.

Two operations rules: long training runs must be detached from the terminal session (`nohup ... & disown`; v9 once
lost its weights at 40% when the host process exited); save a checkpoint every 500k steps.

## Headless mine clearing (three bugs hit in the first v11 training run)

Every time the agent explores an area of the engine that headless mode has never been tested on, it may step on a
new mine — v11 was the first policy to reach level 2 in numbers, and two mines were cleared in one go:

5. **The bat's dive** (first L2 mine, patch 0002): `AddRhino → InitMissileAnimationFromMonster` unconditionally
   dereferences the unloaded monster sprite sheet (an empty optional). Charge missiles must mirror the monster's own
   animation frames and are the only missiles that read monster graphics "at creation time" — so every other missile
   survived and this one alone crashed.
6. **The Butcher's greeting** (second L2 mine, patch 0003): `InitQTextMsg → GetSFXLength` indexes the empty, never
   loaded `sgSFX` vector (the audio system is not initialised in headless mode, while quest dialogue uses the "voice
   length" to compute subtitle scroll speed). **The v1 patch guarded the wrong object** (it guarded the downstream
   pSnd; the real culprit was the whole table) — the probe's four-seed deep L2 sweep "passed" only through survivor
   bias; it never entered the Butcher's room. Lesson: **a regression probe must reproduce the trigger condition,
   otherwise a green light is just the luck of not stepping on the mine.**
7. **The silent hang** (accomplice): after a worker segfault, `SubprocVecEnv.close()` blocks forever on a broken pipe,
   muffling a loud EOFError into a "training freeze mystery" (it hung three times in the same place before it was
   located). train_ppo's finally block now gives close a SIGALRM fuse — training now **dies loudly**.

Two more operations rules: monitoring must check the **timestamp freshness** of `status.json` (frozen data once passed
itself off as "healthy progress"); do not rely on a single pgrep to judge processes (in ERE `\|` is not "or", and `-f`
matches the parent shell's full command line) — monitoring now judges purely from file facts (leaderboard rows /
fatal signatures / freshness / worker count).

## Reproduction

```bash
# From the repo root; the venv lives inside the repo per the README Quickstart (python3 -m venv .venv)
./bootstrap.sh && ./build.sh                 # pinned engine + bridge
.venv/bin/python tests/smoke_random_agent.py
.venv/bin/python train/evaluate.py train/runs/<run>/model_final   # idle machine!
.venv/bin/python train/plot_figures.py   # author side: needs local train/runs/ training logs
                                          # (not distributed with the repo; the figure is provided in docs/assets/)
```

## v22 chapter: policy brain / operator brain (hierarchical SMDP)

**Single-prescription statement**: the only prescription of this generation = the decision layer goes from flat to
hierarchical; reward / observation base / six macros / world rules are all frozen at v20. **Rationale (as revised in
review)**: not "the mode sequence is outside the gradient-reachable domain" (a devil's-advocate critique falsified
that framing — spiral2 can be expressed by a 296-dim memoryless flat policy), but **hierarchy is the only structure in
which the optimised quantity equals the oracle ledger verbatim (undiscounted return with γ_mgr=1) and credit goes only
to the 10-60 decisions per episode**.

Design points: vocabulary FARM/DIVE/RESUPPLY (Discrete(3); the inner loop is the oracle frozen verbatim; drinking is
a brainstem reflex with no option of its own); "drain → dive" is promoted from script to a policy-brain decision (the
exam of this chapter); **a level change always returns control = an explicit invariant** (design decision, measured
in G0.2; the foundation for 16-level scalability: the event-driven decision count grows automatically with campaign
depth, and for a distant 300+ decisions the variance knob is γ_mgr→0.999 with the design unchanged). Devil arm F
(flat 296 dims + BC warm start) is a mandatory pre-registered fourth arm, with surrender clause P7 written down in
advance. Known deviation: the G1a bit-level certificate is replaced by G1b (mechanical fidelity) + G2 (vocabulary
sufficiency) + G0 unit-test coverage, recorded as is.

**Registered predictions (written before running)**: R2 teacher on the 7000 block in [36,46]; R3 main prediction:
hierarchical arm H passes the main metric (≥0.6× the teacher's evaluation-block reference + paired wins against
wrapper-retire ≥22/32) with probability ~55%; R4 if H succeeds, DIVE share in [10%,50%], CAP hits <5%, no option-level
hiding place (share >30% and return ≤0); R5 devil arm: BC replay ≥0.85× teacher (memoryless hypothesis), return
erodes after fine-tuning and H≥F; R6 wrapper-retire evaluation-block mean in [10,20]; R7 never set a paired gate
against wrapper-rush (spiral2 itself only 15/32, a fat-tailed lottery); R8 τ̄ in [80,200], 40k manager steps ≈
[3.2M,8M] micro-steps. **P6 falsification line**: DIVE <2% and median depth = 1 → option-level retirement, launch
insurance arm H-BC. **P7 final architecture review**: F ≥ H on both metrics → formal conclusion "IL + flat is
enough", hierarchy demoted to a control, never dressed up; H > F → the narrow hierarchy claim holds; both lose to
retire → move to the workstation with a curriculum.

**G2 ruling record (before launch)**: mean component 77.5 vs line 36 (2.2×, crushed); paired component 22/32 vs line
24 (2 short). The root cause is not a vocabulary defect: the review fixes (fight on sight / resupply during dives)
strengthened the teacher and the baseline alike (wrapper-retire 29.2 vs the oracle's retirement 15.9), the measured
height of the mountain doubled (teacher evaluation block 101.5), and the paired shortfall belongs to the same family
as the fat-tail artefact already registered in R7. Ruling: the diagnostic purpose was met, launch approved; the
teacher's own 22/32 becomes the empirical anchor of the P3 ≥22 line (H ≥22 = teacher-level paired performance); the P3
mean line is set by the pre-registered formula: 0.6 × 101.5 = 60.9.

**v22 results and P7 final review (pre-registered clauses executed verbatim)**:

| Arm | Gold standard | Combat deaths | Paired wins vs retire |
|---|---|---|---|
| **H hierarchical (policy brain)** | **93.9** (median 103.5) | 2/32 | **24/32** |
| F flat+BC (devil) | 80.2 (125 in training, −36% shrinkage, lesson 8 confirmed again) | 0/32 | 22/32 |
| H-BC insurance arm | 38.5 (collapsed into pure FARM, 3247:1) | 0/32 | 1/32 |
| Teacher script (ceiling) | 101.5 | 25/32 | — |

**P3 main metric: passed** (93.9 ≥ 60.9, paired 24 ≥ 22). **P7: H beats F on both metrics → the narrow claim of
hierarchy (credit assignment under an exact γ=1 ledger) is supported by evidence**; the devil arm lost with honour
(the surrender clause did not trigger, but 80.2 shows flat+BC is a strong control). P6 follow-up: the insurance arm
H-BC turned out far worse than the pure-RL H (38.5 vs 93.9) — the hypothesis "the policy brain cannot discover diving"
**was killed**: the clone fed dive demonstrations degenerated back into not diving and got weaker, while the
self-taught H "tried, priced and refused" — an informed option-level refusal, lesson 17's correction reproduced at
the hierarchical level, and this time with the ceiling measured (teacher 101.5): H traded a 7.5% return gap for a 92%
cut in the death rate. Prediction check: R3 ✓ (main prediction hit), R7 ✓, R8 ✓; R4 ✗ (DIVE share 0.27%), R5 half
wrong (BC replay missed 0.85 on both — the memoryless hypothesis failed at both levels; the root cause includes an
observation defect, the drained flag cannot be fully recovered from the stall clock, recorded as is as a v23
correction candidate), R6 ✗ (retire evaluation block 36.9, outside the predicted band [10,20]).

## v23 chapter: can the operator brain be learned? (one seam, on-policy; pre-registered in [PREREG-v23](../prereg/PREREG-v23.md))

**The exam**: under the frozen v22-H manager, replace the scripted FARM inner loop with a learnable worker — the
replaceability claim, where a tie counts as a win (P1 line 0.95×). The design came out of a structured review
(3 priors × 6 critiques × audit × synthesis), and its four decision points were settled before launch (frozen at a
commit that serves as notarisation).

**Build**: a shared window-core refactor (OptionsEnv and WorkerWindowEnv run the same bookkeeping code, eliminating a
third implementation); the worker sees 298 dims (295 + τ clock + stall clock + drained flag), Discrete(15) with 11/12
always masked; reflex ownership moved up (the worker never observes a pending-reflex state, logged ≡ executed);
**wage stripping fixes the dive arbitrage** (wage w = r − level-change bonus, with the ledger identity
Σw ≡ R − 8×ΣΔdlvl⁺ asserted per window in G0') — the closed-form fix of the only fatal flaw confirmed twice in review.
The pre-launch review confirmed 15 items; missing sentinel wiring (a blocker) voided the first launch and forced a
restart: no car goes on the road without gauges.

**Gate-chain event history (all by pre-registered clauses, zero improvisation)**:
G0 six items green → G0' five items green (606 windows bit for bit) → G0'' (new, stricter than the design review:
32/32 per-seed regression against the pre-refactor probe) → H7=78.5 (R1 [85,105] missed; the appendix A warning came
true) → G1: the assembled BC scored 78.5 = 1.00×H7, **byte-identical to the script** — a forensic review confirmed it
was a genuine result, with the caveat that BC had only been tested on the teacher's distribution and the real exam
was PPO (carried in PREREG-v23 appendix C) → 2M warning check 65.3 (above the 62.8 line, narrowly) → **G2@4M = 42.6,
through the collapse line; stopped and took P2; "no training a little longer" executed verbatim** → retry knob (1),
change the checkpoint: screening 500k/1M/1.5M → **1M = 105.7 (first 16 seeds, +12.6% over the script's 93.9 on the
same subset, 29% divergence)** → G3 full 32: 76.0 = 0.968×H7, 5/32 deaths, all four R4 sentinels green → gold
eligibility established.

**Collapse post-mortem (full sentinel data)**: 95.8% of the training distribution was dry-level windows (wage ≈ 0)
— in a reward desert, entropy is the wind: wandering costs nothing, drift is free, and fresh-level skill erodes with
it. Trajectory: 500k = 94.0 (close to the anchor, 1.8% divergence) → **1M = 105.7 (peak)** → 1.5M = 59.2 → 2M = 65.3
→ 4M = 42.6. The γ=1 value head (residual uncertainty #3) could not hold the anchor. Saving a checkpoint every 500k
(lesson from v9) saved the campaign.

**Final gold-seed review (9000-9031, revealed once, single arm, 1M ckpt)**:

| Metric | v23 learned worker | v22-H (scripted worker) control |
|---|---|---|
| Gold mean | **77.0** (median 91.4) | 93.9 (median 103.5) |
| Combat deaths | 3/32 | 2/32 |
| Divergence from the script | 39.5% | 0 (by definition) |
| FARM level-change rate (arbitrage gauge) | **0.0** | 0.0004 |

**Verdict: P1 missed all three tiers (0.82×). The replaceability claim does not hold — recorded as is.** Three
narrower things do hold: (1) **off-anchor competitiveness exists**: the 1M worker beat the teacher by +12.6% on the
first half of the probe pool with 29% divergence, and spontaneously used keys the teacher never pressed (potion
pickup / equip) — the learned worker is not a parrot, and this is the first real innovation beyond the v22
vocabulary; (2) **lesson 16's engineering won completely**: wage stripping kept the dive arbitrage at zero occurrences
over 8.3M steps and hundreds of thousands of windows; (3) death discipline held (3/32). The cause of defeat is not
"cannot be learned" but "cannot be held" — anchor erosion in the reward desert, with the peak at 1M while the
acceptance budget said 8M. R-line scorecard: R1 ✗, R2 ✓, R3 ✗ (under the registered 8M definition; the 1M ckpt at
0.968× falls inside the band, noted as is), R4 ✓ (all sentinels green throughout), R5 ✗ (77.0 vs [85,105]). Power
statement: n=32, SE≈4, and 0.82× sits about 2.5 SE from the 0.95× line — not noise, a real gap.

**Lesson 18 (draft)**: the anchor inevitably drifts in a reward desert — the erosion speed of a BC anchor is set by
the share of zero-wage states, not by the entropy coefficient alone; peak checkpoints are the norm, not luck, and the
acceptance protocol must write "select the peak" into the pre-registered clauses (this time it was written, and it
saved the run).

**v24 prescriptions (queued under the single-prescription rule)**: (1) intervene in the dry-level window mix during
training (skip/downsample dry-level windows, or a small separate wage for dry-level windows — needs a closed-form EV
audit); (2) peak retention: an entropy schedule / KL regularisation toward the anchor; (3) a DIVE worker (the fat of
the oracle's 44% stall rate, workstation budget); (4) alternating manager-worker freezes (the full architecture,
opening the workstation phase in August).

## v24 chapter: the leash (a crutch for the operator brain, to be dropped gradually)

**Design and pre-registration** ([PREREG-v24](../prereg/PREREG-v24.md), synthesised in a design review; all 22 items
confirmed in the pre-launch review were implemented): add β·CE to the PPO loss (frozen BC teacher, mask renormalised),
fixed annealing 0.5→0.015625→0 (8 legs × 1M steps, an exam between legs), two tripwires (hard 62.8 stops training /
soft 0.97×P\* freezes β); the clauses are executed by the driver, gate_ledger.jsonl records everything, and there is
zero human discretion.

**The eight legs**: legs 1-3 (β≥0.125) = bit-level parrots (93.9, divergence 0); leg 4 (β=0.0625) = 99.5 — the door
of divergence opened 1.4% and the score went over the top; leg 5 = 91.6, a narrow pass; **leg 6 (β=0.0156) divergence
jumped to 29% → 86.0, soft trip; the clause caught the drift as designed**, β frozen; **leg 7 (β frozen at 0.0156) =
110.1, the peak of the campaign**; **leg 8 (β=0) = 55.6, 79% divergence, collapsed as soon as it let go, hard trip,
stopped** — even with a value head fed for 7M steps, the wind of the reward desert still wins at β=0.

**G3 full 32**: leg 7 = 92.0 (**1.17×H7**; v23's best without a leash was 0.97×), 2 deaths; leg 4 = 90.8, 3 deaths;
both candidates cleared the eligibility line (not a lone peak). **Gold (leg 7, single arm, once)**:

| Metric | v24 leashed worker | v22-H (scripted worker) | v23 unleashed worker |
|---|---|---|---|
| Gold mean | **97.2** (median 93.8) | 93.9 (median 103.5) | 77.0 |
| Combat deaths | 2/32 | 2/32 | 3/32 |
| Gold-pool divergence | 0.66% | 0 (by definition) | 39.5% |

**Two verdicts (pre-registered clauses verbatim)**:
- **P1 strong win**: 97.2 ≥ 93.9 and deaths 2 ≤ 4 — "the learned worker reaches and exceeds the H level"; the
  replaceability claim is confirmed, and v23's loss is overturned by the leash.
- **P-crutch-false**: the winner comes from a β>0 leg — **"still on the crutch; the independence claim does not
  hold"**; R-v24.3 (keep the peak after β goes to 0) was cleanly falsified by leg 8, and the dual attribution rules it a
  **capability falsification**, not a schedule falsification. Verdict discipline: gold divergence 0.66% <2%, so under
  R-v24.4 the only permitted wording is **"continuing close to the anchor"**, never "independent of the anchor".

**R-line scorecard**: R-v24.1 ✓ (peak 110.1 ∈ [95,115]); R-v24.2 split (won the 4M head-to-head outright, 99.5 vs
v23's 42.6, but "every leg ≥62.8" was broken by leg 8); R-v24.3 ✗ (lost, capability path); R-v24.4 ✗ (final-leg
divergence 79%, outside the [10,45] band); R-v24.5 ✓ (double-ledger reconciliation consistent); R-v24.6 ✓ (97.2 ∈
[85,110], point 92). Six lines, four hits and two misses, all filed before the reveal.

**Lesson 19 (draft)**: **the end point of annealing is not zero but a feather's weight** — in a reward desert a light
anchor (β≈ε) is permanent equipment, not a temporary crutch: a β=0.0156 leash is enough for the worker to beat the
teacher by +17% (probe) / +3.5% (gold), while β=0 falls off the cliff within 1M steps. The correct formalisation of
"dropping the crutch" is not β→0 but finding the infimum of β; in engineering terms, the hand of the soft tripwire that
froze β is smarter than any annealing schedule.

**v25 prescription queue**: (1) a bisection search for the infimum of β (workstation); (2) the dry-level mix
intervention (root cause of the reward desert; each tested alone before combining with the leash); (3) a DIVE worker
(the fat of the oracle's 44% stall rate); (4) the full architecture with alternating manager-worker freezes (opening
in August).

## v25 chapter: the re-election (manager retrained on top of the leashed worker; pre-registered in [PREREG-v25](../prereg/PREREG-v25.md) v2)

**The exam**: the v22-H manager learned its values against the scripted worker; after the crew was upgraded (the leg-7
leashed worker), does retraining the manager help? The design was frozen after the review (27 confirmed items,
including an n_steps=64 recipe correction, a paired launch criterion and full 32 seeds for both arms) was fully
implemented; the driver executes it, and the gold launch line = a per-seed paired mean difference ≥ +4 against the
v24-G3-leg7 archive and paired wins ≥18/32.

**The election**: G-A0 instrument regression 32/32 with zero mismatches → both arms with the original v22-H recipe,
~40.2k decisions each (24.4/22.4 minutes, throughput matching the v22-H archive to the second) → M-fresh screening
101.7 → **full 32 fell back to 88.9 (the third appearance of the half-pool mirage — exactly why the review clause
required full 32 for both arms)**, 5 deaths, DIVE 0.34/episode (~3× the incumbent, inside the R25.3 band); M-warm
(full v22-H weights as a warm start, lr 1e-4 / ent 0.005) **70.6 — fine-tuning dragged the incumbent itself from 92.0
down to 70.6**, paired warm−fresh = −18.3.

**Launch criterion**: the winner M-fresh has a paired mean difference of **−3.07** against the incumbent and **9/32**
paired wins — far from the +4/18 launch line. **No gold run; v24-golden (97.2) keeps the throne and the leaderboard
does not change.**

**Verdict (P25 no-launch, incumbent retained)**: "the manager is retained; this alternation round brings no gain
(within power)". R-line check: R25.1 ✓ (88.9 ∈ [85,100]); R25.2 ✗✗ (−18.3 vs [0,+4] — the "old values are baggage"
hypothesis was opened in reverse: **touching the old values is the baggage**; fine-tuning cost −21 points, another case
in the anchor-erosion family; the lr/ent confound was pre-registered as non-attributable); R25.3 ✓ (0.34 <0.5; trend
note: the new manager dives 3×); R25.4 not triggered. The DIVE gold-pool definition was not measured because there was
no launch, recorded as is.

**Scientific footnote (the real gain of this chapter)**: the full form of the architecture (learned manager + learned
worker) **was already crowned in v24** — v22-H was itself a learned policy brain. What v25 shows is its **robustness
across crews**: a strategy learned against the scripted worker stays optimal without retraining after the worker
improved by +17%; neither retraining (from scratch / fine-tuning) caught up with it. The manager's knowledge is about
the structure of the world (stall clock, when to farm), not about the identity of the worker — the first evidence for
the "separation of concerns" promise of the hierarchical architecture. Recorded as an observation; whether it becomes
a lesson is still open.

**The v26+ queue is unchanged**: bisection on the infimum of β, the dry-level root cause, a DIVE worker, the full
architecture with alternating freezes (with a longer budget a retrained manager might overturn this — this chapter's
verdict only covers a 40k-decision budget).

## v26 chapter: the oasis (skip-dry; pre-registered in [PREREG-v26](../prereg/PREREG-v26.md))

**The eight legs**: legs 1-3 welded (93.9×3) → leg 4 = 98.6 (8.3% divergence; six times wider than v24 at the same β
and safer) → **leg 5 = 112.2 with 0 deaths (R26.1 point prediction 112, nearly dead centre)** → **leg 6 = 114.5 at
28.6% divergence — reaching the summit exactly where v24 fell off the cliff (86.0 at 29%), the project's highest leg
exam** → leg 7 (β=0) = 83.8 (62% divergence), soft trip → leg 8 (β=0) = 46.9 (92% divergence), hard trip, stopped.
**Lesson 19 survived review: the end point of annealing is not zero, in the oasis too** — the desert amplifies drift
but is not its only source; at β=0 the entropy wind still scatters the anchor on fresh levels, and the oasis only
delays the fall from "collapse within the leg" to "one leg later" (83.8 vs v24's 55.6).

**G3 full 32**: leg 6 = **108.2, 3 deaths, all R4 green, 41.5% divergence** — the project's highest probe score (1.18×
the worker archive's 92.0); leg 5 = 99.5, 1 death. **Paired launch criterion: mean difference +16.2 met, but only 11/32
paired wins < 18 → gold correctly withheld.** Mechanism note: a high-variance strategy with 41.5% divergence — big wins
on 11 seeds, small losses on 21; the mean rises and the median falls. The "paired wins ≥18" clause demands a broad
advantage, not an expected-value advantage, and it protected the gold pool. A driver verdict-text bug (it printed the
[+2,+4) tier by mistake) was corrected in PREREG-v26 appendix 2; the launch decision itself was compliant.

**R-line check**: R26.1 ✓ (114.5 ∈ [100,125]); R26.2 ✗ (leg 8 broke through; exactly 1 soft trip); R26.3 ✓ (zero
dry levels in learning windows throughout); R26.4 not triggered; R26.5 answered: **it still collapses**; R26.6 ✗
(dry-anchor mismatch 32% > 15%, record-only, no verdict — the dry-level behaviour drift did not hurt the return,
supporting "dry levels are low risk" but overturning the "≤15%" expectation). Verdict: **v26 is not an all-round win;
the throne stays with v24-golden 97.2**; v26-leg6 (108.2) joins the workstation rematch queue (the exam: turn an 11/32
rout into a broad 18/32 advantage — "width training").

**v27 branch (pre-registered branch clause executed)**: Otherwise → the v24 recipe ×7M (8 legs × 874,496), with the
launch criterion upgraded to the paired standard. The launch was delayed by 5.8 h, recorded as is.

## v27 chapter: the desert control arm (v24 recipe rerun; pre-registered = PREREG-v26 appendices 2/3)

**The exam**: the Otherwise arm of the pre-registered branch clause. Nominally "the v24 recipe ×7M"; the measured
accounts: LEG = 874,496 (0.875× a v24 leg), about 7M over 8 legs, against the original v24 campaign
(8×1,001,472 ≈ 8M) — **a near-equal-budget redraw**, so it really answers two questions: (1) is the v24 recipe's
throne result reproducible; (2) is v26's +16.2 due to the oasis recipe or to insufficient compute in the desert (a
controlled comparison).

**The eight legs**: legs 1-4 welded (93.9×4, divergence 0) → leg 5 = 96.0 (β=0.03125, 0.1% divergence) → leg 6 = 94.0
with 0 deaths (β=0.015625, **only 9.1% divergence** — at the same β tier v24 had already unwelded to 29% (86.0, soft
trip) and v26 to 28.6% (114.5, summit): the same recipe redrawn shows huge variance in when it unwelds, so a single run
cannot be attributed) → **leg 7 (β=0, tail-cut 500k) = 106.9 with 0 deaths (40.9% divergence) — the first leg in the
whole case that let go without collapsing and peaked instead** → leg 8 (β=0) = 58.4 with 3 deaths (86.6% divergence),
hard trip, stopped (6.25M actually trained).

**Two-point sampling of harvest vs erosion (a post-hoc hypothesis, not pre-registered; leg 8 was its first predictive
test)**: letting go close to the anchor (9.1%) = a harvest of +12.9; continuing far from the anchor (40.9%) = a fall of
−48.5 with divergence exploding to 86.6%. **What kills is not letting go but letting go when already far from the
anchor; letting go can harvest, but cannot settle** (material for an amendment to lesson 19). Also: the v24 β=0 body at
55.6 and the v27 body at 58.4 — a desert-erosion gravity well at ~55-58 reproduced a second time, added to the
post-mortem book.

**G3 full 32**: leg 7 = **93.5 with 1 death** (all R4 green: override 1.8% / level change 0.05% / τ̄ 38.7; 46.2%
divergence) — **the fourth half-pool illusion case: −13.4** (first-16 pool 106.9, last-16 pool inferred ≈80.1), the
same family of cross-section as v25 (−12.8), with v26 (−6.3) the only one spared; leg 5 = 92.6 with 3 deaths (0.4%
divergence, essentially the teacher itself — 4.4M desert steps are worth only +0.6 on the wide pool). **Paired launch
criterion: leg 7 +1.51 / 12 wins, leg 5 +0.62 / 16 wins — far from the +4/18 line; no gold run; the throne stays with
v24-golden 97.2 for the third time** (the tail-cut demotion note travels with the verdict: legs 7-8 are short legs cut
in half by the sps clause).

**Verdict (incumbent retained) and route conclusions**:
(1) **Reproducibility**: under a near-equal-budget redraw, v24's throne path (110.1 → gold 97.2) did not reproduce —
that path depended on the accident of leg 6's soft trip freezing β; this time the draw was a "late unwelding + narrow
β=0 harvest" trajectory, with a breadth gain of only +1.5. **The desert recipe is luck-dependent.**
(2) **Control (the main answer of this chapter)**: oasis 4M (108.2, +16.2) vs desert 6.25M (93.5, +1.51) — **1.56×
the compute cannot buy back the recipe gap; v26's advantage comes from the recipe** (the window-entry distribution
confound is carried as recorded bias (2) of PREREG-v26). Workstation main line confirmed: **the v26 lineage + width
training** (11/32 → 18/32); β_min ≈ 0.016 kept; the desert branch keeps only one research hook: a near-anchor harvest
sprint (a short β=0 run + an immediate stop + a full-pool evaluation).

**Pre-registration check** (v27 has no R table of its own; checked against appendix 2 + inherited clauses): the branch
decision ✓ registered before the results and executed as is; the paired launch criterion ✓ executed as is (not
passed); the β=0 collapse expectation ✓ (leg 8) / **surprise** (leg 7's counter-peak, not registered, recorded as a
post-hoc hypothesis); the sps downgrade clause ✓ (triggered after leg 2, legs 7-8 cut to 500k); wall-clock anomalies
×2 (leg 2 ~1.6 h and leg 6 ~4.4 h of crawling, machine cause unknown) and a verdict more than 5 h late, recorded as is.

## v28 chapter: oasis continuation (pre-registered in [PREREG-v28](../prereg/PREREG-v28.md); same-day post-mortem → two review rounds → early finish in 3h13m)

**The exam**: is v26's width problem (11/32 wins) under-training or intrinsic? The single prescription = the v26-leg6
checkpoint + a constant β=0.015625 (never let go) + skip-dry × 8 legs. The same-day chain: post-mortem (lesion = 6
misread maps accounting for 87% of the losses + 15 maps with −4-point exploration friction) → pre-registration →
first review, 24 items → driver built → smoke test (two definition fixes: teacher divergence on the training
distribution only 3-6%, dry-anchor starting point 63%) → second review, 13 items → frozen at a commit → launch.

**The three legs**: 105.5 (0 deaths, 54.8% divergence, width 6/16) → 85.4 (29.3%, 4/16) → 90.4 (3 deaths, 16.1%,
5/16) → **the early-finish clause fired for the first time** (two consecutive legs <103.1), saving the budget of 5
legs, with zero crashes and zero manual intervention.

**G3 full 32**: leg 1 = **112.4 with 0 deaths (a project high; starting point 108.2 + 4.2)**, paired +20.42, **16/32
wins — 2 short of the launch line, the closest the whole project has come to the gold pool**; won 10/16 on the last 16;
**after four half-pool illusion cases (−29.7/−12.8/−6.3/−13.4), the first reversal ever**: first 16 = 105.5, last 16 =
119.3 — it dominated the weakest half of the archive pool. Leg 3 = 72.5 with 4 deaths (eligibility failed). No gold
run; **the throne stays with v24-golden 97.2 for the fourth time**.

**Verdict and the cost of the tiers**: the driver landed on tier 4 of the frozen P lines ("the width problem is
confirmed intrinsic; the under-training hypothesis is rejected") — the machine was right, but the narrative is
discounted: tier 4 does not distinguish "missed the line" from "did not move", and the actual 11→16 (+5) is a real
move. The cost of the tier design is recorded, and the next version of the P lines adds a "point-estimate width
improvement" tier.

**Mechanism finding (post-hoc label, read together with v27)**: v27 taught "letting go at β=0 can harvest but cannot
settle"; v28 teaches "continuing far from the anchor at β=0.0156 can harvest but cannot stay tethered long" — leg 1
harvested +4.2 on full 32, and in legs 2-3 divergence fell monotonically 54.8→16.1 back toward the anchor, with the
score collapsing toward the teacher's level. **For a strong model already far from its anchor, the teacher anchor is a
slow poison: integrated over a full leg, the tiny 0.0156 pull is enough to dismantle the stay-in-window deep-farming
strategy.** "The anchor follows the throne" is promoted to the first workstation choice.

**R check**: R28.1 ✓ at the lower edge of the band; R28.2 band ✓, point ✗; R28.3 ✗ (median 5, half the sample);
**R28.4 ✓✓ (actual 16 vs point 17)**; R28.5 not triggered; R28.6 moved, recorded without narrative (level-2 seeds
4→7 — a record-side footnote to the goal of breaking through to level 4).

**Workstation queue reordered**: (1) the anchor follows the throne (teacher = v28-leg1 itself, bisection on β_min) —
first choice; (2) a width rematch (16→18, 2 short, starting from v28-leg1); (3) curriculum sampling (targeted at
friction maps); (4) dive-target variants (to be started once approved). Every new mechanism firing for the first time
worked as designed: the early-finish tripwire, the live G-oasis gate, G-CAL record-only, the width probe, the
archive-immutability gate, NEEDS_ATTENTION, and evaluation post-mortem archiving.

## v28 chapter: oasis continuation, detailed record (pre-registered in [PREREG-v28](../prereg/PREREG-v28.md); early finish)

**The exam**: is the width problem (v26 paired wins 11/32) under-training or intrinsic? **Prescription**: continue
the v26-leg6 checkpoint for 8 legs with β constant at 0.015625, never letting go, and add a width probe to the leg exams
(against the first 16 of the anchor, baseline 5/16).

**Record**: leg 1 = 105.5 with 0 deaths (54.8% divergence, width 6/16) → leg 2 = 85.4 (divergence shrank back to
29.3%, width 4) → leg 3 = 90.4 with 3 deaths (16.1%, width 5) — two consecutive legs <103.1, the early-finish clause
fired live for the first time and went to G3 (only 3 of the 8-leg budget used; budget protection worked as designed).
**How legs 2-3 died is the first new knowledge of this chapter: long exposure to the teacher leash dragged the
champion back toward the script (divergence 55→29→16%), and the winning move (stockpiling potions mid-fight) was
confiscated step by step by the anchor — letting go collapses (v27), a long tether binds (v28); the anchor must follow
the throne.**

**G3 full 32**: leg 1 = **112.4 with 0 deaths — the project's highest probe score** (paired +20.4 against the worker
archive's 92.0; width 16/32, last 16 = 10/16; the only reversed archive among all 35 with s16 < full32, the signature
of a real skill); leg 3 = 72.5 with 4 deaths (stopped by eligibility; its divergence consists of eight-direction
footwork — the skill lives in the leg, not the generation). **Launch: 16/32 < 18, two seeds short; no gold run; the
throne stays with v24-golden 97.2 for the fourth consecutive time.** The verdict is stamped with the original text of
pre-registered tier 4: **the width problem is confirmed intrinsic (within power) — the mean exceeds the starting point
of 108.2 but width falls short, the under-training hypothesis is rejected, and the mechanism prescriptions (anchor
follows the throne / curriculum sampling) are promoted to the workstation.**

**R check**: R28.1 ✓ (peak 105.5 ∈ [105,130], at the lower edge; point 118 missed); R28.2 half hit (all legs ≥62.8 ✓;
number of legs <103.1 = 2 at the edge of the band ✓; the point prediction "no early finish" ✗); R28.3 ✗ (median width
over three legs 5 < [6,12] — first-16 width did not move); R28.4 ✓ (16 ∈ [12,22], close to point 17); R28.5 not
triggered; R28.6 **moved, recorded without narrative**: full-32 depth distribution {0:2,1:23,2:7} vs baseline
{0:2,1:26,2:4} — level-2 maps 4→7 (later attributed by the winning-move archaeology: the worker's new behaviour makes
the frozen manager's DIVE more valuable; details in
[docs/forensics/FORENSICS-winning-moves.md](../forensics/FORENSICS-winning-moves.md)).
**Note**: the first-16 width probe stayed flat (5-6) while full-32 wins went 11→16 and the last 16 were won by a wide
margin (119.3 vs 73.8) — the width increment hides in the hard half-pool; the half-pool gauge misled again, adding a
"reversed case" to the illusion family.

**Lesson candidate**: the second half of lesson 19 — **"the leash can neither be slackened to zero nor kept on for
life: the anchor follows the throne"** (three cases converging: v25 bare fine-tuning fell apart / v27 letting go
collapsed / v28 a long tether binds). Champion v28-leg1 is archived at train/models/v28-worker-leg1 (sha
2f7bc9dd810956c3, the SHA-256 prefix before the 2026-09-23 re-save), the crew for the v29 manager re-education.

## v29 chapter: manager re-education (pre-registered in [PREREG-v29](../prereg/PREREG-v29.md); first test of the depth economy)

**The exam**: can the depth economy be learned (the winning-move report: the 8×N descend bonus is cashed at a rate of
≈0 across the whole lineage)? **Prescription**: retrain the manager against the v28-leg1 champion worker, 160k
decisions per arm for two arms (4× v22-H's own budget), anchor = v28-G3-leg1 (112.4, the incumbent assembled agent).

**Record**: G-A0 bit-level regression 32/32; M-fresh finished its arm in 88.7 minutes — nt_zip exactly 160,000 while
status counted 159,988, **the first live firing of the review panel's step-count gate blocker** (unfixed, the
unattended run would have died on a false verdict); M-explore took 105 minutes. Full 32: **fresh 140.3 with 3 deaths**
(depth2 15/32, DIVE 0.69/episode, bonus cashed 4.25/episode), **explore 149.0 with 8 deaths** (depth2 21/32, DIVE 1.44,
**depth_median 2.0 — the median map reaches level 2**, cashed 5.75) → pre-registered residual #2, "high-entropy
reckless descending", came true word for word; **the void line (DIVE>1 and deaths>6) eliminated explore; the
substitution clause (review-panel blocker 2) fired live for the first time, and fresh takes the win under D3-2**;
r29_2 (explore−fresh) = +8.7.

**Verdict**: paired +27.86 / **17/32** wins (line 18, one seed short; the width line blocked for the fourth time in a
row: 11→16→17) → **point-estimate gain tier; no gold run; the throne stays with v24-golden 97.2 for the fifth time**.
**Depth secondary verdict (the exam's main metric): "the depth economy has been learned"** — the 8×N clause, dormant
for half a year, was claimed in bulk for the first time; Mark-I is not reached under its recognition line (learned and
taking the throne); this round's result = mechanism unlocked.

**R check**: R29.1 ✗ **broke through upward** (140.3 > band top 125 — the prediction was too timid, owned as is);
R29.2 ✓ (+8.7 ∈ [−10,+15]); R29.3 ✓ (15, point 12); R29.4 ✓ (0.69, point 1.0); R29.5 not triggered; R29.6 nearly dead
centre (4.25 vs point 4). τ̄ 46.6 recorded as a note only (the new manager, working with the stay-in-window worker,
keeps windows longer, 0.2 above the R4 band; record-only, no verdict).

**Prescription candidates for the next seed (workstation rematch)**: entropy balancing (search between 0.02 and 0.08
for the depth × safety Pareto point), a depth-safety curriculum (learn not to die first, then learn to descend), the
anchor-follows-the-throne worker line (item 2 of the [course plan](ROADMAP-course-plan.md) first). Explore's 149 with
8 deaths shows the ceiling is still higher — the currency to buy it with is safety.

## v30 chapter: worker relay (pre-registered in [PREREG-v30](../prereg/PREREG-v30.md); first live use of "anchor follows the throne")

**The exam**: can the champion worker's deep-level weakness be fixed? **Prescription**: retrain v28-leg1 under
M29-fresh; the only variable between the two arms = the leash teacher (king: static self-anchor / bc: the old
teacher). By design decision, route B went first (the direction review (not published); the decision and its
anchor-compliance rule are written out in [PREREG-v30](../prereg/PREREG-v30.md) D3).

**Record**: G-KL-W 0/1000, G-A0W 32/32; king: 129.8 (6 deaths) → 135.3 (4 deaths), only ~9% from its self-anchor on
the training distribution but 62% from the script — **the leash holds the throne's identity while letting new learning
through; the mechanism worked as designed**; bc: 138.6 → **73.2, stopped by the tripwire** (divergence 54% → 21%, the
third independent replication of "a long tether binds"). The driver died on a metrics dict-key bug (the top-level
fallback recorded it as is), and per the restart protocol a verdict resumer reused all clean assets to finish (the
restart event is in the ledger).

**Full-32 gauges for both arms (the real gain of this chapter)**:
king = 132.1 / 7 deaths / dive 0.81 / depth2 16 / **d2 deaths 7** / a13 **49.2%** / depth median 1.5;
bc = 65.9 / 1 death / dive 0.22 / depth2 4 / a13 **4.7%**.
**R30.6: king−bc = +66.2, 22/32 wins — the first head-to-head of "anchor follows the throne" against the old anchor,
and an overwhelming one.** Within two legs the BC anchor erased the winning move (stockpiling potions) from 39.7% to
4.7%; the self-anchor not only kept it but grew it to 49.2%.

**Verdict**: king stopped by eligibility (7 deaths > 6, not voided); bc is the substitute winner → 65.9 < the floor of
129.1 → **"retraining did not reproduce the starting level"; no launch; the throne (97.2) stays for the sixth time**;
the scientific main verdict (judged on the winner, bc, per the clauses) = out of band / exposure collapse, **and the
verdict carries the gauges of both arms to keep the narrative straight: the treatment arm's (king) real reading is a
worsening of d2 deaths from 3 to 7**.

**Two answers**: (1) **the anchor follows the throne holds** (from the workstation onward, every continuation of a
strong model far from its anchor self-anchors — the second half of lesson 19 is promoted from observation to
interventional evidence); (2) **retraining in FARM windows cannot fix deep-level deaths; it amplifies them** (king kept
its logistics, dived deeper and died more: L1 survival habits do not transfer to the L2 damage spectrum, and the 0.5
brainstem threshold plus zero gear cannot cover it) — **course 3 (gear) / course 4 (potion autonomy) / course 2
(renegotiating wage stripping) are promoted from "queued" to the critical path of deep survival**, and the route-C
post-mortem (11/11 died in farming windows after landing on a deeper level) is re-checked by intervention. R-line
check: R30.6 broke through upward (+66.2 vs band top +15, point +5) — the anchor effect exceeded our imagination for
the third time in a row; R30.5 had the two arms break through the upper and lower edges of the band respectively
(0.492/0.047) — the band was registered by imagining a single arm, the two-arm reality tore it apart, owned as is.
**Fixed next steps (from the direction review, not published, plus this case's evidence)**: next = the course 2
wage-stripping specification + the G0 design draft (for design review); route A keeps a one-time rematch right (the
clause is in the [course plan](ROADMAP-course-plan.md)); workstation = the full self-anchor case + an environment curriculum, started
once approved.
