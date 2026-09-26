# Winning-moves hunt report (2026-07-11, archaeology of three seams; data = all 35 eval-assembled archives + the environment code)

## 1. The two moves identified (cross-generation evidence)

**#1 Potion hoarding in combat (a13, first seen in v26-leg6, deepened in v28-leg1): the only divergence that really
pays.** The script's ancestral rule: potions may only be picked up after the monsters are cleared (dispatch order); the
worker cannot drink by itself (12 is always masked; the right to drink belongs to the brainstem reflex at hp<0.5). The
champion puts 39.7% of all its actions on the potion-pickup macro (the v24 line only 1.1%), and its 42.4% "script
divergence rate" is almost exactly = the a13+a10 share: the entire content of the divergence is this one move.
Effect: the brainstem's magazine always has rounds -> 0 deaths under both definitions -> only then can it afford long
deep-farming windows. The hard half pool went 73.9->119.3.
**It cannot pull the trigger itself, so it learned to load the gun for the one who does.**

**#2 Deep farming inside the window (started in v24, amplified in v26/v28): the amplifier of #1, which does not pay on
its own.** Mechanism = a steady stream of kills feeds the 140-beat no-kill stall clock, merging fragmented windows into
long windows of tau 146 (7031: 73 windows, 40 beats, 52 kills -> 19 windows, 146 beats, 116 kills). Counter-evidence:
the v27 desert line has the highest tau of all yet is only worth 93.5: staying in a window without fault tolerance
inside it is staying for nothing.

**The signature of a real move**: v28-leg1 is the only one of all 35 archives with s16<full32 (105.5<112.4): a real
move wins on hard maps; a phantom wins on easy maps (the 16 pool is systematically ~+15 too high, four confirmed cases
on record).

**Counter-examples on display (discipline)**: at the same ~46% divergence, v27's content is eight-direction
footwork, and its pairing is only +1.5 = idling; in the same campaign v28-leg3's divergence turned into footwork, 72.5
with 4 deaths. **The content of a divergence pays, its amount does not; the move lives in the leg, not the
generation.**

**Case 7024**: the only map cracked across three generations: v24/v26 died here in byte-identical fashion in a
RESUPPLY dead loop, while v28 first fought 14 windows, went down 2 levels with D and took 135.5 with 51 kills, and
survived.

**An invariant**: farming micro-beats per episode are constant at ~2600 (the time budget is welded shut), so all
progress = the kill rate per thousand beats, 12.9->17.0 (+32%). This is an efficiency race, not a duration race.

## 2. List of untaught lessons (sorted by the value of the seam)

The worker really knows 3.5 of its 15 keys (9/13/10 + occasional direction keys).
1. **The depth economy (the biggest unopened seam)**: the 8xN descent bonus has been in place since v17, and its payout
   rate across the whole lineage is ~0 (median depth 1); the frozen v22-H almost never chooses DIVE, and a dead
   episode of 106 idle windows like seed 7023 has no cure. 100% of the descents to depth=2 were made by the manager's
   D, and the v28 worker's behavior change makes DIVE more valuable (4 wins on 5 new level-2 maps): unfreezing/
   retraining the manager is the lever with the highest return ceiling, and also the right road to "breaking through
   to level 4".
2. **Making DIVE a worker skill**: the inner loop is currently a three-line script (thresholds all guessed); the
   prerequisite = renegotiating the wage-stripping formula (the worker currently bears the depth death penalty
   without enjoying the descent bonus, so the optimal deep-level play is to slack).
3. **Reviving the equip key**: a14 was used 0 times in 50,000 calls in v28: the worker has retired the equipment
   lesson (v24-golden still pressed it 50 times), and the delta-AC shaping has become a dead clause; a prerequisite
   subject for deep-level survival.
4. **Potion autonomy**: the brainstem's hard 0.5 threshold degenerates into reactive life support at depth; teach it
   bundled with the DIVE lesson.
5. **Dry-window mismatch**: skip_dry means the worker never trained in the drained state, yet takes the evaluation in
   it unprepared.
6. No lesson planned: RESUPPLY (the script is near optimal); the town economy (mana/spells/shops/gold are dark matter
   the environment already models but with zero interface; v29+ environment engineering).

*Prepared by three read-only reviews: course-plan inventory, champion diff, and lineage archaeology.*

## #3 Root cause established (2026-07-15, probe-gear-value, no protocol contact)

Paired counterfactual of armed vs gear_available removed (H x script, 7000-7031): paired differences all 0, both
variants on bit-identical trajectories: the armed variant actually executed a14 **0 times**, while equipment
opportunities ran as high as 464.7 beats/episode. Root cause = the priority order of the dispatch farm branch (story >
clearing monsters > picking up potions > equipping), i.e. **teacher priority starvation** (the teacher never
demonstrates it), which comes before any unit-price mismatch (H3->H3'); the premise of 3C, "the demos naturally contain
a14 samples", is **falsified** and 3C is shelved automatically under item 4 of the decision; the viable routes change
to 3D, renegotiating teacher priority (dispatch is a frozen pure function, so a protocol-level action needs a dedicated
case) / 3E, synthetic demonstration injection (a separate case on the training side). Data:
train/runs/probe-gear-value/probe_report.json (not published); the full original verdict is in appendix A of
[DESIGN-gear-and-potion-autonomy](../design/DESIGN-gear-and-potion-autonomy.md).
