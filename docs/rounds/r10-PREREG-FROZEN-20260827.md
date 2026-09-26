# R10 pre-registration design draft v0.1 (draft, to be frozen after approval)

Scope set on 2026-08-27: restructure the reward function toward the actual goal of completing the game,
and parallelise training across 32 cores; the next training batch starts once both are done.
Basis: the R9 final verdict + the behavioural-collapse forensics
([r9-FORENSICS-behavioral-collapse-20260827.md](r9-FORENSICS-behavioral-collapse-20260827.md)): the
manager-side tools are exhausted, and the root cause is the wage system.

## 1. Course 2: DIVE wage-system reform (environment side)

North star: completing the game (L16 / the quest chain). The wage system serves completion, no longer a
kill farm.

Candidate options (combinable; final choice under review):
- A. Stair bonus: a one-off reward for first reaching a new deepest level, increasing with depth (e.g.
  base*depth or exponential);
- B. Depth wage multiplier: the same kills/experience are multiplied on deeper levels (farming shallow
  monsters loses value);
- C. Death penalty repricing: today the worker bears an 8xN death penalty with no dive compensation =
  diving has negative EV (the v20 case); tie the penalty to "was the agent pushing into a new depth when it
  died" (exploratory deaths lighter, idle deaths heavier);
- D. Anti-idling clause: a decay coefficient for time spent on the same level (overstaying without
  descending gives diminishing marginal wages).

New pre-registration criteria (two lessons from the R9 forensics that must be written into the rules):
1. Command-wage distinguishability probe: verify before and during training that the expected return
   difference between the manager's three commands in the same state is detectable (otherwise the system
   is judged still indistinguishable, and the run stops early);
2. Behavioural identity pre-check: before a control-arm exam, run the argmax agreement probe; agreement
   >99% declares the paired test powerless, and no exam is wasted.

Invariants (unchanged): the certified worker stays frozen (sha 2837288d); REF_BITEQ world continuity; seed
discipline (the 7000/8000/12000/9000 pools stay untouched, however tempting the final pools are); the G0
gate series.

## 2. Parallelisation (infrastructure)

- Environment parallelism: num-envs 4 -> 24-32 (SubprocVecEnv, leaving room for the system and
  evaluation; capped below the 48 logical threads rather than filling them);
- Re-legislate the seven constants (the R8 decision list): n_steps/env, batch, rollout rhythm, curriculum /
  beta annealing switched from rollout index to global steps (removing the coupling to the number of
  environments), eval interval, checkpoint interval, logging granularity;
- A numpy fast path for the worker: bypass the torch Python wrappers (there is a precedent on the manager
  side); it must pass a bit-level reconciliation gate (G0': torch vs numpy forward passes on the same seed
  agree bit for bit before it goes into service);
- Throughput target: >=8x the current rate (88 steps/min -> on the order of 700+ steps/min), as measured
  in the G0 smoke test; missing it does not affect the science, only patience.

## 3. Process

1. Refine this draft -> approval -> freeze (FREEZE_SHA);
2. Infrastructure first: parallelism + the fast path pass the bit-level gate (pure engineering, the
   science is untouched);
3. G0 smoke test of the wage reform (a short run verifying the distinguishability probe has signal);
4. Official R10 launch (arm design decided then: candidate = retrain the manager under the reformed wage,
   control = the current wage, with a depth gate among the criteria).

Status: DRAFT v0.1, new file, not frozen, under review (2026-08-27).

---
## Revision v0.2 (2026-08-27, review decisions applied)

Decision: A/B/D approved; C is replaced by a revised version: the death penalty is a monotone decreasing
function of depth (heaviest on level 1, lighter the deeper), merged with B into a single depth price
table. The original C (classification by motive) is dropped, because the revised version is a pure
function of depth: it needs no machinery to judge the motive of a death, and it is simpler, cannot be
gamed and is easy to read.

Unified depth price table (to be tuned in G0):
- Wage multiplier m(d): increasing in d (candidates: 1+beta*(d-1) or gamma_wage^(d-1));
- Death penalty P(d) = P0 * gamma_death^(d-1), candidate range for gamma_death [0.6, 0.85];
- Stair bonus A(d): paid once on first reaching a new deepest level in the episode, increasing with d;
- Anti-idling D: marginal wage decay after staying on one level beyond a threshold.
Self-balancing logic: A pulls downward, B rewards survival at depth, C removes the fear of dying deep, D
pushes the agent off shallow levels: the four together form a potential surface that points downward.

New open question (raised in review): gear / skills / stat points. A worker without gear cannot survive at
depth, so the incentive reform may hit a "gear wall" and collapse again. A scouting audit of the action
space and the item mechanics has been dispatched (worker_env, the bridge protocol, the old course-3 case);
after the audit report we decide whether gear capability goes with R10 or gets its own R11 (the
single-variable principle favours the latter; the audit facts decide).

---
## Revision v0.3 (2026-08-27, findings from the parallelisation survey)

Facts:
- SubprocVecEnv is already in place (train_ppo.py:9005, multiple processes whenever num_envs>1), so
  scaling needs no new pipeline;
- The numpy fast path for the worker exists end to end (the --worker-npz branch, PREREG-v25 clause D), but
  the certified risk64 worker has an asymmetric architecture (A12/BC-aux, actor pi[64,64] vf[64,64],
  Discrete(15)) that the current six-matrix NPZ format cannot hold, so export_worker_npz correctly refuses
  to export it;
- The seed-discipline gate works: base 123 collided with a registered pool and was stopped on the spot;
  base 22 plus 32 indices is clean throughout (measured).

New rules:
1. Worker NPZ format v2 (for asymmetric actor inference only): slicing specification + the two actor
   layers + the action head + a contract JSON; an acceptance clause for NumpyManager v2; before going into
   service it passes the G0' bit-level reconciliation gate (torch vs numpy on the same seed, every action
   identical bit for bit, including the mask path and the a12 autonomy branch);
2. Unit reform: every schedule measured in rollouts (distill-anneal-actor-rollouts, sentinel-every,
   dry-anchor-every, distill-ce-probe-every, drywin-metrics-every, checkpoint/eval intervals) is measured in
   global steps or converted by 256/(64*N), removing the num_envs coupling (implementing the seven-constant
   list from the R8 decision);
3. The degree of parallelism is set by measurement: a stress matrix of num-envs {4,16,32} (certified
   worker zip path, seed 22, three rollouts per setting); the throughput knee sets the official N; a
   measured RAM guard rail (one engine instance per environment).

Stress test in progress; results are added in a later section.

---
## Revision v0.4 (2026-08-27, gear audit landed)

Audit conclusions (the full audit report has the details; key points):
- a14 (pick up + equip) exists and is in production: atomic gear swaps, full rollback protection, a
  560-dim gear observation, a release gate on record; after its extinction in v28 (0 uses in 50,000 calls)
  it was revived by three measures (logit bonus 2.5 + a BC forced-recall class + a gear_grace forced
  decision window) and now produces 401 requests / 50 successes per 256 episodes;
- The root cause is price, not capability: the whole gear-swap reward is capped at 1.0 = about one
  monster, 0.19% of the return (note: the +0.5xAC in the old design document is out of date; the current
  rule is min(1, delta_utility/4096));
- a12 (drinking) is permanently closed to the worker in the released configuration, backstopped by the
  brainstem reflex (automatic drink at hp<0.5, 2.41 times per episode): life-saving drinks at depth are
  already covered by the reflex, so unlocking autonomy is an optimisation, not a survival item;
- Skills / spells / shops / inventory / stat points: absent at the protocol level (not gated); town is
  sealed off in three layers, and stat points are allocated automatically 3 vitality : 2 strength. Adding
  any of these is a new engine surface and a separate project for R11+ (the original judgement of the
  [roadmap](../design/ROADMAP-course-plan.md) stands).

R10 gear clauses (merged into the reward-economy package):
- E1 gear repricing: raise the scale/cap of the gear utility reward (candidates: cap 1.0 -> 8-16, or tie it
  to the depth multiplier) so that one real upgrade is worth approx several monsters. Audit warning: doing
  only B (depth multiplier) without E1 makes gear relatively cheaper, so the two must ship together;
- E2 the three revival measures stay unchanged (already in production; do not legislate them twice);
- E3 gear distinguishability pre-check at zero cost: use the existing gear ledger metrics directly
  (mask_opportunities/requests/native_successes, already computed by _gear_progression_gate); they can be
  read during the smoke test, with no new probe;
- E4 allocation table (the heart of the design, to be finalised): how each item of the new economy (A stair
  bonus / B depth multiplier / C decreasing penalty / D anti-idling / E1 gear price) is split between the
  manager's and the worker's wage. Today the worker's wage explicitly strips the 8xN descent bonus (to
  prevent arbitrage by descending mid-fight to escape a penalty, options_env.py:31-33); the new rules must
  rewrite that clause explicitly and keep its anti-arbitrage meaning;
- Potion autonomy (the 4B pair: improved teacher collection + auxiliary BC CE injection) is explicitly moved
  out of R10 and listed as an R11 candidate (it needs a re-certified worker, and its blast radius exceeds
  the single-variable principle).

Scope decision: R10 = the reward-economy package (A/B/C/D + E1, in one protocol version bump and one
re-anchoring; economically inseparable); R11+ = autonomy unlocks and new engine surfaces such as shops /
inventory / skills.

---
## Revision v0.5 (2026-08-27, parallel stress test record)

Stress test results (certified worker zip path, seed 22, n-steps 64, three rollouts per setting,
OMP/MKL=1, timestamps from steady-state rollout intervals):
| num-envs | rollout steps | steady interval | throughput | relative to env4 | per-env efficiency |
|---|---|---|---|---|---|
| 4  | 256  | ~37s   | ~6.9 steps/s (415/min)  | 1.00x | 1.73 steps/s/env |
| 16 | 1024 | ~60.5s | ~16.9 steps/s (1016/min) | 2.45x | 1.06 |
| 32 | 2048 | ~79s   | ~25.9 steps/s (1554/min) | 3.74x | 0.81 |
RAM: ~1.4 GB in total for 32 engines (far below the available memory); rc=0 for all three settings.

Reading and suggested decision:
- Scaling works but is sub-linear (the straggler cost of the SubprocVec synchronisation barrier; per-env
  efficiency 1.73->0.81); 32 is the current sweet spot, and 48 can be explored later;
- The smoke test did not attach the full rule machinery (sentinels / dry anchor / probes / checkpoints),
  so the absolute numbers are optimistic; the relative ratios within the table are reliable;
- At the env32 smoke rate, 160k steps per arm takes approx 1.7 h (an estimated 3-4 h after production
  overhead), compared with 30-33 h per arm in R9: the iteration cycle shrinks by an order of magnitude;
- Suggested rule for R10: num-envs=32 + global-step accounting (clause 2 of v0.3), rollout quantum 2048.

---
## Final chapter v0.6 (2026-08-27, G0 calibration record and the frozen campaign design)

### G0 probe series (seed discipline self-checked throughout; engineering seeds 60-75 / 555-586)
1. rev1 (A24/B0.5/idle900): DIVE-FARM constant difference -29.3, no sign flip → rework;
2. rev2 (A48/B0.25/idle300): sign flip +8.2 at n=16 → falsified with a thicker n=32 (-23.5): the n=16
   flip was noise, recorded as is;
3. Threshold definition corrected: always-DIVE is a suicidal straw man (31/32 died); the real proposition
   becomes "a situational dive (SMART: dive when the level is exhausted or one level up) must lose under v1
   and win under v2";
4. rev3b final (margin=1, n=32, base 555):
   - always-DIVE: v1 -74.7 → v2 -23.5 (suicide still loses; no arbitrage hole);
   - SMART: v1 -19.05 → v2 +5.56 (prudent diving turns positive, although its death rate is 31/32 > 23/32);
   - DiD shift +24.61 (sd 71.7, t=1.94, n=32; 19 episodes shifted, 14 positive).
   Result: calibration passed. The shape of the landscape = reckless loses / prudent wins / idling is
   squeezed, which is exactly the goal of the rules.

### Frozen parameters (identical to REWARD_ECONOMY_V2 in the code)
descend_unit=48.0; kill_depth_beta=0.25; death=2+24x0.7^(d-1) (the revised, decreasing C); gear
scale=1024/cap=12; idle: farm income x0.5 after 300 steps; all booked to the manager's account; the
worker wage-stripping clause carries over (unit prices follow the economy specification; the identity and
the anti-arbitrage meaning are unchanged).

### Infrastructure freeze
num-envs=32 (stress test 3.74x, RAM 1.4 GB); n-steps=64 (rollout quantum 2048); the R10 recipe has no
rollout-measured schedules (verified, so the unit-reform rule is postponed); the numpy worker fast path is
moved to R11 because of the asymmetric-architecture format limit (this round uses the zip-torch path).

### Campaign design (frozen)
- Anchor: M29 (legacy-v3) x certified worker (sha 2837288d), economy v2, pools 2_114 (a) / 2_115 (b) of
  128 each, each run twice and cleared only if bit-identical;
- Candidate: a brand-new MaskablePPO manager (64,64) raw-v4, economy v2, seed 22, total-steps 327,680, ent
  0.02, lr 3e-4, training fuse 21,600 s;
- Behavioural identity pre-check (new R9 rule): candidate vs M29 argmax agreement >=99% declares the
  verdict uninformative and stops early (run at the analysis stage);
- Exam: the candidate on the same pools and seeds, paired with the anchor;
- Gates (applied at verdict time): 1. depth superiority (paired depth LCB>0 over both pools); 2. return
  floor (pooled ret_mean >= anchor x0.90); 3. anti-suicide guard rail (died<=120/128 and mean window count
  >=5); 4. the gear ledger recorded as is (E1 effect reading, not a gate);
- The confirmation pool 2_116 is spent only after all gates pass, on a manual command;
- Environment verification: 864 tests all green (v1 bit-for-bit fidelity), G0 receipts complete.

### Freeze statement
Nothing below this line changes; the master copy is r10-PREREG-FROZEN-20260827.md, and its sha256 is
authoritative (a sha256 recorded for this file refers to the pre-translation text).
Approved on 2026-08-27 (A/B/D, the revised C, booking to the manager's account), with clearance to start
the run.
