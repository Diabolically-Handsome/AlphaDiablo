# R18-B training-arm pre-registration (draft of 2026-09-07; identities are filled in and approved before freezing)

## 0. Design decisions (2026-09-07)

1. Loot selling enters training (it is the core of the snowball plan) → R18-B3 revises the training contract; sustain-loot-v1 enters earned-dive-suffix training.
2. Whole-map hunting is turned off on level 2 (otherwise the model still cannot position itself and is simply beaten to death) → hunt_scope=l1-only enters training arm v1 (wired in R18-B5).
3. Town Portal scroll: first allowed, then removed after the full-interface control row because it is too expensive → training arm v1 has **no portal** (the R18-B5 wiring is still done, kept for a later "buy it when there is money" rule).
4. All the listed mechanisms must be added, otherwise the model plays a distorted Diablo → the chest/barrel sweep (R18-H1) and Cain identify (R18-H2)
   land first, pass zero-training paired acceptance and certification, and only then is the arm launched; both enter the training arm as L1 income sources (wired in R18-B6).
5. Launch time and step budget: not decided (§5 of this file gives suggested values).

## 1. Problem and hypothesis

The R18-X verdict (`r18-X-DEPTH-DIAGNOSIS-20260907.md`) answered "why does it not descend": not for lack of training.
(a) The worker was never paid for descending (R16 escrow never vested, 244/244), so DIVE windows were learned as "keep fighting"; (b) depth is only one action away, but going to level 3 at character level 3 with AC 9
is certain death (counterfactual: level-3 arrivals 2→21, survivors 15→5); (c) snowballing on level 2 is slow and deadly (25 kills per 9000-tick game, hazard 0.37 per 1000 ticks).

**Hypothesis H-B**: in a world where "descending can vest escrow, level 2 has no whole-map hunting, the agent can retreat/open a portal, and selling loot/opening chests/smashing barrels/identifying feed the economy",
a worker whose learning windows open only on level 2 and above will improve both level-2 survival (lower hazard) and growth (kills, clvl), and go down to level 3 when strong enough
(more level-3 arrivals, and more survivors among them).
**Counter-hypotheses** (distinguishable): it only learns to descend, not to survive (repeating the counterfactual: depth up, survival down); or it only learns to hang on without descending (depth unchanged). The four gates and the paired re-test distinguish the three outcomes.

## 2. World (training = measurement, item by item)

| Item | Value | Source |
|---|---|---|
| Starting weights | 7e31dc54 (`r16-arm-a-constitution/model_candidate.zip`) | incumbent worker |
| Prefix | the same frozen 7e31dc54 sampling policy plays level 1 until an "earned descent" (earned-dive-suffix-v1); learning windows only on level 2 and above | R17.1/R18-B |
| Protocol / purchases / economy | l2-town-v1 / full / **sustain-loot-v1** (R18-B3) | decision 1 |
| Readiness rule / manager | coach-v03 (six rules, scripted manager `resource_option_choice`); **the zero-training probe of coach-v04 (per-level thresholds) was negative (survival 19→14, ledger R18_K1_COACH_V04_PROBE_NEGATIVE); not adopted** | R17.1 readiness rule 3 |
| Clock | completion-l2-r18c (arrival 12000 / observation 6000 / 9000 after the first descent) | R18-B2 |
| Retreat / portal | retreat-v1 / **portal-v1 off** (design decision of 2026-09-07 to drop the scroll as too expensive; full-interface control row: with the portal 11/48 survived, hazard 0.415, net −8 against "retreat + hunting off", because the 200-gold scroll used up the money for armor and potions; the R18-B5 wiring is still done, and a "buy it when there is money" portal rule gets its own pre-registration) | R18-A/F, ledger R18_ARM_CONTROL_FULL_INTERFACE_INTERACTION_FOUND |
| Hunt scope | **hunt_scope=l1-only** (R18-B5) | R18-G, decision 2 |
| Income sources | **resource_sweep=sweep-v1** (chests/barrels, R18-H1), **resource_identify=cain-v1** (R18-H2), **resource_weapon_upgrade=smith-v1** (buying a weapon from the smith, R18-K2b, bridge change) (wired in R18-B6) | decision 4 |
| Retreat timing | retreat-v1 unchanged; **retreat-v2 (per-tick reflex + retreat early when far away) was negative with zero training (survival 19→17, deaths on the retreat path 15→14); not adopted**; the next step is the "farm anchored next to the stairs" rule (R18-M, next) | ledger R18_L_RETREAT_V2_PROBE_NEGATIVE |
| Boss avoidance | **not in v1**: after its fix round boss_avoidance=avoid-v1 moved survival 19→20 and hazard 0.228→0.206, but Butcher kills stayed 3→3 and missed the target (the avoidance radius did not cover the descend macro and the retreat walk); v2 next (covering all three path planners + avoiding the Butcher before character level 6) | ledger R18_J_BOSS_AVOIDANCE_V1_FINAL_NOT_MERGED |
| Aggro cap / target selection | off / off | R18-D/E null results |
| Escrow | 0.5 / power 1.6 / readiness gate / ruler v2; windows triggered by retreat or the portal forfeit (R18-B/B5) | R16, R17.0, R18-B |
| Other wage items | identical item by item to the fixed R16 dictionary (hp_loss_price 0.1, a11/a13/a14 logit priors 2.0/2.0/2.5, potion pickup bonus 2.0, no-progress counted as zero) | the fixed earned-dive-suffix-v1 dictionary |
| PPO | mppo, CPU, lr 1e-4, ent 0.005, target-kl 0.01, n-steps 512, batch 256, gamma 1.0, 4 environments (suggest raising to 8) | R16 contract |

## 3. Gates (they crown nothing; they only decide whether the lesson took)

- **Gates A–D**: following the R16/R17 rules, re-cast the controls on the six `r17-anchor-*` volumes (exam protocol in the old 3000-tick form, protocol off): the worker must not regress in the old exam room.
- **E paired re-test (primary criterion)**: the trained worker vs 7e31dc54, with the same interface (the whole §2 set) and the same pool 2_133 (48 seeds, paired, seeded sampling),
  probe `probe_r17_deployment.py` (version r17-deployment-v3-r18m2). Control row = **the v1-world control row** (7e31dc54 × loot selling/coach-v03/r18c/retreat v1/hunting off/sweep/identify/weapon buying; merged tree + merged bridge, 2026-09-07, rows sha 0e5a1acd…):
  20 survived, 19 L2 deaths, hazard 0.222, 4 reached level 3, 894 L2 kills, 253 chests opened, 231 barrels smashed, 16 items identified, 29 weapons bought; paired against "retreat + hunting off": saved 7 / lost 6.
  After installing into the main tree and rebuilding the incumbent bridge, a 16-seed slice must confirm this row bit for bit, otherwise the whole row is re-run.
  - Primary-1: L2 hazard per 1000 ticks (exposure-normalized) ≤ 0.7 × control;
  - Primary-2: paired survival saved − lost ≥ +6/48 and one-sided UCB95 < 0;
  - Primary-3, growth: total L2 kills ≥ 1.5 × control, or the number of games ending at clvl ≥ 4 ≥ 1.5 × control;
  - Depth (informative, expected direction): level-3 arrivals and "survival of level-3 arrivals" rise together; if arrivals rise while survival falls → recorded as counter-hypothesis one.
  - Level-1 prefix: both arms use the same prefix policy, so trajectories before the first descent should be bit-identical (the prefix_through_first_l2 check); if not, this is reported but not judged.
- **F mechanism sub-metrics** (no tautologies): a11 press rate in DIVE windows (natural 2.3%), a11 rate within 5 tiles of the stairs (natural 2.9%),
  retreat trigger count and share dying on the way, portal round trips, chests opened/barrels smashed by the sweep, items identified and the sale-price increase, escrow vested/forfeited amounts.

## 4. Identities (filled in at freezing)

Protocol bundle sha, bridge sha, engine sha, patch stack (0001–0014) sha, worker zip sha, probe version, control-row rows sha, this file's sha → ledger.

## 5. Budget and time (suggested; to be decided)

- Time box: about 8 hours after launch, with a checkpoint + sentinel every 63 488 steps (the same value as R16, set in the launch command); the gates run on the last checkpoint.
- Measured throughput (R18-B3b smoke, light load): **37 learning steps per second with 4 environments** (the environment count is pinned by the R16 parent contract and may not change in migration); 8 hours ≈ 1.0M learning steps.
- Learning-step target: **1 048 576** (=512×2048; one R16 run was 325 632), stopping at whichever of this and the time box comes first; prefix micro ticks do not count as learning steps.
- Prefix budget (per environment, lifetime): attempts 1000, microsteps 100 000 000 (the same as the B3b smoke; only a safety valve).
- Warm start: `train/migrate_loot_candidate.py` schema `diablogym-resource-warm-start/2`, operation `r16-to-sustain-loot-v1-completion-l2-r18c-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v2`, whose world key includes hunt_scope / sweep / identify / weapon_upgrade; the launch command is a local script (not published) (casting the candidate + training; the R18-B6 smoke of 4096 steps had rc=0, **24 escrow vests / 48 forfeits: the first vest in the lineage**, 14 sweep window closes, 7 retreats), wrapped by `launch_r18b.sh` (not published) with a 10-hour anti-zombie fuse (R9 precedent).
- Known gap (does not affect the gates of this round): the evaluation archive schema cannot yet name the loot-selling world (eval_contract rejects sustain-loot-v1 archives); neither the four-gate exams (the old form with the protocol off) nor the paired re-test (probe) go through that path; deployment-form archives are a separate item.
- Training seed: explicit `--seed` (registered); evaluation pool 2_133; the virgin pools 2_116–119 and 2_126–128 are untouched.

## 6. Pre-launch order (fixed)

1. R18-B3 patch → R18-B5 patch → R18-H1/H2 patches → R18-B6 wiring, merged into the main tree (new version strings; the old default paths stay unchanged bit for bit);
2. full suite + probe regression (16 seeds, rows sha 33023de1…) + two-way re-bake 4/4 + 4/4 + re-cast of the six-volume exam room;
3. control row re-run (7e31dc54 × full interface × 2_133); zero-training paired acceptance of H1/H2 (loot income, AC, survival, depth);
4. freeze this file (sha into the ledger) → launch → ledger.
