# R17 Gate T0′ verdict (2026-09-06)

Run on instruction the same day: merge the R17.1-D loot monetisation (candidate in a local work directory,
not published; three-way merge, all 8 conflicts resolved as "keep both"; patch 0012 regenerated against the
CRLF baseline, drift gate PASS) → final-byte certification (full suite 1933/1/0; probe regression 33023de1…
equal; G0 with zero RuntimeError; two-way re-bake in progress, 2/4 PASS on each side at the time of
writing) → T0′. Driver `r10-staging/run_t0_prime.py`; outputs `r10-staging/t0-prime/` (both not
published); new bridge 5693800d…; pool 2_133, paired on the same seeds; clock completion-l2-v1 (a hard
requirement of the gear-selling service: the first L2 cut-off at 12000 actual steps, then 1800 steps of
observation, observation denominator 6000). One seam was fixed during construction: the service handed a
character that was still walking back to the worker → under coach-v03, inside a RESUPPLY window, it now
first waits with an audited script until it stands still (options_env); veto-v1 is unaffected; under
coach-v03, the L2 arrival check of the completion clock no longer requires pretransition_ready.

## 1. Results (48 seeds)

| Group | Clock | Survived | Deaths | L2 reached | L2 hazard per thousand steps | Forced-unready descent share | Final gold | First descent + 1800 observable |
|---|---|---|---|---|---|---|---|---|
| (d′) full + **gear selling** sustain-loot-v1 | 12000 | 13 | 35 | 37 | 0.756 | **0.184** | **125** | 40 |
| (d″) full, no selling, sustain-v6 | 12000 | 15 | 33 | 38 | 0.623 | 0.359 | 95 | 41 |
| Reference (d) full, no selling (T0) | 6000 | 34 | 14 | 20 | 0.760 | 0.500 | 81 | 8 |
| Reference A0′-arm, no channel (T0) | 6000 | 26 | 22 | 24 | 0.890 | — | 100 | 16 |

Paired (d′) vs (d″): saved 1 / lost 3, net −2; death-rate difference +0.042, UCB95 +0.110. **Decision:
T0′_FAIL** (criteria 1 and 2 not met; criterion 4 met).

## 2. Gear selling did its job; survival did not follow

- Money: final gold 95 → 125; forced-unready descents 36% → **18%** (50% in T0 without gear selling). The
  objection recorded on 2026-09-04 holds completely at the level of "readiness".
- Survival: 13 vs 15, hazard 0.76 vs 0.62, no improvement.

## 3. The decisive split: ready and unready characters die equally fast on L2

Split by whether the six-condition law was met at descent (three arms):

| Arm | Ready descents n / L2 deaths / hazard | Forced-unready n / L2 deaths / hazard |
|---|---|---|
| (d) 6000 | 10 / 3 / 0.642 | 10 / 4 / 0.881 |
| (d″) 12000 | 24 / 16 / 0.609 | 14 / 9 / 0.650 |
| (d′) 12000 | 30 / 24 / 0.835 | 7 / 3 / 0.430 |

Pooling the 95 descents of the three arms, **L2 survivors and L2 deaths have almost the same panel at
descent**: AC 10.2 vs 10.8, clvl 3.06 vs 3.08, damage 8.2 vs 7.8, belt 4.25 vs 4.46, HP 84 vs 85. The 21
descents with AC ≥ 13 still died 71% of the time; only 7 descents had clvl ≥ 4 (5 died). Median time to
death on L2: 467 steps.
**In the current range of values, the six-condition readiness law does not separate life from death.** The
panel's objection at the time to option 3, the "inverted ruler" (AC 9 = starting armour + 2, ready by
definition), is confirmed by the data: readiness can be bought, survival cannot. R14's conclusion (h(2) ≈
0.55 is a skill wall, not a price wall) still holds once the resource channel is complete.

## 4. Mechanism summary (measured)

1. At 6000 steps the channel "saves lives" mainly by descending less and later (steps on L2 351 → 192); at
   12000 steps every character that reaches L2 fights there until it dies (33-35/48).
2. The money chain is connected: pick up gold → town trip → buy / repair / equip → sell idle gear → second
   town trip, and the forced-unready share went down in three steps, 50% → 36% → 18%.
3. The per-step death rate on L2 is 0.6-0.8 per thousand steps, the same for characters with AC 10-13, clvl
   3 and four potions. The lever is not purchasing but how the agent plays L2: the hunt macro seeks fights
   actively on L2 (roster 100+), with no retreat, no drinking rhythm and no choice of engagements.

## 5. Next steps (candidates, for decision)

- **Bring R18 forward: an L2 survival course** (the usable parts of the panel's option 2, whose condition,
  "potions and armour available", is now met): a retreat macro, engagement choice, a drinking rhythm; a
  deep-start curriculum (the R9 mcurr machinery).
- **A zero-training mechanism probe (cheap, next)**: same pool, same clock, (d″) with hunt turned off on L2
  (or explore-only, no fight-seeking on L2); if the hazard drops clearly, that proves the lever is the
  intensity of fight-seeking.
- **Recalibrate the readiness law (with care)**: the current data cannot give an AC/clvl threshold that
  separates life from death (AC 13 still dies 71% of the time); the L1 XP pool of 6-9k caps clvl at about
  3-4, and higher thresholds are unreachable on L1. Changing the ruler again is not recommended.
- Training arm: T0′ failed, no launch.

## 6. Seeds and files

2_133 consumed by 2 more groups (6 groups in total). New files: this verdict, and the outputs and scripts in
a local work directory (`t0-prime/`, `run_t0_prime.py`, `t0p_split.py`, `t0p_survivors.py`; not published).
No frozen artefact overwritten. The recast of the six r17-anchor sheets and the two-way re-bake are running
on the final bytes; results are recorded separately in the ledger.
