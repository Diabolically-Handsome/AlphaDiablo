# R18-D/E pre-registration: zero-training probes of the aggro cap hold-v1 and target selection threat-v1 (2026-09-07)

## 0. Background

The R18-C verdict: retreat makes the cycle turn, but the character cannot win fights on L2; the bottleneck has moved from "cannot retreat" to "cannot win". Design requirement:
the model is suspected of aggroing a whole level at once, so an aggro cap of 3–5 monsters is set, finishing them in place before engaging others; and target selection prefers monsters with low HP and a high threat.
Evidence: when retreat triggers in T0″ there are on average 7.5 live monsters within 6 tiles; the median roster on leaving L2 is 134 monsters.

## 1. Interface (default off = unchanged bit for bit)

- **R18-D `aggro_cap="hold-v1"`** (`python/diablogym/aggro_cap.py`, `env.controller_action_context`): on main-line L2+, when a9 is executable
  (an engageable target exists) and **≥ 5 visible live monsters are within 6 tiles**, a10 is masked (explore/whole-map hunt = the only pulling action). The `mask[9]` guard guarantees that an attacking action always remains,
  so it cannot deadlock into exhausted. Counter `_aggro_cap_fired` (number of mask evaluations).
- **R18-E `engagement_priority="threat-v1"`** (`python/diablogym/engagement.py`, `env._engage_candidate_for_action9`): on main-line L2+,
  a9's target changes from "the first engageable wire row" to a ranking over the same frozen candidate set: ranged AI first → threat weight (by the runtime `ai`, not the type; unique monsters override ai)
  → maximum damage → **lowest current HP** → distance → id. The observation, wire order and reward keys are unchanged; the rule is a pure function of the snapshot, and the reward's approach term agrees with the macro.
  Counters `_engagement_decisions` / `_engagement_reordered` (number of choices different from the canonical one).
- **v1 scope = main-line L2+**: the L1 prefix is bit-identical to the retreat arm, so the paired comparison isolates only the L2 effect (verified in the smoke run: seed 2133001 first descent 5181 and L2 arrival 8021, the same as the retreat arm).
  In a smoke run the rules were once applied to all levels: the L1 farm dynamics changed a lot (the cap fired hundreds of times on L1, and one game ended early with a service termination), so the scope was narrowed.
- The thresholds (5 monsters / 6 tiles) and the retreat rule are not changed again in this round; a change means a new pre-registration.

## 2. Probe

Driver `r10-staging/run_r18de_probe.py` (not published); pool 2_133 with same-seed pairs; worker 7e31dc54; clock completion-l2-r18c (9000-tick observation window);
manager = coach-v03; economy sustain-loot-v1; retreat retreat-v1 fully on. Four arms in parallel:

| Arm | Configuration |
|---|---|
| ctl `r18de-retreat` | retreat (= the R18-C retreat-arm configuration, re-run on the R18-D/E bytes) |
| A `r18de-cap` | retreat + hold-v1 |
| B `r18de-threat` | retreat + threat-v1 |
| AB `r18de-cap-threat` | retreat + both |

### Criteria

- **Regression (hard)**: the ctl rows (minus the new None keys) are bit-identical to the R18-C retreat-arm rows; the descents prefix up to the first L2 arrival is identical game by game in all four arms. Otherwise VOID.
- **Main hypotheses (reported, not judged; directions fixed in advance)**: H1 the L2 hazard per 1000 ticks of AB and A is lower than ctl; H2 AB's paired survival saved−lost > 0 with UCB95 < 0;
  H3 the mean number of live monsters nearby when retreat triggers (monsters_near0) is lower in A/AB than in ctl (the aggro cap really limits pulling);
  H4 L2 kills per 1000 ticks (fighting efficiency) in B/AB are not lower than ctl; H5 the maximum depth distribution (number of L3+ arrivals).
- Mechanism report: cap trigger count, share of re-selected targets, retreat count/success/deaths on the way, L2+ occupancy after going upstairs, games that descend again, level of the survivors when the clock runs out.

This probe is not a training launch gate. It decides which interface set the R18-B training arm uses (retreat / retreat + cap / retreat + target selection / everything on).

## 3. Certification chain (before launch)

Full suite with 0 failures (including `tests/test_aggro_engagement.py` and `tests/test_resource_retreat_training.py`); probe regression 33023de1… equal;
two-way re-bake 4/4 + 4/4 (mirror root refreshed); the sha256 of this file recorded in the ledger.

## 4. Seeds and files

2_133 consumes 4 more groups (14 in total); the virgin pools are untouched. New files: this pre-registration, `run_r18de_probe.py` (not published), `aggro_cap.py`, `engagement.py`,
the test files, `r18de-probe/` (not published). Changed protocol files: `env.py` (two kwargs, three lazy row fields, the mask rule, the selector wrapper), `options_env.py` (worker windows also carry retreat telemetry),
the probe (two keys, telemetry fields, version `r17-deployment-v3-r18de`). Pre-change copies were kept in a local work directory (not published).
