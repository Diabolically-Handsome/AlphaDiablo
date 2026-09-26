# R18-A Gate T0″ verdict (2026-09-06)

Pre-registration `r18-A-PREREG-FROZEN-20260906.md` (sha256 f8ac8ba4aedabfa4e9a8238218437b854f4d0fbe2c92cb9bca59ac8daa6a8d87, digest of the pre-translation text; frozen before launch);
driver `r10-staging/run_t0_double_prime.py` and outputs `r10-staging/t0-double-prime/` (not published); bridge 20a2ad7e…; pool 2_133 with same-seed pairs;
clock completion-l2-v1; worker 7e31dc54; manager = the resource coach coach-v03. Pre-launch certification chain: full suite 1933/1/0 + 69 retreat unit tests;
probe regression 33023de1… equal; two-way re-bake 4/4 + 4/4 (mirror root refreshed to the R18-A bytes).

## 1. Results (48 seeds)

| Group | Survived | Died | Reached L2 | L2 deaths | L2 total ticks | L2 hazard per 1000 ticks | L1 deaths | Final gold | clvl |
|---|---|---|---|---|---|---|---|---|---|
| (e) loot selling + **retreat** | **31** | 17 | 37 | **10** | 19 894 | **0.503** | 7 | 216 | 2.96 |
| (d′) loot selling | 13 | 35 | 37 | 27 | 35 724 | 0.756 | 7 | 125 | 3.25 |

Paired (e) vs (d′): **saved 19 / lost 1, net +18**; death-rate difference −0.375, one-sided UCB95 −0.249.
The (d′) re-run rows are bit-identical to the T0′ t0p-full-loot rows (minus the two new None keys): **"off = unchanged bit for bit" is confirmed at deployment level**.

Criteria: 1 ✓ (net +18 ≥ +6, UCB < 0); 2 ✓ (L2 deaths 10 ≤ 0.7 × 27 = 18.9); 4 ✓ (regression equal); 3 (reported, not judged) is below.
**Verdict: T0″_PASS.**

## 2. Mechanism (with one telemetry bookkeeping correction)

36 retreat attempts (35 games), all triggered by `low_hp` (the pressure trigger was off); mean HP at the trigger 0.46·max (minimum 0.28).

| Outcome | Count | Notes |
|---|---|---|
| Reached the upper level | 25 (69%) | median 75 ticks; 20 drinks on the way; after arriving **none of the 24 games died again**, all survived until the clock ran out |
| Died on the way | **10 (28%)** | the telemetry outcome is empty (death closes the window first, so the script had no chance to record it); checked game by game: 10 games died on L2 during the retreat, lasting 18–244 ticks and drinking 1–6 potions on the way |
| Truncated by the clock | 1 | the clock ran out during a retreat |

`died_during_retreat = 0` in the driver SUMMARY is a bookkeeping gap, not a fact; the true share dying on the way is 28%, slightly above the pre-registered reporting line of 25%.
The fix goes into R18-B (fill in the outcome of open attempts at game end); this verdict follows the table above.

**Where the triggers come from:** 26 of the 36 triggers came in DIVE windows: on L2 the coach chooses DIVE (rushing for the level-3 stairs) once coach-v03 judges the character "ready",
and the worker is beaten to half HP on the way to the stairs. Retreat rescues exactly this most dangerous route.

**After retreating:** the median interval from reaching the upper level to the clock running out is 1465 ticks (shortest 17, longest 1693); only 1 game went down to L2 again; final gold 216 vs 125
(from living that much longer on the upper level/in town). So this probe proves the one thing the pre-registration stated: **retreat turned deaths on L2 into survival**;
it did not, and cannot, prove that "retreating and descending again makes progress": the 1800-tick observation window after the first descent cannot hold a full retreat-recover-descend cycle.

## 3. Answer to the design hypothesis

The design diagnosis ("the problem is not the worker; the manager did not tell the worker what to do") is confirmed by the data:
the same worker, the same seeds, the same equipment and funds; only a "retreat" word was added to the manager (not even a new word: it borrows RESUPPLY),
and survival went from 13 to 31, L2 deaths from 27 to 10. The worker's hands were already sufficient; what was missing was the manager's word.

## 4. Next steps (candidates)

1. **R18-B (required before training)**: wire retreat into the training windows and receipts (WorkerWindowEnv currently rejects retreat-v1 explicitly),
   put `resource_retreat.py` into the protocol bundle, add game-end bookkeeping, and re-cast the six-volume exam room (in progress on the final R18-A bytes).
2. **Clock recipe R18-C**: widen the observation window after the first descent beyond 1800 ticks (a new recipe and a new pre-registration) to measure whether "retreat-recover-descend again" makes progress.
3. **The manager's second word**: 26/36 triggers come from DIVE windows on L2; the coach rule "rush for level 3 once ready" on L2 deserves its own case;
   the trigger threshold (50%) and the pressure trigger (off) are not changed in this round; a change means a new pre-registration.
4. Training arms: T0″ passed, but retreat is a scripted hand and the worker has not yet learned in a world with retreat; R18-B comes first, then a launch order.

## 5. Seeds and files

2_133 consumes 2 more groups (8 in total); the virgin pools are untouched. New files: this verdict, the pre-registration, `run_t0_double_prime.py` and
`t0-double-prime/` (not published), `python/diablogym/resource_retreat.py`, `tests/test_resource_retreat.py`; the analysis script `t0pp_analysis.py` (local, not published).
No frozen artifact was overwritten; `r17-anchor-xdevil-b.json` (pre-change bytes) was renamed to `.superseded-pre-retreat.json`.
