# v29 "manager re-education": pre-registration (final text, frozen after the panel's two lenses and 15 items were implemented; the verdict appendix is at the end)

**The question**: can the depth economy be learned? The winning-moves report
([FORENSICS-winning-moves](../forensics/FORENSICS-winning-moves.md)) already verified that the payout rate of the 8xN
descent bonus is ~0 across the whole lineage, because the frozen v22-H (trained against the script worker, 40k
decisions) almost never chooses DIVE; while the v28-leg1 worker's in-fight potion hoarding + deep farming inside the
window makes DIVE win on 4 of 5 new maps: **the ladder is in place; what is missing is a manager willing to go
downstairs.** **The single prescription: retrain the manager on the new crew (v28-leg1).** Two arms = a control within
the same prescription's budget (normal entropy vs high-entropy exploration; v25 proved the fine-tuning arm is poison,
-18.3, and it is never set again). Hard rules as in v25: zero changes on the environment side; the gold seeds at most
once in the whole case; control values cited from archives. Launch decision: 2026-07-11.

## D1 Cast and instruments (zero new code paths)

- Worker = **v28-worker-leg1** (frozen): the zip = the model itself behind the full-32 = 112.4 archive; policy.npz
  exported on launch day, **parity 0/1000**.
- **Anchor = v28-G3-leg1.json (112.4 per seed, sha16 = 6fc6a44c7862424a pinned, asserted at launch)**: the full 32 of
  the incumbent v22-H x the same worker, the natural reference row for a succession comparison.
- **G-A0 (launch gate)**: npz worker + the default v22-h manager on the full 32, per seed == the anchor (ret+/-0.01 /
  died / mode_seq); a mismatch STOPs for a manual re-anchor. It also proves npz fidelity + an unbroken instrument.
- Zero new tools (the full v25 set is in the repo: the --worker-npz subprocess numpy worker, the --options guard
  assertions, export_manager_npz parity, eval_assembled --manager-npz).

## D2 Election recipe and budget

Shared: `--options --algo mppo --gamma 1.0 --max-steps 3000 --n-steps 64
--num-envs 4 --total-steps 160000 --worker-npz <v28 npz>`
(**4x v22-H's own budget**; clock anchor: 27.5 decisions/s was measured with the leg7 worker, the leg1 worker has
+16% beats/window (19.4 vs 16.7, computed from the two G3 archives), which converts to ~23-24 decisions/s ~ **112
minutes/arm**, two arms in series ~3.8h; training timeout 4h/arm, slack >=2x; panel calibration).

- **M-fresh**: lr 3e-4, ent 0.02, --seed 22 (the closest replica of the original v22-H recipe).
- **M-explore**: lr 3e-4, ent 0.08, --seed 24 (the high-entropy arm: the depth economy needs trying out the expected
  value of DIVE; entropy is the single variable between the two arms).
- The R29.2 verdict definition = a whole-recipe comparison of "normal-entropy recipe vs high-entropy recipe";
  attribution to anything other than ent is forbidden.

**Clone difference table (run_v25_election.py -> run_v29_relection.py)**:
(1) cast/anchor fully replaced (previous section); anchor sha asserted at launch; (2) the M-warm arm and the warm-sd
export stage are physically removed; (3) FLOOR_REPRO 85 -> **103.4** (=0.92x112.4, following the v25 ratio
precedent); (4) tags v29-*; exam() refuses to overwrite + half-written .void rotation + evaluation logs archived (v28
clause); (5) pairing always **joins on the seed key + a seed-set assertion** (the v28 fix, dropping v25's row-position
pairing); (6) operational guard rails (the full v28 set + panel reinforcements): top-level exception catch-all
DRIVER_EXCEPTION + NEEDS_ATTENTION, training arm timeout 4h/evaluation 30 minutes (timeouts booked as crashes, **the
kill includes the process group**: keeps SubprocVecEnv grandchildren from being orphaned), preflight (anchor sha/worker
in place/target archives including v29-golden absent); **the step-target gate uses the num_timesteps of
model_final.zip as the single step-count source and asserts exactly =160000** (160000 is divisible by 256 with no
quantization slack, and the throttled status count always lags by a dozen or so steps: a panel blocker; v25's 40192
over-sampling was lucky); timestamped evaluation logs archived (a retake does not overwrite the first exam's autopsy);
the driver restart protocol follows v28 residual #0; (7) the GOLDEN_AUTHORIZED event carries the exact gold-evaluation
command (with --manager-npz) + both shas; (8) full-32 events gain depth instrumentation: seeds with depth>=2,
DIVE/episode, descent bonus paid/episode.

## D3 Verdict and launch

1. **Early abandonment gate**: both arms' 16-seed (7000-7015) means <75 -> "training failed, the succession
   proposition was not examined" (full 32 skipped).
2. Both arms take the full 32 (7000-7031); **eligibility is judged per arm** (deaths <=6 and not void and the sentinel
   gates, or let through by dual attribution); **winner = the highest full-32 mean among the eligible arms** (a +/-0.05
   tie -> fewer deaths -> still tied, take M-fresh); if the arm with the higher mean is blocked by eligibility, a
   substitution event is recorded and the eligible arm takes its place and goes down the full ladder as usual
   (including launch); **both arms ineligible -> the no-winner tier** ("both arms ineligible, proposition unanswered
   (outside statistical power)"; the depth instrumentation is recorded only, with no side verdict).
3. **Reproduction floor**: winner <103.9 (=0.9239x112.4, **strictly following the v25 ratio 85/92**, the panel's
   correction) -> "retraining did not reproduce the reference level, the succession proposition was not examined".
4. **Launch line (paired criterion, per seed against the 112.4 anchor)**: mean diff >=+4 and wins >=18/32 and deaths
   <=6 and the sentinels (descend <=2.04% / cap <5% / override <3% are gates, >=8% voids; tau-bar recorded only; with
   DIVE >1/episode only an override crossing takes the dual attribution, and DIVE >1/episode and deaths >6 -> run
   void, the v25 clause unchanged). **The dual-attribution decision comes before the opening**: a GOLDEN_AUTHORIZED let
   through by dual attribution must carry a "decide first, then spend" flag, and the gold-standard evaluation may be
   launched manually only after a dual_attr_ruling event is written back by hand.
5. **No-launch tiers (exhaustive; eligibility failures are already covered by the no-winner/substitution of D3-2, so
   the winner is always eligible)**: mean diff >=+4 and wins <18 -> **"mean gain but width not reached: a
   point-estimate gain, does not spend the gold run"** (the tier added after v28); mean diff in [+2,+4) -> a
   probe-level improvement, does not spend the gold run; <+2 -> "the incumbent stays, re-education brings no gain
   (power-limited)". The last two tiers must carry the paired win count, and with wins >=14/32 they add a "width-shift
   note (the tier does not change)": the form this case gives to the width-tier instruction of the v28 appendix.
6. **Gold standard (if launched)**: single arm, once, assembled agent = the winner's manager npz x the v28-leg1
   worker; P lines against the throne 97.2 (v28 definition, **decided in order**, the GOLDEN_AUTHORIZED event carries a
   P-line quick reference so the opener can check it on the spot): deaths >6 revert; gold >=101.2 and deaths <=4 ->
   **P29 takes the throne**; in (97.2,101.2) and deaths <=4 -> point-estimate gain, the throne does not move; >97.2 and
   deaths 5-6 -> tie (safety qualified); in [93.9,97.2] -> tie; <93.9 -> revert. Gold-pool opening history: actually
   opened three times so far, v22/v23/v24; if this case launches it is the 4th (corrected definition), and the
   fixed-pool bias goes with the verdict.
7. **Depth side verdict (the main metric of the question; a scientific conclusion that does not move the throne; three
   exhaustive tiers)**: the winner's full-32 seeds with depth>=2 >=12 and DIVE/episode in [0.5,3] and deaths <=6 ->
   "**depth economy learned**" (even without a launch); depth>=2 <=7 (baseline) -> "re-education did not unlock depth";
   **every other combination (including depth2 on the line with DIVE outside the band/deaths over the line) -> the
   out-of-band tier; the verdict carries the reason and it is recorded without narrative**. The side verdict must end
   with the note "the throne and Mark-I determinations follow D3-6 and the roadmap clause
   ([docs/design/ROADMAP-course-plan.md](../design/ROADMAP-course-plan.md))" (guarding against a no-launch run being
   narrated as the complete system being done; a panel clause). **Mark-I determination line (coupled to the throne) =
   depth side verdict "learned" and P29 takes the throne**, synchronized with the roadmap
   ([docs/design/ROADMAP-course-plan.md](../design/ROADMAP-course-plan.md)).

## R lines

| # | Prediction | Number |
|---|---|---|
| R29.1 | winner full 32 | in [95,125], point 112 (matching the incumbent is already respectable) |
| R29.2 | paired (explore-fresh, full 32, same seeds) | in [-10,+15], point +4 |
| R29.3 | **depth line**: the winner's seeds with depth>=2 | baseline 7; in [6,22], point 12 |
| R29.4 | the winner's DIVE/episode | baseline 0.31; in [0.2,3], point 1.0 |
| R29.5 | gold (if launched) | in [98,120], point 108 |
| R29.6 | descent bonus paid/episode (recorded; **reconstructed from the final depth, so it is a lower bound of what was actually paid**: trajectories that climb back/return to town are underestimated; the tool is frozen, registered as is) | baseline 1.75 (=8x7/32); point 4 |

Verdict discipline: carry the depth distribution, DIVE share, descent bonus paid, paired wins against the anchor
(width), a mode_seq summary, the farm tau-bar, a re-check of the seed-7023 idling case (recorded), and the inherited
entry-distribution confound note.

## Residual uncertainty

1. The two arms differ only in ent, so attribution is limited; the lr/seed confound is removed (same lr; seeds 22/24
   follow the replica convention).
2. The risk of the high-entropy arm pressing keys at random is covered by the R4 void line and the override gate.
3. A manager with 4x the budget has no overfitting channel to the probe pool (the training seed rejection ranges stay
   as before; the evaluation pool never entered training).
4. The anchor 112.4 is a reversed archive with 16 < full32 (the only v28 case), so the win line of 18/32 paired against
   it is stricter than in earlier versions; registered as is, the line is not lowered.
5. The worker is frozen and the environment unchanged; the manager's observation/mask/reward are unchanged (the descent
   bonus always belonged to the manager's ledger).
6. **The winner is a max-of-2 order statistic**, so the type-I error rates of R29.1 and the launch line are inflated
   accordingly (~x2); the verdict must carry both arms' raw full-32 numbers and the r29_2 paired difference (panel note
   discipline).
7. **The mechanism lens did not complete** (recorded as is): its review items (the manager training path/guard
   assertions/seed neighbourhoods/G-A0 npz equivalence/OOD risk) are covered by the pass notes of the clause and
   statistics lenses and the v25 precedent (G-A0 32/32, seed assertions, npz parity 0/1000); the plateau risk of an lr
   without a schedule at 4x budget is backstopped by the abandonment gate and the floor clause.

*Final text frozen: 2026-07-11. Critic panel (the clause and statistics lenses completed: 2 blocker/2 major/11 minor
all implemented; the interruption of the mechanism lens is recorded) + synchronized driver revision; the commit
timestamp is the notarization.*

## Appendix: verdict record (booked 2026-07-12)

G-A0 32/32; fresh 88.7 min (nt_zip=160000/status=159988, the first live use of the step-gate blocker fix) / explore
105 min. Full 32: fresh 140.3, 3 deaths (depth2=15, dive=0.69, bonus=4.25) / explore 149.0, 8 deaths (depth2=21,
dive=1.44, depth_median=2.0, bonus=5.75) -> killed by the void line, **the first live use of the substitution clause,
fresh is crowned**; r29_2=+8.7. **Launch: +27.86 / 17/32 wins (one short) -> the point-estimate gain tier, the gold
run not spent, the throne stays for the fifth time; depth side verdict: learned.** The R reconciliation and candidate
prescriptions are in the v29 chapter of DESIGN.md. Zero contact with the gold pool.
