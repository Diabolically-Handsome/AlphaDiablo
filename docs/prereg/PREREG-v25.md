# v25-ALT "manager election": pre-registration (v2, frozen after all 27 items confirmed by the critics were implemented)

**The question**: the first exam of the complete two-brain architecture: retrain the manager on top of the frozen
v24-leg7 leash worker. Claim: **a change of manager brings a gain** (v22-H's values were learned against the script
worker, and the crew has changed).

Hard rules: (1) zero changes on the environment side (python/diablogym); (2) single prescription = retrain the
manager; fresh/warm are **pre-registered control arms of the same prescription** (precedent: the v22 devil arm F, with
the arm-selection rule written in advance; the gold standard is still a single arm, once); (3) the gold seeds
9000-9031 at most once in the whole case, and not spent if the probe does not reach the launch line; (4) the control
values 97.2/92.0 are cited from archives (deterministic argmax replays on the same seeds, no re-burning: this is the
mechanism that makes versions comparable under the "at most once" discipline).

## D1 Cast and instruments (critic blocker fix: every change recorded)

Worker = v24-leg7, numpy forward from npz (policy.npz already exported, parity 0/1000).

| File | Change |
|---|---|
| train/train_ppo.py | (1) --worker-npz (default None); (2) the options branch of make_env: when worker_npz is set, construct NumpyManager **inside the function body** (the spawned subprocess) and OptionsEnv(workers={0: net.choose}), wrapped in a thin _SeedDiscipline (reset seed=None -> sample_train_seed rejects the 7000/9000 ranges, per-episode seeds go into info); (3) an --options guard assertion block (mppo/gamma 1.0/3000 micro-steps/n_steps 64/--seed mandatory + neighbourhood collision rejection), ending the era of "--options relies on human memory" |
| train/eval_assembled.py | --manager-npz (default v22-h, so the old-archive regression is provable) + an .npz branch in load_worker |
| train/export_manager_npz.py | parameterized zip/npz paths (argv like export_worker_npz) |
| train/export_manager_sd.py (new) | v22-H zip -> full policy state_dict (including the value head) .pt, for the M-warm injection |
| train/run_v25_election.py (new) | the sole executor of the clauses (interlock: an exam only after the step count is reached; ledger trace) |

**G-A0 (before launch, three in one)**: (a) regression of the new evaluation path: `--worker npz(leg7) --manager
default` on the full 32, per-seed ret/died/depth/mode_seq == the v24-G3-leg7.json archive (the G0'' 32/32 precedent;
it also proves the instrument is not broken + the npz worker is bit-faithful); on a mismatch -> fallback clause: unify
the representation on both sides and re-anchor the reference row. (b) Lightweight-subprocess clause (correction: the
torch module inevitably enters the subprocess with train_ppo's top-level import, same as the v23 precedent; the
enforceable requirement = do not pickle network objects, do not construct SB3 models in the subprocess, and the worker
runs the numpy forward beat by beat, guaranteed by constructing NumpyManager inside make_env). (c) After each arm is
trained: export_manager_npz export + 1000-obs parity (the G-KL-C criterion); no exam without a pass.

## D2 Election recipe (the v22-H status.json archive verbatim; critic major fix)

Shared: `--options --algo mppo --gamma 1.0 --max-steps 3000 --n-steps 64 --num-envs 4
--total-steps 40192 --worker-npz train/models/v24-worker-leg7/policy.npz`
(SB3 quantization ~40.2k decisions, registered as is; the two arms run in series; decision rate anchored on the
measured v22-H 29/s ~ 25 minutes/arm).

- **M-fresh**: from scratch, lr 3e-4, ent 0.02, --seed 22 (the original v22-H seed, the closest replica).
- **M-warm**: the fresh model + `--bc-init with the full v22-H policy sd (including the value head; a warm value head
  does not step on the v22 mine, freeze=0)`, lr 1e-4, ent 0.005, --seed 23. --resume-from is disabled (the seal-5
  assertion would crash and the semantics would carry old state; ruled out by the critics).
- The R25.2 verdict definition = a whole-recipe comparison of "warm-start recipe vs from-scratch recipe"; "the old
  values are a burden/an asset" may only be opened as a case, not attributed (the lr/ent confound is recorded).

## D3 Verdict and launch (critic blocker/major fixes: both arms get the full 32, paired criterion)

1. **Early abandonment gate**: each arm screened on 16 seeds (7000-7015); only if **both are <75** is training judged
   failed and the full 32 skipped; otherwise both arms take the full 32 (7000-7031): the R25.1/R25.2 definitions close
   automatically, and the 16-seed mirage (v23: 105.7->76.0) and arm-selection bias are removed together.
2. **Winner** = the higher full-32 mean; a +/-0.05 tie -> fewer deaths -> still tied, take M-fresh.
3. **Gold launch line (paired criterion, per seed against v24-G3-leg7.json)**: paired mean difference >= **+4** (~1
   SE of pool extrapolation; anchors: v24 probe->gold +5.2, v23 -29.7) and paired wins >=18/32 and deaths <=6 and the
   sentinels (next item). Middle tier [+2,+4) -> "probe-level improvement, does not spend the gold run; rematch left for
   the workstation line". Winner mean >=85 without reaching the line -> "incumbent stays, no gain this round
   (power-limited)"; <85 -> **"retraining did not reproduce the reference level, the succession proposition was not
   examined"** (this version adds no retraining).
4. **Sentinels (R4 unpacked; critic major: tau-bar is a function of the manager's strategy)**: level-change rate
   <=2.04% (the worker arbitrage instrument, still valid with the worker frozen), cap <5%, override <3% (>=8% voids the
   data) are gates; **the farm tau-bar band is downgraded to recorded-only + a mandatory note in the verdict**.
   Conditional clause: when the winner's full 32 has DIVE >1/episode, a tau-bar/override crossing takes the dual
   attribution (mix drift vs real degradation, the v24 P-crutch-no precedent), and a new DIVE-definition void line is
   added: **DIVE >1/episode and deaths >6 -> run void** (aligned with the spirit of v23 R4).
5. **Gold standard (if launched)**: single arm, once; the losing arm, a non-launched arm and every other ckpt of this
   version never touch 9000-9031.

## P lines (a complete partition of the space, exactly one outcome; the death dimension completed)

Decided in order, gold = G (if launched):
1. deaths >6 -> **P25-revert** (a death-count breakthrough goes into the verdict separately, not mixed into the mean
   narrative).
2. G >= 101.2 (=97.2+1SE) and deaths <=4 -> **P25-gain (significant)**: v25-golden takes the throne, verdict "the
   succession gain holds; the complete two-brain architecture wins its first case".
3. G in (97.2, 101.2) and deaths <=4 -> **P25-point-estimate gain**: "a point-estimate gain, not judged significant
   within the power", **the v24 throne does not move**, recorded and left for a workstation rematch.
4. G >97.2 and deaths in {5,6} -> **P25-tie (safety qualified)**: the verdict adds "safety below v24".
5. G in [93.9, 97.2] -> **P25-tie**: "incumbent stays, no significant gain".
6. G <93.9 -> **P25-revert**: "the probe was misleading, the succession failed"; the spent gold run is recorded without
   embellishment.

Throne clause: only P25-gain (significant) changes the active assembled agent; in every other tier (including no
launch) v24-golden stays and its leaderboard row does not move, and v25-golden only adds a control row. Board row
format: notes must carry the manager's provenance (mgr=v25-fresh/warm) + a note on its decision divergence rate from
the v22-H manager (guarding against a hollow victory, v24 precedent).

## R lines

| # | Prediction | Number |
|---|---|---|
| R25.1 | M-fresh full 32 | in [85,100], point 93 |
| R25.2 | paired (warm-fresh, full 32, same seeds) | in [0,+4], point +2 |
| R25.3 | the winner's DIVE share (measured on the full 32 and the gold pool, both recorded; a disagreement is itself recorded) | <0.5/episode as predicted; [0.5,1] recorded at the margin, no narrative; >1 opens a case (a narrative requires a mechanism autopsy, otherwise only "share anomaly, mechanism unknown" may be written) |
| R25.4 | gold (if launched) | in [93,103], point 97.5; **the band includes the [93,93.9) revert zone: the prediction band covers the possibility of failure after launch, booked in advance, with no conflict with the P lines** |

Verdict discipline: every P25 verdict carries the DIVE share + a mode_seq summary + the farm tau-bar number.

## Residual uncertainty

1. Manager training has no intermediate ckpt (save_freq > 40k); a half-finished finally output is intercepted by the
   driver's step-count interlock; 40k is a single draw with no retry clause, and a training crash falls into the
   "proposition not examined" tier, judged as is.
2. The bias of the old value head carried in by M-warm against the new worker's distribution; direction unknown.
3. The third opening of the gold pool 9000-9031; cross-version throne comparisons carry a fixed-pool selection bias,
   and the verdict carries this qualification.
4. Seed rejection covers the training episodes; the evaluation/gold-pool definitions are isomorphic to earlier versions.

*v2 frozen: 2026-07-10, all items from the critics (3 lenses, 27 items) implemented. The commit timestamp is the
notarization.*
