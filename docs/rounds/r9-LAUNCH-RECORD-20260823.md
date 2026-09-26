# R9 launch record (2026-08-23)

- Date: 2026-08-23.
- Launch: a bare call of `train/run_r9_reeducation.py` (no arguments). The campaign's direction target was a
  depth of about level 4; the verdict still follows the frozen criteria of
  [PREREG-R9](../prereg/PREREG-R9-manager-reeducation.md).
- Environment: 864 tests passed; torch 2.12.1+cpu.

## Addendum 1 (same day, before the relaunch)

- First attempt: halted by the operator on 2026-07-31 for throughput. The certified worker's torch forward pass
  gave about 12 games/min, so one arm needed 7-18 h against a 4 h timeout guard. The three anchor stages and
  G0-6 REF_BITEQ were recorded intact; the mfresh arm was stopped after about 2 h 43 min and 1,927 games, with no
  candidate evaluated.
- Throughput fix (least invasive option, schedule only): training timeout 14,400 s -> 72,000 s. This is a hang
  guard in the driver, not a scientific parameter; steps, environment count, seeds and gate lines are unchanged.
  A GPU speed-up for torch was left as an R10 topic.
- Restart procedure: the three anchor archives were rotated to `.void` (the driver re-burns them
  deterministically; in the REF_BITEQ world the values are expected to be bit-identical); `r9-mfresh` and
  `r9-reeducation` were archived as `._halted-20260731-throughput`.

## Addendum 2 (2026-08-24, before the third launch)

- Second attempt: the mfresh arm ran 21.2 h and hit the 20 h fuse (rc=124) at 112,692/160,000 steps. Under the
  v25 inheritance clause the driver recorded OPERATIONAL_FAILURE (hypothesis not tested, no extra retraining).
- Throughput (measured): about 88.6 manager steps/min, so 160k steps need about 30 h per arm. The bottleneck is
  not the torch forward pass (16.9 microseconds per call; a benchmark ruled out the threading hypothesis) but
  about 1.9 ms per micro-tick of Python/SB3/wrapper overhead. A numpy fast path for the worker became an R10
  infrastructure topic (the project already had a bit-level reconciliation of a numpy forward pass for the
  manager).
- Training-side reading (not a verdict): over six thousand games mfresh plateaued at reward 102.6, about equal to
  the anchor's 101.976, with no sign of unlocking depth. The open question of R9 rested on the mcurr course arm,
  which had not run yet.
- Timeout amendment: fuse 72,000 s -> 216,000 s (60 h). It only guards against a true deadlock; the measured need
  is about 30 h per arm. The run was approved without a schedule limit.
- The third bare-call launch followed this addendum.

## Addendum 3 (2026-08-27, verdict day)

- The third launch ran without tripping a fuse: mfresh 160k steps in 29.7 h, mcurr 160k steps in 33.4 h; both
  arms rc=0 and g_parity passed.
- Exams and verdict: all four paired_analysis depth gates failed (wage non-inferiority and death
  non-inferiority both passed). VERDICT_PATH: both arms failed the replication gate, no winner, and the
  re-education hypothesis stayed unanswered (outside statistical power). golden_authorized=false; the final pool
  2_125 was not used.
- Post-verdict forensics found behavioural collapse in both arms: on the real observation domain the two arms'
  argmax agree 100% and always emit action 0, so the four exams are pairwise bit-identical. See
  [r9-FORENSICS-behavioral-collapse-20260827.md](r9-FORENSICS-behavioral-collapse-20260827.md). The verdict
  numbers stand, and the hypothesis in effect received a strong negative answer: the manager-side means are
  exhausted, and the key to depth lies in the environment's wage scheme (course 2, then awaiting review).
