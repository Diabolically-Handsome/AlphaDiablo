# R12 launch record (2026-08-30)

- Case: R12 in-place worker re-education with two coach arms (route A). Approved 2026-08-30 as a two-arm
  comparison so that a control arm exists.
- Pre-registration: [r12-PREREG-FROZEN-20260830.md](r12-PREREG-FROZEN-20260830.md).
- Arms: arm 1 `r12-arm1-m29` (regular coach); arm 2 `r12-arm2-devil` (a "devil" coach whose manager always
  chooses DIVE).
- Driver: `run_r12_campaign.py` (not published).
- Pool accounting: the paired baseline reuses the existing v2 exams on 2_114/2_115, so no new pool is consumed;
  2_116 is kept for the final review.
- Precondition: the full 864-test suite green on an idle machine (still running at launch).

## Addendum 1 (same day, amendment 2)

- Arm 1 (M29 regular coach) finished training and its four exams on schedule. M29 group: l3+ 23/32 (baseline
  14/17, depth reach nearly doubled); kills 43 -> 33 (a style shift; the kill item of gate C, 0.81 < 0.95, did not
  pass). Its exam under the devil coach was bit-identical to the old worker's: a shallow coach does not change
  deep behaviour, so the control arm did its job.
- Arm 2 (constant-DIVE devil coach) failed structurally. A manager that always dives produces zero FARM windows,
  but in-place worker training uses the farm window as its classroom; the guard refused after 8 games (rc=1;
  post-mortem recorded in the ledger). Lesson: deep-level training needs a coach that dives first and then
  teaches.
- Amendment 2: arm 2's coach was replaced by the scripted coach `--manager-heuristic level-margin-1` (the
  repository's canonical rule: dive when the level is exhausted or the character level is at least one above
  the dungeon level, otherwise farm). Changes: scripted-coach support in `WorkerWindowEnv`, a new contract field
  `manager_heuristic`, and an extended `allow-manager-change` whitelist. Arm 2 was renamed `r12-arm2-smart`;
  driver `run_r12_arm2.py` (not published).
