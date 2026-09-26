# PREREG-R2 anchoring: start of protocol-v3, anchor re-measurement on the gold seeds (final version)

Status: **final version** (passed the critic panel: three seats; all 4 blocker / 13 major / 15 minor
findings addressed). Freeze discipline follows R1/v30: **the commit is the notarisation**, the text
contains no self-referential sha, and the freeze sha goes only into the ledger's FREEZE_SHA event.

## Rationale and authorisation

R1 is closed (the v3 teacher is in place; this is a **prerequisite of the sequence**, not a technical
dependency of this case: of the five runs only A5 loads a BC product, and the substantive link between R1
and this case is the clause "assembled evaluation deferred to R2", see A5/R-G1). The v2 absolute
baselines were deliberately voided by the external audit. This case is the third step of the
[roadmap](../design/ROADMAP-course-plan.md) sequence, "anchor re-measurement"; **once its verdict appendix
is recorded**, course campaigns are unlocked.

**Gold-seed authorisation (panel correction)**: gold seeds 9000-9031 are used once per case, launched
manually, and need a pre-registration plus notice (carried over from R1). The roadmap era-switch note and
the R1 verdict appendix had already announced that R2 would re-measure on the gold seeds and would need a
new pre-registration and notice; the go-ahead was given on 2026-07-12. This case trains nothing and is a
sequence prerequisite for restarting training. The panel proposed a stricter option (have the freeze
commit reviewed once more before launch); following the v29/v30 precedent (campaign-level notice + a
frozen pre-registration authorise the launch, with no second per-case gold-seed sign-off), the drafter
did not adopt it; the choice and its reasons are recorded here. Launch discipline: after the freeze commit
is recorded, the operator starts the driver manually once, which is the "manual launch"; the five runs
are pre-registered and run in series within that one launch.

## One campaign, one prescription

Prescription = "four-way anchoring measurement + one gate evaluation, each spending the gold seeds once;
the numbers are the anchors". **Anchoring is a measurement, not a contest**: no win/loss verdict, no gold
eligibility spent, no change to the throne / incumbent titles (title inheritance follows the
anchor-compliance rule of PREREG-v30 D3 verbatim). No training, no environment-side change, and the probe
pool 7000-7031 is not touched.

## D1 world identity (prerequisite, machine-executable)

- **W1 freeze notarisation**: the porcelain output must be empty after removing the single exempt line
  `?? train/leaderboard-assembled-v3.md` (an unavoidable product of gold evaluation with --board; the
  protocol-v3 board is created by this case; the exempt line's text goes into the ledger, and the board
  file is included in the verdict-appendix commit); assert that `git log -1 --format=%H` of this document
  and of the driver `train/run_reanchor_r2.py` (not published) both == HEAD. **FREEZE_SHA is recorded only
  after every W-line assertion has passed, in the same batch as PREFLIGHT_OK; a failed preflight leaves no
  FREEZE_SHA.** The ledger allows several FREEZE_SHA entries (chained re-freeze, P6): a resumed run asserts
  HEAD == the last one; a new FREEZE_SHA must carry {prev_sha, reason}, where the reason comes from the
  file `train/runs/reanchor-r2/REFREEZE_REASON` written by the operator (non-empty).
- **W2** `PROTOCOL_VERSION==3 ∧ SCHEMA_VERSION==2`.
- **W3** `engine_binary_path` unique; `DEVILUTIONX_REF` unset; bootstrap.sh contains verbatim
  `ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"`.
- **W4** diabdat.mpq present (sha into the ledger).
- **W5 implementation-bundle sampling**: `train_ppo._implementation_bundle_sha256()` is sampled before and
  after every run. **Pre-launch sample ≠ the case's first value → no launch; the whole case halts and
  reports** (drift is present, re-spending is pointless, and no P2 quota is used; CASE_HALT_IMPL_DRIFT,
  exit code 5); post-run sample ≠ the case's first value → that run is OPERATIONAL (P2; a -b relaunch must
  still pass the pre-launch sample, i.e. only once the drift has been reverted).
- **W6** `runtime_versions_identity()` recorded in full in the ledger.
- **W7 artefacts pinned** (full sha values are the driver's frozen constants; a preflight mismatch is
  **P4**, no launch):
  - worker v28-leg1 npz = `976b6c05…f27f8a`; worker v24-leg7 npz = `a31fa7c6…fbc2a`; manager M29
    (**archived under the asset discipline** as `train/models/v29-manager-mfresh/policy.npz`,
    byte-identical to the original in runs, included in the freeze commit) = `89441388…7b66d6`; manager
    v22-H == `DEFAULT_MANAGER_SHA256`; A5's v3 BC teacher `train/runs/bc-worker/policy_sd.pt` =
    `f052067a…ff98e6` (R1 receipt).
  - **Prior archives pinned** (PRIORS; forensic references are sealed before they are cited, v30
    precedent): the full-text sha values of v29-mfresh-full32.json / v28-G3-leg1.json / v24-golden.json are
    asserted against driver constants, with a PRIOR_REFERENCE event in the ledger.
- **W8 idle machine + driver mutual exclusion**: before every run, the matches of `pgrep -f
  'train/(bc_|train_ppo|eval_assembled|run_v[0-9]+|run_reanchor)'` minus the driver's own pid must be
  empty; the driver holds the exclusive lock `train/runs/reanchor-r2/.driver.lock` throughout
  (exclusive_lock); a lock conflict → no launch, no quota used, report (P3 type).
- **W9 first-burn prerequisite**: when the ledger has no FIRING_START for a run, assert that neither
  `<tag>.json` nor `<tag>-b.json` exists under eval-assembled; if one exists → halt and report (so that a
  manual archive from before the case is never silently accepted or takes the first-run tag).
- **W10 resume reconciliation**: when the ledger already has PREFLIGHT_OK, assert that the current
  implementation / engine_binary / diabdat sha values equal those of the first PREFLIGHT_OK verbatim
  (build/ is outside porcelain, so it must be guarded by ledger reconciliation); a mismatch → halt and
  report.
- **W11 identity restated per run**: before every launch, require that run's worker/manager artefact sha
  == the W7 frozen constants (the v30 per-leg require_sha256 precedent), and restate the W1 git assertion
  (covering in-case drift of protocol files outside the W5 bundle, such as eval_assembled.py).

## D2 recipe for five runs (quota per run = 2 launches, counted in the ledger; protocol = eval_assembled verbatim)

| Run | tag | worker | manager (explicit --manager-npz every time, following the v30 panel blocker discipline; the default fallback is forbidden) | v2 prior (pool labelled as is) |
|---|---|---|---|---|
| A1 science anchor | `r2-science` | v28-worker-leg1 npz | v29-manager-mfresh npz | 140.3 (**7000-7031 probe pool**, sha16 08633101; no prior on the gold seeds) |
| A2 launch anchor / incumbent | `r2-launch` | v28-worker-leg1 npz | v22-h-manager npz | 112.4 (**7000-7031 probe pool**, sha16 6fc6a44c; no prior on the gold seeds) |
| A3 throne | `r2-throne` | v24-worker-leg7 npz | v22-h-manager npz | **97.2 (9000-9031 gold pool**, sha16 d9387dcb) |
| A4 script reference | `r2-script` | `script` | v22-h-manager npz | **93.9 (9000-9031 gold pool**, the recorded ppo-hier-v22-h row of leaderboard-hier, median 103.45, died 2/32; a v22 evaluator reading with no archive sha to cite, evidence level = recorded side evidence; the 7000-pool comparison H7 78.5 is a reference only, not a prior) |
| A5 teacher assembled evaluation (gate run, **not an anchor**) | `r2-bcworker` | `bc` (v3 teacher policy_sd.pt) | v22-h-manager npz | no prior (new in v3) |

Command form: `.venv/bin/python train/eval_assembled.py --worker <spec> --manager-npz <npz> --seeds
9000-9031 --board --tag <tag>`. In series, with a wall-clock limit of 1 h per run (on timeout killpg+wait;
the ProcessLookupError kill race does not change the P2 classification); logs
`train/runs/reanchor-r2/logs/<tag>-{ts}.log`. The GOLDEN_AUTHORIZED ledger event (the five command lines,
the notice source and freeze_sha) is recorded after preflight passes and before the first run (v28 D2-9 /
v29 precedent).

## D3 R lines

- **R-V (validity, per run, machine-executable)**: the input freeze at launch time is carried by the
  evaluator's internal identity freeze, reserve_output and the archive's self-verification; **at
  re-verification time** (for an idempotent resume, before launch; for a new run, right after it ends)
  the driver takes its own `freeze_eval_identity(ROOT, worker_spec, manager)` snapshot as a third-party
  check, runs `read_eval_archive(archive, **expected)` with `expected_eval_identity(snapshot, tag,
  seeds=9000-9031)`, and `verify_eval_identity(snapshot, ROOT)` must pass, **and it asserts that the
  snapshot's worker/manager sha == the W7 frozen constants** (the worker sha of script/bc is derived from
  the protocol bundle / gate report, so self-consistency with the snapshot is enough) → the run is valid,
  and agg.ret_mean is the anchor value; any step fails → OPERATIONAL (P2). **A scientific reading is final
  once validly filed, and re-spending is forbidden**; if the ledger already has FIRING_VALID but the
  current snapshot fails R-V → runtime drift during the case, the whole case halts
  (CASE_HALT_RUNTIME_DRIFT, exit code 6), and switching to -b is forbidden.
- **R-O (weak-order prediction, registered, not decided)**: predicted `r2-science ≥ r2-launch ≥
  r2-script`; the relative order of the throne and the launch anchor is not predicted. Whether the order
  holds is recorded by the driver in an R_O_OBSERVATION ledger event and copied into the verdict appendix.
  **A broken order is no reason for any re-measurement, extra spending or reconsideration; the gold seeds
  are not re-exposed because of the R-O result.**
- **R-Δ (cross-world shift, recorded, not decided)**: **only A3 (97.2) and A4 (93.9) are same-pool
  cross-world differences on the gold pool**; the priors of A1/A2 are from the 7000 probe pool, so their
  differences are a composite "cross-world × cross-pool" quantity, recorded only, and reading them as a
  shift is forbidden; the attribution candidates (back-jump confinement / level-change settlement fix /
  engine patches / change of seed pool) cannot be separated. The comparison table in the verdict appendix
  must have two separate columns, "same pool, cross-world" and "cross-pool, cross-world". The driver
  records an R_DELTA event.
- **R-G1 (the freeze obligation from the R1 verdict appendix, "assembled evaluation deferred to R2")**:
  `r2-bcworker.agg.ret_mean ≥ 0.85 × the r2-script reading` → registered as "the learned worker's
  assembled capability meets the bar (gold-pool definition)"; < 0.85 → recorded as is, no halt, no anchor
  granted, no title change. **This run's reading is never called an anchor.** This clause also resolves
  the internal tension between R1's "four-way announcement" and its "half gate deferred"; the decision is
  recorded here. The driver records an R_G1 event.
- **Granting the anchors (case verdict)**: all five runs valid → an ANCHOR_GRANT ledger event (at most one
  per case; after the case closes the driver exits idempotently): `v3-science-anchor := that run's valid
  archive` (tag r2-science or its P2 relaunch r2-science-b, as given by FIRING_VALID.tag), and likewise
  `v3-launch-anchor`; each anchor object must contain {firing, tag, archive_path (repo-relative),
  archive_sha256, seeds, worker{kind,path,sha256}, manager{path,sha256}, ret_mean, v2_prior, freeze_sha}
  for later campaign drivers to consume following the read_comparable_reference precedent (assert the sha
  before reading; **guessing the path from the tag is forbidden**, because leftover archives and real
  anchors may sit in the same directory). No title changes; **throne incumbent anchor value := the
  r2-throne reading** (every later gold P line compares the throne against this value); **script
  reference value := the r2-script reading** (the source for recasting the baselines of later
  eligibility / proportional lines); "recorded for reference" only means no anchor title is granted, and
  does not remove the duty to compare against it. **The completion mark that unlocks course campaigns and
  makes the A rematch right usable = the commit that records this case's verdict appendix** (verbatim per
  the R1 freeze clause; ANCHOR_GRANT is a machine prerequisite and a verdict input and does not unlock
  anything on its own; no course campaign may launch before the appendix is recorded).

## D4 P lines (exhaustive; exit-code table at the end)

- **P1 top-level exception / interruption** → DRIVER_EXCEPTION + NEEDS_ATTENTION. Idempotent acceptance on
  resume needs **two conditions**: the tag's archive passes R-V in full ∧ the ledger has a FIRING_START for
  the tag and no FIRING_INVALID/RESIDUE_SEALED record; an archive without a ledger record may not be
  accepted (W9).
- **P2 launch quota (ledger-based)**: the launch count of a run = the number of its FIRING_START entries
  in the ledger (including launches that never finished, cumulative across resumes); count at 2 with no
  valid archive → no further launch, CASE_HALT_OPERATIONAL. Whether leftover archives are on disk is not a
  basis for the quota. **A -b relaunch is allowed (closed enumeration, once per run)**: (a) the subprocess
  exits non-zero / crashes with no archive; (b) killed on timeout; (c) the archive exists but fails R-V,
  and that run's post-run W5 sample ≠ the case's first value (drift after the run; the drift must be
  reverted before -b); (d) reserve_output refuses to overwrite a leftover archive. **Must halt, no -b**:
  any assertion fails before launch (W5 pre-launch drift / W11 mismatch / git not clean / machine not
  idle), the ledger is unparseable, or R-V fails while the run's W5 samples before and after both equal
  the case's first value and the archive is structurally consistent (unknown cause; re-spending adds no
  information; report). Any archive that was produced but failed R-V, on a first run or a resume, is
  recorded as RESIDUE_SEALED{tag, sha, why} (deduplicated), and is neither deleted nor changed.
- **P3 machine not idle / driver lock conflict** → no launch, no quota used, report.
- **P4 preflight W line fails** → PREFLIGHT_FAIL event, no launch, report (exit code 3, not through the
  DRIVER_EXCEPTION channel).
- **P5 halt policy (option A, the panel's choice between two options)**: a run's quota exhausted → the
  whole case halts, later runs are not launched: a failure is interpreted first as a world-level anomaly,
  and under-sampling is preferred to careless spending; runs that are already valid are kept and carried
  over into the continuing case under the P6 chain clause (carried over if they still pass
  re-verification, and re-spending is forbidden: the finality of scientific readings takes precedence
  over the form of the freeze).
- **P6 chained re-freeze**: if the driver or the document needs fixing after a halt, continue with a
  correction + a new commit; the ledger allows several FREEZE_SHA entries (carrying prev_sha + the reason
  from the REFREEZE_REASON file), and a resumed run asserts HEAD == the last one.

**Exit codes**: 0 = case closed (ANCHOR_GRANT) / case already closed, idempotent exit; 2 = P2/P5 quota
exhausted, halt; 3 = P4 preflight failed; 4 = P3 not idle / lock conflict; 5 = W5 pre-launch drift, halt;
6 = R-V runtime drift during the case, halt; any other non-zero or abnormal death = P1.

## D5 ledger, board and verdict

Ledger `train/runs/reanchor-r2/gate_ledger.jsonl` (not published; JSONL, append-only; a resumed run
asserts every line parses). Recorded per run: FIRING_START{firing, tag, impl_before},
FIRING_EXIT{exit_code, log, impl_after}, FIRING_VALID (with the full agg table, archive_path/sha,
worker/manager identity, impl_after) or FIRING_INVALID{why = first line of the exception text},
RESIDUE_SEALED (deduplicated, with why). Case level: PREFLIGHT_OK (the first one is the reconciliation
base), PRIOR_REFERENCE, GOLDEN_AUTHORIZED, R_O_OBSERVATION, R_DELTA, R_G1, ANCHOR_GRANT (unique).

**Board handling**: leaderboard-assembled-v3.md keeps one row per successful archive (the row key
tag@sha16 binds the archive, and the contract refuses rewrites); rows of voided first runs and of leftover
archives are all kept for reference, neither deleted nor changed; the scientific criteria come only from
the ledger, and the number of board rows is no basis for a verdict. The anchor-value table in the verdict
appendix is shown in the fixed order A1-A5, **sorting by value is forbidden**; the appendix notes that
this case's board rows register archives of an anchoring measurement, not ranks, not challenges, and
change no incumbent title.

**Archives committed** (the v28 D2-10 stop-gap precedent): the verdict-appendix commit must include, with
`git add -f`, all five valid archives and the sealed leftover archives (train/runs/eval-assembled/r2-*.json)
plus the board file; the sha256 values cited by ANCHOR_GRANT must be re-verifiable verbatim within that
commit, and if re-verification fails the appendix may not be recorded.

## Follow-ups (not executed in this case)

Verdict appendix recorded → course campaigns unlocked, the A rematch right usable. Candidates for the next
training case (each with its own pre-registration): course 5, the dry-window course (training side), or
the A rematch right; the design memo for courses 3 and 4 (environment side) goes through its own decision.

---

## Verdict appendix (case closed 2026-07-12; the commit of this appendix is the notarisation = the moment course campaigns are unlocked)

Freeze commit `50e356ef7c9d43044bc6af1b60059be4da59538a`; ledger `train/runs/reanchor-r2/gate_ledger.jsonl`
(not published); **all five runs took 7 minutes** (≈70-100 seconds each, the basis for budget calibration
after R2). All five runs succeeded at the first launch (zero -b, zero OPERATIONAL, driver exit code 0,
no defects).

**Anchor-value table (fixed order A1-A5, not sorted by value; not ranks, not challenges, no title
changes)**

| Run | v3 gold-pool reading | v2 prior (pool) | Comparison column | Archive sha16 |
|---|---|---|---|---|
| A1 science anchor | **152.8** (median 153.1, died 6/32, depth median **2.0**, kills 61.1) | 140.3 (7000 pool) | cross-world × cross-pool, no shift reading | 8d6a05d3517a6481 |
| A2 launch anchor / incumbent | **133.9** (median 133.4, died 4/32, depth median 1.0) | 112.4 (7000 pool) | cross-world × cross-pool, no shift reading | 8ab6b51065105a67 |
| A3 throne | **97.7** (median 93.8, died 2/32, kills 41.2) | 97.2 (gold pool) | **same pool, cross-world: Δ=+0.5** | 2324648a416cbc0c |
| A4 script reference | **94.2** (median 103.05, died 2/32) | 93.9 (gold pool, recorded leaderboard side evidence) | **same pool, cross-world: Δ=+0.3** | 71c298e05b6bf19e |
| A5 teacher assembled evaluation (not an anchor) | 94.2 (died 2/32) | no prior | gate run | 0f5970acc26f200e |

- **Anchors granted**: `v3-science-anchor := r2-science.json` (sha `8d6a05d3…e273f5`);
  `v3-launch-anchor := r2-launch.json` (sha `8ab6b510…42dff1`). **Throne incumbent anchor value := 97.7;
  script reference value := 94.2.** No title changes. The ANCHOR_GRANT event carries every field needed to
  consume it (archive_path/sha / worker and manager identity / seeds / freeze_sha); later drivers may not
  guess paths from tags.
- **R-O: prediction holds** (152.8 ≥ 133.9 ≥ 94.2); the relative order of the throne and the launch
  anchor was not predicted; measured launch 133.9 > throne 97.7, recorded, not decided.
- **R-Δ (same pool, cross-world: the only two legitimate shift readings)**: throne +0.5, script +0.3. The
  seven patches + back-jump confinement disturb the gold-pool measurement of **shallow-living line-ups**
  almost not at all, the hardest support for "the v2 same-engine paired conclusions are internally
  valid"; the world changes bite in the deep economy (A1 depth median 1.0→2.0, deaths 3→6 type changes,
  observed only and not read as shifts because they cross pools).
- **R-G1: PASS** (94.2 ≥ the line 80.07 = 0.85×94.2). It is also a **perfect clone**: 32 episodes, 42,571
  deployment calls, worker_divergences=0, and agg is identical to A4 bit for bit: R1's G1 data-side
  top1=1.000 turns into behavioural equivalence under the gold-pool assembled definition. The obligation
  deferred from R1 is discharged; the verdict qualification is lifted: "the learned worker (in BC teacher
  form) can replace the script under the gold-pool assembled definition" is now supported. This run's
  reading is never called an anchor.
- **Archives committed**: the five archives + the board `train/leaderboard-assembled-v3.md` were included
  in this appendix's commit with `git add -f`; the sha values cited by ANCHOR_GRANT are on record and
  re-verifiable.
- **Sequence complete**: the three-step sequence of the external audit (BC regeneration → teacher recast →
  anchor re-measurement) is fully recorded with this commit: **course campaigns are unlocked, and the A
  rematch right is usable** (the seal of the era-switch note in the
  [roadmap](../design/ROADMAP-course-plan.md) is lifted from here on).
