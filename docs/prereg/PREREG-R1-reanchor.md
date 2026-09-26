# PREREG-R1 re-anchoring: start of protocol-v3, BC base regeneration (final version)

Status: **final version** (passed the critic panel: 4 critics; all 3 blocker / 10 major / 12 minor
findings addressed). Freeze discipline follows the PREREG-v30 precedent: **the commit is the
notarisation**. The freeze commit = the commit that includes the final text of this document and the
driver `train/run_reanchor_r1.py` (not published); no self-referential sha is written back into the text,
the freeze sha is written only by the driver into the ledger's FREEZE_SHA event, and the verdict appendix
cites it.

## Rationale

The external audit (dossier 86c7270) sealed protocol-v2. None of the three existing bc_report.json files
has a v3 identity receipt, and train_ppo._validate_bc_report refuses BC products without a receipt. The
reasons for regeneration **differ by artefact rather than being uniform** (panel correction: the audit's
conclusions must be restated faithfully):

- **worker**: the audit re-verified held-out top1=1.000 on the day (quality confirmed again), so
  regeneration is purely an identity requirement. **Provenance limitation of the prior**: the key set of
  bc-worker/bc_report.json on disk (mtime 2026-07-12) can be produced neither by the v23 generator nor by
  the current v3 generator; the generator was never committed, and it overwrote the original v23 report in
  place (no _previous copy). On review, its policy_sha256/demos_sha256 match the artefacts on disk from
  2026-07-10 verbatim, so it is taken to be a leftover of the external audit's re-verification, with
  evidence level = **side evidence of unclear provenance**: top1=1.000 serves only as a prior for the R-W
  prediction and may not serve as evidence for any gate.
- **manager**: the v2 report's same-pool ratio 0.6498 FAIL (2026-07-09, same-pool definition
  50.379/77.531, comparable) stands on its own; regeneration has two reasons, identity and a re-check of
  the hypothesis.
- **flat**: the definition behind the old ratio 0.5974 (2026-07-09) is judged biased by the v3 script note
  ("it treats differences in seed difficulty as policy loss"; the denominator is the teacher's mean on the
  demonstration pool), and it **cannot be converted directly** to this case's same-pool gate; regeneration
  has two reasons, identity and a correction of method.

The external audit's conclusions about the BC trio (worker re-verified at 1.000; manager/flat not
trustworthy) have not appeared in any archive on record; this paragraph is their transcription into the
record. The generation of the three existing reports is determined **by field set** (none has a
schema_version/protocol_version key = pre-v3); mtime is side evidence only.

## One campaign, one prescription

This case's prescription = "recast the BC base in the v3 world and accept its receipts". This case **does
not spend the gold seeds 9000-9031** (anchoring is R2), **changes not a single line on the environment
side**, and **trains no PPO**. The probe pool 7000-7031 is used only as the built-in replay baseline of the
manager/flat scripts (gate use, compliant). Panel check: no creep on any of gold seeds / environment side /
PPO.

## D1 world identity (prerequisite; all machine-executable, asserted item by item in the driver's preflight)

- **W1 freeze notarisation in three steps** (following PREREG-v30, "the commit is the notarisation"):
  1. finalise after the panel's findings are addressed; one commit includes the final text of this
  document + the driver + the [roadmap](../design/ROADMAP-course-plan.md) note (after which the text of
  D1-D5 cannot be changed); 2. the launch preflight asserts that `git status --porcelain` is empty and
  writes `git rev-parse HEAD` verbatim into the ledger's FREEZE_SHA event; 3. assert that
  `git log -1 --format=%H -- <path of this document>` == HEAD, the same commit (the verdict appendix is a
  new commit after the verdict and is not bound by this preflight). The driver **does not embed** the
  freeze sha; any "sha copied from a commit" is forbidden.
- **W2** `eval_contract.PROTOCOL_VERSION == 3` and the evaluation contract's `SCHEMA_VERSION == 2` (note
  that this and R-S's BC report schema_version==1 are **two different constants**, asserted separately
  and never mixed up).
- **W3** `engine_binary_path(ROOT)` resolves uniquely; assert that the environment variable
  `DEVILUTIONX_REF` is not set and that bootstrap.sh contains verbatim
  `ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"` (the line itself contains
  the env override syntax, and the default value pins the SHA; the binary's byte identity is bound
  separately by the W5 bundle).
- **W4** the game data directory contains `diabdat.mpq` (full-content world; sha256 recorded in the
  ledger).
- **W5** `train_ppo._implementation_bundle_sha256()` (a private function in train/train_ppo.py;
  eval_contract has no API of that name) is sampled at **stage granularity**: once before each stage's
  launch and once after it ends, recorded in the ledger (six times over three stages); if a stage's two
  values differ, or any sample ≠ the case's first value, or ≠ implementation_sha256 in that stage's
  bc_report.json → that stage's products, **including a scientific FAIL report**, are voided together and
  recorded as OPERATIONAL. Note: the scripts' built-in drift assertion covers only the PASS path that
  writes weights; the world identity of a scientific FAIL report is backstopped by this clause's
  before/after sampling in the driver.
- **W6** the ledger records `eval_contract.runtime_versions_identity()` in full (it pins the six packages
  numpy/gymnasium/torch/stable-baselines3/sb3-contrib/tensorboard plus the Python implementation /
  version / cache_tag; the function itself fails loudly on any drift).
- **W7 collection fixture (the frozen v2 manager)**: `train/models/v22-h-manager/policy.npz` is present,
  with `sha256 == eval_contract.DEFAULT_MANAGER_SHA256`
  (`0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7`), recorded in the ledger.
  **Legitimacy note**: this file is a frozen product of the v2 era; this case uses it only as the S1
  demonstration-collection fixture (a window scheduler, a fixed function input). It is not the subject of
  any scientific conclusion of this case and does not constitute reuse of a v2 baseline; the v3
  evaluation contract names it as the default manager (DEFAULT_MANAGER_RELATIVE, same path, same
  constant), so it does not conflict with the start of v3. bc_worker.py builds WorkerWindowEnv without
  passing the manager_sha256 pin, so this preflight is the only prior gate on that dependency; at load
  time train_ppo re-checks the report's manager_npz_sha256 against the file on disk. Replacing this
  fixture with a v3 generation belongs to R2 and later cases.
- **W8 sealing the priors**: before launch, the sha256 + mtime of each existing bc_report.json,
  policy_sd.pt and demos.npz under train/runs/bc-{worker,manager,flat}/ go into the ledger (PREV_ARTIFACT
  events; including the fact that the bc-worker report's mtime is later than its weights, recorded as is,
  without interpretation or action). Declaration: each script's begin_output_attempt will move them to
  _previous/<time_ns>/ in the same directory; that move is **archive preservation** (location changes,
  bytes unchanged) and does not violate the run-archive discipline; the verdict appendix must contain a
  "prior sha ↔ _previous location" table, and any byte difference → OPERATIONAL.

## D2 recipe for the trio (scripts executed verbatim, not a single line changed; order = critical path first)

Execution order: **worker → manager → flat** (the worker is the v3 season's teacher, so it scores first).

| Stage | Command | Built-in gate (in the script, fail-loud) | Expected duration (estimate, not a basis) |
|---|---|---|---|
| S1 | `.venv/bin/python train/bc_worker.py` | **G1 data side** (= the data half-gate of PREREG-v23 G1): held-out top1≥0.95 ∧ recall≥0.85 for threshold classes (≥300 samples in the full set); a class-weighted retrain = the only retry; demonstration seeds exactly 100-227 with no fallback; the always-masked 11/12 never mixed in | ~1-2 h |
| S2 | `.venv/bin/python train/bc_manager.py` | memoryless ratio gate: BC (7000 pool) / teacher (same pool) ≥ 0.85 | ~0.5-1 h |
| S3 | `.venv/bin/python train/bc_flat.py` | ratio gate of the same type ≥ 0.85 | ~1-2 h |

**Operations discipline** (the driver's responsibility; table clause: driver = `train/run_reanchor_r1.py`,
included in the freeze commit, orchestration only, it changes no script under train/; the text matches the
driver as it is):

- **Concurrency**: the three stages run strictly in series; before each stage's launch assert the machine
  is idle, i.e. `pgrep -f 'train/(bc_|train_ppo|eval_assembled|run_v[0-9]+)'` has zero matches, otherwise
  no launch (P6-type report). No other training or evaluation job may start on the machine while the
  case runs. Reason: the ratio gates are return-based and unaffected by load, but the 4 h wall clock is
  directly affected, and a false timeout under load would waste the single rerun quota of a stage.
- **Timeout**: each stage has a wall-clock limit of 4 h; every launch uses
  `Popen(..., start_new_session=True)`, and after a timeout `os.killpg(SIGKILL)` it must `proc.wait()`
  to reap (following the run_v30_relay precedent). On inspection the three scripts are single-process (the
  engine loads in-process, no DataLoader workers), so the risk of orphaned engines is zero; this clause is
  a discipline against future changes. Engine scratch left by SIGKILL is reclaimed automatically at the
  next initialisation under the env registry lock, and the driver **must not** delete scratch by hand;
  half-written \*.tmp.\* files are overwritten by the next run and heal themselves, so they are not
  cleaned by hand either.
- **Measured durations**: each stage's actual start/end wall clock goes into the ledger (the priors have
  no durations, so the table's durations are estimates); before rerunning S1 after a timeout, check the
  log's "collected n/128 episodes" progress lines to tell a hang from slowness and record it in the
  ledger, but the 4 h hard line is not waived because of progress; measured durations are the basis for
  budget calibration in R2 and later cases.
- **Logs**: stdout+stderr merged into `train/runs/reanchor-r1/logs/S{n}-{role}-{ts}.log`.

## D3 R lines (falsifiable predictions + verdicts, written before running)

**Verdict source clause**: the only source of a stage verdict = the verdict keys of bc_report.json
(data_gate/hypothesis/memoryless_hypothesis) + the R-S receipt gate; **the script's exit code is not a
basis for the verdict**. Verdict key ∈ {PASS, FAIL} and the receipt passes the gate → scientific verdict;
missing report, verdict key RUNNING/missing, missing/mismatched receipt → OPERATIONAL. The three outcomes
{PASS, FAIL, OPERATIONAL} cover every exit (a failed W line = P6, no launch; a driver crash = P1).

- **R-W (worker teacher)**: predicted **PASS** (prior: the audit's re-verification top1=1.000, with the
  provenance limitation in the rationale).
  - PASS → register "v3 teacher (data side) in place"; the verdict must carry (pairs, held_out_top1, the
    per-class recall table, class_weighted_retry) and policy_sha256/demos_sha256. **G1 verdict limitation
    (the obligation of v23 appendix C carried over)**: the G1 data side only proves that BC can reproduce
    the teacher on the teacher's own trajectory distribution; the assembled-replay half-gate of v23-G1
    (eval_assembled ≥ 0.85×baseline) belongs to the anchoring case because the baseline does, so it is
    **deferred to measurement in R2**; the R1 verdict may not claim that "the learned worker can replace the
    script".
  - FAIL (including after the single retry) → **the whole case halts and reports**. Rationale: the same
    recipe scored perfectly in the previous world, so a FAIL is interpreted first as a world-level anomaly;
    S2/S3 readings could not be interpreted against that background, and running the worker first makes
    halting cost nothing. After a halt, running S2/S3 needs a written decision, recorded as a supplementary
    registration (without changing this case's R lines), and its conclusions must be labelled "collected
    with the teacher missing".
- **R-M (manager memoryless hypothesis)**: predicted **FAIL** (prior 0.6498, comparable same-pool
  definition; mechanism candidate = the env.exhausted wrapper state is unobservable).
  - FAIL → register: "under the registered recipe (64×64 MLP / 10 epochs / demonstration pool 100-227),
    the manager teacher cannot be cloned by a 303-dim memoryless BC at a ratio ≥0.85 (v3 world); mechanism
    candidate = the env.exhausted wrapper state is unobservable. The removal rests on **engineering
    unusability**, not on a proof that 'it is not a memoryless function'." The M-BC arm is removed for the
    v3 era, and future elections are fresh-only; revival needs a teacher-redesign case (its own
    pre-registration; if it touches the environment side, its own approval).
  - PASS → register: "in the v3 world, this teacher can be cloned by a 303-dim memoryless BC with the
    registered recipe at ratio=<value> ≥0.85"; the difference from the v2 prior is **not attributed to a
    single cause**, and the candidate explanations are recorded side by side: (a) the v2 back-jump /
    level-change settlement defects; (b) v3 semantic changes weaken the teacher's state dependence (the
    prior may be a true property of the v2 world); (c) statistical fluctuation. A ratio ∈ [0.85, 0.90)
    adds "marginal pass". The arm's continued eligibility rests only on the v3 reading and does not endorse
    an attribution.
  - The verdict must carry (pairs, bc_replay_7000, teacher_7000, ratio).
- **R-F (flat memoryless hypothesis)**: predicted **FAIL**, but **with lower confidence than R-M** (see the
  definitional limitation of the prior in the rationale: the old 0.5974 cannot be converted to the
  same-pool gate and is directional evidence only; whichever way it opens, it is recorded as is, with no
  after-the-fact endorsement). The verdict template is of the R-M type (296 dims; mechanism candidate =
  the stall-clock _clock wrapper state is unobservable); it carries (pairs, bc_replay_mean_7000s,
  teacher_replay_mean_7000s, ratio).
- **R-S (receipt gate, each of the three reports; train_ppo's actual code is authoritative, with no weaker
  subset)**:
  - **PASS report**: the driver calls `train_ppo._validate_bc_report(path of policy_sd.pt, the
    corresponding gate name, verify_replay=False)`, and R-S PASS means every assertion passes, including a
    key set **exactly equal** to `_BC_PASS_KEYS[gate name]` (extra keys and missing keys are equally
    wrong), schema_version==1 (the BC report schema `train_ppo._BC_REPORT_SCHEMA_VERSION`, not W2's
    evaluation-contract SCHEMA_VERSION==2), protocol_version==3, implementation_sha256 == the current
    bundle, **generator_sha256 == the sha of the corresponding current train/bc_*.py file (equality, not
    just presence)**, byte-level binding of policy/demos, recomputation of the worker evidence, and the
    worker report bound to the currently frozen manager NPZ. Recomputation by replay (verify_replay=True,
    the full definition) is left to the consuming gates: R2 and season training run the same validator in
    full at load time.
  - **FAIL report** (the validator is designed to refuse FAIL, so the driver checks by hand against the
    same standard): the key set must be exactly `_BC_PASS_KEYS[gate name]` minus policy_sha256 (the worker
    also minus demos_sha256; **manager_npz_sha256 is permanent worker provenance, so it must be present on
    FAIL too and equal the W7 ledger value**); the four receipt keys (schema_version / protocol_version /
    implementation_sha256 / generator_sha256) are all present and each equals the current world value
    (generator under the same equality definition).
  - Any condition unmet → OPERATIONAL (products voided; archiving is ensured by the script's
    begin_output_attempt at the next launch). **A scientific FAIL is not exempt from the receipt gate.**

**Case verdict (exhaustive)**: R-W PASS ∧ R-S all pass → **case closed** (PASS or FAIL for R-M/R-F are
both legitimate outcomes); R-W FAIL → halt and report; any stage still OPERATIONAL after its rerun quota
is exhausted → halt and report; ANOMALY (see D4) → halt and report. The case verdict must carry: the three
stage verdicts + every receipt sha + the ledger path + the freeze sha.

## D4 P lines (operational incidents, exhaustive)

- **P1 driver top-level exception / manual interruption** → DRIVER_EXCEPTION in the ledger +
  NEEDS_ATTENTION. **Idempotent resume decision** (per stage; the only source of truth = the canonical
  bc_report.json; the scripts have no built-in skip, and each bc_*.py archives the canonical products to
  _previous at start and reruns from scratch, so the skip must be decided by the driver, and a wrong
  launch is forbidden):
  - (a) final PASS: the R-S PASS branch passes in full → skip the stage;
  - (b) final scientific FAIL: the R-S FAIL branch passes in full → **S1: whether first run or resume,
    the R-W halt and report is triggered at once, and S2/S3 must not launch**; S2/S3: no rerun (no extra
    attempts); record STAGE_SKIP_FAIL in the ledger and continue;
  - (c) everything else (RUNNING leftovers / missing keys / extra keys / unparseable / missing file) →
    treated as unfinished and rerun; the rerun counts against the stage's OPERATIONAL quota (**1 per stage
    in total, whatever the P1-P7 cause; any second OPERATIONAL → halt and report**). RUNNING leftovers are
    not cleaned by hand (the script archives them on rerun; on a halt they are sealed as they are, and the
    train_ppo validator naturally refuses them, so there is no risk of consuming them by mistake).
- **P2 exclusive_lock conflict** → OPERATIONAL. The lock is an fcntl.flock advisory lock that the kernel
  releases automatically when the process dies (including kill -9); a leftover .bc.lock file holds no lock
  and is harmless, and **deleting the lock file as a clean-up is forbidden** (deleting a held lock file
  breaks mutual exclusion). A conflict must mean a live holder exists: locate it with `lsof` (the pid= line
  inside the lock file is only a hint about the previous holder, not evidence), confirm it is not a
  process of this case, terminate it and retry once; another conflict → halt and report. The lock is taken
  at the script entry, before begin_output_attempt, so P2 does not harm products on disk.
- **P4 provenance drift** (triggered by a built-in script assertion or by the W5 stage-granularity
  sampling) → OPERATIONAL, rerun once; a repeat → halt and report.
- **P5 timeout (>4 h)** → killpg + wait to clear, OPERATIONAL, rerun once; a repeat → halt and report.
- **P6 preflight fails** (any W-line assertion fails / machine not idle) → no launch, report.
- **P7 non-zero exit without a FAIL verdict in the report** (RUNNING placeholder / missing / incomplete
  keys; typical: a seed-discipline assertion) → OPERATIONAL, rerun once; a repeat → halt and report. The
  criterion for a scientific FAIL (exhaustive) = the report's verdict key == "FAIL" ∧ the R-S FAIL branch
  passes in full; anything short of that is OPERATIONAL and may not be registered as a scientific
  conclusion.
- **ANOMALY (scientific anomaly, uses no rerun quota)**: the stage log shows "ratio gate undefined" (the
  same-pool teacher's mean return ≤0) → no rerun (with fixed seeds a rerun adds no information, and "the
  teacher's mean return is ≤0 in the v3 world" is a scientific anomaly to report at once); halt and report
  immediately.

## D5 ledger, logs and the sources of the disciplines

Ledger = `train/runs/reanchor-r1/gate_ledger.jsonl` (not published; JSONL, one event per line,
append-only; a resumed run first reads the old ledger and asserts every line parses; unparseable → halt
and report). Each stage records at least: stage name, launch/end wall clock, exit code, log path, the two
implementation_bundle_sha256 values before and after, verdict, receipt sha values (policy / demos /
manager_npz where applicable), and the scientific numbers (D3 verdict discipline). NEEDS_ATTENTION lives
in the same directory.

**Two independent disciplines and their sources** (panel correction: no dangling external references):
1. **document freeze discipline**: after the panel review and freeze, the text of D1-D5 cannot be changed;
verdicts and corrections are always appended as appendices at the end, an appendix cannot be changed once
recorded either, and later corrections are made as new correction entries (precedent: the PREREG-v30
appendix, where the commit timestamp is the notarisation). 2. **run-archive discipline**: pre-v3 run
archives are immutable forensic records (source: the project README of the time, "Pre-v3 archives remain
useful as immutable forensic records"); the only operation this case may perform on them is the
_previous preservation move after W8 registration.

## Preview of later cases (not executed in this case)

**Alignment with the sequence in the roadmap's era-switch note**: the three steps "BC base regeneration →
teacher recast → anchor re-measurement" fold into two cases: **R1 covers the first two steps** (the worker
BC is the recast of the v3 season's teacher; the manager npz teacher is a frozen artefact and is not
recast, with its identity locked by the receipt sha; the scripted teacher needs no recast, only
re-measurement in R2), and **R2 covers the third step** (anchor re-measurement). There is no hidden case
beyond the three steps; "no course campaign may launch before all of it is complete" uses the recording
of the R2 verdict as its completion mark.

**R2 anchoring case** = re-measure four parties on the gold seeds in the v3 world: 1. the science anchor =
the 140.3 line-up (M29-fresh × v28-worker-leg1, v29's historical high); 2. the launch anchor = **the
incumbent assembled agent's 112.4 line-up (under the anchor-compliance rule of PREREG-v30 D3; the v30
verdict item 5, re-election, on record)**; 3. the throne v24-golden; 4. the scripted teachers. This sets up
the v3 launch/science dual anchors; the correspondence between old and new titles follows the
anchor-compliance rule of PREREG-v30 D3 verbatim, and re-measurement changes no incumbent title. Gold
seeds are used once per case, launched manually, and need a new pre-registration plus notice. **The A
rematch right** (one time only; a clause of the direction review (not published), kept in the
[roadmap](../design/ROADMAP-course-plan.md)) has its unlock condition deferred likewise: it can be used
only after R2's new v3 anchor is in place, with anchor = the incumbent assembled agent as determined under
the compliance rule at that time; neither this case nor R2 consumes the right. Only after R2 can the design
cases for courses 3, 4 and 5 (environment side, each with its own approval) be queued.

---

## Verdict appendix (case closed 2026-07-12; the commit of this appendix is the notarisation, and it may not be changed once recorded)

Freeze commit `106eb16954c5ba55f304acaea41587cc5c9785bf`; ledger `train/runs/reanchor-r1/gate_ledger.jsonl`
(not published); **13 minutes in total** (the clause "expected durations are estimates" came true: S1
measured 5.6 min / S2 3.2 min / S3 4.1 min, the basis for budget calibration in R2 and later cases). Case
verdict: **case closed** (R-W PASS ∧ all three R-S pass). All three R-line predictions **hit**.

- **R-W: PASS (prediction hit)**. pairs=166383, held-out top1=**1.000**, threshold-class recall {9: 1.0},
  zero retries (class_weighted_retry=false). Receipts: policy
  `f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6`, demos
  `3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363`, manager_npz = the W7 ledger value,
  impl `76dba1eb…`, generator `505ade6f…` (full values in the ledger). **v3 teacher (data side) in
  place**; the G1 verdict limitation applies: the assembled evaluation is deferred to R2, and this verdict
  does not claim that "the learned worker can replace the script". Observation recorded: the threshold
  classes shrank from v2's {9,10} to {9} (action 10 has <300 samples in the full set): the demonstration
  distribution really changes with the world, so regeneration is not a formality.
- **R-M: FAIL (prediction hit)**. ratio=**0.6478** (51.124/78.922). Verdict as in the registered D3 text:
  under the registered recipe, the manager teacher cannot be cloned by a 303-dim memoryless BC at ≥0.85
  (v3 world); mechanism candidate = the env.exhausted wrapper state is unobservable; the removal rests on
  engineering unusability. **Cross-world observation**: it reproduces the v2 prior 0.6498 (same-pool
  definition) almost digit for digit, which strongly weakens candidate explanation (a), "caused by v2
  world defects", and puts the world-invariance of the teacher's state dependence on top (an
  observation; the registered verdict is unchanged). The M-BC arm is removed for the v3 era, elections are
  fresh-only, and revival needs a teacher-redesign case.
- **R-F: FAIL (prediction hit; the "lower confidence" clause honoured)**. ratio=**0.4786**
  (18.758/39.198): the flat teacher's first reading under the **same-pool definition** (v2's 0.5974 used
  the biased old definition and is not comparable, see the rationale). The flat BC baseline is removed for
  the v3 era; mechanism candidate = the stall-clock _clock wrapper state is unobservable.
- **Prior ↔ _previous table (W8 obligation)**: bc-worker {report `f7f825b3…`, demos `91d9aec3…`, policy
  `4de43826…`} → `_previous/1783858471165715000`; bc-manager {report `374760c4…`, policy `0772d85e…`} →
  `_previous/1783858810262880000`; bc-flat {report `5909671a…`, policy `1f4b5f0c…`} →
  `_previous/1783859000643200000`. Bytes preserved; full sha values in the ledger's PREV_ARTIFACT events.
- **Incident record (no blame)**: the driver's top-level guard wrongly caught `SystemExit(0)` and left a
  spurious DRIVER_EXCEPTION ledger event and a NEEDS_ATTENTION entry after CASE_CLOSED; judged harmless
  (the case was closed, the real exit code was 0). Hot fix: `sys.exit` moved out of the try (in the same
  commit as this appendix), and a handling note was appended to NEEDS_ATTENTION.
- **Sequence progress**: the first two of the roadmap's three steps (BC base regeneration + teacher
  recast) are completed by this case; next case: **R2 anchoring** (gold-seed re-measurement, which needs a
  new pre-registration plus notice).
