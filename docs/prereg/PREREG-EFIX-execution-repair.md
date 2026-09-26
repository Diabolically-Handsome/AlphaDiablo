# PREREG E-fix: DIVE execution repair (2026-07-31)

> Approval chain (2026-07-31): "repair the legs first, then retrain the manager" (the recommended option) + three switches "A: keep the old endpoints", "anchor re-burn
> folded into R9", "full-set definition" + launch authorised directly once the design was finalised (authorised to proceed straight
> to R9 training). The full foundation and design investigation is a separate record (not published); root cause: the R9 foundation review.

## 1. What is repaired (two places, each with an explicit old-behaviour endpoint, v32 polarity)

- **Fix A · fallback promotion** (_macro_descend in env.py; knob
  DiabloGymEnv(descend_fallback_promotion=True), False = old-behaviour endpoint):
  the pocket-reachable region of the monster-avoiding BFS produced a partial path where "going back the way you came is closer", which forms a deterministic limit cycle
  with the lenient planner (R8 seed 2122004: a 6-window, 24-micro-step period, burning ~900
  micro-steps). The fix ports the rule already legislated in _macro_progression: when the monster-avoiding path cannot step onto
  a stairs tile, consult the lenient planner, and promote only if its end point is strictly closer; when it can step onto the stairs the second BFS
  does not happen, and the behaviour is bit-for-bit the old one.
- **Fix B · a real no-progress clock for DIVE** (options_env.py; knob
  OptionsEnv(dive_stall_protocol="no-progress-v1"), "tau-v3" = old
  endpoint): the old window-closing criterion tau>=KILL_PATIENCE only counted elapsed time and looked at no progress at all -- long routes
  (on a 112x112 map a window covers at most ~48 tiles) and clearing monsters along the way were all shot down (measured stall rate 75%).
  New criterion: only "the historical minimum distance to the target strictly improves / a kill / the full positive_progress set"
  extends the window, and the window closes after KILL_PATIENCE without progress; **bare displacement does not count as progress** (a limit cycle moves
  every beat); the KILL_PATIENCE constant (frozen observation-normalisation denominator) and the TAU_CAP=600
  hard cap are unchanged. When stubs/shells are not configured, the old semantics are the defensive default.

## 2. G0 proof items (case files in train/runs/efix-g0-evidence/, E0-MANIFEST.sha256; not published)

- **E0 pre-implementation baseline**: the official R7/R8 final-exam archives (cryptographically frozen products of the pre-implementation code)
  + the 16-game bit-level replay probe and seven-piece data set of the foundation investigation (sealed before any change, with a sha list).
- **G0-1 bit-level identity at the old endpoints (passed, measured)**: with both knobs of the new code set to the old values, replaying the R8 final-exam
  seeds already used, 2122000-2122007 -- **8/8 exact=True** (including a faithful reproduction of depth=1 in the limit-cycle game 2122004).
- **G0-2a**: declared N/A (zero training-side knobs, train_ppo untouched).
- **G0-2b functional smoke test (passed, measured)**: at the new endpoints (1) the limit cycle is gone (2122004 old:
  10 dive windows, 9 stalled → new: 1 dive window, 0 stalled; after 596 steps of real travel it died in combat -- execution repaired,
  the outcome is down to the recipe); (2) stall rate 24/32=75% → **0/12=0%**; (3) **4 windows** with tau>140 still dived successfully
  (existence proof for fix B); (4) old endpoint = old behaviour (i.e. G0-1).
- **G0-ghost**: the existing 9024/9005 KATs stay green (full suite 826+8 pass); the limit-cycle
  predicate of 2122004 has zero hits at the new endpoint.
- **G0-domain**: the first-dive prefix is consistent per domain 8/8 (zero first-action divergence in FARM/RESUPPLY windows) --
  a behavioural proof that incentives are untouched; static proof on the diff: reward/wage/mask/observation encoding
  touched on zero lines.
- **G0-6 full-table REF_BITEQ**: the sample of 8 seeds at the old endpoints has passed; the full-table replay (256x2 archives)
  belongs with the R9 anchor re-burn and runs with the R9 case (as approved: anchor folded into R9).

## 3. Handling of the impact surface

- env.py/options_env.py are both in the implementation bundle and the protocol bundle → the bundle sha changes; **BC pool
  rotation is deferred**: R9 is a manager campaign and does not collect worker BC; the old 2_142/2_143 artifacts are refused
  automatically by data_gate (fail-closed), and legislating the rotation to 2_144/2_145 is postponed to the next worker BC
  collection case, so the final, frozen R7/R8 campaign files are not touched.
- The assembled-anchor re-burn is folded into the R9 pre-registration (as approved); the 2_123 range is suggested for the new-protocol anchor.
- Tests: 83 pass on the impact surface; new tests/test_efix_execution.py with 8 tests
  (protocol branches / target distance / knob contract); full suite 826+8 pass.
