# R18-C pre-registration: a zero-training probe with a long observation window (2026-09-07)

## 0. Background

An open-ended training run was proposed to see whether there is any progress. It was not launched: the training windows do not include retreat yet (R18-B is not done, and WorkerWindowEnv explicitly rejects retreat-v1), so launching a training arm in a hurry would break the protocol rules.
Instead, a **zero-training mechanism probe with a long observation window** answers the underlying question: T0″ showed that retreat turns deaths on L2 into survival,
but the 1800-tick window after the first descent cannot hold one "retreat → recover → descend again" cycle; widen the window five-fold and see whether the cycle turns, and where it leads.

## 1. Change (default off = unchanged bit for bit)

`python/diablogym/completion_clock.py`: a new immutable recipe `completion-l2-r18c` (a frozen subclass, fields init=False):
arrival deadline 12000, observation denominator 6000, collection/service/farm constants the same as v1, **9000 ticks of observation after the first descent** (v1: 1800).
`CompletionClock` only accepts the two registered recipes. `python/diablogym/options_env.py`: r18c is accepted wherever v1 is;
the clock is built from the recipe; the telemetry reports the actual recipe. `worker_env.py` is not changed: training keeps rejecting r18c.
Known minor flaw (reported only): the `time_protocol` field of `SustainCompletionService.telemetry()` still writes the v1 string (same constants, label not changed).

## 2. Probe

Driver `r10-staging/run_r18c_probe.py` (not published); pool 2_133 (a consumed pool, same-seed pairs); worker 7e31dc54; manager = the resource coach coach-v03;
economy sustain-loot-v1; decoding: sample; two arms: (e-long) retreat-v1 + r18c; (d′-long) control + r18c.
The physical cap per game is 12000 + 9000 = 21000 ticks.

### Items reported, not judged (P1–P6)

- P1 descending again: the number of games in the retreat arm that "go down to L2 again after reaching the upper level", and how many of them survive until the clock runs out;
- P2 depth: the maximum main-line depth distribution of both arms (number of L3+ arrivals);
- P3 survival: paired saved − lost and the one-sided UCB95 (same method as T0″), also reported under the fd+9000 and fd+1800 definitions;
- P4 exposure: L2 hazard per 1000 ticks, total L2 ticks;
- P5 cycles: histogram of successful retreats per game, the number of second attempts and their success rate;
- P6 prefix identity: the first-descent tick of every (d′-long) game must match the T0″ control arm game by game (same seeds and arrival deadline → trajectories before the first descent should be bit-identical);
  otherwise this probe is VOID.

This probe is not a training launch gate; its output is evidence of what the R18-B training arm should reward (descending again? depth? survival?).
**No threshold (retreat rule 50%/75%, observation window 9000) changes in this round; a change means a new pre-registration.**

## 3. Certification chain (before launch)

Full pytest suite with 0 failures (including `tests/test_completion_r18c.py`); probe regression 33023de1… equal; two-way re-bake 4/4 + 4/4 (mirror root refreshed);
a three-lens review (identity / new path / experiment design) with no unrefuted blocker; the sha256 of this file recorded in the ledger.

## 4. Seeds and files

2_133 consumes 2 more groups (10 in total); the virgin pools are untouched. New files: this pre-registration, `run_r18c_probe.py` and `r18c-probe/` (not published), `tests/test_completion_r18c.py`.
Pre-change copies of `completion_clock.py` and `options_env.py` were kept in a local work directory (not published).
