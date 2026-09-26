# R17 pre-registration draft (2026-09-06): the resource channel "as actually built" + Gate T0

Status: **draft, not frozen**. Once it is frozen, its sha256 goes into the ledger, and any byte change voids
it and requires a new freeze.
Basis: [`r17-DIRECTION-PANEL-20260902.md`](r17-DIRECTION-PANEL-20260902.md) (the panel skeleton, §4),
[`r17-0-VERDICT-20260906.md`](r17-0-VERDICT-20260906.md), decisions 1-6 (ledger of 2026-09-06, lines
443/444), the two-way gate (line 451), the retroactive registrations (lines 452-458), and two review memos
of 2026-09-06 (a review of R18-R23 and a comparison of readiness laws; ledger and memos not published).

## 1. Rationale

R16 pushed the root-cause chain to the dead end of "cleared but not ready"; R17.0 measured that the manager
hardly matters in deployment (survival 17/17/17/16 under four scripted managers), so the lever must be the
affordances of the environment. A separate line of work (R17.1-A..F, all not certified) has built the
physical affordances the panel asked for: picking up gold, going upstairs back to town, buying / equipping
/ repairing without the UI, healing at Pepin, and town macros; it measured a town round trip at 614-1263
steps (median about 780), with only 17%-23% of returns meeting the bar, and 70/104 of the misses due purely
to lack of cash. **Gate T0 has never run.** This pre-registration freezes the channel in the form in which
it was actually built, together with the readiness law as decided, and does only one thing: answer, with a
zero-training factorial probe, "once the hands are built, does the danger of L2 go down?", and use that as
the only criterion for launching any training arm.

## 2. Rules (frozen items)

1. **Resource protocol** `resource_protocol = l2-town-v1`; five purchase modes `none / heal / potions /
   armor / full` (the factor of T0).
2. **Readiness law** `resource_readiness_law = coach-v03` (decision 3): six conditions for L1→L2
   readiness: clvl≥2, AC≥9, dmg≥6, instant potions in the belt ≥4, minimum durability ≥15 across the four
   slots (head / left hand / right hand / chest), one usable weapon; **HP is not part of the law**; it is
   used only for the drinking reflex / HP-loss pricing and the manager's town-trip trigger (the trigger keeps
   the seven native conditions, including HP≥80%).
   The L1→L2 branch of the engine's `ResourceTransitionGuard` is advisory: it only writes a receipt
   (`reason ∈ {ready, forced_unready}`, `pretransition_ready_law`) and does not veto; `target>2` is advisory
   as well (`deeper_advisory`), and L3+ is decided by the Python-side table (v0.2, sixteen levels). **The
   forced-descent escape hatch is kept, as the panel decided**: `m[FARM] = not forced_dive` (the frozen rule
   text); forced-unready descents are counted separately and do not pay escrow (escrow pays according to
   `pretransition_ready_law`).
3. **Service recipe** `resource_service_policy = sustain-v6` (the most complete form, already evaluated in
   that separate work: a 1500-step trip limit, a 450-step gold pick-up window, buy armour / equip armour /
   unequip to keep readiness / repair / buy potions / free healing); the FARM trigger stays at 3600 steps
   (the R19 pairing results do not support 1800).
   **Decision point A**: this can be changed to legacy-v1 (600-step limit), but R19 measured 0/48 trips
   completing within 600 steps, so under that recipe T0 would measure zero round trips.
4. **Clock**: the frozen 6000 steps (`worker_time_protocol = legacy`), **not** completion-l2-v1 (12000
   steps), to stay comparable with A0′ and the R16 deployment.
5. **Deployed form** (unchanged from R16): worker 7e31dc54 (the R16 constitutional worker) and the
   certified worker; autonomy on; economy v4; hunt on; farm_scene_cap 3600; window clock reset; sampled
   decoding; with the protocol on, manager = `resource_option_choice` (scripted), with the protocol off,
   manager = the readiness-v3 coach (the A0′ form).
6. **Exam protocol**: the old 3000-step form unchanged; the six anchor sheets are recast under the new
   protocol bundle as **new files** `r17-anchor-{xdevil,xm29,xm29full}-{a,b}` (protocol off, R16 exam
   flags); the old `r16-anchor-*` files are never overwritten.
7. **Seeds**: T0 pool = **2_133 (2133000-2133047)**, registered on 2026-09-06, unused; A0′ is re-probed on
   the same pool (paired design). The fresh pools 2_116-119 and 2_126-128 are not touched; 2_129 has been
   declared burned.

## 3. Gate T0 (zero training; the only criterion for launching a training arm)

**Combinations** (48 seeds each, 6000 steps, sampled, single-threaded Torch):
- A0′: protocol off, readiness-v3 coach × {7e31dc54, certified worker}: 2 groups (baseline, re-probed on
  the new pool).
- T0 rows × 7e31dc54: (a) `none` (pick up gold only, no town trip), (b) `potions`, (c) `armor`, (d)
  `full`: 4 groups; (e) `heal` (healing only) as an information row: 1 group; (d) × certified worker as an
  information row: 1 group.
- 8 groups in total ≈ 40 minutes of machine time; zero training; consumes 2_133.

**Main metrics** (panel §4.1):
- Main-1 survival (exposure-normalised): L2 hazard per thousand steps = L2 deaths / Σ steps spent on L2;
  survival at first descent step + 1800.
- Main-2 growth retention: L2 reached ≥ 0.8 × A0′; mean clvl ≥ A0′ − 0.1; L1 kills per thousand dungeon
  steps ≥ 0.85 × A0′ (town steps excluded).
- Pareto count alive ∧ L2.

**Criteria** (panel §4.2, verbatim):
(d) against A0′: paired saved − lost ≥ +6/48 and UCB95 < 0; L2 reached ≥ 0.8 × A0′; L2 hazard per thousand
steps ≤ 0.7 × A0′; (d) − (a) survival ≥ +4/48 (a channel effect rather than observation drift); (b) and (c)
reported separately, and the verdict must name the load-bearing lever; the forced-unready descent share of
(d) ≤ 25%. Any unmet → no training arm is launched; R17 verdict = channel not confirmed / not attributable.
**T0 crowns nothing.**

**Mechanism sub-metrics** (informational; tautologies forbidden): distribution of trip trigger reasons
(cleared / cap; idle-clock must be 0); median trip steps; AC and durability bought; potions bought; empty
trips; belt/AC at descent; gold picked up per L1 clear; attribution of each arm's cash shortfall;
`forced_unready` and `ready` receipt counts; size of the L2 roster.

## 4. Certification prerequisites (before launching T0, in fixed order)

1. The full final-byte suite + a two-way re-bake (the July library + the new `build-res` library)
   reproducing the four anchors bit for bit (passed once on 2026-09-06 before the construction for decision
   3; must pass again after it).
2. Write the protocol source bundle sha, bridge sha, engine sha and patch-stack sha into the ledger and
   freeze this file.
3. Recast the six `r17-anchor-*` sheets (about 50 minutes).
4. Launch order (consumes 2_133).

## 5. If T0 passes: the R17.1 training arm (under its own launch order; this file only previews it)

Continue training 7e31dc54, coach-v03, escrow ruler v2, T = round2048(266240/(1−s)); four gates against
r17-anchor; deployment against A1 = T0(d) and A2 = certified worker / R17 environment; crowning rule = the
R16 wording (survival not worse than A2 and growth significantly better than A2).

## 6. Identities (filled in at freeze)

- Protocol bundle sha256 (22 files, `eval_contract.source_bundle_sha256`):
  `fb7d651e9c2e466e0fc1ec4704b8a99b345b58dbb0ff6570bae636f5be58d671`;
  bridge `_diablogym` (build-res, 2026-09-06): `a7e47ebb1e42ef264d065808f9e23cd86dec7a939f9023fdc3b0145bc7333ab9`;
  engine `liblibdevilutionx_so.so`: `a57a2ca3621480aacfa3c82eae2ac2b233148437c0e8c4665e117cfcdd0dc4d5`;
  patches 0001-0011 sha256: see ledger line 451 (0009 de0303d3…, 0011 2c8873f1…).
- Final-byte certification (ledger R17_1_FINAL_BYTES_CERTIFICATION_PASS): two-way re-bake reproduced the four
  anchors bit for bit; suite on the official tree 1790/53/9 (9 = symbols missing from the old library);
  suite on the mirror root with the new library all green (including 9 native readiness tests); G0 receipt
  smoke rc=0; probe regression receipt 33023de1… reproduced.
- sha256 of this file: computed at freeze and written into the ledger.

## 7. Corrections to this pre-registration from the W/P adversarial review (2026-09-06)

Handling of the four HIGH findings of the W/P adversarial review of 2026-09-06 (not published):
- H1 A0′ (2026-09-02) was run while the code tree was changing → that table is a historical reference only;
  re-probing A0′ on 2_133 in T0 replaces it.
- H2 `alive_at_first_descent+1800` is structurally unmeasurable at 6000 steps (mean first descent 4535
  steps + 1800 > 6000); the 0.214/0.276/0.200 in the delivery table are censoring artefacts, and the
  uncensored true value is ≈ 0 → **decision point D**: this pre-registration turns that metric into
  "steps survived after the first descent (with right censoring; report the observed denominator
  `fd_plus_1800_observed_n`)" as information; **the main metrics are now only the exposure-normalised L2
  hazard per thousand steps** (unaffected by censoring) and the three growth-retention items; the clock is
  not extended (comparability with R16).
- H3 the receipt identity was only enforced on the timeout branch and lacked a death term → fixed: both
  assembly points in worker_env now also emit `worker_credited_terminal_death_reward` and
  `worker_policy_reward_receipt="v1"`; leashed_ppo checks the six-term identity on every marked transition
  (a missing key raises). Added to the final-byte gate.
- H4 the probe was changed on 09-04 without a regression receipt → added after step 1 of the certification
  prerequisites: rerun the probe's regression segment (16 seeds, readiness-v3 form), check rows sha
  `33023de1…`, register the new `probe_sha256`; the two definitions of M2, `mask_forced_descent_share` and
  `farm_masked_at_descent_share`, are listed separately.

## 8. Decisions (2026-09-06: approved as proposed, and T0 may be shortened)

- A. Service recipe **sustain-v6 / 1500 steps**.
- B. **No information rows**; T0 is shortened to 4 groups: A0′ × {7e31dc54, certified worker} + (a) `none`
  + (d) `full` (both 7e31dc54); (b) `potions` and (c) `armor` are deferred to T0′, after the gear-selling
  economy (R17.1-D) is merged. The prior view recorded on 2026-09-04 (in the ledger): the worker neither
  picks up nor sells gear, so it has no money to spend in town → expect no significant improvement of (d)
  over A0′; T0's value is to turn "lack of money" into per-arm numbers and to serve as the baseline for T0′.
- C. **Freeze** (the sha256 of this file goes into the ledger).
- D. "First descent + 1800" is demoted to censored information; main metrics = exposure-normalised L2 hazard
  per thousand steps + the three growth-retention items.
- E. **The new bridge becomes the official runtime**: `build/` → the new `build-res` build (a symlink to a
  local build directory, not published), with the July build kept alongside; the anchor recast and T0 all
  run on the same runtime (bridge a7e47ebb…).
- The T0 criteria stay as written in §3; the shortened version can only decide the two items (d) vs A0′ and
  (d) − (a); (b) and (c) wait for T0′.
