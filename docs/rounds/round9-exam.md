# Round 9 exam (2026-09-24): released model vs candidate on 27 new worlds

**Read first.**

- **The exam was started with the project owner's approval, not by its own gate.** The candidate adapter
  `d13facts5` failed one of its pre-set offline criteria (N2: 81.77% agreement on the old test sets, required
  at least 82.31%), and the training-world gate paused on a new loop signature (1 of 19 games). The project owner
  approved starting the exam anyway. (An earlier candidate, `d10facts4`, had failed its offline gate and was given
  an exception; it was then replaced, and that exception did not carry over.)
- **The project owner stopped the exam after 3 of its 4 waves.** Wave 4 (candidate, second half of the seeds) was
  stopped about 10 minutes in; its 16 games ended as `operator_stop` and are not results. This affects only the
  candidate: the released model (`d9facts3`) finished both halves, 32 games. The pre-registered adoption rule and
  the one confirmatory test could not be run as written. "Candidate not adopted" is the owner's decision after
  seeing the first half (8/16 vs 12/16), not the output of the adoption rule.
- **The model is not the whole agent.** It picks one option from a menu that code builds, while the game is
  paused; code computes the fact lines and executes every move and swing. See
  [What the model does and what code does](#what-the-model-does-and-what-code-does).

Per-game statistics: [`round9-exam-results.json`](round9-exam-results.json). Code: [`option_brain/`](../../option_brain/README.md).
Videos: release [`round9-stable-kills-20260924`](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/tag/round9-stable-kills-20260924).
Round numbers here count the option-brain training rounds of 2026-09-22 to 09-24; they are unrelated to the
campaign papers `r9-*` to `r19-*` in this folder.

## Result in one table

| Arm | Adapter | Seeds | Skeleton King killed | Butcher killed | Both missions | Deaths |
|---|---|---|---|---|---|---|
| Old (released, "version 9") | `d9facts3` | half A (w2) | 7 / 8 | 5 / 8 | 12 / 16 | 1 |
| Old (released, "version 9") | `d9facts3` | half B (w3) | 6 / 8 | 5 / 8 | 11 / 16 | 5 |
| **Old, total** | `d9facts3` | **A + B** | **13 / 16** (81.2%, Wilson 95% 57.0–93.4) | **10 / 16** (62.5%, 38.6–81.5) | **23 / 32** (71.9%, 54.6–84.4) | **6 / 32** (18.8%, 8.9–35.3) |
| New (candidate) | `d13facts5` | half A (w1) | 3 / 8 | 5 / 8 | 8 / 16 | 3 |
| New (candidate) | `d13facts5` | half B (w4) | — | — | stopped by the owner, not a result | — |

"Killed" means the game's target boss was killed with the hero alive (status `success`; the runner checks for
death before it checks for the kill, and ends the game at the kill). Intervals are Wilson 95% intervals, computed
after the fact.

## What was compared

Both arms ran **the same round-9 code**: the same runner, menu (`r9`), undo guard, fact-line code, hands, broker
and bridge/engine binaries, with the same hands flags:
`--executor r3 --r3-chain --r3-target --r3-chain-guard post_hit --r3-hold-floor 0.5 --undo-guard --r3-chase --facts --auto-belt --allow-heldout`,
6,000 decisions at most. The only difference was the adapter and the fact-line version it was trained with:

- **Old arm, `d9facts3`**: the round-8 adapter (trained 2026-09-23), fact lines version 3. This is the model
  released as "version 9".
- **New arm, `d13facts5`**: trained 2026-09-24 in round 9d on 4,455 rows, fact lines version 5.

Because the round-9 menu hides more options than the round-8 menu (for example the stairs down on level 2 in the
Butcher mission), the old arm's numbers here are **not** the same experiment as its round-8 held-out numbers
(Skeleton King 9/14, Butcher 8/16), and the two are not pooled.

## Protocol (summary)

The exam followed an internal pre-registration (in Chinese), frozen before the first game: its frozen copy has
sha256 `6da34cd41104ddac42474a1e7a7b830a100bdbcf568af039401aef34f6d4dfcf`, and five revisions were appended
afterwards (two code fixes and the new-arm change before the freeze, the owner's approval to start, the owner's
stop). There is no external timestamp, and the document itself is not published because it contains internal
notes and local paths.

- **Seeds.** Counting up from 4150064, the first 16 seeds with the Skeleton King quest and the first 16 with the
  Butcher quest, by a copy of the engine's quest draw (`quest_lottery.py`), checked against the bridge (King) and
  a tick-0 engine probe (Butcher; the probe started those 16 games and read one tile at tick 0, a declared
  exception to "never used"). 27 distinct seeds; 5 seeds (064, 067, 081, 082, 086) appear in both lists with
  different missions.
  - King: 064 067 069 070 071 074 081 082 | 086 089 093 094 095 096 097 098
  - Butcher: 064 065 066 067 068 072 075 076 | 078 079 080 081 082 083 086 087
  - (seed prefix 4150; the bar separates half A from half B)
- **Never used before.** A seed audit over every training, teacher and game record found 0 rows for the exam
  seeds (last run at the final freeze: 541 game folders, 699 files, 124,875 lines). What a seed does not change:
  the town, the Skeleton King's tomb (a fixed map with fixed monster placement) and the Butcher's room layout are
  the same in every Diablo I world, and tomb states from training worlds are in the training data. The exam
  therefore tests the road to the boss in a new world; the boss arenas are not new.
- **Waves.** Four waves of 16 games (8 King + 8 Butcher) in ABBA order: w1 new/A, w2 old/A, w3 old/B, w4 new/B,
  one inference process per wave, games of a wave running concurrently. Queueing only changes wall-clock time:
  the game is paused while it waits for the brain.
- **Limits.** 150,000 ticks for King games, 80,000 for Butcher games (20 ticks per second), 6,000 decisions,
  720 minutes of wall clock.
- **Scoring.** Intention to treat: only `success` counts; every other end (death, tick budget, loop stops) is 0.
  In a Butcher game a Skeleton King kill does not count; in a King game a Butcher kill is neutral. Infrastructure
  failures would have been re-run once; none occurred. No game in waves 1–3 was stopped by hand.
- **Freeze.** Every file that runs (code, scripts, config, bridge, engine, game data, adapters, executor weights,
  base-model config files) was listed with its sha256 before the exam: 361 files, list sha256
  `5b732f5eb0d1ba68bd5c7915f52210ed89d8724b814fb625d54399a59dc7a9ef`, checked by `sha256sum -c` before every wave
  and again after the exam (361/361 OK; the 10 base-model shards 10/10 OK). One gap was found afterwards: two
  runtime modules of the frozen Gymnasium package were not in the list (see `option_brain/PROVENANCE.md`).
- **Adoption rule (not executed).** The candidate would have been adopted only if, over both halves, it killed at
  least as many bosses as the old arm (and no more than one fewer per mission), died no more often, and met a
  retreat-behaviour criterion. The one confirmatory test was a one-sided exact sign-flip test over the 27 seeds.

## Results by wave

| Wave | Arm | Half | King | Butcher | Deaths | Other ends |
|---|---|---|---|---|---|---|
| w1 | new `d13facts5` | A | 3 / 8 | 5 / 8 | 3 | 4 loop stops (`no_progress_cycle_40`), 1 tick budget |
| w2 | old `d9facts3` | A | 7 / 8 | 5 / 8 | 1 | 1 loop stop, 2 tick budget |
| w3 | old `d9facts3` | B | 6 / 8 | 5 / 8 | 5 | — |
| w4 | new `d13facts5` | B | — | — | — | 16 `operator_stop` (exam stopped by the owner; not a result) |

## Paired comparison on half A (descriptive only)

The 16 pairs of half A (same seed, same mission, both arms):

| Mission | Both killed | Only old | Only new | Neither | Exact McNemar p (two-sided, descriptive) |
|---|---|---|---|---|---|
| Skeleton King | 2 | 5 | 1 | 0 | 0.22 |
| Butcher | 2 | 3 | 3 | 0 | 1.00 |
| Both | 4 | 8 | 4 | 0 | 0.39 |

With 16 pairs, differences of this size are not statistically significant; the pre-registered test could not be
run because half B has no candidate games. The practical reading is the owner's: the candidate did not improve on
the released model where it could be compared (King 3/8 vs 7/8), so it was not adopted.

Post-hoc, the candidate lost mainly through three habits, counted over half A: leaving the tomb when it saw the
Skeleton King and the fact line said "ready" (311 of 596 such decisions, old arm 13 of 254), walking back to a
doorway with no enemy in sight (3.57 vs 0.76 times per 100 dungeon decisions), and holding a tile (228 vs 79
times). Its four loop stops came from the doorway habit. The improvement the candidate showed on the 19
training-world gate seeds (8, then 11, then 14 of 19 over three rounds) was largely overfitting to those seeds:
three rounds of fixes and retraining were judged on the same 19 seeds, one round added gate positions to the
training data, and the old arm, which received the same code fixes, was never run on the gate.

| Training-world gate (19 seeds; not the exam) | Kills | Deaths |
|---|---|---|
| `d9facts3`, round-8 code | King 4/11, Butcher 1/8 | — |
| `d10facts4` | 8 / 19 | 2 |
| `d11facts5` | 11 / 19 | 3 |
| `d13facts5` | 14 / 19 | 2 |

## Per-seed results

Columns: final character level, deepest dungeon level reached (3 = level 3 and the tomb), game time in minutes
(ticks / 1,200; the game is paused while the model decides, so thinking time is not included), decisions, healing
potions left in the belt at the end, and whether the game was replayed from its native journal.

**Old arm, `d9facts3` (the released model)**

| Seed | Mission | Half | Wave | Result | Char. level | Deepest | Game min | Decisions | Belt potions | Replayed |
|---|---|---|---|---|---|---|---|---|---|---|
| 064 | Skeleton King | A | w2 | **killed** (also killed the Butcher on the way) | 8 | 3 | 23.3 | 835 | 4 | yes |
| 067 | Skeleton King | A | w2 | **killed** | 8 | 3 | 73.5 | 1439 | 6 | yes |
| 069 | Skeleton King | A | w2 | **killed** | 8 | 3 | 29.5 | 867 | 6 | yes |
| 070 | Skeleton King | A | w2 | died | 4 | 3 | 15.2 | 550 | 0 | yes |
| 071 | Skeleton King | A | w2 | **killed** | 8 | 3 | 25.3 | 957 | 7 | yes |
| 074 | Skeleton King | A | w2 | **killed** | 9 | 3 | 44.2 | 1093 | 6 | yes |
| 081 | Skeleton King | A | w2 | **killed** | 8 | 3 | 40.4 | 1155 | 8 | yes |
| 082 | Skeleton King | A | w2 | **killed** | 8 | 3 | 38.9 | 1008 | 8 | yes |
| 086 | Skeleton King | B | w3 | died | 3 | 2 | 12.4 | 400 | 0 | yes |
| 089 | Skeleton King | B | w3 | died | 3 | 2 | 11.2 | 333 | 0 | yes |
| 093 | Skeleton King | B | w3 | **killed** | 7 | 3 | 37.2 | 1052 | 7 | yes |
| 094 | Skeleton King | B | w3 | **killed** | 7 | 3 | 29.9 | 1049 | 8 | yes |
| 095 | Skeleton King | B | w3 | **killed** | 7 | 3 | 29.9 | 920 | 5 | yes |
| 096 | Skeleton King | B | w3 | **killed** | 7 | 3 | 24.4 | 771 | 4 | yes |
| 097 | Skeleton King | B | w3 | **killed** (video) | 6 | 3 | 21.9 | 801 | 3 | yes |
| 098 | Skeleton King | B | w3 | **killed** | 8 | 3 | 28.7 | 1004 | 2 | yes |
| 064 | Butcher | A | w2 | **killed** | 7 | 2 | 34.7 | 962 | 8 | yes |
| 065 | Butcher | A | w2 | **killed** | 5 | 2 | 25.5 | 729 | 1 | yes |
| 066 | Butcher | A | w2 | loop stop | 6 | 2 | 46.5 | 820 | 8 | — |
| 067 | Butcher | A | w2 | **killed** | 5 | 2 | 20.0 | 624 | 5 | yes |
| 068 | Butcher | A | w2 | tick budget | 4 | 2 | 66.7 | 812 | 8 | — |
| 072 | Butcher | A | w2 | **killed** | 6 | 2 | 32.7 | 797 | 7 | yes |
| 075 | Butcher | A | w2 | tick budget | 6 | 2 | 66.7 | 1011 | 8 | — |
| 076 | Butcher | A | w2 | **killed** | 5 | 2 | 20.1 | 602 | 5 | yes |
| 078 | Butcher | B | w3 | **killed** (video) | 5 | 2 | 14.8 | 503 | 3 | yes |
| 079 | Butcher | B | w3 | **killed** | 4 | 2 | 19.7 | 564 | 1 | yes |
| 080 | Butcher | B | w3 | died | 2 | 2 | 6.8 | 214 | 0 | yes |
| 081 | Butcher | B | w3 | **killed** | 6 | 2 | 26.6 | 703 | 3 | yes |
| 082 | Butcher | B | w3 | **killed** | 6 | 2 | 21.2 | 553 | 4 | yes |
| 083 | Butcher | B | w3 | died | 4 | 2 | 38.1 | 628 | 0 | yes |
| 086 | Butcher | B | w3 | died | 4 | 2 | 24.2 | 640 | 0 | yes |
| 087 | Butcher | B | w3 | **killed** | 4 | 2 | 17.1 | 443 | 1 | yes |

**New arm, `d13facts5` (candidate), half A only**

| Seed | Mission | Wave | Result | Char. level | Deepest | Game min | Decisions | Belt potions | Replayed |
|---|---|---|---|---|---|---|---|---|---|
| 064 | Skeleton King | w1 | died | 3 | 2 | 10.6 | 427 | 0 | — |
| 067 | Skeleton King | w1 | died in the tomb (had killed the Butcher earlier) | 7 | 3 | 39.5 | 1145 | 0 | yes |
| 069 | Skeleton King | w1 | loop stop | 6 | 3 | 26.0 | 639 | 8 | — |
| 070 | Skeleton King | w1 | **killed** | 7 | 3 | 35.2 | 1118 | 6 | yes |
| 071 | Skeleton King | w1 | tick budget | 8 | 3 | 125.0 | 1673 | 8 | — |
| 074 | Skeleton King | w1 | **killed** | 7 | 3 | 39.4 | 986 | 1 | yes |
| 081 | Skeleton King | w1 | **killed** | 7 | 3 | 42.3 | 990 | 6 | yes |
| 082 | Skeleton King | w1 | loop stop | 7 | 3 | 40.1 | 1011 | 7 | — |
| 064 | Butcher | w1 | died | 3 | 2 | 10.8 | 449 | 0 | — |
| 065 | Butcher | w1 | loop stop | 1 | 1 | 8.8 | 213 | 1 | — |
| 066 | Butcher | w1 | **killed** | 5 | 2 | 24.8 | 588 | 6 | yes |
| 067 | Butcher | w1 | **killed** | 6 | 2 | 30.1 | 794 | 3 | yes |
| 068 | Butcher | w1 | **killed** | 6 | 2 | 33.2 | 867 | 3 | yes |
| 072 | Butcher | w1 | loop stop | 4 | 2 | 15.6 | 603 | 3 | — |
| 075 | Butcher | w1 | **killed** | 6 | 2 | 63.5 | 996 | 6 | yes |
| 076 | Butcher | w1 | **killed** | 6 | 2 | 41.9 | 808 | 7 | yes |

**Old-arm details.**

- Skeleton King kills: median 35,874 ticks (29.9 minutes of game time), range 26,289–88,216; character level 6–9
  at the kill. Butcher kills: median 24,796 ticks (20.7 minutes), range 17,701–41,633; character level 4–7.
- All six deaths happened with no healing potion left in the belt: five on dungeon level 2 at character level
  2–4, one on level 3 at level 4. Nobody in the old arm died in the tomb: all 13 old-arm games that entered it
  ended with a Skeleton King kill.
- Per 100 decisions of the old arm: the model chose to drink 1.2 times (295 drinks, every one of them a model
  choice), to spend an attribute point 3.1 times, and to fight a named enemy or all visible enemies 45 times.

## What the model does and what code does

- **Lockstep.** The game is paused while the model decides; the runner refuses to continue if the native state
  advanced while it was thinking. In the old arm there was one decision per 46.7 game ticks on average (about
  2.3 s of game time; 24,839 decisions over 1,161,188 ticks). A human cannot pause like this.
- **The menu is code.** `options.py` lists what can be executed now (attack a named enemy, fight all, hold, back
  off, drink, explore in one of eight directions, stairs or quest entrance, pick up, open, talk, buy, sell,
  repair, identify, equip, belt moves, attribute point, resume). The round-9 rules hide some options: the stairs
  down on level 3 (the bridge refuses deeper levels) and, in the Butcher mission, on level 2; options that just
  had no effect or failed to deliver while nothing changed; selling an item bought in the same town visit; an
  attribute point while an enemy is adjacent. The undo guard also hides options that would just undo the previous
  action. In the old arm, the menu rules hid at least one option in 26.6% of decisions and the undo guard in
  16.8%.
- **The fact lines are code.** `facts.py` computes them from the public game state and the engine's own formulas
  and data: the hit chance against the boss, whether the current weapon can kill him within a set number of
  swings, how many healing potions are needed, and a verdict such as "ready" or "not ready yet (character level
  5)". The thresholds are ours (for the released model: Skeleton King at character level 7 with 8+ potions and at
  most 60 swings; Butcher at level 5 with 8+ potions and at most 45 swings). The model's fight-or-leave choices
  largely follow this verdict.
- **The hands are code.** Walking (shortest path over the explored map), attack chains and target choice among
  the enemies the model named, chasing to the last-seen tile, moving potions from pack to belt when no enemy is
  near (auto-belt, 103 times in the old arm) and closing a shop window when the model chose something else (32
  decisions). The model directly orders drinking, attribute points, and buying, selling, repairing and
  identifying. The executor may not drink.
- **Where the labels came from.** The adapters were trained by imitation learning on labels from Claude-based
  teacher workflows (two independent teacher runs per state and a third to adjudicate, over several DAgger
  rounds; the teachers also read written guidance the student never sees, `option_brain/TEACHER.md`) and on
  earlier demonstrations by an OpenAI Codex agent ("Astra") that had played two worlds through the same executor.
  No reinforcement learning or reward signal was used for the adapters.

## Verification

**Replay from native journals (CPU only, no model).** Every native command a game issued is in its journal with a
sha256 hash chain that includes the engine's checkpoint hash. `option_brain/replay/replay_verify.py` feeds the
journal back, row by row, into the same engine and bridge binaries and checks every row's tick, receipt and hash,
and every decision's observation digest.

- Round-9 exam: 38 games replayed, 38 passed on the first pass (no timing waits needed): every old-arm game with a
  boss kill or a death (29: 23 target kills, 6 deaths), every candidate game with a target kill (8), and candidate
  King seed 067 (a Butcher kill, then death). 1,539,627 journal rows and 30,488 decisions matched; replayed
  journals and combat telemetry were byte-identical to the originals (38/38); final states equal `result.json`
  (38/38). Kills and deaths were read from the engine's own counters, and the reverse check (every kill or death
  seen in a replay is in the results, and vice versa) matched exactly.
- Not replayed: 10 games with neither a kill nor a death in scope (old Butcher 066, 068, 075; candidate King 064,
  069, 071, 082 and Butcher 064, 065, 072).
- Round 8 (the earlier held-out check of the same adapter): all 50 games replayed, 2,681,486 journal rows and
  41,684 decisions matched; 48 on the first pass, and the two games of seed 4150021 after adding a 6-second wait
  before one pickup (the engine ignores a second pickup of the same item within 6 s of wall-clock time, and the
  replay runs far faster than the live game).

**Adapter re-scoring (GPU).** A replay shows that the logged commands produce the logged states; it cannot show
which model chose them. For that, logged prompts were run again through the frozen inference code with each
arm's adapter:

- Round-9 exam: 2,000 decisions per arm, stratified by game (sample seed 20260923): `d9facts3` 2,000 of 2,000
  identical (from 24,839 decisions of 32 games), `d13facts5` 2,000 of 2,000 identical (from 13,307 decisions of
  16 games); maximum probability difference 0 in both.
- Round 8: 39,743 of 41,684 decisions re-scored, all identical; the other 1,941 (training-world games) were not
  re-scored.

**Command census.** The native journals of the 48 completed exam games (2,103,086 rows) contain only 18
ordinary command kinds (`tick`, `walk`, `attack_stand`, `attack`, `pickup`, `operate`, `stat`, `drink`, `buy`,
`equip`, `talk`, `dismiss`, `sell`, `belt`, `unbelt_exact`, `repair`, `reset`, `identify`). None of the bridge's
test probes (for example `probe_invincible` or `probe_add_experience`) was called.

**What this does not prove.** The hash chain is unkeyed: it shows that files were not corrupted after recording,
not that nobody rewrote a whole log. The logs, adapters and executor weights are not published, so these checks
cannot be repeated by others at this time; the code that ran and the sha256 of every input are published.

## Limitations

- No full-game clear. Only two bosses, the Butcher (dungeon level 2) and the Skeleton King (his tomb below
  level 3). The bridge used here refuses main dungeon levels below 3.
- Warrior, Normal difficulty, single player, one class of worlds; small samples (16 games per boss) and wide
  intervals.
- Not end-to-end: menus, fact lines and hands are code; much of the strategy lives in the fact-line verdicts and
  in the teachers' guidance.
- The game is paused while the model decides, and the model sees structured text (including exact hit points of
  visible monsters), not pixels.
- The engine is patched for headless running (crash guards, read-only interfaces, a shop-transaction refactor
  with the same prices and random-number use, and one upstream save-load fix); no combat, drop, price, experience,
  quest or map-generation rule or data was changed.
- The exam is internally pre-registered without an external timestamp, and it was stopped early by the owner
  (affecting the candidate arm).
- The videos are replay renders of two winning games, not live captures, and neither covers a whole game.
