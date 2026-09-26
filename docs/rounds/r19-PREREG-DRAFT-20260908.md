# R19 pre-registration draft (2026-09-08; draft, never frozen): the "fair-gear Butcher" campaign

Stages: graduate from level 1 -> build up on level 2 (avoiding the Butcher) -> fight the Butcher at level 6 with
fair gear -> only then consider level 3. Each stage freezes its own pre-registration and passes its own gates.

## 0. Campaign goals (2026-09-08)

1. Goal: the **first recorded, reproducible "fair-gear" Butcher kill** (defined in section 1).
2. One step at a time: make level 1 solid, then build up on level 2, and go down to level 3 only after the Butcher
   can be beaten reliably; going to level 3 earlier just gets the character killed.
3. The Butcher's room is **recognised from its layout**, not from monster coordinates in the engine (no god's-eye
   view).
4. Manager options are completed per the 2026-09-07 list. To keep the agent from turning into a scripted AI, every
   piece of scaffolding is registered as "learned / written" with a removal condition.
5. The end point is **one network that plays all 16 levels**. Specialists, parent prefixes and composite
   deployments are scaffolding that is eventually folded back into one network through a curriculum or
   distillation. The definition of a "full clear" goes into the project's standing rules.
6. Human knowledge: study videos of human full clears and reverse-engineer common sense as "trigger / action /
   why / cost", so the agent gets a complete learning environment.

## 1. "Fair gear" and "reliable" (candidate standing rules; thresholds to be set)

- **Fair gear**: starting gear unchanged; no attribute edits, no item duplication, no experience-table edits; all
  gold and gear earned in game; only the warrior's own means (melee, scrolls, potions), no spells; single player,
  normal difficulty; the Butcher appears under the original rules, and **his door is neither disabled nor
  hidden**; **one continuous game** from town. Town trips count; loading a save does not.
- **Reliable**: pre-registration first; fresh seeds; 48 paired games; publish the kill rate and the survival rate
  after the kill (suggested thresholds: kill >= 50% and killer survival >= 80%); every game's seed, save and full
  trajectory published and replayable; a GitHub tag and an aispeedrun.ai entry; wording "the first recorded".
- **Prior work (searched 2026-09-08, checked three ways)**: the develop branch of rouming/DevilutionX-AI reports
  a 73.7% per-level spawn success rate over 16 levels (50.2% on level 2) and **disguises the Butcher's door as a
  wall** to avoid him, without fighting; lciesielski/DeepDungeon targets the Butcher but with overpowered gear and
  the Doom spell, and reports no success rate; NiteKat/DAPI, a purely rule-based system, reportedly clears the
  whole game; zero academic papers. **Nobody has a recorded Butcher kill.**

## 2. Current baseline (source: `r18-B-GATES-REPORT-20260908.md`, control rows sha 0e5a1acd..., parent 7e31dc54 in the v1 world, 48 seeds)

| Item | Now |
|---|---|
| clvl >= 3 at game end / before death | 39/48 (81%) |
| Leaves level 1 alive | 34/48 (71%); another 7 games alive but did not descend within 12,000 ticks (1 game with 0 kills throughout) |
| Dies on level 1 | 7/48 (15%): all before the first town trip (ticks 559-2433), at clvl 1-2, belt 0, 100 gold, armour 7, with 5-10 monsters nearby; in 5/7 the top damage source is a **skeleton captain** |
| Level 1 cleared (<= 3 left) | 4/41 (10%); on average 90-97 kills with 9 left |
| At the first descent to level 2 | clvl 3 (70%) / 4 (30%), AC 9-15, belt mostly 4 potions |
| Level 2 | hazard 0.222 per 1,000 ticks; 26 kills per visit (level 2 holds about 120); level 2 deaths 19/48; killed by the Butcher 3/48 |
| Experience budget | clvl 6 = 18,258 XP; clearing level 1 gives about 6k (clvl 3); the missing 12k can only come from level 2 |

Three lessons from R18-B: the agent learns when the reward is there (descend-macro a11 rate 2.4% -> 13.9%,
level-3 arrivals 4 -> 15); opening learning windows only on level 2 and above makes level 1 regress (deployed
alone, level-1 deaths 25/48); learning to descend without learning to survive only moves the deaths deeper.

## 3. Stages, rules and gates

### Stage A: graduate from level 1 (arm `r19-A`)

Rules (new; zero-training paired probes first):
- **A1 early retreat**: extend the retreat-v1 trigger (hp <= 0.5 max, or empty belt and hp <= 0.75 max) to level 1;
  on level 1 the up staircase is the way to town. Aimed directly at the 7 early deaths.
- **A2 buy potions at the start**: spend the starting 100 gold on potions before entering the dungeon (Pepin's
  price and purchasable count to be checked). It is the first thing a human does.
- **A3 level-1 retention (training shape)**: learning windows cover level 1 (see section 6), so the learner no
  longer has zero level-1 data.
- Kept: sweep-v1, cain-v1, smith-v1, sustain-loot-v1, hunt_scope=l1-only, coach-v03, the r18c clock.

Gates (48 paired seeds; control = parent in the same world; values are suggestions):
- level-1 deaths <= 5% (now 15%); level-1 clear rate >= 50% (now 10%); games with clvl >= 3 at the first descent
  >= 90%, with belt >= 4;
- paired survival not worse than control (UCB95 < 0, or net >= 0 and not significantly worse);
- no regression on the six-exam set (needs amendment 3 or a new exam format, see section 6);
- mechanism metrics: cause-of-death catalogue of early deaths, number of opening potion purchases, number of
  level-1 retreats and deaths during retreat.

### Stage B: build up on level 2 (arm `r19-B`)

Rules:
- **B1 Butcher-room exclusion v2 (layout recognition, section 4)**: all three path finders (a10 hunt/explore, a11
  descent path, retreat walk) avoid it; `boss_near` retreat trigger (6 tiles). R18-J v1 covered only a10 and
  Butcher deaths stayed 3 -> 3, so it does not count.
- **B2 farm near the stairs** (R18-M idea): fight on level 2 within R tiles of the up staircase until the
  targets are met, then widen. Aimed at deaths on the retreat path (15 of 22 level-2 deaths happen while walking
  back, with a median distance of 22 tiles to the stairs at trigger time).
- **B3 decouple the retreat check**: the retreat rule is currently evaluated only at DIVE window boundaries (70 of
  85 triggers); check it every tick instead, **without changing thresholds** (retreat-v2 changed the thresholds,
  was negative, and is dropped). Lesson from K1: turning off voluntary DIVE also turns off the retreat check, so
  decouple first and add stage gates afterwards.
- **B4 no descent from level 2**: in this stage DIVE (2 -> 3) is masked entirely (the stage goal is build-up, not
  depth). A threshold/mask change that needs sign-off.
- Everything from stage A is kept.

Gates: level-2 hazard <= 0.15 (now 0.222); games ending with clvl >= 6 >= 50% (now 4%); **Butcher deaths = 0**;
level-2 kills >= 1.5x; paired survival not worse than control.

### Stage C: fight the Butcher (arm `r19-C`)

- **C1 Butcher readiness table** (scaffolding with a registered removal condition): clvl >= 6, AC >= X, belt >= 6,
  weapon damage >= Y, gear sorted in town. Only when all hold is the exclusion lifted, and the manager gains
  option 4 **BOSS** (enter the Butcher's room and fight).
- **C2 tactics**: reverse-engineered from human videos (holding the doorway, potion rhythm, positioning after the
  door triggers "Ah, fresh meat"), preferably as observation features and reward terms the worker learns rather
  than a fixed script; any part that must be scripted is registered as scaffolding.
- **Removal condition**: after stage C passes its gates, feed the readiness signals (clvl/AC/belt/weapon) into the
  manager's observation, remove the hard mask, and let the manager learn when it is ready. The exclusion keeps
  only its perception role (recognising the room).
- Gates: on 48 fresh paired seeds, kill rate >= 50% and killer survival >= 80% (thresholds to be set); all
  trajectories published.

### Stage D: level 3, with a pre-registration only after stage C passes.

## 4. Layout recognition of the Butcher's room (2026-09-08 survey; engine and bridge facts)

Engine (DevilutionX; each point has file:line evidence in a survey note next to
`train/runs/r18-reports/gates-20260908/`, not published):
- The Butcher quest Q_BUTCHER is fixed on cathedral level 2; the room is the prefab `levels/l1data/rnd6.dun`
  (6x6 large tiles = 12x12 world tiles), stamped by `InitSetPiece()`, and it occurs once per map.
- There are only **five** positions (chosen by SelectChamber): prefab origin W in {(20,48), (48,48), (76,48),
  (48,20), (48,76)}; walkable interior W+4..W+9; the only opening is in the **east wall** (large tile (5,3)); the
  Butcher always spawns at W+(4,4), i.e. at one of five fixed coordinates.
- `AddTortures()` places 12 torture/corpse objects at fixed offsets around the anchor micro-tile dPiece 366. Seven
  of them (TNUDEM1-4, TNUDEW1-3) are solid and leave a fixed 7-point hole in the walkable map; the 5 racks are
  not solid. On cathedral levels only the THEME_TORTURE theme room also places one of them (TNUDEM2), so **any
  TORTURE1-5 or TNUDEW\* means the Butcher's room**.
- Wall tiles 87/88/91/92/123/126 occur only in rnd6.dun; the room is its own 8x8 light zone (W+3..W+10) and lights
  up entirely on entry; no other monsters, barrels or sarcophagi spawn inside.
- The Butcher: type 51 (MT_CLEAVER), AI 13, 220 HP, resists fire and lightning, cannot open doors.

Bridge (diablogym) today:
- The observation has **no** tile ids, no explored/lit map and no room or light-zone ids; `local_map` has only
  five channels (walkable/door/closed_door/hazard/explosive), and walkable is ground truth that sees through
  darkness; the `objects` channel (sweep-v1) exports only five classes of chests/barrels/sarcophagi and no torture
  objects; the monster list exposes the coordinates of unlit monsters (**ruled out; not used**).

Proposal **butcher-room-v1** (respects partial observability):
1. Add a class `gore` to the `objects` channel (the 12 object ids 29-40), with the same `visible = IsTileLit` tag;
2. only when at least one torture object **has been seen under light** (the same "seen before it counts" memory as
   sweep) is W inferred from its fixed offset and the 12x12 footprint registered as an exclusion zone, avoided by
   all three path finders;
3. before that nothing is known, and like a human the agent may wander in; that is the price of fairness, and the
   `boss_near` retreat trigger is the fallback;
4. **no** template matching of "five candidate positions x the 7-point hole in the walkable map", which sees
   through darkness (the same class as the god's-eye view), unless decided otherwise.

To check (next probe): whether the east-wall opening carries an OBJ_L1LDOOR door object; whether dPiece 366 is
unique on the certified seed set; the false-positive rate.

## 5. Manager options and rules (learned / written / removal condition)

| Item | Now | Learned/written | R19 plan / removal condition |
|---|---|---|---|
| FARM / DIVE / RESUPPLY options | chosen by the scripted coach coach-v03 in the resource world | written | from stage C2, feed readiness signals into the manager's observation and learn them |
| BOSS (new option 4) | none | - | introduced in stage C; entry guarded by the C1 readiness table (scaffolding), removed in C2 |
| open chests / break barrels, sweep-v1 | yes, level 1 | written | keep; registered as the "economy pipeline" |
| identify cain-v1 / sell / buy potions / buy armour / buy weapons smith-v1 | yes | written | keep; a human only clicks a few times here too |
| pick up gear from the floor, a14 | yes | **learned** | keep |
| stat points | fixed in the bridge at 3 vitality : 2 strength | written (C++) | keep and register; could become an option later |
| kill priority threat-v1 / aggro cap | present but off (R18-D/E negative) | written | keep off; the monster catalogue becomes a manager observation feature (C2) |
| retreat retreat-v1 | yes, level 2+; checked only at window boundaries | written | A1 extends it to the start of level 1; B3 checks every tick without changing thresholds |
| town portal portal-v1 | present but off (too expensive) | written | a separate "buy when rich" rule |
| Butcher-room exclusion | none (J v1 not merged) | written (perception) | B1 layout recognition v2; after C2 only perception remains |
| hunt range hunt_scope | l1-only | world setting | keep |
| farm near the stairs | none | written | B2 |
| big picture (small monsters first, then the boss) | the coach's six rules, no boss awareness | written | staged readiness tables -> learned in C2 |
| NPCs | Griswold / Pepin / Cain (Adria follows the gate) | - | the Butcher fight needs no Wirt/Ogden/Farnham/Gillian |

## 6. Training shape and tools

- **Level-1 retention** (A3): learning windows must cover level 1. Two options: (a) earned-dive-suffix-v2, where
  the parent still plays the prefix but a fixed share of live level-1 windows goes to the learner; (b) farm-dive-v1
  warm-started directly from the parent, with all windows live. Composite deployment v0 is diagnostic only; the
  formal deployment shape is one network.
- **Escrow**: kept (R18-B showed that payouts work); each stage's payout condition follows the stage goal (stage B
  does not descend, so escrow moves to "clear the room / level up" goals). A threshold change that needs
  sign-off.
- **Exams**: the formal exam rejects the weights-only lineage (amendment 3 draft pending); if R19 warm-starts with
  critic migration (with a warm-up receipt), the exam tools can stay as they are. One of the two.
- **Budget**: about 1M learning steps / 10 h per stage; soak for >= 30 min before launch; >= 20 GB free disk.
- **Review**: changes to rules and probes go through code review; only certified bytes enter the main tree; new
  files never overwrite frozen files.

## 7. Open decisions

1. The two definitions and threshold numbers in section 1; 2. amendment 3 (exam admission) or a switch to critic
migration warm starts; 3. level-1 retention via (a) or (b); 4. exclusion recognition from "seen torture objects"
only (recommended) or a template that sees through darkness; 5. the 2 -> 3 mask and escrow conditions of stage B;
6. the exact numbers of the stage gates; 7. whether to run the zero-training probes (A1, A2, B1) before freezing
stage A.

## 8. Proposed order (if approved)

Zero-training probes (A1 level-1 retreat, A2 opening potion purchase, B1 exclusion v2 -> Butcher deaths 3 -> 0)
-> review -> certification -> freeze the `r19-A` pre-registration -> launch (daytime launch, 30 min soak) -> close
the same day.
