# AlphaDiablo R15 readiness table by level, draft v0.1

**Scope**: the whole table assumes Diablo I single player on Normal (monster HP = the multiplayer table
value ÷ 2; DevilutionX reproduces the original by default; reports 1/2/3 cross-checked). Our character is
**a fixed warrior, starting at level 1 in every episode, with v20 automatic stat allocation 3 vitality : 2
strength (magic and dexterity are never raised)** (report 4, diablogym.cpp:2069-2079). This environment
fact makes every guide strategy that depends on spells (Fire Wall / Holy Bolt / Mana Shield) or high
dexterity (block / to-hit) **unavailable or weaker for us**, and the table has been corrected for it.

**Confidence legend**: ★★★ = verified item by item against source code / Jarulf; ★★ = explicit citation
from a community guide; ★ = value derived from mechanics (no guide gives the number directly).

---

## 1. Readiness table by level (to be reached before entering the level)

The "damage per hit" column is the **hit-recovery line**: the maximum damage of a single hit must be ≥ the
highest resident mlvl on the level + 3 to cause hit recovery (confirmed in DevilutionX monster.cpp,
★★★). The HP column is derived as "survive 5-8 of the level's largest single hits" (★).

| dlvl (section) | clvl threshold | AC threshold | damage-per-hit threshold | HP threshold | Threat baseline on the level (single player) | Confidence |
|---|---|---|---|---|---|---|
| 1 Cathedral | 1 | 12-17 (unequipped) | ≥5 | 70-90 | mlvl 1-2, largest single hit 7 | clvl/damage ★★★, AC ★★ |
| 2 Cathedral | 3 | 15-25 | ≥8 | 90-110 | largest single hit 15; the Butcher's room (see hard gates) | same |
| 3 Cathedral | 5 | 30-40 | ≥10 | 100-130 | Black Death (single hit 22, permanent HP loss); the Skeleton King's side cave | damage ★★★, AC ★★ (FuriousPaul: AC≥35) |
| 4 Cathedral | 7 | 35-50 | ≥12 | 120-150 | Gloom ToHit 70; first goat archers | ★★ |
| 5 Catacombs | 8 | 40-55 | ≥14 | 140-170 | section jump 1: mlvl→8-10, invisible monsters + archer packs + open halls | ★★ |
| 6 Catacombs | 9 | 45-60 | ≥16 | 150-180 | acid spitters (magic damage) in numbers | ★★ |
| 7 Catacombs | 10 | 50-65 | ≥17 | 160-200 | horned demon charge 5-32 (ToHit 500, always hits; avoid by positioning) | ★★★ (charge mechanics) |
| 8 Catacombs | 11 | 55-70 | ≥19 | 180-220 | Illusion Weaver 16-24 per hit; Toad HP 67-80 | ★★ |
| 9 Caves | **12** | 60-80 | **≥22** | 200-240 | **section jump 2, widely held to be the steepest**: ToHit 60-100, HP nearly doubled, no corridor chokepoints | jump ★★★ (data), terrain ★★ |
| 10 Caves | 13 | 65-85 | ≥26 | 210-250 | Slayer ToHit 100; Obsidian charge 20-50; character-panel to-hit should be ≥150% (GOG) | ★★ |
| 11 Caves | 14 | 70-90 | ≥25 | 220-260 | Guardian ToHit 110; snakes with fast charges | ★★ |
| 12 Caves | 15 | 80-100 | ≥27 | 240-280 | first succubi (blood star = magic damage) and black knights; magic resistance starts to matter | ★★ |
| 13 Hell | **17** | 90-110 (see the AC note on its value) | ≥29 | 260-300; fire resistance ≥50 | section jump 3: Balrog 22-30 / ToHit 130; blood knight 25-35 / AC 85 | ★★ |
| 14 Hell | 18 | ≤120 (diminishing returns) | ≥31 | 280-320 | **enforced to-hit floor of 20%**: however high the AC, 20% of hits land | floor mechanics ★★★ |
| 15 Hell | 20 | ≤130 | ≥33 | 300-350; fire resistance 75 + magic resistance 50 | floor 25%; Lazarus's lair (must fight, see hard gates) | ★★★/★★ |
| 16 Hell | **22** (natural community pace 24-26) | 110-130 cap | ≥33 (hit recovery on blood knights) | 350-400+; fire resistance 75 | **to-hit floor 30%**; Advocates casting fireballs from off screen; Diablo | ★★★/★★ |

**AC note** (★★★): monster to-hit = 30+ToHit+2(mlvl−clvl)−AC−Dex/5, with a floor of 15% (enforced
20/25/30% on d14/15/16). So the marginal value of AC drops sharply after d13; the community rule of thumb is
"in Hell, AC below 100 might as well be zero". Our Dex is always 20 (no points added), so the Dex/5 term is
always 4, and block in Hell is only about 40-50%: **HP and potions are our only reliable defence in Hell,
so the HP threshold should take the upper end of the table, or even 15% above it**.

---

## 2. Draft power formula power(raw)

### 2.1 Observability mapping (based on the report 4 survey)

| Guide quantity | raw field | Notes / error |
|---|---|---|
| clvl | `char_level` | directly available ★★★ |
| HP | `max_hp` | directly available (the manager currently sees only the HP ratio, not the absolute value) |
| AC | `armor_class` | GetArmor = gear AC + bonuses + Dex/5; a constant 4 points above the guides' "gear AC" definition, negligible |
| Maximum damage per hit | `item_max_damage + damage_mod` (+ `item_bonus_damage` if it is an additive affix) | **the meaning of damage_mod (whether it includes the character bonus Str×clvl/100) needs a one-frame probe to verify**; the hit-recovery check uses this sum |
| Expected output | `((item_min+item_max)/2 + damage_mod) × melee_to_hit/100` | melee_to_hit is the panel value, before monster AC |
| Three resistances | `fire/lightning/magic_resist` | directly available |
| Block | `block_chance`, `block_enabled` | directly available |
| Overall combat scalar | `gear_combat_utility` | the project's designated source of truth for gear combat power (uint32), strictly monotone on upgrades; **its typical distribution has not been sampled, so the normalising denominator is open** |

Conclusion: all four threshold quantities are **available in raw, none needs a proxy**; the only proxy risk
is the meaning of damage_mod, which one probe can pin down.

### 2.2 Formula

```
power_ratio(raw, d) = min(
    char_level                        / C(d),
    max_hp                            / H(d),
    min(armor_class, AC_CAP(d))       / A(d),      # AC_CAP: no cap for d≤13; for d≥14 capped at A(d)
    (item_max_damage + damage_mod)    / D(d),
)
# from d≥13 add resistance gates: fire_resist ≥ FR(d)  (FR: 50 on level 13, 75 on levels 15/16; level 15 also requires magic_resist≥50)
release criterion: power_ratio ≥ 1 and the resistance gates pass
```

C/H/A/D are the **lower bounds** of the columns of the table in section 1. The minimum, not a weighted sum:
readiness is weakest-link logic (enough HP but not enough damage = no hit recovery = mobbed; enough damage
but not enough HP = one wave kills you), so any term <1 means not ready.

**Role of gear_combat_utility**: a fifth, soft signal (it already contains expected throughput, AC×1024,
the three resistances ×512, block and a maxHP component, a ready-made composite score on the engine side),
but since its scale has not been sampled empirically, **the first version uses it only for monotonicity
monitoring and as a stall-clock signal, not as a hard gate**; first collect its distribution from recent
trajectories, then set G(d).

### 2.3 Implementation path (pure Python, no engine change)

In the extras of `options_env.py _mgr_obs`, append normalised scalars from `self.env._raw`: `max_hp/400`,
`(item_max_damage+damage_mod)/40`, `fire_resist/100`, `magic_resist/100`, `gear_combat_utility` hi/lo
(÷65536, the same definition as the controller wire), and the ready-made `power_ratio` clipped to [0,2]/2.
**Constraint**: the legacy-v3 view of M29 is frozen (dim286 has a bit-level assertion), so new dimensions
can only be appended after 295 or be raw-v4-only, and the meaning of the existing 295 dimensions must not
change (report 4).

---

## 3. Hard-gate nodes (four fights)

| Node | Location | Numbers (single player, Normal) | Decision | Constraints specific to us |
|---|---|---|---|---|
| **Butcher** | dlvl2 (absent from the whole game with probability 1/3) | HP 110, 6-12 per hit, ToHit 50, cannot open doors | **Go around.** Skip whenever clvl<6 or HP<110; once qualified (clvl 6-8 + a belt full of red potions) he can be fought, and the Butcher's Cleaver he drops is an overpowered weapon for dlvl2 | A warrior has no Fire Wall, so the close-the-door-and-burn trick is out; fighting head-on or across a fence (we have no bow strategy) → skip by default, and leave the cleaver for qualified episodes |
| **Skeleton King** | dlvl3 side cave (appears only 50% of the time in single player) | HP 120, 6-16 per hit, AC 70, immune to magic, opens doors, endlessly revives small skeletons | **Go around.** Community threshold AC≥35 + damage per hit ≥20 (FuriousPaul); the standard solution is Holy Bolt | **Our mana is always 10, so we can never learn Holy Bolt**; AC 70 means a very low hit rate, and he revives minions: pure melee is very poor value, so skip him always (the crown he drops is not needed) |
| **Lazarus** | dlvl15 lair | HP 300, 30-50 per hit, immune to magic + fire and lightning resistant, teleport + fireballs; with Red Vex/Blackjade (HP 200 each, blood star magic damage) + a pack of Hell Spawn | **Must fight**: without killing him the pentagram from 15 to 16 does not open. Threshold clvl≈22-26 (reports 2/3 say 19-23 and 24-27; take the overlap), fire resistance 75, magic resistance ≥50, HP ≥320 | The quest chain is a multi-step macro process: pick up the staff → return to town and give it to Cain → the red portal → read two books and teleport through three sections → kill. **Whether the current action/skill system covers "pick up a specific quest item + return to town and talk" needs its own survey**; besides the numbers, this is the biggest engineering risk on the road to completion |
| **Diablo** | dlvl16 | HP 833, AC 90, ToHit 220, melee 30-60 + knockback; ranged Apocalypse: a fixed 40-point physical check, **all three resistances useless**, can be blocked / missed (30% minimum hit rate) | **Must fight.** Threshold clvl≥26, HP 350-400+, fire resistance 75 (for clearing the level), a belt full of red potions | His to-hit ≈250−AC: AC below 235 means nothing, so **give up stacking AC; the criterion looks only at HP + potions + block**; pull the 4 levers first, clear every Advocate on the level, then fight him alone. We have no Mana Shield / Stone Curse, so fighting head-on is the only way, and the HP threshold should be 400+ |

---

## 4. Comparison with the current SMART coach (level-margin-1)

Current release rule: clvl ≥ dlvl+1. Against C(d) of this table:

| Levels | clvl at coach release | Readiness table requirement | Gap | Measured evidence |
|---|---|---|---|---|
| d1 | 2 | 1 | meets it | h(1)=0.16, acceptable |
| d2 | 3 | 3 (and HP≥90, damage per hit ≥8) | **clvl barely, gear/HP not checked at all** | **h(2)=0.55**: more than half die on level 2; margin-1 nominally qualifies but lets an unequipped character into the Butcher's level |
| d3-4 | 4-5 | 5-8 | **-1 to -3 levels** | **h(3)=0.67**: released clearly too early |
| d5-8 Catacombs | 6-9 | 8-13 | -2 to -4 levels | (no per-level h data yet; the best arm reached l3+ only 19-23/64) |
| d9-12 Caves | 10-13 | 12-17 | -2 to -4 levels, and level 9 is the steepest jump | — |
| d13-16 Hell | 14-17 | 17-26 | **-3 to -9 levels** | — |

Conclusions: (1) the community never had a "clvl=2×dlvl" formula, but the actual pace translates into a
**level margin of +3 to +5 that grows with depth**, so margin-1 releases systematically too early from d3
on; (2) the more fundamental defect is **a single dimension**: h(2)=0.55 happens where the margin is
nominally met, which shows the deaths come from weak links in HP/gear/damage rather than level, and a pure
clvl gate cannot block them; (3) suggestion: upgrade the coach to the multi-dimensional `power_ratio ≥ 1`
gate of section 2 (its clvl term already contains a margin that grows with depth), keeping margin-1 only as
a lower-bound backstop. Also suggested: evaluate the v20 automatic stat allocation. 3 vitality : 2 strength
keeps Dex at 20, leaving gaps in both to-hit and block in Hell (the community standard for panel to-hit of
150-170% can only come from affixes); if necessary change it to a mix of vitality:strength:dexterity. That
changes the MDP and has to go through the protocol process.

---

## 5. Implementation note: the structural conflict between the 3000-micro-step protocol and completing the game

- **Current state** (report 4): PROTOCOL_MAX_STEPS=3000 is pinned at train/eval_contract.py:39; the
  depth median of the last four exam sheets is only 1-2 levels, and the two r14 arms have micro_steps_mean
  2650-2715 (the budget is essentially used up). Completing 16 levels needs ~187 steps per level on
  average, while the current FARM budget on the starting level alone is 1800 steps (FARM_SCENE_CAP,
  options_env.py:67-71): **an order of magnitude apart; both the skill wall and the step budget make it
  unreachable**.
- **Implied step requirement of this readiness table**: at the table's pace (train each level fully before
  going down), a conservative estimate is 1000-1800 steps per level + town round trips + the Lazarus quest
  chain, about **30000-50000 micro-steps** in total.
- **Suggested reform**: do not change the current 3000-step exam (it anchors historical comparability and
  calibrated thresholds; eval_contract.py:33-38 says fail-closed explicitly). **Open a separate "campaign
  protocol" (campaign track)**: a new PROTOCOL_VERSION, max_steps on the order of 50000, an independent
  archive namespace. Reference points to update together (report 4 found about 20):
  eval_contract.py:39/640/790/797/947/1145, eval_assembled.py:34/629/1661-1672/1769,
  tests/test_eval_pipeline.py:273/2061-2106; hidden coupling: the calibration of FARM_SCENE_CAP=1800 as
  "a 3000-step episode leaves 1200 for DIVE" must be redone for the new budget, and the manager
  observation's level-steps/1500 normalisation (options_env.py:1933) needs review (the remaining-time
  dimension already adapts to max_steps and needs no change). The independent max_steps of the various
  run_*.py/train_ppo.py on the training side must be synchronised separately. This is a
  constitution-level change; a separate pre-registered case is suggested.

---

## 6. Data conflicts and confidence notes

1. **Butcher HP**: report 2 says 160 (320/2), reports 1/3 say 110 (220/2) per monstdat. **We use 110**
   (the source-code definition); report 2's number looks like a typo.
2. **Skeleton King HP**: report 2 says 70, reports 1/3 say 120 (240/2) per monstdat. **We use 120**.
3. **clvl before Lazarus**: report 2 (19-23 on level 15) and report 3 (24-27) differ; the table takes the
   overlap, 22-26.
4. **Every value in the HP threshold column is derived** (almost no guide gives per-level HP numbers), and
   we have no spells for self-protection, so the upper ends are suggested throughout.
5. **Items to verify with a probe**: the warrior's first-frame four stats (30/10/20/25 from engine
   defaults, not restated in the repository), the meaning of damage_mod, and the empirical distribution of
   gear_combat_utility; each can be settled with one frame / one batch of trajectories, suggested before the
   readiness gate goes live.
6. dlvl16 is generated specially (only black knights / blood knights / Advocates + Diablo); random unique
   elites on each level (2.5-3× HP) are not included in the thresholds, one more reason to take the upper
   end of each range.
