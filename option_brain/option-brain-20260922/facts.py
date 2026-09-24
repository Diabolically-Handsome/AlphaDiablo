"""Digested memory for the option brain: at most 3 short conclusion lines inserted just before "Options:".

The brain has no memory; round 4 showed it loops (talk/close the same trader, stairs up/down, swap gear back)
and paces badly, and the probe showed that a raw activity log without training makes it worse. This module
turns the game history into a few pre-digested English conclusions instead:
  a) Pace line     (dungeon): character level vs the pace of the CURRENT level and of going further down / the King.
     Town readiness (town, when the cathedral stairs are listed) takes this slot instead: ready to go down only with
     HP 75%+ and 2+ healing potions (potions only count while gold allows buying one). Character level says nothing
     in town (level 1 is always "ready" for dungeon level 1), so it is not used there.
  b) Town-visit    (town only): what is already done during the current town visit, with how many times each
     trader was opened and whether the reopenings bought or sold anything.
  c) Last-move     (only if one of the previous 1-3 decisions did it): stairs taken, shop opened/reopened/closed,
     townsperson talked to, gear taken off. The most recent event wins; a "closed" event is never shown while a
     shop window is open now.

Pure and deterministic. The facts for decision n use ONLY decisions 1..n-1 (their executed actions and outcomes,
all known when decision n is asked) plus decision n's own prompt; never decision n's result or anything later.

Normalized history record (one per executed decision):
  n            decision number
  scene_before (floor, quest)   from that decision's own prompt Location line; quest=True in the tomb
  kind, text   the chosen option (game logs) / Astra's labelled option (Astra rows)
  result       outcome reason of that decision
  scene_after  (floor, quest) or None when not known
  level_after  character level after that decision, or None when not derivable without the future
  shop         vendor code whose window was open when that decision was asked ('smith', 'healer', ...) or None
  actions      executed town/gear actions, in order: {'act': 'talk', 'who'} | {'act': 'buy'} | {'act': 'sell'}
               | {'act': 'repair'} | {'act': 'identify'} | {'act': 'dismiss', 'shop'}
               | {'act': 'equip', 'off': [names], 'on': [names]}
facts_lines() reads only scene_before and actions of the history (plus the current prompt), so the same game
gives the same facts whichever adapter built the records.

Choices beyond the brief (see tests_facts.py for each):
  - Stairs facts come from the scene chain (scene_before of k+1, or the current prompt for k = n-1); in every log
    this equals receipt.scene_after, and it works the same for Astra rows, whose last scene_after is unknown.
  - Pace thresholds (TEACHER.md round 3): dungeon level 2 from character level 3, level 3 from 4, the tomb from 4
    (peek), the Skeleton King from 6. On level 2 the line states the pace of level 2 itself and of going below it;
    on level 3 (Skeleton King mission) the pace of level 3 and, separately, of the King; in the tomb the pace of the
    tomb and of the King. Level 1 states only the pace of going below it (level 1 itself suits every level).
    No pace line on level 4+ or on level 2+ of the Butcher mission (no teacher threshold; going deeper is never
    part of that mission).
  - Only completed talks/buys/sells/repairs/identifies/closes/equips count. "Healed at Pepin" = a completed talk to
    Pepin. A trade (buy/sell/repair/identify) belongs to the trader whose window is open at that moment.
  - A townsperson's menu that the 'npc' option closes by itself is not reported as a closed shop.
  - Astra goals often chain commands (e.g. sell, sell, close, talk to Pepin); with snapshots the town/gear actions
    are every command it executed, not just the labelled first one (kind/text stay the labelled option).
  - Fact lines never mention option letters, so option_sft's shuffle augmentation keeps them valid.
Version 4 (round 9, FACTS_VERSION=4) adds engine-number lines (boss verdict, Weapon, Boss, Retreat); see its block
below. Versions 1-3 print exactly what they printed before round 9.
Version 5 (round 9b, FACTS_VERSION=5, opt-in) adds the Butcher waiting at the stairs: a 'Butcher:' memory line and a
plan in town / on dungeon level 1, a 'Butcher:' stairs line on dungeon level 2, the Until clause and two trims; see its
block after version 4. It is computed from the version-4 lines, so versions 1-4 print exactly what they printed before.
"""
import json
import math
import os
import re
from fractions import Fraction

# Version 2 (2026-09-23 04:20): the King verdict also needs potions (king1-facts/s4150012 fought him with 0
# potions after 'ready'), and on dungeon level 3 / in the tomb a weapon line says swords do half damage to the
# undead there and blunt weapons 50% more (engine: player.cpp, MonsterClass::Undead). FACTS_VERSION=1 restores
# the exact version-1 lines (the d6facts2 model was trained on them).
FACTS_VERSION = int(os.environ.get('FACTS_VERSION', '2'))
KING_LEVEL_V2 = 7
KING_POTIONS_V2 = 8
SWORD_WORDS = ('Sword', 'Sabre', 'Falchion', 'Scimitar', 'Blade', 'Claymore')
BLUNT_WORDS = ('Club', 'Mace', 'Morning Star', 'Flail', 'Maul', 'Hammer')

# Version 3 (2026-09-23 06:10, opt-in with FACTS_VERSION=3). king2-d8: heroes reached the tomb at character level 7
# with 8+ potions but all 30 attribute points in strength (dexterity 20) and a sword. Engine numbers (DevilutionX
# player.h GetMeleeToHit, player.cpp PlrHitMonst, txtdata warrior baseMeleeToHit 70, monstdat MT_SKING armour 70,
# unique HP 240 halved in single player = 120): chance to hit = character level + dexterity/2 + 70 - 70 (5..95%);
# damage per hit = weapon damage + character level * strength / 100, halved with a sword, +50% with a club/mace
# against the undead. Version 3 adds (a) a 'Hit chance' line on dungeon level 3 / in the tomb and whenever
# attribute points are unspent, (b) the swings needed to kill the King in the King verdict (ready only at
# <= 60 swings), (c) the weapon line for every weapon type and in town, (d) 10+ potions for town readiness on the
# King mission from character level 6.
V3_SWORD_WORDS = SWORD_WORDS + ('Dagger',)          # daggers are swords in the engine (ItemType::Sword)
V3_AXE_WORDS = ('Axe', 'Cleaver')
KING_HP = 120
KING_ARMOUR = 70
BUTCHER_ARMOUR = 50
WARRIOR_MELEE_TO_HIT = 70
KING_SWINGS_READY = 60
KING_TOWN_POTIONS = 10
KING_TOWN_LEVEL = 6
# Butcher (monstdat MT_CLEAVER armour 50, class Demon: no weapon-type modifier; unique HP 220 halved = 110). Astra
# killed him at character level 5 with dexterity 40, a Blade and 16 potions: about 40 swings by the formula below.
BUTCHER_HP = 110
BUTCHER_LEVEL = 5
BUTCHER_POTIONS = 8
BUTCHER_SWINGS_READY = 45
BUTCHER_TOWN_LEVEL = 4

# Version 4 (2026-09-23 round 9, opt-in with FACTS_VERSION=4; spec PLAN-annotations-v4.md sections 1, 3, 4 and
# ROADMAP-20260923.md section 2). Engine numbers replace the swing counts, because the bosses regenerate: with a sword
# the Skeleton King regenerates faster than he is hurt, so version 3's "about 170 swings" really meant "never".
# Formula sheet (PLAN section 1; SRC = devilutionX/Source, TSV = devilutionX/assets/txtdata):
#   F1  hit chance p = clamp(L + Dex/2 + visible to_hit + 70 - boss armour, 5, 95)%    player.h:573-576, player.cpp:549-555
#   F2  one hit: r = U[lo..hi]; r += r*damage%/100; += L*Str/100; x2 with chance L% (warrior critical strike);
#       then against the undead a sword does x - x/2 and a mace/club x + x/2 (a mace in either hand wins), animals the
#       reverse, demons x1                                                   player.cpp:567-606, items.cpp:2555,2596
#   F3  E = exact expectation over r and the critical strike (fractions)
#   F5  bare hands 1-1, 1-3 with a shield                                     items.cpp:2494-2514
#   F6  unidentified items show no bonuses and none apply                     items.cpp:2833-2850
#   F7  swing period in the chained attack: sword / mace / bare 9 frames, axe 10, staff 11   animations.tsv:6-19
#   F8  boss hits you h = min(100, max(15, boss toHit + 2(boss level - L) + 30 - your armour))%   monster.cpp:1162-1204
#   F9  you block blk = clamp(Dex + 30 + 2L - 2*boss level, 0, 100)% with a shield in EITHER hand   monster.cpp:1198-1207,
#       player.h:619-625, items.cpp:2656-2673 (CalcPlrBlockFlag: isHoldingItem(Shield))
#   F10 his hit stuns you when damage >= L: pst = ((hi-L)*64+1)/((hi-lo)*64+1) (1 if L <= lo)   monster.cpp:1219,
#       player.cpp:2655-2661
#   F11 an interrupted swing costs 10.5 frames (6 stun or 2x3 block + half a swing): calibration constant (PLAN G1/G3)
#   F12 he regenerates floor(boss level/2)/64 HP per frame, 20 frames per second   monster.cpp:4271-4276
#   F13/F14 boss rows (V4_BOSSES)   F15 your hit staggers him when damage >= boss level + 3   monster.cpp:988-998
#   F16 Potion of Healing: 2*(M/8 + rnd(M/4)) of max HP M, mean 2*floor(M/8) + floor(M/4) - 1   player.cpp:1743-1752
# Model (not engine constants): intr = h*(blk + (1-blk)*pst); swings per second S = max(0, 20 - A*intr*10.5)/T_sw;
# damage per second g = S*p*E; he hurts you D = A*h*(1-blk)*(lo+hi)/2 per second; T = boss HP/(g - regen);
# potions P = ceil(T*D/heal). Bands: bow or no melee weapon -> melee_first; g <= regen -> cannot; g < 1.5*regen ->
# barely (no seconds: the time is unstable near the regeneration balance); T > 120 s -> long; else ok, and ready
# only with min(P+3, 20) potions carried (boss_need_v4). Seconds are rounded up to 10.
# Version 4 lines: (a) K, the boss verdict of the Pace line (dungeon level 3 / tomb on the King mission, dungeon
# level 2 on the Butcher mission), with the best weapon in reach when it does better; (b) the Weapon line W; (c) the
# Boss line B while the Skeleton King or the Butcher is visible; (d) the Hit line counts visible to_hit; (e) the town
# potion target comes from K; (f) the Retreat line (dungeon); (g) on the Butcher mission dungeon level 3+ and the tomb
# are outside the mission; (h) money: how many of the missing potions the gold buys and where to earn the rest; (i) the
# weapon may sit in either hand and a shield in the weapon-hand slot, so the hero has "no weapon" only when neither hand
# holds a melee weapon (version 3 missed it for 488 decisions of heldout-king-v3/s4150049 and 165 of butcher1-v3/s4150029).
# 2026-09-23 review of version 4 (before any training on it): the Retreat line uses the town threshold on every dungeon
# level and waits while the mission boss is at half HP; the Boss line next to a Retreat line is short and says how often
# an adjacent boss interrupts the walk; K asks for min(P+3, 20) and the town target follows K at every level; the gold
# fact says where to earn it; the remedy follows the Weapon line's order within a band; P is pluralised; the Weapon line
# shortens itself for long magic names and yields to the town line's potions (+120-token budget, test_t11_synthetic).
# 2026-09-23 15:05 (owner decisions D1, D2): the Retreat line fires at 1 potion or fewer, or at 2 or fewer when
# surrounded, and never while a boss is adjacent (see RETREAT_POTIONS); so the Boss line's short form next to a Retreat
# line only occurs with the boss in sight but not adjacent, and no longer mentions interrupted walks.
V4_FPS = 20                                        # game frames per second (bridge diablogym.cpp:1986 tickRate 20)
V4_SWING = {'Sword': 9, 'Mace': 9, 'Axe': 10, 'Staff': 11, 'Weapon': 9}   # F7; 'Weapon' = unknown base, like a sword
V4_INTERRUPT = Fraction(21, 2)                     # F11
V4_BANDS = ('melee_first', 'cannot', 'barely', 'long', 'ok')
V4_LONG_SECONDS = 120
V4_SPARE_POTIONS = 3                               # ready: P + 3 potions carried (escorts and summons are not modelled)
V4_TOWN_MAX = 20                                   # town potion target cap
V4_WEAPON_CHARS = 240                              # above this length the Weapon line shortens itself step by step ...
V4_WEAPON_CHARS_BOSS = 200                         # ... and above this one next to the Boss line; every example of PLAN
# section 3 keeps its full text (#1248 is 158 characters with the King in sight, #604 194, #319 231); two long magic
# names had made the line 84 tokens
# monstdat.tsv:52-53, unique_monstdat.tsv:3,11; unique level +5 (monster.h:395-405), unique HP halved in single player
# (monster.cpp:210-214). The King attacks for 16 frames and starts with 17% per standing frame (monster.cpp:2385-2427,
# 3330-3334): one attack per 21 frames; the Butcher attacks every 11 frames when adjacent (monster.cpp:2511-2526,
# 4322-4330).
V4_BOSSES = {
    'king': dict(who='the Skeleton King', mlvl=14, hp=120, armour=70, to_hit=60, lo=6, hi=16, cls='undead',
                 rate=Fraction(20, 21)),
    'butcher': dict(who='the Butcher', mlvl=6, hp=110, armour=50, to_hit=50, lo=6, hi=12, cls='demon', rate=Fraction(20, 11)),
}
# itemdat.tsv: name -> (itemType, two-handed) for weapons :120-157, shields :73-78, starting and quest items :3-8, :33,
# :38. The longest base name contained in an item name wins ("Spiked Club" before "Club"); no magic prefix or suffix
# (item_prefixes.tsv, item_suffixes.tsv) contains a base name.
V4_BASES = {
    'Dagger': ('Sword', False), 'Short Sword': ('Sword', False), 'Falchion': ('Sword', False), 'Scimitar': ('Sword', False),
    'Claymore': ('Sword', False), 'Blade': ('Sword', False), 'Sabre': ('Sword', False), 'Long Sword': ('Sword', False),
    'Broad Sword': ('Sword', False), 'Bastard Sword': ('Sword', False), 'Two-Handed Sword': ('Sword', True),
    'Great Sword': ('Sword', True), "Griswold's Edge": ('Sword', False),
    'Small Axe': ('Axe', True), 'Axe': ('Axe', True), 'Large Axe': ('Axe', True), 'Broad Axe': ('Axe', True),
    'Battle Axe': ('Axe', True), 'Great Axe': ('Axe', True), 'Cleaver': ('Axe', True),
    'Mace': ('Mace', False), 'Morning Star': ('Mace', False), 'War Hammer': ('Mace', False), 'Spiked Club': ('Mace', False),
    'Club': ('Mace', False), 'Flail': ('Mace', False), 'Maul': ('Mace', True),
    'Short Bow': ('Bow', True), "Hunter's Bow": ('Bow', True), 'Long Bow': ('Bow', True), 'Composite Bow': ('Bow', True),
    'Short Battle Bow': ('Bow', True), 'Long Battle Bow': ('Bow', True), 'Short War Bow': ('Bow', True),
    'Long War Bow': ('Bow', True),
    'Short Staff': ('Staff', True), 'Long Staff': ('Staff', True), 'Composite Staff': ('Staff', True),
    'Quarter Staff': ('Staff', True), 'War Staff': ('Staff', True),
    'Buckler': ('Shield', False), 'Small Shield': ('Shield', False), 'Large Shield': ('Shield', False),
    'Kite Shield': ('Shield', False), 'Tower Shield': ('Shield', False), 'Gothic Shield': ('Shield', False),
}
# unique_itemdat.tsv:2,10,12-64,81-86: identified unique weapons and shields -> base (through itemdat's uniqueBaseItem).
V4_UNIQUE_BASES = {
    "The Butcher's Cleaver": 'Cleaver', "Griswold's Edge": "Griswold's Edge", 'The Rift Bow': 'Short Bow',
    'The Needler': 'Short Bow', 'The Celestial Bow': 'Long Bow', 'Deadly Hunter': 'Composite Bow',
    'Bow of the Dead': 'Composite Bow', 'The Blackoak Bow': 'Long Bow', 'Flamedart': "Hunter's Bow", 'Fleshstinger': 'Long Bow',
    'Windforce': 'Long War Bow', 'Eaglehorn': 'Long Battle Bow', "Gonnagal's Dirk": 'Dagger', 'The Defender': 'Sabre',
    "Gryphon's Claw": 'Falchion', 'Black Razor': 'Dagger', 'Gibbous Moon': 'Broad Sword', 'Ice Shank': 'Long Sword',
    "The Executioner's Blade": 'Falchion', 'The Bonesaw': 'Claymore', 'Shadowhawk': 'Broad Sword', 'Wizardspike': 'Dagger',
    'Lightsabre': 'Sabre', "The Falcon's Talon": 'Scimitar', 'Inferno': 'Long Sword', 'Doombringer': 'Bastard Sword',
    'The Grizzly': 'Two-Handed Sword', 'The Grandfather': 'Great Sword', 'The Mangler': 'Large Axe', 'Sharp Beak': 'Large Axe',
    'Bloodslayer': 'Broad Axe', 'The Celestial Axe': 'Battle Axe', 'Wicked Axe': 'Large Axe', 'Stonecleaver': 'Broad Axe',
    "Aguinara's Hatchet": 'Small Axe', 'Hellslayer': 'Battle Axe', "Messerschmidt's Reaver": 'Great Axe', 'Crackrust': 'Mace',
    'Hammer of Jholm': 'Maul', "Civerb's Cudgel": 'Mace', 'The Celestial Star': 'Flail', "Baranar's Star": 'Morning Star',
    'Gnarled Root': 'Spiked Club', 'The Cranium Basher': 'Maul', "Schaefer's Hammer": 'War Hammer', 'Dreamflange': 'Mace',
    'Staff of Shadows': 'Long Staff', 'Immolator': 'Long Staff', 'Storm Spire': 'War Staff', 'Gleamsong': 'Short Staff',
    'Thundercall': 'Composite Staff', 'The Protector': 'Short Staff', "Naj's Puzzler": 'Long Staff', 'Mindcry': 'Quarter Staff',
    'Rod of Onan': 'War Staff', 'The Deflector': 'Buckler', 'Split Skull Shield': 'Buckler', "Dragon's Breach": 'Kite Shield',
    'Blackoak Shield': 'Small Shield', 'Holy Defender': 'Large Shield', 'Stormshield': 'Gothic Shield',
}
# Retreat line (dungeon). FAILURE-CENSUS-20260923 section 2 and the round-8 logs (37,402 dungeon decisions, 50 games):
# all 6 deaths came after the potions fell to 2 or fewer on dungeon level 2/3; 4+ enemies adjacent cost 4.9 HP per
# second (1.4 with one) and, with 2 or fewer potions, 27% of such decisions were within 20 decisions of death (0.6%
# with 3+ potions: then the hero drinks and survives).
# Thresholds of 2026-09-23 15:05 (owner decisions D1 and D2, REPORT.md; they replace the 14:34 version, which fired at
# "fewer potions than the town asks for", i.e. 2 or fewer from character level 3, and so fired on dungeon level 1 for
# every hero who had drunk one potion there: the town and the dungeon sent the hero back and forth):
#   potion trigger: RETREAT_POTIONS = 1 or fewer healing potions (belt + pack), at every character level, on every
#     dungeon level and in the tomb. The town asks for 3 from character level 3 and 2 before (town_base_v4), so a hero
#     the town lets go down never meets it, a hero who meets it on dungeon level 2 still meets it on level 1, and from
#     character level 3 a whole spare potion lies between the two numbers. With gold for no potion it needs HP below
#     50% (Pepin heals for free); it stays off while the mission's boss is in sight at half his HP or less and a potion
#     is left (TEACHER.md, Butcher fight: left alone he regenerates to full).
#   crowd trigger: RETREAT_CROWD_POTIONS = 2 or fewer potions and "surrounded": RETREAT_ADJACENT = 4+ enemies adjacent
#     (steps 0 on the Visible enemies line) or RETREAT_NEAR = 8+ listed enemies within RETREAT_NEAR_STEPS = 5 steps
#     (adjacent ones included). Any gold, any HP. With 3+ potions a crowd is fought, not fled.
#   D2: no Retreat line while the Skeleton King or the Butcher is adjacent (adjacent_boss_v4): the Boss line and the
#     Pace verdict decide there, as TEACHER.md's 'The Butcher next to the hero' does (walking away from an adjacent
#     boss seldom works: round 8 chose the stairs 899 times with the Butcher adjacent and got through in 31%, and
#     heldout2-butcher-v3/s4150046 died running from him).
# With a boss in sight but not adjacent the Retreat line leaves the gold to the town line (+120-token budget).
RETREAT_POTIONS = 1
RETREAT_CROWD_POTIONS = 2
RETREAT_ADJACENT = 4
RETREAT_NEAR = 8
RETREAT_NEAR_STEPS = 5
RETREAT_HP_PCT = 50
TOWN_POTIONS_V4 = 3                                # from character level ENTER_LEVEL[2] = 3 (dungeon level 2 on pace)
_STATS = re.compile(r'unspent attribute points (\d+)\. Strength (\d+), dexterity (\d+)')
_WEAPON = re.compile(r'^Equipped: .*?weapon hand: ([^(;]+?) \(damage (\d+)-(\d+)', re.M)

TOWN = (0, False)
LOOKBACK = 3                      # last-move line looks at the previous 1-3 decisions
DESCENT_LEVEL = {1: 3, 2: 4}      # TEACHER.md round-3 pacing: leave level 1 at char level 3+, level 2 at 4+
ENTER_LEVEL = {2: 3, 3: 4}        # ... i.e. be on level 2 from char level 3+, on level 3 from 4+
TOMB_LEVEL = 4                    # peek into the tomb at 4+
KING_LEVEL = 6                    # fight the Skeleton King at char level 6+
TOWN_HP_PCT = 75                  # town readiness: HP 75%+ (Pepin heals for free) ...
TOWN_POTIONS = 2                  # ... and 2+ healing potions (the starting kit), unless gold cannot buy one
POTION_PRICE = 50                 # Potion of Healing at Pepin
VENDOR_NAME = {'smith': 'Griswold', 'healer': 'Pepin', 'witch': 'Adria', 'cain': 'Deckard Cain'}
VENDORS = frozenset(VENDOR_NAME.values())
SHOP_PHRASE = {'smith': "Griswold's shop", 'healer': "Pepin's shop", 'witch': "Adria's shop", 'cain': "Deckard Cain's window"}
SHOP_OF = {VENDOR_NAME[k]: v for k, v in SHOP_PHRASE.items()}
NPC_KIND_NAME = dict(VENDOR_NAME, other='a townsperson')
NPC_TEXT_KEYS = (('Griswold', 'Griswold'), ('Pepin', 'Pepin'), ('Adria', 'Adria'), ('Deckard Cain', 'Deckard Cain'),
                 ('townsperson', 'a townsperson'))
TRADES = ('buy', 'sell', 'repair', 'identify')
FACT_PREFIXES = ('Pace: ', 'Ready to go down: ', 'Already done this town visit: ', 'You just ', 'Weapon: ', 'Hit chance: ',
                 'Boss: ', 'Retreat: ', 'Butcher: ')   # 'Butcher: ' (version 5) for every version: no logged or training
# prompt has a line starting with it (tests_facts_v5.test_prefix_scan), so strip_facts of versions 1-4 is unchanged
OPTIONS_MARK = '\nOptions:\n'

_LOC = re.compile(r'^Location: (.*)$', re.M)
_HERO = re.compile(r'^Hero: level (\d+),', re.M)
_VITALS = re.compile(r'^Hero: level \d+, HP \d+/\d+ \((\d+)%\), gold (\d+),.*?healing potions (\d+) in belt and (\d+) in pack', re.M)
_MISSION = re.compile(r'^Mission: (.*)$', re.M)
_SHOP = re.compile(r'^A shop window is open \((.*)\)\.$', re.M)
_OPTION = re.compile(r'^[A-Za-z]\. (.*)$')


# ----------------------------------------------------------------------------------------------- prompt parsing
def _head(prompt):
    i = prompt.find(OPTIONS_MARK)
    if i < 0:
        raise ValueError('prompt has no "Options:" line')
    return prompt[:i]


def parse_scene(prompt):
    """(floor, quest) from the Location line: town (0, False), dungeon level D (D, False), tomb (D, True)."""
    m = _LOC.search(_head(prompt))
    if not m:
        raise ValueError('prompt has no Location line')
    loc = m.group(1)
    if loc.startswith('Town'):
        return TOWN
    q = re.match(r'Quest area below dungeon level (\d+)\b', loc)
    if q:
        return (int(q.group(1)), True)
    d = re.match(r'Dungeon level (\d+)\.', loc)
    if d:
        return (int(d.group(1)), False)
    raise ValueError('unknown Location line: ' + loc)


def parse_level(prompt):
    m = _HERO.search(_head(prompt))
    if not m:
        raise ValueError('prompt has no Hero line')
    return int(m.group(1))


def parse_vitals(prompt):
    """(HP percent, gold, healing potions in belt + pack) from the Hero line."""
    m = _VITALS.search(_head(prompt))
    if not m:
        raise ValueError('prompt has no parsable Hero HP/gold/potions')
    pct, gold, belt, pack = map(int, m.groups())
    return pct, gold, belt + pack


def parse_mission(prompt):
    m = _MISSION.search(_head(prompt))
    text = m.group(1) if m else ''
    if 'Skeleton King' in text:
        return 'skeleton_king'
    if 'Butcher' in text:
        return 'butcher'
    raise ValueError('unknown mission line: ' + text)


def parse_shop(prompt):
    m = _SHOP.search(_head(prompt))
    return m.group(1) if m else None


def parse_option_texts(prompt):
    i = prompt.find(OPTIONS_MARK)
    if i < 0:
        raise ValueError('prompt has no "Options:" line')
    out = []
    for line in prompt[i + len(OPTIONS_MARK):].split('\n'):
        m = _OPTION.match(line)
        if m:
            out.append(m.group(1))
    return out


def _scene_from_list(scene):
    return (int(scene[0]), bool(scene[1]))


def _depth(scene):
    floor, quest = scene
    return floor + (0.5 if quest else 0)


def _scene_name(scene):
    floor, quest = scene
    if floor == 0:
        return 'town'
    if quest:
        return "the Skeleton King's tomb"
    return f'dungeon level {floor}'


# ----------------------------------------------------------------------------------------------- actions
def npc_from_text(text):
    for key, name in NPC_TEXT_KEYS:
        if key in text:
            return name
    return 'a townsperson'


def parse_equip(text):
    """'Equip X in SLOT (currently Y)' -> (off names, on names). Y may be 'nothing' or 'A; B'."""
    idx = text.rfind(' (currently ')
    if not text.startswith('Equip ') or idx < 0 or not text.endswith(')'):
        return [], []
    on = text[len('Equip '):idx].rsplit(' in ', 1)[0]
    now = text[idx + len(' (currently '):-1]
    off = [] if now == 'nothing' else now.split('; ')
    return off, [on]


def actions_from_option(kind, text, result, shop):
    """Actions of one executed option (game logs, and Astra rows without a snapshot). Only completed ones count."""
    if result != 'completed' or not kind:
        return []
    if kind == 'npc':
        return [dict(act='talk', who=npc_from_text(text or ''))]
    if kind in TRADES:
        return [dict(act=kind)]
    if kind == 'dismiss':
        return [dict(act='dismiss', shop=shop)]
    if kind == 'equip':
        off, on = parse_equip(text or '')
        return [dict(act='equip', off=off, on=on)]
    return []


def actions_from_astra_snapshot(snap):
    """Every command Astra's goal actually executed (a goal can chain talk, sells, buys, close, equips).

    Executed = all commands if the goal completed, else those before result.service_index (same rule as
    relabel.py uses for operated objects). Equipment changes come from the state diff before/after the goal.
    """
    goal = snap.get('goal') or {}
    if goal.get('mode') != 'service':
        return []
    res = snap.get('result') or {}
    reason, si = res.get('reason'), res.get('service_index')
    done = [c for i, c in enumerate(goal.get('commands', [])) if reason == 'completed' or (si is not None and i < si)]
    npcs = dict(snap.get('npc_kinds') or {})
    shop = snap.get('vendor') if snap.get('vendor') not in (None, '', 'none') else None
    acts = []
    after_talk_other = False
    equip_at = None
    for c in done:
        k = c.get('kind')
        if k == 'move':
            continue
        if k == 'talk':
            kind = npcs.get(c['args']['id'], 'other')
            acts.append(dict(act='talk', who=NPC_KIND_NAME.get(kind, 'a townsperson')))
            shop = kind if kind in VENDOR_NAME else None
            after_talk_other = kind not in VENDOR_NAME
            continue
        if k == 'dismiss':
            if not after_talk_other:  # the runner closes a townsperson's menu inside the same 'npc' option
                acts.append(dict(act='dismiss', shop=shop))
            shop = None
        elif k in TRADES:
            acts.append(dict(act=k))
        elif k == 'equip' and equip_at is None:
            equip_at = len(acts)
            acts.append(None)
        after_talk_other = False
    if equip_at is not None:
        off, on = snap.get('equip_diff') or ([], [])
        acts[equip_at] = dict(act='equip', off=list(off), on=list(on))
    return acts


# ----------------------------------------------------------------------------------------------- adapters
def record_from_game_row(row):
    """One normalized record from a game-log / live-runner decision row (uses nothing from later rows)."""
    prompt = row['prompt']
    shop = parse_shop(prompt)
    receipt = row.get('receipt') or {}
    after = row.get('after') or {}
    sa = receipt.get('scene_after')
    rec = dict(n=int(row['n']), scene_before=parse_scene(prompt), kind=row.get('kind'), text=row.get('text'),
               result=row.get('result'), scene_after=_scene_from_list(sa) if sa is not None else None,
               level_after=after.get('level'), shop=shop,
               actions=actions_from_option(row.get('kind'), row.get('text'), row.get('result'), shop))
    if FACTS_VERSION >= 5:
        rec.update(sighting_v5(prompt))   # version 5: b5, up5, ex5 of that decision's own prompt
    return rec


def history_from_game_log(rows):
    """rows = decisions 1..n-1 of one game log (decisions.jsonl rows, in order)."""
    out = []
    for i, r in enumerate(rows):
        if int(r['n']) != i + 1:
            raise ValueError(f'game log rows must be decisions 1..k in order; got n={r["n"]} at position {i}')
        out.append(record_from_game_row(r))
    return out


def _labelled(row):
    if not row.get('label'):
        return None
    return next((o for o in row['options'] if o['label'] == row['label']), None)


def history_from_astra(rows, snapshots=None):
    """rows = Astra relabel rows of ONE world with number < current, sorted by number.

    kind/text = the labelled option of the row (what Astra did, as the option menu expresses it).
    With snapshots ({number: slim snapshot}, see load_astra_snapshots) the town/gear actions come from every
    command Astra's goal really executed; without them only from the labelled option.
    """
    out = []
    for i, r in enumerate(rows):
        if i and int(r['number']) != int(rows[i - 1]['number']) + 1:
            raise ValueError('Astra rows must be consecutive decisions in number order')
        opt = _labelled(r)
        kind, text = (opt['kind'], opt['text']) if opt else (None, None)
        shop = parse_shop(r['prompt'])
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        snap = (snapshots or {}).get(int(r['number']))
        acts = actions_from_astra_snapshot(snap) if snap is not None else actions_from_option(kind, text, r.get('result_reason'), shop)
        out.append(dict(n=int(r['number']), scene_before=parse_scene(r['prompt']), kind=kind, text=text,
                        result=r.get('result_reason'), scene_after=parse_scene(nxt['prompt']) if nxt else None,
                        level_after=parse_level(nxt['prompt']) if nxt else None, shop=shop, actions=acts))
        if FACTS_VERSION >= 5:
            out[-1].update(sighting_v5(r['prompt']))   # version 5: b5, up5, ex5 of that row's own prompt
    return out


def load_astra_snapshots(path):
    """Slim view of strategy-brain-repair snapshots.jsonl: {number: {goal, result, vendor, npc_kinds, equip_diff}}.

    Everything kept is known right after that decision executed (its goal, outcome and resulting equipment).
    """
    from options import item_name  # local import: facts.py itself needs no game code
    out = {}
    with open(path) as fh:
        for line in fh:
            s = json.loads(line)
            before = s['before']
            res = s.get('result') or {}
            kinds = {int(n['id']): n.get('kind') for n in (s.get('known') or {}).get('0,0', {}).get('npcs_memory', {}).values()}
            kinds.update({int(n['id']): n.get('kind') for n in before.get('npcs', [])})
            diff = None
            st = res.get('state')
            if st is not None:
                worn_b = [x for x in before['equipment'] if not x['empty']]
                worn_a = [x for x in st['equipment'] if not x['empty']]
                ids_b = {tuple(x['identity']) for x in worn_b}
                ids_a = {tuple(x['identity']) for x in worn_a}
                pack_a = {tuple(x['identity']) for x in st['inventory'] if not x['empty']}
                diff = ([item_name(x) for x in worn_b if tuple(x['identity']) not in ids_a and tuple(x['identity']) in pack_a],
                        [item_name(x) for x in worn_a if tuple(x['identity']) not in ids_b])
            out[int(s['number'])] = dict(goal=s.get('goal'), result=dict(reason=res.get('reason'), service_index=res.get('service_index')),
                                         vendor=before.get('vendor'), npc_kinds=kinds, equip_diff=diff)
    return out


# ----------------------------------------------------------------------------------------------- facts
def _plural(k, word):
    return f'{k} {word}' if k == 1 else f'{k} {word}s'


def _ordinal(k):
    suffix = 'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')
    return f'{k}{suffix}'


def _here(level, what, need):
    return f"{what} is recommended from character level {need}: you are {'on' if level >= need else 'below'} pace here."


def _next(level, what, need):
    return f"{what} is recommended from character level {need}: {'ready' if level >= need else 'not ready yet'}."


def parse_stats(prompt):
    """(unspent points, strength, dexterity) from the Hero line, or None."""
    m = _STATS.search(_head(prompt))
    return tuple(int(x) for x in m.groups()) if m else None


def parse_weapon(prompt):
    """(name, min damage, max damage) of the equipped weapon, or None."""
    m = _WEAPON.search(_head(prompt))
    return (m.group(1).strip(), int(m.group(2)), int(m.group(3))) if m else None


def weapon_class(name):
    if any(w in name for w in BLUNT_WORDS):
        return 'blunt'
    if any(w in name for w in V3_SWORD_WORDS):
        return 'sword'
    if any(w in name for w in V3_AXE_WORDS):
        return 'axe'
    return None


def hit_pct(level, dex, armour=KING_ARMOUR, to_hit=0):
    return max(5, min(95, level + dex // 2 + to_hit + WARRIOR_MELEE_TO_HIT - armour))


def king_swings(level, strength, dex, weapon):
    """Expected swings to kill the Skeleton King (120 HP) with the equipped weapon, or None without one."""
    if weapon is None:
        return None
    name, lo, hi = weapon
    per_hit = (lo + hi) / 2 + level * strength // 100
    per_hit *= {'sword': 0.5, 'blunt': 1.5}.get(weapon_class(name), 1.0)
    n = KING_HP / (hit_pct(level, dex) / 100 * per_hit)
    return int(round(n)) if n < 20 else int(5 * round(n / 5)) if n <= 100 else int(10 * round(n / 10))


def butcher_swings(level, strength, dex, weapon):
    if weapon is None:
        return None
    name, lo, hi = weapon
    per_hit = (lo + hi) / 2 + level * strength // 100
    n = BUTCHER_HP / (hit_pct(level, dex, BUTCHER_ARMOUR) / 100 * per_hit)
    return int(round(n)) if n < 20 else int(5 * round(n / 5)) if n <= 100 else int(10 * round(n / 10))


def _butcher_v3(level, potions, prompt):
    st, wp = parse_stats(prompt), parse_weapon(prompt)
    swings = butcher_swings(level, st[1], st[2], wp) if st else None
    ok = level >= BUTCHER_LEVEL and potions >= BUTCHER_POTIONS and swings is not None and swings <= BUTCHER_SWINGS_READY
    missing = []
    if level < BUTCHER_LEVEL:
        missing.append(f'character level {level}')
    if potions < BUTCHER_POTIONS:
        missing.append(_plural(potions, 'healing potion'))
    with_ = f'about {swings} swings with your {wp[0]}' if swings is not None else 'no weapon'
    if swings is None or swings > BUTCHER_SWINGS_READY:
        missing.append(with_)
    head = (f"Fighting the Butcher (on this level) is recommended from character level {BUTCHER_LEVEL} with {BUTCHER_POTIONS}+ "
            f"healing potions and a weapon that kills him in {BUTCHER_SWINGS_READY} swings or fewer: ")
    return head + (f'ready ({with_}).' if ok else f"not ready yet ({'; '.join(missing)}).")


def _king_v3(level, potions, what, prompt):
    st, wp = parse_stats(prompt), parse_weapon(prompt)
    swings = king_swings(level, st[1], st[2], wp) if st else None
    ok = level >= KING_LEVEL_V2 and potions >= KING_POTIONS_V2 and swings is not None and swings <= KING_SWINGS_READY
    missing = []
    if level < KING_LEVEL_V2:
        missing.append(f'character level {level}')
    if potions < KING_POTIONS_V2:
        missing.append(_plural(potions, 'healing potion'))
    with_ = f'about {swings} swings with your {wp[0]}' if swings is not None else 'no weapon'
    if swings is None or swings > KING_SWINGS_READY:
        missing.append(with_)
    head = (f"{what} is recommended from character level {KING_LEVEL_V2} with {KING_POTIONS_V2}+ healing potions and a "
            f"weapon that kills him in {KING_SWINGS_READY} swings or fewer: ")
    return head + (f'ready ({with_}).' if ok else f"not ready yet ({'; '.join(missing)}).")


def _king(level, potions, what, prompt=None):
    if FACTS_VERSION < 2:
        return _next(level, what, KING_LEVEL)
    if FACTS_VERSION >= 4 and prompt is not None:
        return verdict_v4(prompt, 'king', what)
    if FACTS_VERSION >= 3 and prompt is not None:
        return _king_v3(level, potions, what, prompt)
    ok = level >= KING_LEVEL_V2 and potions >= KING_POTIONS_V2
    missing = []
    if level < KING_LEVEL_V2:
        missing.append(f'character level {level}')
    if potions < KING_POTIONS_V2:
        missing.append(_plural(potions, 'healing potion'))
    return (f"{what} is recommended from character level {KING_LEVEL_V2} with {KING_POTIONS_V2}+ healing potions: "
            + ('ready.' if ok else f"not ready yet ({', '.join(missing)})."))


def pace_line(scene, level, mission, potions=0, prompt=None):
    """Dungeon only: the pace of the current level (level 2+, the tomb) and of the next step (going below / the King)."""
    floor, quest = scene
    head = f'Pace: character level {level}.'
    if floor == 0:
        return None
    if FACTS_VERSION >= 4 and prompt is not None and mission == 'butcher' and (quest or floor >= 3):
        return ' '.join([head, outside_mission_v4(prompt, scene)])  # version 4 (g): before the tomb's King verdict
    if quest:
        return ' '.join([head, _here(level, "You are in the Skeleton King's tomb. The tomb", TOMB_LEVEL),
                         _king(level, potions, 'Fighting him', prompt)])
    if mission == 'skeleton_king' and floor == 3:
        return ' '.join([head, _here(level, 'Dungeon level 3', ENTER_LEVEL[3]),
                         _king(level, potions, 'Fighting the Skeleton King (his tomb entrance is on this level)', prompt)])
    if mission == 'butcher' and floor == 2 and FACTS_VERSION >= 4 and prompt is not None:
        return ' '.join([head, verdict_v4(prompt, 'butcher', 'Fighting the Butcher (on this level)')])
    if mission == 'butcher' and floor == 2 and FACTS_VERSION >= 3 and prompt is not None:
        return ' '.join([head, _butcher_v3(level, potions, prompt)])
    if mission == 'butcher' and floor >= 2:
        return None  # the Butcher lives on level 2: going deeper is never part of this mission
    need = DESCENT_LEVEL.get(floor)
    if need is None:
        return None
    if floor == 1:  # level 1 itself suits every character level
        return ' '.join([head, _next(level, 'Going below dungeon level 1', need)])
    return ' '.join([head, _here(level, f'Dungeon level {floor}', ENTER_LEVEL[floor]), _next(level, 'Going below it', need)])


def town_ready_line(prompt, option_texts):
    """Town, when the cathedral stairs are listed: ready to go down = HP 75%+ and 2+ healing potions.

    Potions only block while gold can buy one (otherwise the hero could never leave town).
    Version 3: on the Skeleton King mission from character level 6 the potion target is 10+.
    """
    if not any('stairs DOWN' in t for t in option_texts):
        return None
    if FACTS_VERSION >= 4:
        return town_ready_v4(prompt)
    pct, gold, pots = parse_vitals(prompt)
    need = TOWN_POTIONS
    if FACTS_VERSION >= 3 and parse_mission(prompt) == 'skeleton_king' and parse_level(prompt) >= KING_TOWN_LEVEL:
        need = KING_TOWN_POTIONS
    elif FACTS_VERSION >= 3 and parse_mission(prompt) == 'butcher' and parse_level(prompt) >= BUTCHER_TOWN_LEVEL:
        need = BUTCHER_POTIONS
    hp_ok = pct >= TOWN_HP_PCT
    can_buy = gold >= POTION_PRICE
    pots_ok = pots >= need or not can_buy
    have = _plural(pots, 'healing potion')
    if hp_ok and pots_ok:
        if pots >= need:
            return f'Ready to go down: yes (HP {pct}%, {have}).'
        return f'Ready to go down: yes (HP {pct}%; only {have}, and gold {gold} cannot buy another).'
    why, help_ = [], []
    if not hp_ok:
        why.append(f'HP {pct}%, needs {TOWN_HP_PCT}%+')
        help_.append('heals for free')
    if not pots_ok:
        boss = {KING_TOWN_POTIONS: ' for the Skeleton King', BUTCHER_POTIONS: ' for the Butcher'}.get(need, '') if need != TOWN_POTIONS else ''
        why.append(f'{have}, needs {need}+{boss}')
        help_.append(f'sells potions for {POTION_PRICE} gold')
    return f"Ready to go down: not yet ({'; '.join(why)}). Pepin {' and '.join(help_)}."


def _visit_start(history, i):
    """Index of the first record of the town visit that record i belongs to (i itself when it is not in town)."""
    if tuple(history[i]['scene_before']) != TOWN:
        return i
    j = i
    while j > 0 and tuple(history[j - 1]['scene_before']) == TOWN:
        j -= 1
    return j


def _visit_stats(visit):
    """Replay the actions of one town visit in order.

    Returns (talks {who: completed talks, in order of first talk}, sessions [[trader, trades]] one per completed talk
    to a trader, ordinal {(record position, action index): k-th talk to that who}, bought, sold). A trade belongs to
    the trader whose window is open: the one talked to last, as long as the next decision still shows a window open
    and no close came in between.
    """
    talks, sessions, ordinal = {}, [], {}
    bought = sold = 0
    cur = None
    for p, rec in enumerate(visit):
        shop = rec.get('shop')
        if shop is None or (cur is not None and sessions[cur][0] != VENDOR_NAME.get(shop)):
            cur = None
        for q, a in enumerate(rec['actions']):
            act = a['act']
            if act == 'talk':
                who = a['who']
                talks[who] = talks.get(who, 0) + 1
                ordinal[(p, q)] = talks[who]
                if who in VENDORS:
                    sessions.append([who, 0])
                    cur = len(sessions) - 1
                else:
                    cur = None
            elif act in TRADES:
                bought += act == 'buy'
                sold += act == 'sell'
                if cur is not None:
                    sessions[cur][1] += 1
            elif act == 'dismiss':
                cur = None
    return talks, sessions, ordinal, bought, sold


def _talked_phrase(who, k, sessions):
    if who == 'a townsperson':
        return who if k == 1 else f'townspeople {k} times'
    if k == 1:
        return who
    trades = [t for w, t in sessions if w == who]
    note = ''
    if not any(trades):
        note = ' (nothing bought or sold)'
    elif not any(trades[1:]):
        note = ' (nothing bought or sold after the first time)'
    return f'{who} {k} times{note}'


def town_visit_line(history):
    """Conclusions over the current town visit = the trailing decisions that were asked in town."""
    start = len(history)
    while start > 0 and tuple(history[start - 1]['scene_before']) == TOWN:
        start -= 1
    talks, sessions, _, bought, sold = _visit_stats(history[start:])
    talked = [_talked_phrase(who, k, sessions) for who, k in talks.items()]
    return (f"Already done this town visit: healed at Pepin: {'yes' if 'Pepin' in talks else 'no'}; "
            f"talked to: {', '.join(talked) if talked else 'nobody'}; bought: {_plural(bought, 'item')}; sold: {_plural(sold, 'item')}.")


def _events(history, i, scene_next):
    """[(tag, sentence)] for record i in execution order: shop/dialog closed or opened, townsperson talk, gear taken
    off, then the stairs if the scene changed."""
    rec = history[i]
    ev = []
    ordinal = None
    for q, a in enumerate(rec['actions']):
        act = a['act']
        if act == 'dismiss':
            shop = a.get('shop')
            ev.append(('closed', f"You just closed {SHOP_PHRASE.get(shop, 'a shop window' if shop else 'a dialog window')}."))
        elif act == 'talk':
            if ordinal is None:
                v = _visit_start(history, i)
                ordinal = _visit_stats(history[v:i + 1])[2]
                pos = i - v
            k, who = ordinal[(pos, q)], a['who']
            if who in VENDORS:
                ev.append(('talk', f'You just opened {SHOP_OF[who]}.' if k == 1 else
                           f'You just reopened {SHOP_OF[who]} ({_ordinal(k)} time this visit).'))
            else:
                ev.append(('talk', 'You just talked to a townsperson.' if k == 1 else
                           f'You just talked to a townsperson ({_ordinal(k)} townsperson talk this visit).'))
        elif act == 'equip' and a.get('off'):
            on = ((',' if len(a['off']) > 1 else '') + f" and equipped {' and '.join(a['on'])}") if a.get('on') else ''
            ev.append(('took_off', f"You just took off {' and '.join(a['off'])}{on}."))
    frm = tuple(rec['scene_before'])
    if scene_next != frm:
        if _depth(scene_next) < _depth(frm):
            ev.append(('stairs', f'You just came up from {_scene_name(frm)}.'))
        else:
            ev.append(('stairs', f'You just went down from {_scene_name(frm)}.'))
    return ev


def last_move_line(history, current_scene, current_shop=None):
    """Most recent stairs / shop / townsperson / take-off event among the previous LOOKBACK decisions, or None.

    While a shop window is open now, a "closed" event is stale (something reopened a window since) and is skipped.
    """
    start = max(0, len(history) - LOOKBACK)
    events = []
    for i in range(start, len(history)):
        nxt = tuple(history[i + 1]['scene_before']) if i + 1 < len(history) else current_scene
        events += _events(history, i, nxt)
    if current_shop is not None:
        events = [e for e in events if e[0] != 'closed']
    return events[-1][1] if events else None


def parse_potions(prompt):
    m = re.search(r'healing potions (\d+) in belt and (\d+) in pack', prompt)
    return int(m.group(1)) + int(m.group(2)) if m else 0


def weapon_line(prompt, scene, mission):
    """Version 2: on dungeon level 3 and in the tomb (Skeleton King mission), the weapon type against the undead."""
    if FACTS_VERSION < 2 or mission != 'skeleton_king' or scene[0] != 3:
        return None
    m = re.search(r'^Equipped: .*?weapon hand: ([^(;]+?) \(', prompt, re.M)
    if not m:
        return None
    name = m.group(1).strip()
    if any(w in name for w in SWORD_WORDS):
        return (f'Weapon: your {name} is a sword: it does half damage to the undead here (the Skeleton King and his '
                f'skeletons); a club, mace, morning star or flail does 50% more.')
    if any(w in name for w in BLUNT_WORDS):
        return f'Weapon: your {name} is a blunt weapon: it does 50% more damage to the undead here (the Skeleton King and his skeletons).'
    return None


def weapon_line_v3(prompt, scene, mission):
    """Version 3, Skeleton King mission: the weapon type against the undead, on dungeon level 3 / in the tomb for
    every weapon, and in town (from character level 5) when the weapon is not blunt. Says when a blunt weapon is for
    sale in the open shop."""
    if mission != 'skeleton_king':
        return None
    wp = parse_weapon(prompt)
    town = scene == TOWN
    if town and parse_level(prompt) < 5:
        return None
    if not town and scene[0] != 3:
        return None
    blunt_buy = [t[len('Buy '):].split(' (')[0] for t in parse_option_texts(prompt)
                 if t.startswith('Buy ') and weapon_class(t[len('Buy '):].split(' (')[0]) == 'blunt' and 'cannot use' not in t]
    sale = f' {blunt_buy[0]} is for sale in this shop.' if blunt_buy else ''
    undead = 'the undead (the Skeleton King and his skeletons)'
    if wp is None:
        return f'Weapon: none equipped; a club, mace, morning star or flail does 50% more damage to {undead}.{sale}'
    name, lo, hi = wp
    cls = weapon_class(name)
    if cls == 'blunt':
        return None if town else f'Weapon: your {name} (damage {lo}-{hi}) is a blunt weapon: it does 50% more damage to {undead}.'
    if cls == 'sword':
        what = f'is a sword: it does half damage to {undead}; a club, mace, morning star or flail does 50% more (3 times a sword)'
    else:
        what = f'does normal damage to {undead}; a club, mace, morning star or flail does 50% more'
    if town and not sale:
        sale = ' Griswold often sells one (a Club costs about 20 gold).'
    return f'Weapon: your {name} (damage {lo}-{hi}) {what}.{sale}'


def hit_line(prompt, scene, mission):
    """Version 3: chance to hit the mission boss, on dungeon level 3 / in the tomb (King mission) and whenever
    attribute points are unspent."""
    st = parse_stats(prompt)
    if st is None:
        return None
    unspent, strength, dex = st
    king = mission == 'skeleton_king'
    if not unspent and not (king and scene[0] == 3):
        return None
    level = parse_level(prompt)
    who, armour = ("the Skeleton King", KING_ARMOUR) if king else ('the Butcher', BUTCHER_ARMOUR)
    per_dmg = -(-100 // max(level, 1))
    pts = f'{unspent} attribute points unspent. ' if unspent else ''
    to_hit = loadout_v4(prompt)['to_hit'] if FACTS_VERSION >= 4 else 0   # version 4: visible to_hit of the worn items (F1)
    return (f'Hit chance: {pts}About {hit_pct(level, dex, armour, to_hit)}% of your swings hit {who} (armour {armour}) at character '
            f'level {level} and dexterity {dex}; every 2 dexterity add 1% (and block chance), while {per_dmg} strength add '
            f'only 1 damage per hit at your level.')


# ----------------------------------------------------------------------------------------------- version 4
# Every version-4 line is a pure function of the current prompt (no history): the Hero, Equipped and Visible enemies
# lines (render.py) and the option texts (options.py). Numbers are exact fractions; printed numbers round half up.
_HERO_V4 = re.compile(r'^Hero: level (\d+), HP (\d+)/(\d+) \((\d+)%\), gold (\d+), armor (-?\d+), damage (\d+)-(\d+), '
                      r'healing potions (\d+) in belt and (\d+) in pack, unspent attribute points (\d+)\. '
                      r'Strength (\d+), dexterity (\d+)', re.M)
_EQUIPPED = re.compile(r'^Equipped: (.*)$', re.M)
_VISIBLE = re.compile(r'^Visible enemies(?: \((\d+)\))?: (.*)$', re.M)
_ENEMY = re.compile(r'^(.*) #(\d+) hp (-?\d+)/(\d+) \((.*)\)$')
_ATTACK_BOSS = re.compile(r'^Attack (.*\[SKELETON KING\]|The Butcher) #\d+ \(hp (-?\d+)/(\d+),')
_PICKUP = re.compile(r'^Pick up (.*) \((\d+) steps away\)$')
_BUY = re.compile(r'^Buy (.*) for (\d+) gold(?: \(\d+ in stock\))?$')
_WAY_UP = re.compile(r'^(?:Take|Walk next to) the (stairs UP to the previous level/town|exit back to the main dungeon) '
                     r'\((\d+) steps away\)$')
_ITEM_DAMAGE = re.compile(r'^damage (\d+)-(\d+)$')
_ITEM_ARMOR = re.compile(r'^armor (-?\d+)$')
_ITEM_BONUS = re.compile(r'^([a-z_]+) ([+-]\d+)$')
HANDS = ('weapon hand', 'shield hand')
WHERE_ORDER = {'pack': 0, 'floor': 1, 'buy': 2}
V4_VERDICT_KEYS = (('melee_first', ': not ready: equip a melee weapon first.'), ('cannot', ' cannot kill him (you deal'),
                   ('barely', ' barely hurts him (you deal'), ('long', ': not ready: with your '),
                   ('ready', ': ready: with your '), ('not_ready_potions', ': not ready yet: with your '))


def _cdiv(a, b):
    """C++ integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _half_up(x, places=0):
    """Exact decimal rounding (half up) of a non-negative number: an int for places=0, else a string like '4.8'."""
    k = math.floor(Fraction(x) * 10 ** places + Fraction(1, 2))
    return k if not places else f'{k // 10 ** places}.{k % 10 ** places:0{places}d}'


def _seconds(t):
    """Fight time rounded UP to 10 seconds (PLAN section 1)."""
    return 10 * math.ceil(Fraction(t) / 10)


def item_base_v4(name):
    """itemdat base name of an item name, or None: identified uniques by name, else the longest base name inside it."""
    if name in V4_UNIQUE_BASES:
        return V4_UNIQUE_BASES[name]
    hits = [b for b in V4_BASES if b in name]
    if not hits and re.search(r'\bStaff of ', name):
        return 'Short Staff'  # a long spell-staff name uses shortName: 'Staff of <spell>' (items.cpp:1106-1117 GenerateStaffName);
        # every staff base is a two-handed Staff, and the damage numbers come from the item text anyway
    return max(hits, key=lambda b: (len(b), b)) if hits else None


def parse_item_v4(text):
    """options.item_name text 'Name (armor 5, damage 1-8, durability 3/45, unidentified, to_hit +5, cannot use)' ->
    dict(text, name, base, type, two, lo, hi, armor, cannot, bonus). Bonuses are shown only on identified items (F6); a
    'cannot use' item counts for nothing (items.cpp:2833-2835 skips items without _iStatFlag). An item with damage but
    no known base is a melee weapon of type 'Weapon' (no class modifier, never offered as a candidate)."""
    name, _, rest = text.partition(' (')
    it = dict(text=text, name=name, lo=None, hi=None, armor=0, cannot=False, bonus={})
    for s in (rest[:-1].split(', ') if rest.endswith(')') else []):
        m = _ITEM_DAMAGE.match(s)
        if m:
            it['lo'], it['hi'] = int(m.group(1)), int(m.group(2))
            continue
        m = _ITEM_ARMOR.match(s)
        if m:
            it['armor'] = int(m.group(1))
            continue
        if s == 'cannot use':
            it['cannot'] = True
            continue
        m = _ITEM_BONUS.match(s)
        if m:
            it['bonus'][m.group(1)] = int(m.group(2))
    base = item_base_v4(name)
    kind, two = V4_BASES.get(base, (None, False))
    if kind is None and it['lo'] is not None:
        kind = 'Weapon'
    if kind in V4_SWING and it['lo'] is None:
        kind = None  # a weapon name without damage numbers is no weapon we can compute
    it.update(base=base, type=kind, two=two)
    return it


def parse_hero_v4(prompt):
    m = _HERO_V4.search(_head(prompt))
    if not m:
        raise ValueError('prompt has no parsable Hero line')
    lvl, cur, mx, pct, gold, ac, lo, hi, belt, pack, unspent, strength, dex = map(int, m.groups())
    return dict(L=lvl, cur=cur, mx=mx, pct=pct, gold=gold, ac=ac, lo=lo, hi=hi, pots=belt + pack, unspent=unspent,
                str=strength, dex=dex)


def parse_equipped_v4(prompt):
    """[(slot, item)] of the Equipped line (render.py gear_text)."""
    m = _EQUIPPED.search(_head(prompt))
    body = m.group(1) if m else 'nothing.'
    body = body[:-1] if body.endswith('.') else body
    if body == 'nothing':
        return []
    out = []
    for part in body.split('; '):
        slot, _, item = part.partition(': ')
        out.append((slot, parse_item_v4(item)))
    return out


def loadout_v4(prompt):
    """The hero's fighting kit: the Hero line numbers, the melee weapon / bow / shield in EITHER hand, and the visible
    to_hit and damage% of all usable worn items (F1, F2). Item (i): the weapon may sit in the shield-hand slot and a
    shield in the weapon-hand slot (heldout-king-v3/s4150049: Buckler in the weapon hand, Mace in the shield hand);
    the engine blocks with a shield in either hand (items.cpp:2656-2673) and adds up both hands' damage
    (items.cpp:2833-2840), so the hero has no weapon only when neither hand holds a usable melee weapon."""
    ld = parse_hero_v4(prompt)
    eq = [(slot, it) for slot, it in parse_equipped_v4(prompt) if not it['cannot']]
    hands = [it for slot, it in eq if slot in HANDS]
    ld['weapon'] = next((it for it in hands if it['type'] in V4_SWING), None)
    ld['bow'] = next((it for it in hands if it['type'] == 'Bow'), None)
    ld['shield'] = next((it for it in hands if it['type'] == 'Shield'), None)
    ld['to_hit'] = sum(it['bonus'].get('to_hit', 0) for _, it in eq)
    ld['dmg_pct'] = sum(it['bonus'].get('damage_percent', 0) for _, it in eq)
    return ld


def potion_heal_v4(max_hp):
    """(least, most, mean) HP one Potion of Healing restores to a warrior with max_hp (F16: 2*(M/8 + rnd(M/4)))."""
    a, b = max_hp // 8, max(max_hp // 4 - 1, 0)
    return 2 * a, 2 * (a + b), 2 * a + b


def damage_dist_v4(lo, hi, level, strength, dmg_pct, wtype, cls):
    """{damage: probability} of one landed hit (F2, player.cpp:567-606), exact over the roll and the critical strike."""
    dmod = level * strength // 100                    # warrior _pDamageMod (items.cpp:2596), added at player.cpp:573
    crit = Fraction(min(level, 100), 100)             # CriticalStrike (attributes.tsv:2): x2 with chance L% (575-580)
    out = {}
    for r in range(lo, hi + 1):
        base = r + _cdiv(r * dmg_pct, 100) + dmod    # 569-570: roll, + roll * damage% / 100
        for d, w in ((base, 1 - crit), (2 * base, crit)):
            if not w:
                continue
            if (cls == 'undead' and wtype == 'Sword') or (cls == 'animal' and wtype == 'Mace'):
                d -= _cdiv(d, 2)                      # 591-593, 598-600
            elif (cls == 'undead' and wtype == 'Mace') or (cls == 'animal' and wtype == 'Sword'):
                d += _cdiv(d, 2)                      # 594-596, 601-603
            out[d] = out.get(d, 0) + w / (hi - lo + 1)
    return out


def fight_v4(ld, boss, boss_hp=None):
    """The section-1 model for kit ld against a boss (V4_BOSSES key) with boss_hp HP left (default full):
    dict(h, blk, D, regen, intr, band [, S, E, stagger, p, g, T, P]); intr = chance that one of his attacks interrupts
    the hero (a block, or a hit that stuns: F9, F10), whatever the hero is doing (a swing, a walk)."""
    b = V4_BOSSES[boss]
    lvl = ld['L']
    f = dict(boss=boss, hp=b['hp'] if boss_hp is None else boss_hp, regen=Fraction(b['mlvl'] // 2 * V4_FPS, 64))  # F12
    f['h'] = Fraction(min(100, max(15, b['to_hit'] + 2 * (b['mlvl'] - lvl) + 30 - ld['ac'])), 100)            # F8
    f['blk'] = Fraction(max(0, min(100, ld['dex'] + 30 + 2 * lvl - 2 * b['mlvl'])), 100) if ld['shield'] else Fraction(0)  # F9
    f['D'] = b['rate'] * f['h'] * (1 - f['blk']) * Fraction(b['lo'] + b['hi'], 2)
    if lvl <= b['lo']:                                                                                        # F10
        pst = Fraction(1)
    elif lvl > b['hi']:
        pst = Fraction(0)
    else:
        pst = Fraction((b['hi'] - lvl) * 64 + 1, (b['hi'] - b['lo']) * 64 + 1)
    intr = f['intr'] = f['h'] * (f['blk'] + (1 - f['blk']) * pst)
    w = ld['weapon']
    if w is None:
        f['band'] = 'melee_first'
        return f
    f['S'] = max(Fraction(0), V4_FPS - b['rate'] * intr * V4_INTERRUPT) / V4_SWING[w['type']]
    dist = damage_dist_v4(w['lo'], w['hi'], lvl, ld['str'], ld['dmg_pct'], w['type'], b['cls'])
    f['E'] = sum(d * p for d, p in dist.items())
    f['stagger'] = sum(p for d, p in dist.items() if d >= b['mlvl'] + 3)                                      # F15
    f['p'] = Fraction(hit_pct(lvl, ld['dex'], b['armour'], ld['to_hit']), 100)                                # F1
    f['g'] = f['S'] * f['p'] * f['E']
    if f['g'] <= f['regen']:
        f['band'] = 'cannot'
    elif f['g'] < Fraction(3, 2) * f['regen']:
        f['band'] = 'barely'
    else:
        f['T'] = f['hp'] / (f['g'] - f['regen'])
        f['P'] = math.ceil(f['T'] * f['D'] / max(potion_heal_v4(ld['mx'])[2], 1))
        f['band'] = 'long' if f['T'] > V4_LONG_SECONDS else 'ok'
    return f


def _rank(f):
    return V4_BANDS.index(f['band'])


def weapon_options_v4(prompt):
    """[(item, where, option text)] usable melee weapons the options can put in hand: Equip (options.py:312, in the
    pack), Pick up (options.py:341, any distance), Buy (options.py:282, in the open shop).
    where = ('pack', 0, replaced item texts) | ('floor', steps, None) | ('buy', price, None)."""
    out = []
    for t in parse_option_texts(prompt):
        if t.startswith('Equip '):
            off, on = parse_equip(t)
            if not on:
                continue
            it, where = parse_item_v4(on[0]), ('pack', 0, tuple(off))
        elif t.startswith('Pick up '):
            m = _PICKUP.match(t)
            if not m:
                continue
            it, where = parse_item_v4(m.group(1)), ('floor', int(m.group(2)), None)
        elif t.startswith('Buy '):
            m = _BUY.match(t)
            if not m:
                continue
            it, where = parse_item_v4(m.group(1)), ('buy', int(m.group(2)), None)
        else:
            continue
        if it['type'] in V4_SWING and it['type'] != 'Weapon' and not it['cannot']:
            out.append((it, where, t))
    return out


def swap_v4(ld, it, where):
    """Kit after putting melee weapon it in hand: it replaces the worn weapon or bow; a two-handed one (or an Equip
    option that names the shield among the replaced items) also takes off the shield: no block and armour minus the
    shield's (PLAN section 3). Visible bonuses of the replaced items go with them."""
    off = where[2] if where[0] == 'pack' else None
    new = dict(ld, weapon=it, bow=None)
    gone = [x for x in (ld['weapon'], ld['bow']) if x is not None and (off is None or x['text'] in off)]
    sh = ld['shield']
    if sh is not None and (it['two'] or (off is not None and sh['text'] in off)):
        gone.append(sh)
        new['shield'] = None
        new['ac'] = ld['ac'] - sh['armor']
    for key, bonus in (('to_hit', 'to_hit'), ('dmg_pct', 'damage_percent'), ('str', 'strength'), ('dex', 'dexterity')):
        new[key] = ld[key] - sum(x['bonus'].get(bonus, 0) for x in gone) + it['bonus'].get(bonus, 0)
    return new


def _where(where):
    return {'pack': 'in your pack', 'floor': f'on the floor ({where[1]} steps)', 'buy': f'for {where[1]} gold here'}[where[0]]


def better_weapons_v4(prompt, ld, boss):
    """[(item, where, fight)] options whose damage per swing frame (E / T_sw) against the boss beats the worn weapon's
    (0 for a bow or no weapon), best first, one per name and damage (PLAN section 3 W)."""
    now = fight_v4(ld, boss)
    base = now['E'] / V4_SWING[ld['weapon']['type']] if ld['weapon'] else 0
    rows = []
    for it, where, text in weapon_options_v4(prompt):
        f = fight_v4(swap_v4(ld, it, where), boss)
        rate = f['E'] / V4_SWING[it['type']]
        if rate > base:
            rows.append(((-rate, WHERE_ORDER[where[0]], where[1], text), it, where, f))
    out, seen = [], set()
    for _, it, where, f in sorted(rows, key=lambda r: r[0]):
        if (it['name'], it['lo'], it['hi']) not in seen:
            seen.add((it['name'], it['lo'], it['hi']))
            out.append((it, where, f))
    return out


def visible_enemies_v4(prompt):
    """[(name, hp, max hp, steps)] of the Visible enemies line (the first 10; render.py:62-74); steps = 0 for
    'adjacent/in attack range', None when unreachable."""
    m = _VISIBLE.search(_head(prompt))
    if not m or m.group(2) == 'none.':
        return []
    body = m.group(2)
    body = re.sub(r' \(\+\d+ more\)$', '', body[:-1] if body.endswith('.') else body)
    rows = []
    for part in body.split('; '):
        e = _ENEMY.match(part)
        if e:
            w = e.group(5)
            steps = 0 if w == 'adjacent/in attack range' else int(w.split()[0]) if w.endswith(' steps') else None
            rows.append((e.group(1), int(e.group(3)), int(e.group(4)), steps))
    return rows


def visible_boss_v4(prompt):
    """(boss, hp, max hp) of a visible Skeleton King or Butcher: from the attack options first (the King is always
    listed first, options.py:216-220), else from the Visible enemies line ('Skeleton King SKELETON KING', render.py:71);
    None when neither is visible."""
    for t in parse_option_texts(prompt):
        m = _ATTACK_BOSS.match(t)
        if m:
            return ('king' if m.group(1).endswith('[SKELETON KING]') else 'butcher', int(m.group(2)), int(m.group(3)))
    for name, hp, mx, _ in visible_enemies_v4(prompt):
        if name.endswith(' SKELETON KING'):
            return 'king', hp, mx
        if name == 'The Butcher':
            return 'butcher', hp, mx
    return None


def way_up_v4(prompt, scene):
    """'stairs UP 14 steps away' / 'tomb exit 3 steps away' from the nearest listed way up, or that none is listed."""
    best = None
    for t in parse_option_texts(prompt):
        m = _WAY_UP.match(t)
        if m and (best is None or int(m.group(2)) < best[0]):
            best = (int(m.group(2)), 'tomb exit' if m.group(1).startswith('exit') else 'stairs UP')
    if best is None:
        return 'no tomb exit listed yet' if scene[1] else 'no stairs UP listed yet'
    return f'{best[1]} {_plural(best[0], "step")} away'


def earn_where_v4(level, mission):
    """Where to earn the gold for missing potions: the dungeon levels on pace for the hero (ENTER_LEVEL) inside the
    mission (the Butcher mission ends at dungeon level 2)."""
    top = max([1] + [d for d, need in ENTER_LEVEL.items() if level >= need and (mission == 'skeleton_king' or d <= 2)])
    return 'dungeon level 1' if top == 1 else f'dungeon levels {top - 1}-{top}'


def money_v4(gold, missing, where, town=False):
    """Item (h): how many of the missing healing potions the gold buys, when it cannot buy them all (049/053 went
    between the tomb and town 59 times with under 50 gold), and where the rest comes from: the dungeon levels of
    earn_where_v4 (not a trip to town with nothing to buy), and in town loot from the pack, never the worn gear (the
    shop menu also offers 'Sell equipped ...', options.py:349-353)."""
    k = gold // POTION_PRICE
    if missing <= 0 or k >= missing:
        return ''
    rest = f'sell loot from your pack or earn more on {where}' if town else f'earn more on {where} first'
    return f" Your {gold} gold buys {k or 'none'} of the {missing} missing: {rest}."


def boss_need_v4(f):
    """Healing potions to carry into fight f (ok band): P + 3 spare, at most 20, the town line's cap (TEACHER.md 13:20:
    never more than 20), so the Pace verdict never asks for more than the town lets the hero buy (review m5)."""
    return min(V4_TOWN_MAX, f['P'] + V4_SPARE_POTIONS)


def _weapon_name(ld):
    return ld['weapon']['name'] if ld['weapon'] else 'bare hands'


def _verdict_parts(f, wname, pots):
    """(main text, short text) of a fight's band (PLAN section 3 K templates; short = 'Beating him now: ...')."""
    band = f['band']
    if band == 'melee_first':
        return 'not ready: equip a melee weapon first.', 'equip a melee weapon first.'
    if band in ('cannot', 'barely'):
        how = 'cannot kill him' if band == 'cannot' else 'barely hurts him'
        return (f"not ready: your {wname} {how} (you deal about {_half_up(f['g'], 1)} damage per second, he regenerates "
                f"{_half_up(f['regen'], 1)} per second).", f'your {wname} {how}.')
    if band == 'long':
        s = f'over 2 minutes and far more healing potions than your {pots}.'
        return f'not ready: with your {wname} it takes {s}', s
    s = f"about {_seconds(f['T'])} seconds and {_plural(f['P'], 'healing potion')}"
    if pots >= boss_need_v4(f):
        s += f' (you carry {pots}).'
        return f'ready: with your {wname} {s}', s
    s += f"; carry {boss_need_v4(f)}+ (you carry {pots})."
    return f'not ready yet: with your {wname} {s}', s


def _remedy_key(f, it, where, text):
    """Sort key (smaller = better) of a weapon option for the Pace verdict's remedy: the higher band first, then the
    Weapon line's own order (damage per swing frame, pack before floor before shop, nearer, text). Both lines therefore
    name the same weapon unless the Weapon line's first one fights in a lower band (a two-handed weapon that takes off
    the shield: its note says so); before 2026-09-23 the remedy took the most damage per second within the band and 239
    of 4,481 logged prompts with both lines named two different weapons (review m6)."""
    return (-_rank(f), -f['E'] / V4_SWING[it['type']], WHERE_ORDER[where[0]], where[1], text)


def remedy_v4(prompt, ld, boss, f, hp=None):
    """(item, where, fight) of the best weapon option (_remedy_key) when it fights in a better band than kit ld's fight
    f, else None: the weapon the Pace verdict names."""
    best = None
    for it, where, text in weapon_options_v4(prompt):
        g = fight_v4(swap_v4(ld, it, where), boss, hp)
        key = _remedy_key(g, it, where, text)
        if best is None or key < best[0]:
            best = (key, it, where, g)
    return best[1:] if best is not None and _rank(best[3]) > _rank(f) else None


def verdict_v4(prompt, boss, what):
    """K: '{what}: <verdict>' for the Pace line, with the gold fact when potions are short and the best weapon in
    reach when it reaches a better band (PLAN section 3 K)."""
    ld = loadout_v4(prompt)
    seen = visible_boss_v4(prompt)
    hp = seen[1] if seen is not None and seen[0] == boss else None
    f = fight_v4(ld, boss, hp)
    out = f'{what}: ' + _verdict_parts(f, _weapon_name(ld), ld['pots'])[0]
    if retreat_line(prompt, parse_scene(prompt)) is not None:
        return out  # the Retreat line (with its own gold fact) outranks the extras; keeps the +120-token budget
    if f['band'] == 'ok':
        return out + money_v4(ld['gold'], boss_need_v4(f) - ld['pots'], earn_where_v4(ld['L'], parse_mission(prompt)))
    best = remedy_v4(prompt, ld, boss, f, hp)
    if best is not None:
        it, where, g = best
        if g['band'] == 'ok':
            how = f"about {_seconds(g['T'])} seconds and {_plural(g['P'], 'healing potion')} (you carry {ld['pots']})"
        else:
            how = {'cannot': 'still cannot kill him', 'barely': 'barely hurts him', 'long': 'over 2 minutes'}[g['band']]
        out += f" With the {it['name']} {_where(where)}: {how}."
    return out


def outside_mission_v4(prompt, scene):
    """Item (g), Butcher mission on dungeon level 3+ or in the tomb (7 of 24 round-8 Butcher games went there, 2 of
    them killed the King instead)."""
    where = "The Skeleton King's tomb" if scene[1] else f'Dungeon level {scene[0]}'
    king = ' Killing the Skeleton King does not complete it.' if scene[1] else ''
    return f'{where} is outside your mission: the Butcher is on dungeon level 2; go back up ({way_up_v4(prompt, scene)}).{king}'


def boss_line(prompt, scene, mission):
    """B (version 4): how fast a visible Skeleton King or Butcher hurts the hero. The other mission's boss is 'not
    needed for your mission'; on the King mission a visible Butcher also gets the short verdict of beating him now."""
    seen = visible_boss_v4(prompt)
    if seen is None or scene == TOWN:
        return None
    boss, hp, mx = seen
    b = V4_BOSSES[boss]
    ld = loadout_v4(prompt)
    f = fight_v4(ld, boss, hp)
    other = (boss == 'butcher') != (mission == 'butcher')
    block = f"you block {_half_up(f['blk'] * 100)}%" if ld['shield'] else 'you have no shield to block'
    last = f"; your {ld['cur']} HP last about {_half_up(ld['cur'] / f['D'])} seconds" if f['D'] else ''
    head = f"Boss: {b['who']} ({hp}/{mx} HP{', not needed for your mission' if other else ''})"
    if retreat_line(prompt, scene) is not None:
        # Next to a Retreat line (the boss is in sight but not adjacent: D2 keeps the Retreat line off while he is):
        # the damage rate and the time the HP leaves. Shorter than the full line for the +120-token budget (the King
        # mission's dungeon level 2 had only a Pace line in v3). Until D2 (2026-09-23 15:05) this form also said, with
        # the boss adjacent, how often his attacks interrupt the walk to the stairs.
        return f"{head}: about {_half_up(f['D'], 1)} HP per second{last}."
    line = (f"{head} hits you in {_half_up(f['h'] * 100)}% of his attacks for {b['lo']}-{b['hi']} and {block}: about "
            f"{_half_up(f['D'], 1)} HP per second{last}.")
    if other and boss == 'butcher':
        line += ' Beating him now: ' + _verdict_parts(f, _weapon_name(ld), ld['pots'])[1]
    return line


def weapon_line_v4(prompt, scene, mission):
    """W (version 4): damage per hit against the mission boss for the worn weapon and the 2 best better options.
    King mission: dungeon level 3 and the tomb; town from character level 5 when something better is listed or the
    worn weapon is no mace/club. Butcher mission: town and dungeon levels 1-2, only when something better is listed
    and the Butcher is not in sight (then the Pace verdict names the best weapon in reach); in town one option, and
    only while the hero carries the potions the town line asks for. Never next to a Retreat line: fleeing outranks
    weapon advice. Above V4_WEAPON_CHARS characters (V4_WEAPON_CHARS_BOSS next to the Boss line) the line drops its
    second option, then calls the worn weapon 'your weapon', then drops its stagger note; in town the second option only
    while the town line is not asking for potions (its heal range and gold fact come first), and on the King mission no
    line then if the worn weapon is a mace already. These limits keep the prompt within the +120-token budget (PLAN
    section 3; tests_facts_v4.test_t11_synthetic checks worst cases built from the longest affix names of the tables)."""
    king = mission == 'skeleton_king'
    town = scene == TOWN
    ld = loadout_v4(prompt)
    if king and ((town and ld['L'] < 5) or (not town and scene[0] != 3)):
        return None
    if not king and not town and (scene[1] or scene[0] > 2 or visible_boss_v4(prompt) is not None):
        return None
    if not town and retreat_line(prompt, scene) is not None:
        return None
    boss = 'king' if king else 'butcher'
    short = town and ld['pots'] < town_target_v4(prompt)[0]
    if short and not king:
        # Butcher mission in town (version 3 had no weapon line there; +120-token budget): potions first, no weapon
        # advice while the hero carries fewer than the town line asks for (it then carries the heal range and the gold)
        return None
    better = better_weapons_v4(prompt, ld, boss)[:1 if town and (short or not king) else 2]  # town: the best one
    mace = ld['weapon'] is not None and ld['weapon']['type'] == 'Mace'
    if not better and (not king or (town and mace)):
        return None
    if short and mace:
        return None  # King mission in town with a mace already (version 3 had no line then): potions first
    now = fight_v4(ld, boss)
    head = ('Weapon: damage per hit against the undead (the Skeleton King and his skeletons): ' if king else
            'Weapon: damage per hit against the Butcher (a demon: the weapon type does not matter): ')

    def build(opts, worn):
        """The line with options opts; worn = 'name' (PLAN text), 'weapon' (the worn weapon or bow called 'weapon' /
        'bow') or 'bare' (also without its stagger note). F15: 9+ damage staggers the Butcher; the first stagger note
        says so in full."""
        long_note = [True]

        def notes(f, extra):
            n = []
            if not king and f is not None:
                s = _half_up(f['stagger'] * 100)
                n.append(f"{s}% of your hits stagger him: {V4_BOSSES[boss]['mlvl'] + 3}+ damage" if long_note[0] else f'{s}%')
                long_note[0] = False
            n += extra
            return f" ({'; '.join(n)})" if n else ''

        if ld['weapon'] is not None:
            name = ld['weapon']['name'] if worn == 'name' else 'weapon'
            parts = [f"your {name} {_half_up(now['E'], 1)}{notes(now if worn != 'bare' else None, [])}"]
        elif ld['bow'] is not None:
            parts = [f"your {ld['bow']['name'] if worn == 'name' else 'bow'} shoots arrows (no melee numbers)"]
        else:
            parts = ['you hold no melee weapon']
        for it, where, f in opts:
            extra = (['two-handed: no shield'] if it['two'] and ld['shield'] is not None else []) + \
                (['slower swing'] if it['type'] in ('Axe', 'Staff') else [])
            parts.append(f"{it['name']} {_where(where)} {_half_up(f['E'], 1)}{notes(f, extra)}")
        return parts, head + '; '.join(parts) + '.'

    if better:
        # +120-token budget: the line stays within V4_WEAPON_CHARS (V4_WEAPON_CHARS_BOSS next to the Boss line); long
        # magic names first cost the second option, then the worn weapon's name, then its stagger note (every PLAN
        # example keeps its full text)
        cap = V4_WEAPON_CHARS_BOSS if not town and visible_boss_v4(prompt) is not None else V4_WEAPON_CHARS
        for opts, worn in ((better[:2], 'name'), (better[:1], 'name'), (better[:1], 'weapon')):
            line = build(opts, worn)[1]
            if len(line) <= cap:
                return line
        return build(better[:1], 'bare')[1]  # at most about 260 characters with the longest names of the item tables
    parts = build([], 'name')[0]
    best = ', the best of your options.'
    if not town and ld['weapon'] is not None:
        seen = visible_boss_v4(prompt)
        hp = seen[1] if seen is not None and seen[0] == boss else None
        f = fight_v4(ld, boss, hp)
        if f['band'] != 'ok' and remedy_v4(prompt, ld, boss, f, hp) is not None:
            # the Pace verdict names a weapon that fights better although it does less damage per hit (its to_hit, or
            # fewer interrupted swings without a shield): not "the best" then (111 logged prompts said both)
            best = ', the most damage per hit of your options.'
    line = head + parts[0] + (best if ld['weapon'] is not None else '; no melee weapon is in your options.')
    if town and not mace:
        club = parse_item_v4('Club (damage 1-6)')      # itemdat.tsv:142, value 20
        f = fight_v4(swap_v4(ld, club, ('buy', 20, None)), boss)
        if ld['weapon'] is None or f['E'] / V4_SWING['Mace'] > now['E'] / V4_SWING[ld['weapon']['type']]:
            line += f" A Club would do {_half_up(f['E'], 1)}; Griswold often sells one for about 20 gold."
    return line


def town_base_v4(level):
    """Potions to leave town without a boss target: 3 from character level 3, 2 before (the starting kit). The
    Retreat line's potion trigger is 1 or fewer (RETREAT_POTIONS, D1 of 2026-09-23 15:05): the town never lets a hero
    go down into a Retreat line, and from character level 3 a spare potion lies between the two numbers, so town and
    dungeon do not send the hero back and forth."""
    return TOWN_POTIONS_V4 if level >= ENTER_LEVEL[2] else TOWN_POTIONS


def town_target_v4(prompt):
    """(potions needed to go down, boss or None). The floor: 10 on the King mission from character level 6, 8 on the
    Butcher mission from character level 4 (PLAN section 3 Town), town_base_v4 before. When K says the worn weapon kills
    the mission's boss (ok band) the target is the verdict's own number boss_need_v4 = min(P+3, 20), never below the
    floor, at EVERY character level (review m5: below those levels the town said 3+ while the verdict said 'carry 19+'
    in 78 logged prompts): the same function as the dungeon verdict, so the town never lets a hero go down with fewer
    potions than the verdict asks for."""
    ld = loadout_v4(prompt)
    mission = parse_mission(prompt)
    boss = 'king' if mission == 'skeleton_king' else 'butcher'
    if mission == 'skeleton_king' and ld['L'] >= KING_TOWN_LEVEL:
        floor = KING_TOWN_POTIONS
    elif mission == 'butcher' and ld['L'] >= BUTCHER_TOWN_LEVEL:
        floor = BUTCHER_POTIONS
    else:
        floor = None
    f = fight_v4(ld, boss)
    if f['band'] == 'ok':
        return max(floor or town_base_v4(ld['L']), boss_need_v4(f)), boss
    return (floor, boss) if floor else (town_base_v4(ld['L']), None)


def town_ready_v4(prompt):
    """Town readiness (version 4): the v4 potion target, the heal range of one potion and the gold fact when short.
    Without gold for another potion the hero may go down, and the line says where to earn the rest (review m2: it said
    'yes' and 'sell items or earn gold first' in one breath)."""
    ld = parse_hero_v4(prompt)
    pct, gold, pots = ld['pct'], ld['gold'], ld['pots']
    need, boss = town_target_v4(prompt)
    where = earn_where_v4(ld['L'], parse_mission(prompt))
    who = {'king': ' for the Skeleton King', 'butcher': ' for the Butcher', None: ''}[boss]
    hp_ok = pct >= TOWN_HP_PCT
    pots_ok = pots >= need or gold < POTION_PRICE
    have = _plural(pots, 'healing potion')
    if hp_ok and pots_ok:
        if pots >= need:
            return f'Ready to go down: yes (HP {pct}%, {have}).'
        return (f'Ready to go down: yes (HP {pct}%; only {have}, and gold {gold} cannot buy another). '
                f'{need}+ are needed{who}: earn the gold on {where} or sell loot from your pack.')
    why, help_ = [], []
    if not hp_ok:
        why.append(f'HP {pct}%, needs {TOWN_HP_PCT}%+')
        help_.append('heals for free')
    if not pots_ok:
        why.append(f'{have}, needs {need}+{who}')
        help_.append(f'sells potions for {POTION_PRICE} gold')
    line = f"Ready to go down: not yet ({'; '.join(why)}). Pepin {' and '.join(help_)}."
    if not pots_ok:
        lo, hi, mean = potion_heal_v4(ld['mx'])
        line += (f" A Potion of Healing restores {lo}-{hi} of your {ld['mx']} HP (about {mean})." +
                 money_v4(gold, need - pots, where, town=True))
    return line


def adjacent_boss_v4(prompt):
    """'king' / 'butcher' when the Skeleton King or the Butcher is next to the hero (his attack option says 'in attack
    range now', options.py:274-276, or the Visible enemies line 'adjacent/in attack range', render.py:69), else None."""
    for t in parse_option_texts(prompt):
        m = _ATTACK_BOSS.match(t)
        if m and t.endswith(', in attack range now)'):
            return 'king' if m.group(1).endswith('[SKELETON KING]') else 'butcher'
    for name, _, _, steps in visible_enemies_v4(prompt):
        if steps == 0 and (name.endswith(' SKELETON KING') or name == 'The Butcher'):
            return 'king' if name.endswith(' SKELETON KING') else 'butcher'
    return None


def surrounded_v4(prompt):
    """(adjacent, near, surrounded) from the Visible enemies line (the enemies it lists, at most 10: render.py:62-74):
    the enemies at steps 0, the enemies within RETREAT_NEAR_STEPS steps (the adjacent ones included), and whether the
    hero is surrounded: RETREAT_ADJACENT+ adjacent or RETREAT_NEAR+ near (D1 of 2026-09-23 15:05). Unreachable
    enemies do not count."""
    steps = [e[3] for e in visible_enemies_v4(prompt) if e[3] is not None]
    adj = sum(1 for s in steps if s == 0)
    near = sum(1 for s in steps if s <= RETREAT_NEAR_STEPS)
    return adj, near, adj >= RETREAT_ADJACENT or near >= RETREAT_NEAR


def retreat_line(prompt, scene):
    """Version 4 (f), every mission and dungeon level (D1/D2 of 2026-09-23 15:05; thresholds and data: see
    RETREAT_POTIONS): go up to town now, with the way up and what the gold buys, when
      - the healing potions (belt + pack) are RETREAT_POTIONS (1) or fewer, and the gold buys a potion or HP is below
        50%, and the mission's boss is not in sight at half his HP or less with a potion left; or
      - the hero is surrounded (surrounded_v4: 4+ enemies adjacent or 8+ within 5 steps) with RETREAT_CROWD_POTIONS (2)
        or fewer potions, any gold and HP;
    never while the Skeleton King or the Butcher is adjacent (adjacent_boss_v4): then the Boss line and the Pace
    verdict decide, and walking away from him seldom works."""
    if scene == TOWN or adjacent_boss_v4(prompt) is not None:
        return None
    ld = parse_hero_v4(prompt)
    adj, near, surrounded = surrounded_v4(prompt)
    low = ld['pots'] <= RETREAT_POTIONS
    seen = visible_boss_v4(prompt)
    mission_boss = 'king' if parse_mission(prompt) == 'skeleton_king' else 'butcher'
    if low and ld['pots'] and seen is not None and seen[0] == mission_boss and 2 * seen[1] <= seen[2]:
        low = False       # finish him: at half HP or less he is worth a potion more (TEACHER.md, Butcher fight)
    crowd = surrounded and ld['pots'] <= RETREAT_CROWD_POTIONS
    if not (crowd or (low and (ld['gold'] >= POTION_PRICE or ld['pct'] < RETREAT_HP_PCT))):
        return None
    why = f"{_plural(ld['pots'], 'healing potion')} left"
    if adj >= RETREAT_ADJACENT:
        why += f' and {adj} enemies next to you'
    elif near >= RETREAT_NEAR:
        why += f' and {near} enemies within {RETREAT_NEAR_STEPS} steps'
    if seen is not None:
        # with a boss in sight (not adjacent) the Boss line says how fast he hurts; the gold waits for the town line
        # (+120-token budget)
        return f'Retreat: {why}: go up to town now ({way_up_v4(prompt, scene)}).'
    k = ld['gold'] // POTION_PRICE
    buys = (f"Pepin heals for free and your {ld['gold']} gold buys {_plural(k, 'healing potion')}" if k else
            f"Pepin heals for free, but your {ld['gold']} gold buys no potion")   # where to earn it: the town line
    return f'Retreat: {why}: go up to town now ({way_up_v4(prompt, scene)}); {buys}.'


def weapon_choice_v4(prompt, text):
    """For reports (rerender checks): (chosen, best, worn) damage per swing frame against the mission boss when option
    text puts a melee weapon in hand, else None."""
    boss = 'king' if parse_mission(prompt) == 'skeleton_king' else 'butcher'
    ld = loadout_v4(prompt)
    rates = {t: fight_v4(swap_v4(ld, it, where), boss)['E'] / V4_SWING[it['type']] for it, where, t in weapon_options_v4(prompt)}
    if text not in rates:
        return None
    worn = fight_v4(ld, boss)['E'] / V4_SWING[ld['weapon']['type']] if ld['weapon'] else 0
    return rates[text], max(rates.values()), worn


# ----------------------------------------------------------------------------------------------- version 5
# Version 5 (round 9b, 2026-09-23 night, opt-in with FACTS_VERSION=5; spec r9-work/r9b/FACTS-V5-SPEC.md as corrected by
# its review, workflow wf_8332a9fa-a96): the Butcher blocking the stairs. In the loop games (gate9-new-butcher/s4150025,
# gate9-new-king/s4150025, round 8 king3-v3/s4150025 and 6 more; about 21% of the round-8 failures) the hero fled up
# the stairs from him, he waited at the landing, and town said 'yes' and dungeon level 1 'ready' again: down, adjacent,
# up, for the rest of the game, never getting stronger. Engine facts from the training-seed logs:
#   - he does not heal while the hero is on another level: 7 of 7 returns found him within 4 HP of what he had when the
#     hero left (307-1,657 frames away); on his own level he regenerates about 1 HP per second (F12);
#   - when the hero leaves dungeon level 2 by the stairs up with him within 3 steps, he waits at the bottom of those
#     stairs: 381 of 388 returns saw him at once (4 of 1,152 otherwise);
#   - getting past him to stairs 2-3 steps away cost a median 21% and at most 58% of the hero's HP in the 271 escapes
#     made with potions; the 56 escapes that cost 60-95% had no potion after the hero had sold its armour.
# So the lines tell the hero to prepare what can be prepared (a better weapon in the pack, potions for the gold, what is
# left of dungeon level 1) and then to fight him AT THE STAIRS, every wound staying: with 3+ potions fight on and drink
# at 40%; with 2 fight on until HP is below 50%, then take the stairs up with those 2 for the way; with 1 or none go up;
# finish him when his HP is at half or less and the hero's own HP plus one spare potion covers the rest.
# Every version-5 line is computed from the version-4 lines (facts_lines_v5): versions 1-4 are unchanged.
# History: record_from_game_row / history_from_astra add b5, up5, ex5 (sighting_v5 of that decision's own prompt), so
# the live FactsTracker and a rerender of the logged rows see the same memory.
V5_GUARD_STEPS = 3      # he was within 3 steps in the last decision on dungeon level 2 before going up -> he waits there
V5_STAIRS_NEAR = 3      # 'at the stairs': the nearest stairs-UP option 3 steps away or fewer and he within 3 steps
V5_DRINK_MIN = 3        # at the stairs with 3+ potions: fight on, drink at 40% HP
V5_TRIP_MIN = 2         # a trip down to fight him needs 2+ potions (they are kept for the way up)
V5_TRIP_HP = 50         # with exactly 2 potions: fight on until HP is below 50%, then go up
V5_DRINK_PCT = 40       # drink at 40% HP next to him (TEACHER.md 'The Butcher next to the hero' rule 1)
V5_VIABLE = ('ok', 'long')
# the evidence numbers above as TEACHER.md's round-9b section states them (tests_facts_v5.py recomputes the first four
# from the training-seed logs and binds all of them to the text): returns without healing (7 of 7, within 4 HP), returns
# that met him at the landing after leaving with him within 3 steps (381 of 388; 4 of 1,152 otherwise), escapes with
# potions (271: median 21%, at most 58% of max HP), escapes without potion after selling the armour (56: 60-95%, armour
# 9-10, his hit chance 74-77% by F8 at character level 3-4)
V5_EVIDENCE = dict(no_heal=(7, 7), no_heal_hp=4, landing=(381, 388), landing_else=(4, 1152), escapes_with_potions=271,
                   escape_median_pct=21, escape_max_pct=58, escapes_bad=56, escape_bad_pct=(60, 95), bad_armour=(9, 10),
                   bad_hit_pct=(74, 77))
V5_UNTIL = ' Until then gain levels on this level, away from his room.'
V5_MEMORY = ('Butcher: waiting at the stairs down to dungeon level 2 with {hp}/{mx} HP (he does not heal while you are on '
             'another level).')
V5_WAITS = 'the Butcher waits at the stairs'
V5_BLOCKED = 'he stands between you and the stairs up'
V5_STAYS = ' do not walk away from them.'
_UP_STAIRS_V5 = re.compile(r'^(?:Take|Walk next to) the stairs UP to the previous level/town \((\d+) steps away\)$')
_ATTACK_BUTCHER_V5 = re.compile(r'^Attack The Butcher #\d+ \(hp (-?\d+)/(\d+), (in attack range now|(\d+) steps away)')
_EARN_MORE_V5 = re.compile(r' Your \d+ gold buys (?:\d+|none) of the \d+ missing: earn more on dungeon levels? [0-9-]+ first\.$')
_EXPLORE_V5 = 'Explore toward '


def sighting_v5(prompt):
    """dict(b5, up5, ex5) from one decision's own prompt (its fact lines, if any, are never read):
    b5  = (hp, max hp, steps) of the Butcher on the Visible enemies line (steps 0 'adjacent/in attack range', None
          'unreachable'), else from his attack option 'Attack The Butcher #N (hp H/M, in attack range now | N steps
          away)' (the Visible enemies line lists only the first 10 enemies in state order, render.py:62-74), else None;
    up5 = steps of the nearest 'Take / Walk next to the stairs UP' option, or None;
    ex5 = number of 'Explore toward' options."""
    b = next(((hp, mx, st) for name, hp, mx, st in visible_enemies_v4(prompt) if name == 'The Butcher'), None)
    texts = parse_option_texts(prompt)
    up, ex = None, 0
    for t in texts:
        m = _UP_STAIRS_V5.match(t)
        if m and (up is None or int(m.group(1)) < up):
            up = int(m.group(1))
        ex += t.startswith(_EXPLORE_V5)
    if b is None:
        for t in texts:
            m = _ATTACK_BUTCHER_V5.match(t)
            if m:
                b = (int(m.group(1)), int(m.group(2)), 0 if m.group(4) is None else int(m.group(4)))
                break
    return dict(b5=b, up5=up, ex5=ex)


def butcher_memory_v5(history, scene):
    """(guard, l1). guard = dict(hp, mx) when the hero's latest stay on dungeon level 2 ended by going up to dungeon
    level 1 or town with the Butcher within V5_GUARD_STEPS steps in that stay's last decision (b5), else None; only
    used in town and on dungeon level 1. l1 = ex5 of the latest decision on dungeon level 1 (None: never there).
    His death clears it (he is no longer in any prompt), and every stay on dungeon level 2 renews it."""
    l1 = next((r['ex5'] for r in reversed(history) if tuple(r['scene_before']) == (1, False)), None)
    k = next((i for i in range(len(history) - 1, -1, -1) if tuple(history[i]['scene_before']) == (2, False)), None)
    if k is None:
        return None, l1
    nxt = tuple(history[k + 1]['scene_before']) if k + 1 < len(history) else tuple(scene)
    b = history[k]['b5']
    if nxt[0] < 2 and not nxt[1] and b is not None and b[2] is not None and b[2] <= V5_GUARD_STEPS:
        return dict(hp=b[0], mx=b[1]), l1
    return None, l1


def _landing_move_v5(rec):
    """A decision that keeps the hero at the landing: attack the Butcher, drink, equip, move a potion to the belt, the
    stairs UP (whatever the result), or resume attacking him / walking to the stairs UP."""
    kind, text = rec.get('kind'), rec.get('text') or ''
    if kind in ('drink', 'equip', 'belt'):
        return True
    if kind == 'fight':
        return text.startswith('Attack The Butcher #')
    if kind == 'exit':
        return 'stairs UP' in text
    if kind == 'resume':
        return 'Attack The Butcher #' in text or 'stairs UP' in text
    return False


def at_stairs_v5(history, prompt, scene):
    """Dungeon level 2 with the Butcher within V5_STAIRS_NEAR steps (or adjacent): dict(b, st, blocked) when the hero is
    at the stairs up: the nearest stairs-UP option V5_STAIRS_NEAR steps away or fewer (st = its steps), or none listed
    while this stay on dungeon level 2 began at the landing (its first decision: stairs UP and the Butcher both within 3
    steps) and every decision since only attacked him, drank, equipped, moved a potion to the belt, chose the stairs UP
    or resumed one of those (blocked: he stands in the way, st None). Else None."""
    if tuple(scene) != (2, False):
        return None
    s = sighting_v5(prompt)
    b = s['b5']
    if b is None or not ((b[2] is not None and b[2] <= V5_STAIRS_NEAR) or adjacent_boss_v4(prompt) == 'butcher'):
        return None
    if s['up5'] is not None:
        return dict(b=b, st=s['up5'], blocked=False) if s['up5'] <= V5_STAIRS_NEAR else None
    j = len(history)
    while j > 0 and tuple(history[j - 1]['scene_before']) == (2, False):
        j -= 1
    stay = history[j:]
    if not stay:
        return None
    first = stay[0]
    b0 = first['b5']
    if first['up5'] is None or first['up5'] > V5_STAIRS_NEAR or b0 is None or b0[2] is None or b0[2] > V5_STAIRS_NEAR:
        return None
    if not all(_landing_move_v5(r) for r in stay):
        return None
    return dict(b=b, st=None, blocked=True)


def potions_beyond_hp_v5(f, ld, cur):
    """P_h: healing potions fight f needs beyond the hero's own HP above the 40% drink line (cur - 4/10 max HP), with
    the mean heal of one Potion of Healing (F16)."""
    heal = max(potion_heal_v4(ld['mx'])[2], 1)
    return max(0, math.ceil((f['T'] * f['D'] - (cur - Fraction(V5_DRINK_PCT, 100) * ld['mx'])) / heal))


def finishable_v5(f, ld, cur, bhp, bmx):
    """The fight can be finished now: ok band, the Butcher at half his HP or less, and the potions cover what the hero's
    own HP above the drink line does not, plus one spare."""
    return f['band'] == 'ok' and 2 * bhp <= bmx and ld['pots'] >= potions_beyond_hp_v5(f, ld, cur) + 1


def budget_v5(ld, f, n):
    """Damage to the Butcher one trip buys: the HP above the 40% line of full HP (0.6 max HP) plus n potions, spent at
    his rate D, times the net damage rate g - regen; 0 outside the ok / long bands."""
    if f['band'] not in V5_VIABLE:
        return Fraction(0)
    if not f['D']:
        return math.inf
    heal = potion_heal_v4(ld['mx'])[2]
    return (Fraction(100 - V5_DRINK_PCT, 100) * ld['mx'] + n * heal) / f['D'] * (f['g'] - f['regen'])


def guard_need_v5(prompt, need_b):
    """(potions to leave town, who) while he waits at the stairs. Butcher mission: his own number at his remembered HP
    (boss_need_v4 = min(P+3, 20)), never below town_base_v4 (a wounded Butcher needs fewer than the full-HP floor 8);
    Skeleton King mission: the larger of the version-4 target and his number; outside the ok / long bands: version 4."""
    need4, boss4 = town_target_v4(prompt)
    who4 = {'king': ' for the Skeleton King', 'butcher': ' for the Butcher', None: ''}[boss4]
    if need_b is None:
        return need4, who4
    if parse_mission(prompt) == 'butcher':
        return max(town_base_v4(parse_hero_v4(prompt)['L']), need_b), ' for the Butcher at the stairs'
    return (need_b, ' for the Butcher at the stairs') if need_b >= need4 else (need4, who4)


def _short_v5(f, wname):
    return {'melee_first': 'you hold no melee weapon', 'cannot': f'your {wname} cannot kill him',
            'barely': f'your {wname} barely hurts him', 'long': f'your {wname} needs over 2 minutes'}[f['band']]


def _remedy_how_v5(g, pots):
    if g['band'] == 'ok':
        return f"about {_seconds(g['T'])} seconds and {_plural(g['P'], 'healing potion')} (you carry {pots})"
    return {'cannot': 'still cannot kill him', 'barely': 'barely hurts him', 'long': 'over 2 minutes',
            'melee_first': 'no melee weapon'}[g['band']]


def plan_v5(prompt, scene, guard, l1):
    """The plan in town / on dungeon level 1 while the Butcher waits at the stairs with guard['hp'] HP: dict(code, ld, f,
    rem, need, who, need_b, finish). Codes, first that holds: weapon (a better weapon in reach: the pack, the floor on
    dungeon level 1, or the open shop when its damage budget beats the potions the gold buys), ready (ok band and the
    potions he needs, or finishable), potions (town: buy) / level1 (explore what is left of dungeon level 1) in that
    order in town and the reverse on dungeon level 1, stairs (ok/long band and V5_TRIP_MIN+ potions), sell (ok/long band,
    fewer), weak."""
    ld = loadout_v4(prompt)
    f = fight_v4(ld, 'butcher', guard['hp'])
    town = tuple(scene) == TOWN
    rem = remedy_v4(prompt, ld, 'butcher', f, guard['hp']) if f['band'] != 'ok' else None
    if rem is not None and town and rem[1][0] == 'floor':
        rem = None
    if rem is not None and rem[1][0] == 'buy':
        it, where, g = rem
        after = budget_v5(swap_v4(ld, it, where), g, ld['pots'] + max(0, ld['gold'] - where[1]) // POTION_PRICE)
        if not after > budget_v5(ld, f, ld['pots'] + ld['gold'] // POTION_PRICE):
            rem = None       # all the gold in potions does more (king1-facts/s4150025#364: 71 against 29)
    need_b = boss_need_v4(f) if f['band'] in V5_VIABLE else None
    need, who = guard_need_v5(prompt, need_b)
    finish = finishable_v5(f, ld, ld['cur'], guard['hp'], guard['mx'])
    ready = f['band'] == 'ok' and (ld['pots'] >= need or finish)
    buy = f['band'] in V5_VIABLE and ld['pots'] < need and ld['gold'] >= POTION_PRICE
    explore = (sighting_v5(prompt)['ex5'] if not town else (l1 or 0)) > 0
    if rem is not None:
        code = 'weapon'
    elif ready:
        code = 'ready'
    elif town and buy:
        code = 'potions'
    elif explore:
        code = 'level1'
    elif buy:
        code = 'potions'
    elif f['band'] in V5_VIABLE and ld['pots'] >= V5_TRIP_MIN:
        code = 'stairs'
    elif f['band'] in V5_VIABLE:
        code = 'sell'
    else:
        code = 'weak'
    return dict(code=code, ld=ld, f=f, rem=rem, need=need, who=who, need_b=need_b, finish=finish)


def butcher_memory_line_v5(plan, guard):
    """(B) town / dungeon level 1: his HP when the hero left and the fight numbers for the gear now (and for the weapon
    the plan names)."""
    ld, f, rem = plan['ld'], plan['f'], plan['rem']
    w = _weapon_name(ld)
    if f['band'] == 'ok':
        v = f"With your {w}: about {_seconds(f['T'])} seconds and {_plural(f['P'], 'healing potion')} (you carry {ld['pots']})."
    elif f['band'] == 'long':
        v = f"With your {w}: over 2 minutes (you carry {ld['pots']})."
    else:
        s = _short_v5(f, w)
        v = s[0].upper() + s[1:] + '.'
    if rem is not None:
        it, where, g = rem
        v += f" With the {it['name']} {_where(where)}: {_remedy_how_v5(g, ld['pots'])}."
    return V5_MEMORY.format(hp=guard['hp'], mx=guard['mx']) + ' ' + v


def money_v5(gold, missing):
    """The gold fact while he waits at the stairs (no 'earn more on dungeon levels 1-2': the plan covers level 1)."""
    k = gold // POTION_PRICE
    if missing <= 0:
        return ''
    if k >= missing:
        return f' Your {gold} gold buys all {missing}.'
    return f" Your {gold} gold buys {k or 'none'} of the {missing} missing: sell loot from your pack."


def town_ready_v5(prompt, plan, guard):
    """(A) the town 'Ready to go down' line while he waits at the stairs (whether or not the stairs DOWN are listed)."""
    code, ld, f, rem, need, who = plan['code'], plan['ld'], plan['f'], plan['rem'], plan['need'], plan['who']
    pct, gold, pots = ld['pct'], ld['gold'], ld['pots']
    have = _plural(pots, 'healing potion')
    hp_ok = pct >= TOWN_HP_PCT
    hp_not_yet = f'Ready to go down: not yet (HP {pct}%, needs {TOWN_HP_PCT}%+). Pepin heals for free.'
    w = _weapon_name(ld)
    if code in ('ready', 'potions'):
        if code == 'ready' and pots < need:          # finishable
            if hp_ok:
                return f"Ready to go down: yes (HP {pct}%, {have}): the Butcher has only {guard['hp']} HP left: finish him at the stairs."
            return hp_not_yet
        if hp_ok and pots >= need:
            return f'Ready to go down: yes (HP {pct}%, {have}).'
        why, help_ = [], []
        if not hp_ok:
            why.append(f'HP {pct}%, needs {TOWN_HP_PCT}%+')
            help_.append('heals for free')
        short = pots < need and gold >= POTION_PRICE
        if short:
            why.append(f'{have}, needs {need}+{who}')
            help_.append(f'sells potions for {POTION_PRICE} gold')
        return f"Ready to go down: not yet ({'; '.join(why)}). Pepin {' and '.join(help_)}." + (money_v5(gold, need - pots) if short else '')
    if not hp_ok:
        return hp_not_yet
    if code == 'weapon':
        it, where, g = rem
        verb = 'equip' if where[0] == 'pack' else 'buy'
        return f"Ready to go down: not yet: {V5_WAITS} and {_short_v5(f, w)}; {verb} the {it['name']} {_where(where)} first."
    if code == 'level1':
        return f'Ready to go down: yes, to explore the rest of dungeon level 1 (HP {pct}%, {have}); not below it: {V5_WAITS}.'
    if code == 'stairs':
        if gold >= POTION_PRICE:       # only a long band with the potions he needs
            return f'Ready to go down: yes (HP {pct}%, {have}): fight the Butcher at the stairs.'
        return f'Ready to go down: yes (HP {pct}%, {have}; gold {gold} cannot buy another): fight the Butcher at the stairs.'
    if code == 'sell':
        return (f'Ready to go down: not yet ({have}; {V5_TRIP_MIN}+ are needed to fight the Butcher at the stairs and gold {gold} '
                f'cannot buy one): sell loot from your pack to Griswold or Adria.')
    return (f'Ready to go down: not yet: {V5_WAITS} and {_short_v5(f, w)}; buy a better weapon from Griswold (sell loot from '
            f'your pack for gold).')


def pace_l1_v5(prompt, plan, guard):
    """(C) the dungeon-level-1 Pace line while he waits at the stairs. A 'go' plan (ready, finish, stairs) says go down
    only with HP 75%+ (the town's test) and a stairs-DOWN option listed; otherwise up to town first."""
    code, ld, f, rem = plan['code'], plan['ld'], plan['f'], plan['rem']
    head = f"Pace: character level {ld['L']}. Going below dungeon level 1: "
    w = _weapon_name(ld)
    if code in ('ready', 'stairs'):
        if ld['pct'] < TOWN_HP_PCT:
            tail = f"not yet: go up to town first (HP {ld['pct']}%; Pepin heals for free), then fight the Butcher at the stairs"
        elif not any('stairs DOWN' in t for t in parse_option_texts(prompt)):
            tail = 'not yet: go up to town first, then come back and fight the Butcher at the stairs'
        elif code == 'stairs':
            tail = f'go down and fight the Butcher at the stairs, keeping {V5_TRIP_MIN} healing potions for the way up'
        elif ld['pots'] < plan['need']:              # finishable
            tail = f"go down and finish the Butcher at the stairs (he has only {guard['hp']} HP left)"
        else:
            tail = 'go down and fight the Butcher at the stairs'
    elif code == 'weapon':
        it, where, g = rem
        verb = {'pack': 'equip', 'floor': 'pick up', 'buy': 'buy'}[where[0]]
        tail = f"not yet: {verb} the {it['name']} {_where(where)} first ({V5_WAITS})"
    elif code == 'level1':
        tail = f'not yet: explore the rest of this level first ({V5_WAITS} and you are not ready for him)'
    elif code == 'potions':
        k = min(ld['gold'] // POTION_PRICE, plan['need'] - ld['pots'])
        tail = f"not yet: go up to town and buy healing potions first (your {ld['gold']} gold buys {k})"
    elif code == 'sell':
        tail = (f'not yet: go up to town and sell loot from your pack for healing potions ({V5_TRIP_MIN}+ are needed to fight '
                f'the Butcher at the stairs)')
    else:
        tail = f'not yet: go up to town and buy a better weapon from Griswold ({_short_v5(f, w)})'
    return head + tail + '.'


def stairs_line_v5(history, prompt, scene):
    """(D) dungeon level 2, either mission, the hero at the stairs up (at_stairs_v5): (line, case) or None. The fight is
    computed at his HP now. Cases, first that holds: equip (a better weapon in the pack that fights in the ok / long band),
    weak (the worn weapon outside those bands), finish (finishable, no Retreat line), fight (V5_DRINK_MIN+ potions),
    fight_on (exactly V5_TRIP_MIN potions, HP V5_TRIP_HP%+, no Retreat line), up."""
    at = at_stairs_v5(history, prompt, scene)
    if at is None:
        return None
    bhp, bmx = at['b'][0], at['b'][1]
    ld = loadout_v4(prompt)
    f = fight_v4(ld, 'butcher', bhp)
    rem = remedy_v4(prompt, ld, 'butcher', f, bhp) if f['band'] != 'ok' else None
    ret = retreat_line(prompt, scene) is not None
    blocked = at['blocked']
    pots = ld['pots']
    st = None if blocked else _plural(at['st'], 'step')
    where = V5_BLOCKED if blocked else f'you are at the stairs up ({st} away)'
    if rem is not None and rem[1][0] == 'pack' and rem[2]['band'] in V5_VIABLE:
        return (f"Butcher: {where}: equip the {rem[0]['name']} in your pack now; with it {_remedy_how_v5(rem[2], pots)}.",
                'equip')
    if f['band'] not in V5_VIABLE:
        s = _short_v5(f, _weapon_name(ld))
        if blocked:
            return (f'Butcher: {s} and {V5_BLOCKED}: drink at {V5_DRINK_PCT}% HP and take the stairs up as soon as they are '
                    f'listed again;{V5_STAYS}', 'weak')
        return (f'Butcher: {s}: go up the stairs now ({st} away), drinking at {V5_DRINK_PCT}% HP on the way, and come back '
                f'stronger.', 'weak')
    if not ret and finishable_v5(f, ld, ld['cur'], bhp, bmx):
        return f'Butcher: {where} and he is nearly beaten ({bhp} HP left): finish him, drinking at {V5_DRINK_PCT}% HP.', 'finish'
    if pots >= V5_DRINK_MIN:
        if f['band'] == 'long':
            although = ' although a whole fight takes over 2 minutes'
        elif pots < boss_need_v4(f):
            although = ' although a whole fight needs more healing potions than you carry'
        else:
            although = ''
        return (f'Butcher: {where} and he does not heal while you are on another level: fight him here{although}, drinking '
                f'at {V5_DRINK_PCT}% HP; when {V5_TRIP_MIN} healing potions are left, go up once your HP is below '
                f'{V5_TRIP_HP}%.', 'fight')
    if pots == V5_TRIP_MIN and ld['pct'] >= V5_TRIP_HP and not ret:
        return (f'Butcher: {where} and he does not heal while you are on another level: fight him here until your HP is '
                f'below {V5_TRIP_HP}%, then go up; your {V5_TRIP_MIN} healing potions are for the way up.', 'fight_on')
    have = _plural(pots, 'healing potion')
    if blocked:
        return (f'Butcher: {have} left and {V5_BLOCKED}: attack him, drinking at {V5_DRINK_PCT}% HP, and take the stairs up '
                f'as soon as they are listed again;{V5_STAYS}', 'up')
    return (f'Butcher: {have} left: go up the stairs now ({st} away), drinking at {V5_DRINK_PCT}% HP on the way; he does '
            f'not heal while you are on another level.', 'up')


def until_v5(prompt, scene, mission, pace):
    """(E) Butcher mission, dungeon level 2, the Butcher not in sight, no Retreat line: the Pace verdict's band at full HP
    is melee_first / cannot / barely / long and no weapon in reach reaches the ok band -> the Pace line gets V5_UNTIL
    (gate9-new-butcher/s4150019 went straight back up on 'far more healing potions than your 3')."""
    if pace is None or mission != 'butcher' or tuple(scene) != (2, False) or visible_boss_v4(prompt) is not None:
        return pace
    if retreat_line(prompt, scene) is not None:
        return pace
    ld = loadout_v4(prompt)
    f = fight_v4(ld, 'butcher')
    if f['band'] not in ('melee_first', 'cannot', 'barely', 'long'):
        return pace
    rem = remedy_v4(prompt, ld, 'butcher', f)
    if rem is not None and rem[2]['band'] == 'ok':
        return pace
    return pace + V5_UNTIL


def facts_lines_v5(history, prompt, lines):
    """Version 5 from the version-4 lines (history records carry b5 / up5 / ex5). Line order: town Ready, Butcher, town
    visit, Weapon, Hit, last move; dungeon level 1 Pace, Butcher, Retreat, Weapon, Hit, last move; dungeon level 2 Pace,
    Retreat, Boss, Butcher (stairs), Weapon, Hit, last move (at most 7 lines)."""
    scene = parse_scene(prompt)
    out = list(lines)
    if scene in (TOWN, (1, False)):
        guard, l1 = butcher_memory_v5(history, scene)
        if guard is None:
            return out
        plan = plan_v5(prompt, scene, guard, l1)
        if scene == TOWN:
            if not (plan['code'] == 'weapon' or plan['ld']['pots'] >= plan['need']):
                out = [x for x in out if not x.startswith('Weapon: ')]   # (G) potions first, as the plan says
            new, head = town_ready_v5(prompt, plan, guard), 'Ready to go down: '
        else:
            new, head = pace_l1_v5(prompt, plan, guard), 'Pace: '
        i = next((k for k, x in enumerate(out) if x.startswith(head)), None)
        if i is None:
            out.insert(0, new)
            i = 0
        else:
            out[i] = new
        out.insert(i + 1, butcher_memory_line_v5(plan, guard))
        return out
    if scene == (2, False):
        mission = parse_mission(prompt)
        i = next((k for k, x in enumerate(out) if x.startswith('Pace: ')), None)
        if i is not None:
            out[i] = until_v5(prompt, scene, mission, out[i])
        sl = stairs_line_v5(history, prompt, scene)
        if sl is not None:
            line, case = sl
            if mission == 'butcher' and i is not None and case in ('equip', 'finish', 'fight', 'fight_on'):
                out[i] = _EARN_MORE_V5.sub('', out[i])   # (F) at the stairs the fight is here, not on dungeon levels 1-2
            j = max((k for k, x in enumerate(out) if x.startswith(('Pace: ', 'Retreat: ', 'Boss: '))), default=-1)
            out.insert(j + 1, line)
    return out


def facts_lines(history, current_prompt):
    """At most 3 digested fact lines (4 in version 2 on level 3 / in the tomb, up to 5 in version 3, up to 6 in
    version 4: Pace, Retreat, Boss, Weapon, Hit, last move in the dungeon; up to 7 in version 5: the 'Butcher:' line
    after them on dungeon level 2)."""
    scene = parse_scene(current_prompt)
    v3 = FACTS_VERSION >= 3
    if FACTS_VERSION >= 4:
        mission = parse_mission(current_prompt)
        if scene == TOWN:
            lines = [town_ready_line(current_prompt, parse_option_texts(current_prompt)), town_visit_line(history),
                     weapon_line_v4(current_prompt, scene, mission), hit_line(current_prompt, scene, mission)]
        else:
            lines = [pace_line(scene, parse_level(current_prompt), mission, parse_potions(current_prompt), current_prompt),
                     retreat_line(current_prompt, scene), boss_line(current_prompt, scene, mission),
                     weapon_line_v4(current_prompt, scene, mission), hit_line(current_prompt, scene, mission)]
        lines.append(last_move_line(history, scene, parse_shop(current_prompt)))
        lines = [x for x in lines if x]
        return facts_lines_v5(history, current_prompt, lines) if FACTS_VERSION >= 5 else lines
    if scene == TOWN:
        lines = [town_ready_line(current_prompt, parse_option_texts(current_prompt)), town_visit_line(history)]
        if v3:
            mission = parse_mission(current_prompt)
            lines += [weapon_line_v3(current_prompt, scene, mission), hit_line(current_prompt, scene, mission)]
    else:
        mission = parse_mission(current_prompt)
        if v3:
            lines = [pace_line(scene, parse_level(current_prompt), mission, parse_potions(current_prompt), current_prompt),
                     weapon_line_v3(current_prompt, scene, mission), hit_line(current_prompt, scene, mission)]
        else:
            lines = [pace_line(scene, parse_level(current_prompt), mission, parse_potions(current_prompt)),
                     weapon_line(current_prompt, scene, mission)]
    lines.append(last_move_line(history, scene, parse_shop(current_prompt)))
    return [x for x in lines if x]


def insert_facts(prompt, lines):
    """Prompt with the fact lines inserted immediately before the "Options:" line (unchanged when lines is empty)."""
    if not lines:
        return prompt
    for x in lines:
        if '\n' in x or not x.startswith(FACT_PREFIXES):
            raise ValueError('not a fact line: ' + repr(x))
    i = prompt.find(OPTIONS_MARK)
    if i < 0:
        raise ValueError('prompt has no "Options:" line')
    return prompt[:i] + '\n' + '\n'.join(lines) + prompt[i:]


def strip_facts(prompt):
    """Inverse of insert_facts: drop fact lines from the part before "Options:"."""
    i = prompt.find(OPTIONS_MARK)
    if i < 0:
        raise ValueError('prompt has no "Options:" line')
    head = [x for x in prompt[:i].split('\n') if not x.startswith(FACT_PREFIXES)]
    return '\n'.join(head) + prompt[i:]


def line_type(line):
    """Short tag for reports."""
    if line.startswith('Butcher: '):     # version 5 (no line of versions 1-4 starts so)
        if line.startswith('Butcher: waiting at the stairs'):
            return 'butcher_memory'
        if ' in your pack now; with it ' in line:
            case = 'equip'
        elif 'he is nearly beaten' in line:
            case = 'finish'
        elif 'fight him here until your HP is below' in line:
            case = 'fight_on'
        elif 'fight him here' in line:
            case = 'fight'
        elif 'come back stronger' in line or f'{V5_BLOCKED}: drink at ' in line:
            case = 'weak'
        else:
            case = 'up'
        return f'butcher_stairs_{case}' + ('_blocked' if V5_BLOCKED in line else '')
    if line.startswith('Pace: ') and 'Going below dungeon level 1: go down' in line:       # version 5 (C)
        return 'pace_guard_go'
    if line.startswith('Pace: ') and 'Going below dungeon level 1: not yet' in line:
        return 'pace_guard_not_yet'
    if line.startswith('Pace: '):
        if 'is outside your mission' in line:
            return 'pace_outside_mission'
        for band, key in V4_VERDICT_KEYS:  # version 4 K: the band of the verdict itself (never the remedy sentence)
            if key in line:
                tag = ('pace_king_' if 'Skeleton King' in line else 'pace_butcher_') + band
                return tag + ('_below_here' if 'below pace here' in line else '')
        tag = 'pace_king_' if 'Skeleton King' in line else 'pace_'
        # 2026-09-23 round 9: version 3's 'ready (about N swings ...).' counts as ready (it was tagged not_ready)
        tag += 'ready' if line.endswith(': ready.') or ': ready (' in line else 'not_ready'
        return tag + ('_below_here' if 'below pace here' in line else '')
    if line.startswith('Boss: '):
        return ('boss_king' if line.startswith('Boss: the Skeleton King') else 'boss_butcher') + \
            ('_other_mission' if 'not needed for your mission' in line else '')
    if line.startswith('Retreat: '):
        crowd = 'enemies next to you' in line or f'enemies within {RETREAT_NEAR_STEPS} steps' in line
        return 'retreat_crowd' if crowd else 'retreat' if 'Pepin heals' in line else 'retreat_boss'
    if line.startswith('Weapon: '):
        return 'weapon_v4' if line.startswith('Weapon: damage per hit against') else 'weapon'
    if line.startswith('Hit chance: '):
        return 'hit'
    if line.startswith('Ready to go down: '):
        return 'town_ready' if line.startswith('Ready to go down: yes') else 'town_not_ready'
    if line.startswith('Already done this town visit'):
        return 'town_visit'
    for start, tag in (('You just came up', 'last_up'), ('You just went down', 'last_down'), ('You just closed', 'last_closed'),
                       ('You just opened', 'last_opened_shop'), ('You just reopened', 'last_reopened_shop'),
                       ('You just talked to a townsperson', 'last_townsperson'), ('You just took off', 'last_took_off')):
        if line.startswith(start):
            return tag
    return 'other'


# ----------------------------------------------------------------------------------------------- live helper
class FactsTracker:
    """Live twin of history_from_game_log + facts_lines for the runner.

    Per decision:   user = render(...);  lines = tracker.facts(user);  user = insert_facts(user, lines)
    After it ran:   tracker.observe(row)   # the same dict the runner writes to decisions.jsonl
    The logged prompt may carry the fact lines; parsing ignores them.
    """

    def __init__(self):
        self.history = []

    def observe(self, row):
        if int(row['n']) != len(self.history) + 1:
            raise ValueError(f'FactsTracker expects decisions 1, 2, ... in order; got n={row["n"]} after {len(self.history)}')
        self.history.append(record_from_game_row(row))

    def facts(self, prompt):
        return facts_lines(self.history, prompt)

    def render(self, prompt):
        return insert_facts(prompt, self.facts(prompt))
