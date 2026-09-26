# R18-F pre-registration: a zero-training probe of the Town Portal scroll portal-v1 (2026-09-07)

## 0. Background

Design requirement: the Town Portal scroll has to be built. R18-C showed that the retreat cycle can turn, but every retreat has to walk back to the stairs, and 25–28% of retreats die on the way;
the scroll replaces "walk to the stairs" with "open a door where you stand", pushing the cost of a town trip to the minimum, at a price of 200 gold per scroll and returning to the spot where it was cast.

## 1. Interface (default off = unchanged bit for bit)

- Engine: the witch Adria enters the headless shop (`patches/0013-headless-witch-vendor.patch`, drift gate PASS); bridge: the portal gate splits into two branches
  (`portal_to_town` L2+→town and `portal_return` town→L2+, both requiring Python authorization; all other teleport messages are still refused); `configure_resource_protocol(..., portal=True)`;
  `configure_resource_portal`; `act_cast_town_portal`; at most 2 per game; the observation always has `portal_enabled`, and when on also the authorization/count/portal state/scroll count.
  Cross-bridge identity: when off, the receipts of the four teleport messages in town and in the dungeon, and the whole resource_state, are bit-identical to the old bridge (only one extra `portal_enabled` key).
- Python: `resource_portal="portal-v1"` requires l2-town-v1 + coach-v03 + retreat-v1; `PortalService`: restocks scrolls at Adria during a town trip
  (buying only when enough is left for a 4-slot potion belt), opens a portal first when the retreat rule fires on L2+ and a scroll is carried (otherwise retreats by the stairs), walks through, runs the town flow as before and comes back through the town-side portal;
  a failure never ends the game. Three-level priority chain: portal → retreat → town service.
- Known open issues: the return lands on the casting spot, facing the surviving crowd, with HP unchanged; a scroll costs 200 gold; a failed placement still consumes the scroll.

## 2. Probe

Driver `r10-staging/run_r18f_probe.py` (not published); pool 2_133 with same-seed pairs; worker 7e31dc54; clock completion-l2-r18c; manager coach-v03;
economy sustain-loot-v1; retreat retreat-v1; two arms: ctl `r18f-retreat` (re-run) and P `r18f-retreat-portal` (+ portal-v1).

Criteria: regression (hard): the ctl rows are bit-identical to the R18-G control rows; the prefix up to the first L2 arrival is identical game by game in both arms; otherwise VOID.
Main hypotheses (reported, not judged; directions fixed in advance): H1 P's L2 hazard per 1000 ticks is lower than ctl; H2 paired saved−lost > 0 with UCB95 < 0;
H3 P's share of "deaths during retreat" (portal and stairs combined) is below ctl's 25%; H4 P's number of games descending again and L2+ occupancy share are not lower than ctl (the portal keeps depth);
H5 scroll economy: scrolls bought, portal count, failure-reason distribution, final gold.

## 3. Certification chain

New bridge (build-res rebuilt from the same source) → full suite with 0 failures (including `tests/test_resource_portal.py`) → probe regression equal → two-way re-bake on the new bridge/final bytes 4/4 + 4/4 →
the sha256 of this file recorded in the ledger.

## 4. Seeds and files

2_133 consumes 2 more groups (18 in total); the virgin pools are untouched. New files: this pre-registration, `run_r18f_probe.py` (not published), `resource_portal.py`, `tests/test_resource_portal.py`,
`patches/0013-*.patch`, `r18f-probe/` (not published). Changed protocol files: `src/resource_protocol.hpp`, `src/diablogym.cpp`, `resource_protocol.py`, `env.py`, `options_env.py`, the probe.
