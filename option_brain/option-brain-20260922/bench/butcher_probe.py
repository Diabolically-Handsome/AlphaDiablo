# Round-9 exam seed check (PREREG-ROUND9.md decision 6; PREREG-ROUND9-REVIEW.md "Butcher quest check"), copied 2026-09-23
# from a local work directory (butcher_probe.py, sha256 27217e40829414441d39f4acf2bde9c8c95d633c8b37203c264e467461c6bc98;
# its 80-seed output run1.json sha256 ce223b1afbfd72731f630a89ef03e596127af367735d3ee54a2742b79a7d7868: the 16 Butcher
# exam seeds 16/16 have the quest, engine and quest_lottery.py agree on 80/80 seeds). One change from the scratch copy:
# the throwaway save directory is created next to the output file (was: next to this script), so that a run from
# bench/ writes nothing into the experiment directory. A declared exception to "never used": it opens each seed at
# tick 0 only (no step, no model, no dungeon), checks the checkpoint digest is unchanged, and writes nothing to live/
# or relabel-*; put the output under $AD_ROOT/seed-audit/. CPU, seconds.
"""Tick-0 engine probe: is Q_BUTCHER available in the game a seed starts?

Reads only. For each seed: bridge reset(seed) (same bridge/assets/data as game_support_r3.Controller),
no step, then read:
  - dMonster over the whole town map via probe_tile (read-only). InitTowners places the Wounded
    Townsman (TOWN_DEADGUY, towners.tsv row 3, tile 24,32) only if
    Quests[Q_BUTCHER]._qactive != QUEST_NOTAVAIL && != QUEST_DONE (towners.cpp:733-737).
  - manual_checkpoint(): king_present (Quests[Q_SKELKING] != NOTAVAIL), DungeonSeeds (seeds[15] = quest seed).
  - butcher_events(): kills / quest_done (must be 0 / False at tick 0).
Checks that the checkpoint digest is unchanged by the probes.
usage: python butcher_probe.py <out.json> <seed> [<seed> ...]
"""
import sys, json, hashlib, time
from pathlib import Path

sys.path.insert(0, '$AD_WORKSPACE/experiments/strategy-brain-sft-20260922')
sys.path.append('$AD_WORKSPACE/experiments/option-brain-20260922')
import game_support_r3 as gs          # noqa: E402  (same import chain live_runner uses)
from game_support_r3 import load_bridge  # noqa: E402
import session as sess                # noqa: E402  (module Controller's Session comes from)
import quest_lottery as ql            # noqa: E402


def canon(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    out = Path(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2:]]
    b, build = load_bridge()
    b.manual_configure()
    save = out.resolve().parent / f'save-{int(time.time()*1000)}'
    save.mkdir()
    b.init(sess.ASSETS, str(save), sess.DATA, 0, False)
    rows = []
    for seed in seeds:
        b.reset(seed)
        obs = b.manual_observe()
        ck0 = b.manual_checkpoint()
        d0 = canon(ck0)
        be = dict(b.butcher_events())
        occ = {}
        for x in range(112):
            for y in range(112):
                m = b.probe_tile(x, y)['monster']
                if m:
                    occ.setdefault(m, []).append((x, y))
        ck1 = b.manual_checkpoint()
        dead = b.probe_tile(24, 32)['monster']
        tavern = b.probe_tile(55, 62)['monster']
        qseed_py = ql.mt19937_outputs(seed, 16)[15]
        gone = ql.missing_quests(seed)
        rows.append(dict(
            seed=seed, tick=obs['tick'], scene=list(obs['scene']),
            deadguy_tile_24_32=dead, tavern_tile_55_62=tavern,
            towner_indices=sorted(occ), towner_positions={str(k): v[:3] for k, v in sorted(occ.items())},
            engine_butcher_available=dead != 0,
            engine_king_present=bool(ck0['king_present']),
            butcher_events=be,
            engine_quest_seed=int(ck0['seeds'][15]), lottery_quest_seed=qseed_py,
            lottery_butcher='butcher' not in gone, lottery_king='skeleton_king' not in gone,
            probes_read_only=canon(ck1) == d0,
        ))
        r = rows[-1]
        r['session_module'] = sess.__file__
        print(seed, 'tick', r['tick'], 'scene', r['scene'], 'deadguy', dead, 'tavern', tavern, 'n_idx', len(occ),
              'engB', r['engine_butcher_available'], 'lotB', r['lottery_butcher'],
              'engK', r['engine_king_present'], 'lotK', r['lottery_king'],
              'qseed_eq', r['engine_quest_seed'] == qseed_py, 'be', be, 'ro', r['probes_read_only'], flush=True)
    b.end_game()
    out.write_text(json.dumps(dict(bridge_sha256=build['bridge_sha256'], engine_sha256=build['engine_sha256'],
                                   assets=sess.ASSETS, data=sess.DATA, is_spawn=bool(b.probe_is_spawn()),
                                   rows=rows), indent=1))


if __name__ == '__main__':
    main()
