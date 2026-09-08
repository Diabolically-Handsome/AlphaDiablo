"""R18-X L2 stairs-geometry replay (2026-09-07 morning) — a DIAGNOSTIC, not an exam.

Replays the R18-C/F retreat control arm (worker 7e31dc54, coach-v03, sustain-loot-v1,
completion-l2-r18c, retreat-v1, hunt all, stochastic decoding, per-episode reseed) on the
already-consumed pairing seeds and records, on main L2 only, how far the down-stairs are and
how close the pair ever gets.  Read-only instrumentation: raw / OptionsEnv state / the
monster-aware BFS planner; no RNG is touched.  The v3 row columns are re-derived so the run
can be checked bit-for-bit against the recorded arm rows.

usage: python run_r18x_l2geom.py SEED_LO SEED_HI OUT.json
"""
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train" / "runs" / "r10-staging"))

import probe_r17_deployment as P  # noqa: E402
import diablogym.options_env as OE  # noqa: E402
from diablogym.options_env import OptionsEnv, FARM, DIVE, RESUPPLY  # noqa: E402

bridge = OE.bridge
WORKER_ZIP = str(ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip")
ENV_OVERRIDES = {"explore_global_hunt": True, "farm_scene_cap": 3600,
                 "reset_layer_clock_on_window": True, "reward_economy": "v4",
                 "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                 "resource_service_policy": "sustain-loot-v1",
                 "resource_readiness_law": "coach-v03",
                 "worker_time_protocol": "completion-l2-r18c",
                 "resource_retreat": "retreat-v1"}
MAX_STEPS = 6000
PATH_EVERY = 10
NEAR = 6
FORCE_A11 = len(sys.argv) > 4 and sys.argv[4] == "force-a11"
RANGED_AI = frozenset({3, 7, 9, 14, 16, 19, 20, 25, 35, 36, 37, 38})  # engagement.RANGED_AI (copied, read-only use)
import os as _os
ENV_OVERRIDES.update(json.loads(_os.environ.get("R18X_OVERRIDES", "{}")))


def cheb(ax, ay, bx, by):
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def stairs_of(raw, msg):
    return [(int(t["x"]), int(t["y"])) for t in raw.get("triggers", []) if t.get("msg") == msg]


def alive_monsters(raw):
    return [m for m in raw.get("monsters", []) if int(m.get("hp", 0)) > 0 and int(m.get("type", -1)) != 109]


class Wrap:
    """Delegates everything to the real callback; only records the returned action."""

    def __init__(self, cb, rec):
        self._cb = cb
        self._rec = rec

    def __call__(self, obs, mask):
        a = self._cb(obs, mask)
        final = self._rec(int(a), mask)
        return a if final is None else final

    def __getattr__(self, name):
        return getattr(self._cb, name)


def run_seed(cb, seed):
    visits, traces = [], []
    st = {"cur": None, "l2_beat_ctx": None, "first_entry": None,
          "last_hp": None, "dmg": {}, "hits": {}, "last_attr": None}
    env_kwargs = dict(max_steps=MAX_STEPS, drink_sovereignty=True,
                      worker_observation_view="dual-v4-asymmetric-v3",
                      manager_observation_view="legacy-v3",
                      dive_live_sovereignty=True, reward_economy="v2")
    env_kwargs.update(ENV_OVERRIDES)

    def rec_action(a, mask):
        cur = st["cur"]
        ctx = st["l2_beat_ctx"]
        if cur is None or ctx is None:
            return None
        avail11 = bool(np.asarray(mask)[11]) if len(mask) > 11 else False
        final = None
        if FORCE_A11 and ctx["opt"] == DIVE and avail11:
            # counterfactual: the descend macro whenever the manager holds a DIVE window on L2+
            final = 11
            a = 11
            cur["forced_a11"] += 1
        cur["actions"][a] += 1
        cur["a11_avail_beats"] += int(avail11)
        if ctx["d"] is not None and ctx["d"] <= 2:
            cur["a11_avail_d_le2_beats"] += int(avail11)
        if ctx["opt"] == DIVE:
            cur["dive_actions"][a] += 1
            cur["dive_beats"] += 1
            cur["a11_avail_dive_beats"] += int(avail11)
            if ctx["d"] is not None:
                cur["dive_d_sum"] += ctx["d"]
                if ctx["d"] <= 5:
                    cur["dive_d_le5_beats"] += 1
                    cur["dive_d_le5_a11"] += int(a == 11)
                    cur["dive_d_le5_a11_avail"] += int(avail11)
        st["l2_beat_ctx"] = None
        return final

    def close_visit(reason, beat, raw):
        cur = st["cur"]
        if cur is None:
            return
        cur["exit_beat"] = beat
        cur["beats"] = beat - cur["entry_beat"]
        cur["exit_reason"] = reason
        cur["kills"] = int(raw.get("monster_kill_total", 0)) - cur["kills0"]
        cur["hp1"] = int(raw.get("hp", 0))
        cur["actions"] = dict(cur["actions"])
        cur["dive_actions"] = dict(cur["dive_actions"])
        cur["path_checks"] = cur["path_checks"]
        visits.append(cur)
        st["cur"] = None

    def attribute_damage(raw, dl):
        """Heuristic damage attribution (diagnostic): hp drop since the last beat goes to the nearest
        adjacent (<=2) alive monster, else the nearest ranged-AI monster within 12, else unknown."""
        hp = int(raw.get("hp", 0))
        last = st.get("last_hp")
        st["last_hp"] = hp
        if last is None or hp >= last:
            return
        drop = last - hp
        px, py = int(raw["player_x"]), int(raw["player_y"])
        mons = alive_monsters(raw)
        near = sorted(((cheb(px, py, m["x"], m["y"]), i, m) for i, m in enumerate(mons)),
                      key=lambda t: (t[0], t[1]))
        pick = None
        for d, _, m in near:
            if d <= 2:
                pick = m
                break
        if pick is None:
            for d, _, m in near:
                if d <= 12 and int(m.get("ai", -1)) in RANGED_AI:
                    pick = m
                    break
        key = ("%d:%d" % (int(pick.get("type", -1)), int(pick.get("ai", -1)))) if pick else "unknown"
        dmg = st["dmg"].setdefault(str(dl), {})
        dmg[key] = dmg.get(key, 0) + drop
        hits = st["hits"].setdefault(str(dl), {})
        hits[key] = hits.get(key, 0) + 1
        st["last_attr"] = {"key": key, "dlvl": dl, "drop": drop, "hp_after": hp,
                           "near_count": sum(1 for d, _, _m in near if d <= 6)}

    def on_beat():
        raw = env.env._raw
        beat = int(env.env._steps)
        if raw.get("dead"):
            return
        dl = int(raw.get("dungeon_level", 0))
        setl = bool(raw.get("is_set_level"))
        attribute_damage(raw, dl)
        cur = st["cur"]
        if not (dl == 2 and not setl):
            if cur is not None:
                close_visit("ascend" if dl < 2 else ("descend" if dl > 2 else "scene"), beat, raw)
            st["l2_beat_ctx"] = None
            return
        px, py = int(raw["player_x"]), int(raw["player_y"])
        down = stairs_of(raw, bridge.WM_DIABNEXTLVL)
        up = stairs_of(raw, getattr(bridge, "WM_DIABPREVLVL", -1))
        d = min((cheb(px, py, sx, sy) for sx, sy in down), default=None)
        mons = alive_monsters(raw)
        near = sum(1 for m in mons if cheb(px, py, m["x"], m["y"]) <= NEAR)
        opt = (env._win or {}).get("opt")
        if cur is None:
            sx, sy = min(down, key=lambda s: cheb(px, py, *s)) if down else (None, None)
            cur = {"seed": seed, "visit": len(visits) + 1, "entry_beat": beat,
                   "d0": d, "dmin": d, "dmin_beat": beat, "hp0": int(raw.get("hp", 0)),
                   "max_hp": int(raw.get("max_hp", 0)), "kills0": int(raw.get("monster_kill_total", 0)),
                   "roster0": len(mons), "near0": near, "stairs": [sx, sy], "player0": [px, py],
                   "up_to_down": (min((cheb(ux, uy, sx, sy) for ux, uy in up), default=None)
                                  if sx is not None else None),
                   "mons_near_stairs0": (sum(1 for m in mons if cheb(sx, sy, m["x"], m["y"]) <= 8)
                                         if sx is not None else None),
                   "mons_between0": (sum(1 for m in mons
                                         if min(px, sx) - 3 <= int(m["x"]) <= max(px, sx) + 3
                                         and min(py, sy) - 3 <= int(m["y"]) <= max(py, sy) + 3)
                                     if sx is not None else None),
                   "actions": Counter(), "dive_actions": Counter(), "dive_beats": 0, "dive_d_sum": 0,
                   "a11_avail_beats": 0, "a11_avail_dive_beats": 0, "a11_avail_d_le2_beats": 0, "forced_a11": 0,
                   "dive_d_le5_beats": 0, "dive_d_le5_a11": 0, "dive_d_le5_a11_avail": 0,
                   "beats_obs": 0, "near_sum": 0, "hp_min": int(raw.get("hp", 0)),
                   "path_checks": {"n": 0, "avoid_reach": 0, "free_reach": 0, "avoid_none": 0,
                                   "avoid_end_d_sum": 0, "free_end_d_sum": 0},
                   "d_le2_beats": 0, "d_le5_beats": 0}
            st["cur"] = cur
            if st["first_entry"] is None:
                st["first_entry"] = {"beat": beat, "d0": d, "roster": len(mons), "up_to_down": cur["up_to_down"],
                                     "mons_between": cur["mons_between0"], "mons_near_stairs": cur["mons_near_stairs0"],
                                     "clvl": int(raw.get("char_level", 0)), "ac": int(raw.get("armor_class", 0))}
        cur["beats_obs"] += 1
        cur["near_sum"] += near
        cur["hp_min"] = min(cur["hp_min"], int(raw.get("hp", 0)))
        if d is not None:
            if d < (cur["dmin"] if cur["dmin"] is not None else 10**9):
                cur["dmin"], cur["dmin_beat"] = d, beat
            cur["d_le2_beats"] += int(d <= 2)
            cur["d_le5_beats"] += int(d <= 5)
        if d is not None and cur["beats_obs"] % PATH_EVERY == 1 and cur["stairs"][0] is not None:
            sx, sy = cur["stairs"]
            pc = cur["path_checks"]
            pc["n"] += 1
            try:
                pa = env.env._plan_descend_path(raw, sx, sy, avoid_monsters=True)
                pf = env.env._plan_descend_path(raw, sx, sy, avoid_monsters=False)
            except Exception as exc:  # planner is diagnostic here; never abort the replay
                pa = pf = None
                pc.setdefault("errors", []).append(repr(exc)[:200])
            if pa is None:
                pc["avoid_none"] += 1
                pc["avoid_end_d_sum"] += d
            else:
                ex, ey = (pa[-1][0], pa[-1][1]) if pa else (px, py)
                de = cheb(ex, ey, sx, sy)
                pc["avoid_end_d_sum"] += de
                pc["avoid_reach"] += int(de <= 1)
            if pf is None:
                pc["free_end_d_sum"] += d
            else:
                ex, ey = (pf[-1][0], pf[-1][1]) if pf else (px, py)
                de = cheb(ex, ey, sx, sy)
                pc["free_end_d_sum"] += de
                pc["free_reach"] += int(de <= 1)
        if cur["beats_obs"] % 10 == 1:
            traces.append([seed, cur["visit"], beat, d, int(raw.get("hp", 0)), near, opt])
        st["l2_beat_ctx"] = {"opt": opt, "d": d}

    wrap = Wrap(cb, rec_action)
    env_kwargs["workers"] = {FARM: wrap, DIVE: wrap}
    env = OptionsEnv(**env_kwargs)
    cb.on_beat = on_beat
    try:
        obs, _ = env.reset(seed=seed)
        cb.episode_reseed(seed)
        done = trunc = False
        while not (done or trunc):
            mask = np.asarray(env.action_masks(), dtype=bool)
            chosen = int(env.resource_option_choice(mask))
            if not mask[chosen]:
                for cand in (FARM, DIVE, RESUPPLY):
                    if mask[cand]:
                        chosen = cand
                        break
            obs, r, done, trunc, info = env.step(chosen)
        raw = env.env._raw
        beat = int(env.env._steps)
        if st["cur"] is not None:
            dl = int(raw.get("dungeon_level", 0))
            if raw.get("dead"):
                close_visit("death", beat, raw)
            elif dl == 2 and not raw.get("is_set_level"):
                close_visit("end", beat, raw)
            else:
                close_visit("ascend" if dl < 2 else "descend", beat, raw)
        resource = getattr(env, "resource_service", None)
        row = {"seed": seed, "depth": int(raw.get("dungeon_level", 0)), "died": bool(raw.get("dead")),
               "micro_steps": beat, "char_level": int(raw.get("char_level", 0)),
               "kills": int(raw.get("monster_kill_total", 0)), "gold": int(raw.get("gold", 0)),
               "max_main_depth": int(getattr(env.env, "_resource_max_main_depth", 0) or 0),
               "first_entry": st["first_entry"], "visits": visits, "traces": traces,
               "dmg_by_floor_type": st["dmg"], "hits_by_floor_type": st["hits"],
               "killer": (st["last_attr"] if raw.get("dead") else None)}
        return row
    finally:
        cb.on_beat = None
        env.close()


def main():
    lo, hi, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ident0 = P.source_identity(WORKER_ZIP)
    cb = P.load_zip_policy(WORKER_ZIP, stochastic=True)
    cb.diablogym_worker_action12_mode = "environment-mask"
    rows = []
    t0 = time.time()
    for seed in range(lo, hi + 1):
        rows.append(run_seed(cb, seed))
        r = rows[-1]
        print(f"seed {seed} depth {r['depth']} died {r['died']} steps {r['micro_steps']} "
              f"visits {len(r['visits'])} first_entry {r['first_entry']} {time.time()-t0:.0f}s", flush=True)
    ident1 = P.source_identity(WORKER_ZIP)
    doc = {"probe": "r18x-l2geom-v1", "note": "diagnostic replay of the R18 retreat control arm; not an exam",
           "force_a11": FORCE_A11,
           "env_overrides": ENV_OVERRIDES, "max_steps": MAX_STEPS, "worker_zip": WORKER_ZIP,
           "source_identity": ident0,
           "source_changed_during_run": sorted(k for k in ident0 if ident0[k] != ident1[k]),
           "seeds": f"{lo}-{hi}", "rows": rows}
    json.dump(doc, open(out, "w"), ensure_ascii=False)
    print("WROTE", out)


if __name__ == "__main__":
    main()
