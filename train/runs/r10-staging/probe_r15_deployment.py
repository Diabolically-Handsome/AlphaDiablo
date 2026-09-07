"""R15 部署形态法证探针(非考卷:不经档案身份系统,结果只入台账法证)。

readiness-v1 脚本经理 × 工人 zip(FARM+DIVE 双注册,主权开,economy v2),
逐种子统计 最深层/死亡/通关/微拍。用法:
  probe_r15_deployment.py <worker.zip> <lo-hi seeds> <out.json> [max_steps=3000] [argmax|stochastic]
max_steps 可放宽(长视野法证:战备表在充足预算下的可爬性)。

R16 修宪(v2):
  C4 尸检污染——单机死亡时引擎剥光装备、金币减半,局末读面板得到尸检值
     (AC=4/dmg=1/gold=50)。本版 died 行的 char_level/armor_class/max_hp/
     gold/xp/kills/hit_damage 一律取「死亡前最后存活快照」:每个工人拍
     (callback 入口,raw 为该拍决策前的活体状态)与每个存活收窗后各刷新
     一次快照;post_death 列另存局末尸检值供审计;snapshot_micro_step/
     snapshot_source 标注快照时刻。存活行 post_death=None,主列即局末值。
  agg 增加存活分层:alive_n/alive_clvl_mean/alive_ac_mean/alive_kills_mean/
     dead_kills_mean/median_steps_dead 等;每千拍归一 xp_per_1k(raw["xp"])。
  C3 stochastic 解码改为与考卷(eval_assembled --worker-decoding sample)
     同款逐局定种:每局 env.reset 后 model.set_random_seed(seed),torch
     单线程;同 zip 同种子两遍逐位可复现。argmax 路径不变。
"""
import hashlib
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

from diablogym.options_env import DIVE, FARM, RESUPPLY, OptionsEnv  # noqa: E402
from diablogym.worker_env import (  # noqa: E402
    readiness_clear_ratio,
    readiness_floor_cleared,
    readiness_power_ratio,
    readiness_power_ratio_v2,
)

# R16 修宪(v3):第 6 参数为 JSON 环境覆写(新法锚/新法臂部署形态),键:
#   explore_global_hunt / explore_global_fallback / progress_far_tiles /
#   farm_scene_cap / reset_layer_clock_on_window / reward_economy /
#   drink_sovereignty(bool)/ manager("readiness-v1"|"readiness-v3")。
# 缺省(不给第 6 参数)= 旧法探针形态逐字不变。
_R16_ENV_KEYS = ("explore_global_hunt", "explore_global_fallback",
                 "progress_far_tiles", "farm_scene_cap",
                 "reset_layer_clock_on_window", "reward_economy")
_R16_MANAGERS = ("readiness-v1", "readiness-v3")

PROBE_VERSION = "r15-deployment-v3"
# 法证溯源(非身份系统):多队并行在位修法期间,记录本次探针实际 import 的
# 协议源码与工人 zip 的 SHA-256,便于把结果归因到具体源码状态。
_SOURCE_FILES = (
    "train/eval_contract.py", "train/leashed_ppo.py",
    "python/diablogym/__init__.py", "python/diablogym/controller_wire.py",
    "python/diablogym/env.py", "python/diablogym/nav.py",
    "python/diablogym/options_env.py", "python/diablogym/worker_env.py",
    "train/runs/r10-staging/probe_r15_deployment.py",
)


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_identity(worker_zip):
    ident = {rel: _sha256(ROOT / rel) for rel in _SOURCE_FILES}
    ident["worker_zip"] = _sha256(worker_zip)
    return ident


def load_zip_policy(path, stochastic=False):
    """stochastic=True:按训练分布采样解码(审计团「argmax 冻结坍缩」判别用),
    RNG 由调用方每局经 choose.episode_reseed(seed) 定种;默认 argmax 与考卷同款。
    choose.on_beat 若被设置,则在每个工人拍(决策前)被调用一次——探针用它
    在活体状态下采快照。"""
    from leashed_ppo import LeashedMaskablePPO
    model = LeashedMaskablePPO.load(
        path, env=None, device="cpu",
        teacher_path=None, teacher_sha256=None)
    if stochastic:
        import torch
        torch.set_num_threads(1)

    def choose(obs, mask):
        hook = choose.on_beat
        if hook is not None:
            hook()
        a, _ = model.predict(
            obs, action_masks=mask, deterministic=not stochastic)
        return int(a)
    choose.on_beat = None
    choose.episode_reseed = (
        (lambda seed: model.set_random_seed(int(seed))) if stochastic
        else (lambda seed: None))
    choose.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    choose.diablogym_worker_action12_mode = "permanently-masked"
    return choose


def panel(raw):
    """面板六件套 + xp(全部取自同一 raw 快照)。"""
    return {
        "char_level": int(raw.get("char_level", 0)),
        "armor_class": int(raw.get("armor_class", 0)),
        "max_hp": int(raw.get("max_hp", 0)),
        "gold": int(raw.get("gold", 0)),
        "xp": int(raw.get("xp", 0)),
        # R15.2 榨取率法证:击杀/单击伤害面板
        "kills": int(raw.get("monster_kill_total", 0)),
        "hit_damage": (int(raw.get("item_max_damage", 0))
                       + int(raw.get("damage_mod", 0))),
    }


def _mean(values, digits):
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _median(values):
    values = list(values)
    if not values:
        return None
    return float(statistics.median(values))


def aggregate(rows):
    n = len(rows)
    alive = [r for r in rows if not r["died"]]
    dead = [r for r in rows if r["died"]]
    agg = {
        "n": n,
        "died": len(dead),
        "victories": sum(1 for r in rows if r["victory"]),
        "depth_hist": {},
        "l3": sum(1 for r in rows if r["depth"] >= 3),
        "l5": sum(1 for r in rows if r["depth"] >= 5),
        "depth_mean": round(sum(r["depth"] for r in rows) / max(1, n), 3),
        # 全体均值(died 行已改为死亡前快照,不再被尸检值污染)
        "clvl_mean": round(
            sum(r["char_level"] for r in rows) / max(1, n), 2),
        "ac_mean": round(
            sum(r["armor_class"] for r in rows) / max(1, n), 1),
        "kills_mean": round(
            sum(r["kills"] for r in rows) / max(1, n), 1),
        "gold_mean": round(
            sum(r["gold"] for r in rows) / max(1, n), 1),
        "hit_damage_mean": round(
            sum(r["hit_damage"] for r in rows) / max(1, n), 1),
        "xp_mean": round(sum(r["xp"] for r in rows) / max(1, n), 1),
        "xp_per_1k_mean": round(
            sum(r["xp_per_1k"] for r in rows) / max(1, n), 2),
        "micro_steps_mean": round(
            sum(r["micro_steps"] for r in rows) / max(1, n), 1),
        # 存活分层(None = 该层为空)
        "alive_n": len(alive),
        "alive_clvl_mean": _mean((r["char_level"] for r in alive), 2),
        "alive_ac_mean": _mean((r["armor_class"] for r in alive), 1),
        "alive_kills_mean": _mean((r["kills"] for r in alive), 1),
        "alive_gold_mean": _mean((r["gold"] for r in alive), 1),
        "alive_xp_per_1k_mean": _mean((r["xp_per_1k"] for r in alive), 2),
        "median_steps_alive": _median(r["micro_steps"] for r in alive),
        "dead_n": len(dead),
        "dead_clvl_mean": _mean((r["char_level"] for r in dead), 2),
        "dead_ac_mean": _mean((r["armor_class"] for r in dead), 1),
        "dead_kills_mean": _mean((r["kills"] for r in dead), 1),
        "dead_gold_mean": _mean((r["gold"] for r in dead), 1),
        "dead_xp_per_1k_mean": _mean((r["xp_per_1k"] for r in dead), 2),
        "median_steps_dead": _median(r["micro_steps"] for r in dead),
        # 审计对照:died 行尸检值均值(旧探针 v1 口径)
        "dead_post_death_ac_mean": _mean(
            (r["post_death"]["armor_class"] for r in dead), 1),
        "dead_post_death_gold_mean": _mean(
            (r["post_death"]["gold"] for r in dead), 1),
        "dead_post_death_hit_damage_mean": _mean(
            (r["post_death"]["hit_damage"] for r in dead), 1),
    }
    for r in rows:
        key = str(r["depth"])
        agg["depth_hist"][key] = agg["depth_hist"].get(key, 0) + 1
    return agg


def run_episode(env, cb, seed, stochastic, manager="readiness-v1"):
    """一局;返回 row。快照策略见模块 docstring。
    manager: readiness-v1(v0.1 表 + 榨干旗逃生,旧法)/ readiness-v3(R16:
    v0.2 表 + 逐层击杀口径清场比 ≥1.0(无活怪)逃生,与 worker_env._mgr_choose 同款)。"""
    snap = {"panel": None, "micro_step": None, "source": None}
    floor_state = {"dlvl": None, "kills_at_entry": 0}

    def take(source):
        raw = env.env._raw
        if raw.get("dead"):
            return
        snap["panel"] = panel(raw)
        snap["micro_step"] = int(env.env._steps)
        snap["source"] = source

    cb.on_beat = lambda: take("beat")
    try:
        obs, _ = env.reset(seed=seed)
        # 局开始时以该局 seed 定种(与 eval_assembled sample 模式同款);
        # argmax 下为空操作。
        cb.episode_reseed(seed)
        take("reset")
        done = trunc = False
        max_depth = 1
        while not (done or trunc):
            raw = env.env._raw
            mask = np.asarray(env.action_masks(), dtype=bool)
            if manager == "readiness-v3":
                dlvl = int(raw["dungeon_level"])
                if floor_state["dlvl"] != dlvl:
                    floor_state["dlvl"] = dlvl
                    floor_state["kills_at_entry"] = int(
                        raw.get("monster_kill_total", 0))
                ready = readiness_power_ratio_v2(raw, dlvl + 1) >= 1.0
                clear = readiness_clear_ratio(
                    raw, floor_state["kills_at_entry"]) >= 1.0
                want = DIVE if (ready or clear) else FARM
            else:
                ready = readiness_power_ratio(
                    raw, int(raw["dungeon_level"]) + 1) >= 1.0
                # v2 逃生:真实清场,不认可白嫖的榨干旗
                want = (DIVE if (ready or readiness_floor_cleared(raw))
                        else FARM)
            if not mask[want]:
                for cand in (FARM, DIVE, RESUPPLY):
                    if mask[cand]:
                        want = cand
                        break
            obs, r, done, trunc, info = env.step(int(want))
            max_depth = max(
                max_depth, int(env.env._raw.get("dungeon_level", 1)))
            take("window")
    finally:
        cb.on_beat = None
    raw = env.env._raw
    monsters = raw.get("monsters") or []
    micro_steps = int(env.env._steps)
    died = bool(raw.get("dead"))
    final = panel(raw)
    if died:
        live = snap["panel"]
        post_death = final
    else:
        live = final
        post_death = None
    return {
        "seed": seed, "depth": max_depth,
        "died": died,
        "victory": bool(raw.get("victory")),
        "micro_steps": micro_steps,
        **live,
        "xp_per_1k": round(1000.0 * live["xp"] / max(1, micro_steps), 3),
        "monsters_left_here": sum(
            1 for m in monsters if int(m.get("hp", 0)) > 0),
        "snapshot_micro_step": (
            snap["micro_step"] if died else micro_steps),
        "snapshot_source": snap["source"] if died else "final",
        "post_death": post_death,
    }


def main():
    worker_zip, seed_span, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    decoding_arg = sys.argv[5] if len(sys.argv) > 5 else "argmax"
    if decoding_arg not in ("argmax", "stochastic", "sample"):
        raise SystemExit(
            f"第 5 参数只允许 argmax|stochastic|sample,收到 {decoding_arg!r}")
    stochastic = decoding_arg in ("stochastic", "sample")
    # R16(v3):第 6 参数 JSON 环境覆写;缺省 = 旧法形态逐字不变
    overrides = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    unknown = set(overrides) - set(_R16_ENV_KEYS) - {"drink_sovereignty",
                                                     "manager"}
    if unknown:
        raise SystemExit(f"第 6 参数含未知键: {sorted(unknown)}")
    manager = str(overrides.get("manager", "readiness-v1"))
    if manager not in _R16_MANAGERS:
        raise SystemExit(f"manager 只允许 {_R16_MANAGERS},收到 {manager!r}")
    drink_sovereignty = bool(overrides.get("drink_sovereignty", False))
    env_overrides = {k: overrides[k] for k in _R16_ENV_KEYS if k in overrides}
    lo, hi = (int(x) for x in seed_span.split("-"))
    identity_before = source_identity(worker_zip)
    cb = load_zip_policy(worker_zip, stochastic=stochastic)
    if drink_sovereignty:
        # 主权开放:工人回调双标签改为环境掩码模式(OptionsEnv 自绑定校验)
        cb.diablogym_worker_action12_mode = "environment-mask"
    rows = []
    for seed in range(lo, hi + 1):
        env_kwargs = dict(
            max_steps=max_steps,
            workers={FARM: cb, DIVE: cb},
            drink_sovereignty=drink_sovereignty,
            worker_observation_view="dual-v4-asymmetric-v3",
            manager_observation_view="legacy-v3",
            dive_live_sovereignty=True,
            reward_economy="v2")
        env_kwargs.update(env_overrides)
        env = OptionsEnv(**env_kwargs)
        try:
            rows.append(run_episode(env, cb, seed, stochastic, manager))
        finally:
            env.close()
    agg = aggregate(rows)
    identity_after = source_identity(worker_zip)
    changed_during_run = sorted(
        k for k in identity_before if identity_before[k] != identity_after[k])
    doc = {"probe": PROBE_VERSION,
           "source_identity": identity_before,
           # 非空即表示探针运行期间源码被改(并行修法),结果归因需谨慎
           "source_changed_during_run": changed_during_run,
           "note": "法证探针,非考卷;readiness-v1 脚本经理部署形态",
           "row_semantics": (
               "died 行主列 = 死亡前最后存活快照(工人拍/存活收窗),"
               "post_death = 局末尸检值;存活行主列 = 局末值,post_death=None"),
           "worker_zip": worker_zip, "seeds": seed_span,
           "max_steps": max_steps,
           # R16(v3):部署形态声明(缺省 = 旧法:readiness-v1/主权关/v2/无覆写)
           "manager": manager,
           "drink_sovereignty": drink_sovereignty,
           "r16_environment": env_overrides or None,
           "decoding": "stochastic" if stochastic else "argmax",
           "decoding_seeding": (
               "per-episode model.set_random_seed(seed) after env.reset; "
               "torch.set_num_threads(1)" if stochastic else None),
           "agg": agg, "rows": rows}
    pathlib.Path(out_path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
