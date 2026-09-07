# 复核修复(2026-09-06):~/r17_work/review/R17-0-WP-ADVERSARIAL-REVIEW-20260906.md H2/H4/M2/L1
"""R17.0 仪表探针(probe_r15_deployment.py v3 的超集;v3 文件字节不动)。

argv 契约与 v3 完全相同:
  probe_r17_deployment.py <worker.zip> <lo-hi> <out.json> [max_steps=3000]
                          [argmax|stochastic|sample] [json-form]
第 6 参数 JSON 键同 v3(_R16_ENV_KEYS + drink_sovereignty + manager);
manager 扩为 readiness-v1 | readiness-v3 | readiness-v3-strict | const-FARM |
const-DIVE。每种经理都走同一条掩码回退阶梯 FARM->DIVE->RESUPPLY,并记录
强制下楼。

逐位契约:manager=readiness-v3(或 readiness-v1)时,每行的 v3 列
(V3_ROW_KEYS)与 probe_r15 v3 在同 zip/同种子/同形态下逐位相等——决策
序列、快照钩子点(每个工人拍 + 每次存活收窗)、采样定种(局开始
model.set_random_seed(seed);torch 单线程)全部同款;R17 新增仪表只读
raw / OptionsEnv 状态,不触碰 RNG。rows_sha_v3(rows) 给出该限制列的
sha256,驱动脚本用它对 r16-deploy-arm.json 做逐位回归。

R17.0 新增仪表(每行):
  descents[]      每次 dungeon_level 增加(收窗 reason=descend 处观测):
                  {beat, from_dlvl, to_dlvl, belt_heals, hp, max_hp, armor_class,
                  char_level, hit_damage, ratio_v2(=readiness_power_ratio_v2(raw,
                  to_dlvl)), ready(ratio_v2>=1), forced(决策时 ¬mask[FARM]),
                  trigger, forced_reason, window_id, kills_so_far,
                  floor_kills_before_descent, floor_beats_before_descent,
                  monsters_left_prev_floor, potions_visible_prev_floor}
                  monsters_left_prev_floor / potions_* 取自下楼前最后一次活体
                  观测(同窗内最多早一拍)。
  death           {beat, dlvl, belt_heals_last_live, belt_heals_post_death,
                  floor_potions_total/visible/reachable(最后活体观测本层地面
                  治疗药:全部 / visible / visible∧reachable),
                  beats_on_current_floor, hp_last_live, max_hp_last_live,
                  last_live_beat, window_opt, window_trigger}
  floors[]        每次楼层驻留:{dlvl, entry_beat, exit_beat, beats, kills,
                  monsters_left_at_exit, potions_visible_at_exit, exit_reason}
  dive_windows[]  每个 DIVE 窗:{window_id, beat0, beat1, tau, end_reason,
                  descended, trigger, forced, forced_reason, ratio_v2, cleared}
  windows         {total, farm, dive, resupply, forced_dive, mask_forced,
                  fallback, coach_dive_wants, coach_cleared_true}
  first_descent_beat / alive_at_fd_plus_1800(None=从未下楼或观察被截断)/
  fd_plus_1800_censored(存活截断早于首降+1800)/ beats_by_dlvl / l2_beats /
  died_on_l2 / l2_death_beat
trigger 口径(下楼所在窗的经理决策):
  coach_ready    教练自愿 DIVE 且 ratio_v2>=1(readiness-v1 按 v0.1 尺 ready)
  coach_cleared  教练自愿 DIVE 因清场子句
  const_dive     const-DIVE 的无条件 DIVE(既不 ready 也不 cleared)
  mask_forced    教练不想 DIVE,掩码 m[FARM]=False 经回退阶梯落到 DIVE
  fallback       其余(如非 DIVE 窗内发生的换层)
forced_reason(决策时 ¬mask[FARM] 的成因,按 options_env.py:739-741 掩码法
反推):handoff(剧情目标交权 _farm_handoff)/ cap(farm_scene_steps>=cap)/
idle_clock(layer_clock>=KILL_PATIENCE)/ exhausted_other / None。
agg 在 v3 全部键之外追加 farm_masked_at_descent_share(下楼时 ¬mask[FARM]
的份额,与教练意愿无关)/ mask_forced_descent_share(trigger=mask_forced,
即真正「经理被推翻」的份额)/ descents_total / ready_descents /
cleared_true_decisions(教练 cleared 子句为真的决策数)/
mean_beat_first_descent / alive_at_first_descent_plus_1800 /
l2_hazard_per_1k(L2 死亡数 / ΣL2 停留拍 × 1000)及下楼面板分布。
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

from diablogym.options_env import (  # noqa: E402
    DIVE, FARM, KILL_PATIENCE, RESUPPLY, OptionsEnv, _farm_handoff)
from diablogym.worker_env import (  # noqa: E402
    readiness_clear_ratio,
    readiness_floor_cleared,
    readiness_power_ratio,
    readiness_power_ratio_v2,
    roster_alive_count,
)

_R16_ENV_KEYS = ("explore_global_hunt", "explore_global_fallback",
                 "progress_far_tiles", "farm_scene_cap",
                 "reset_layer_clock_on_window", "reward_economy",
                 # R17 T0(2026-09-06):资源通道臂的环境旗直通 OptionsEnv;
                 # 不在 v3 行/agg 键内,readiness-v3 逐位路径不受影响。
                 "resource_protocol", "resource_purchase_mode",
                 "resource_service_policy", "resource_readiness_law",
                 # R17 T0′:sustain-loot-v1 要求显式 completion-l2-v1 时钟
                 "worker_time_protocol", "resource_retreat",
                 # R18-D/E (2026-09-07): aggro cap + engagement priority flags
                 "aggro_cap", "engagement_priority",
                 # R18-G: global-hunt scope (the pull mechanism)
                 "hunt_scope",
                 # R18-F: Scroll of Town Portal interface
                 "resource_portal")
_R17_MANAGERS = ("readiness-v1", "readiness-v3", "readiness-v3-strict",
                 "const-FARM", "const-DIVE",
                 # R17 T0:协议开启时的脚本经理 = OptionsEnv.resource_option_choice
                 "resource")

PROBE_VERSION = "r17-deployment-v3-r18f"
V3_PROBE_VERSION = "r15-deployment-v3"
# probe_r15 v3 的每行字段(逐位回归口径);R17 行是其超集。
V3_ROW_KEYS = (
    "seed", "depth", "died", "victory", "micro_steps",
    "char_level", "armor_class", "max_hp", "gold", "xp", "kills", "hit_damage",
    "xp_per_1k", "monsters_left_here", "snapshot_micro_step",
    "snapshot_source", "post_death")
ALIVE_AFTER_FIRST_DESCENT_BEATS = 1800
_SOURCE_FILES = (
    "train/eval_contract.py", "train/leashed_ppo.py",
    "python/diablogym/__init__.py", "python/diablogym/controller_wire.py",
    "python/diablogym/env.py", "python/diablogym/nav.py",
    "python/diablogym/options_env.py", "python/diablogym/worker_env.py",
    "train/runs/r10-staging/probe_r15_deployment.py",
    "train/runs/r10-staging/probe_r17_deployment.py",
)


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_identity(worker_zip):
    ident = {rel: _sha256(ROOT / rel) for rel in _SOURCE_FILES}
    ident["worker_zip"] = _sha256(worker_zip)
    return ident


def rows_sha_v3(rows):
    """限制到 v3 列的行 sha256(与 r16-deploy-*.json 行逐位回归口径)。"""
    payload = json.dumps(
        [{k: r[k] for k in V3_ROW_KEYS} for r in rows],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_zip_policy(path, stochastic=False):
    """与 v3 同款(采样定种 / 单线程 / on_beat 钩子)。"""
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
    """面板六件套 + xp(全部取自同一 raw 快照)。与 v3 逐字相同。"""
    return {
        "char_level": int(raw.get("char_level", 0)),
        "armor_class": int(raw.get("armor_class", 0)),
        "max_hp": int(raw.get("max_hp", 0)),
        "gold": int(raw.get("gold", 0)),
        "xp": int(raw.get("xp", 0)),
        "kills": int(raw.get("monster_kill_total", 0)),
        "hit_damage": (int(raw.get("item_max_damage", 0))
                       + int(raw.get("damage_mod", 0))),
    }


def floor_potions(raw):
    """本层地面治疗药计数(floor_items 的 heal 旗;visible/reachable 为桥侧
    可见性/可达性标志,与 env._policy_floor_items(raw, "heal") 同口径)。"""
    items = raw.get("floor_items") or ()
    heal = [it for it in items if bool(it.get("heal"))]
    visible = [it for it in heal if bool(it.get("visible", True))]
    reachable = [it for it in visible if bool(it.get("reachable", True))]
    return {"total": len(heal), "visible": len(visible),
            "reachable": len(reachable)}


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


def _hist(values):
    out = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


def coach_decide(manager, raw, floor_state):
    """脚本经理的 want。返回 (want, ready, cleared, coach_reason, ratio)。

    readiness-v1 / readiness-v3 分支与 probe_r15 v3 run_episode 内联逻辑逐字
    等价(进层基数按 dungeon_level 变化重置)。"""
    dlvl = int(raw["dungeon_level"])
    if manager == "readiness-v1":
        ratio = readiness_power_ratio(raw, dlvl + 1)
        ready = ratio >= 1.0
        cleared = readiness_floor_cleared(raw)
        want = DIVE if (ready or cleared) else FARM
    else:
        if floor_state["dlvl"] != dlvl:
            floor_state["dlvl"] = dlvl
            floor_state["kills_at_entry"] = int(
                raw.get("monster_kill_total", 0))
        ratio = readiness_power_ratio_v2(raw, dlvl + 1)
        ready = ratio >= 1.0
        cleared = readiness_clear_ratio(
            raw, floor_state["kills_at_entry"]) >= 1.0
        if manager == "readiness-v3":
            want = DIVE if (ready or cleared) else FARM
        elif manager == "readiness-v3-strict":
            want = DIVE if ready else FARM
        elif manager == "const-FARM":
            want = FARM
        elif manager == "const-DIVE":
            want = DIVE
        else:
            raise ValueError(f"未知 manager {manager!r}")
    if want == DIVE:
        coach_reason = ("ready" if ready
                        else ("cleared" if cleared else "const"))
    else:
        coach_reason = "farm"
    return want, bool(ready), bool(cleared), coach_reason, float(ratio)


def forced_reason(env, raw):
    """决策时 m[FARM]=False 的成因(options_env.py:739-741 反推)。"""
    if _farm_handoff(raw):
        return "handoff"
    cap = int(getattr(env, "farm_scene_cap", 0) or 0)
    if cap and int(getattr(env, "farm_scene_steps", 0)) >= cap:
        return "cap"
    if int(getattr(env, "layer_clock", 0)) >= KILL_PATIENCE:
        return "idle_clock"
    if bool(getattr(env, "exhausted", False)):
        return "exhausted_other"
    return "unknown"


def aggregate(rows):
    """v3 agg 全部键(逐字同款计算)+ R17 扩展键。"""
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

    # ---- R17.0 扩展 ----
    descents = [d for r in rows for d in r["descents"]]
    dive_windows = [w for r in rows for w in r["dive_windows"]]
    with_fd = [r for r in rows if r["first_descent_beat"] is not None]
    l2_beats = sum(r["l2_beats"] for r in rows)
    l2_deaths = sum(1 for r in rows if r["died_on_l2"])
    l1_beats = sum(r["beats_by_dlvl"].get("1", 0) for r in rows)
    l1_kills = sum(f["kills"] for r in rows for f in r["floors"]
                   if f["dlvl"] == 1)
    deaths = [r["death"] for r in dead]
    observed_fd = [r for r in with_fd if r["alive_at_fd_plus_1800"] is not None]
    alive_fd = [r for r in observed_fd if r["alive_at_fd_plus_1800"]]
    agg.update({
        "descents_total": len(descents),
        "forced_descents": sum(1 for d in descents if d["forced"]),
        # 决策时 FARM 被掩码(与教练意愿无关);旧名 forced_descent_share
        "farm_masked_at_descent_share": (
            round(sum(1 for d in descents if d["forced"]) / len(descents), 4)
            if descents else None),
        # 真正「经理被推翻」:教练不想 DIVE,靠回退阶梯落到 DIVE
        "mask_forced_descent_share": (
            round(sum(1 for d in descents if d["trigger"] == "mask_forced")
                  / len(descents), 4)
            if descents else None),
        "ready_descents": sum(1 for d in descents if d["ready"]),
        "ready_descent_share": (
            round(sum(1 for d in descents if d["ready"]) / len(descents), 4)
            if descents else None),
        "descent_trigger_hist": _hist(d["trigger"] for d in descents),
        "descent_forced_reason_hist": _hist(
            d["forced_reason"] for d in descents if d["forced"]),
        "episodes_with_descent": len(with_fd),
        "mean_beat_first_descent": _mean(
            (r["first_descent_beat"] for r in with_fd), 1),
        "median_beat_first_descent": _median(
            r["first_descent_beat"] for r in with_fd),
        "alive_at_first_descent_plus_1800": (
            round(len(alive_fd) / len(observed_fd), 4) if observed_fd else None),
        "alive_at_first_descent_plus_1800_n": (
            f"{len(alive_fd)}/{len(observed_fd)}" if observed_fd else None),
        "fd_plus_1800_observed_n": len(observed_fd),
        "fd_plus_1800_censored_n": sum(
            1 for r in with_fd if r["fd_plus_1800_censored"]),
        "l2_reach": sum(1 for r in rows if r["depth"] >= 2),
        "l2_beats_total": l2_beats,
        "l2_deaths": l2_deaths,
        "l2_hazard_per_1k": (
            round(1000.0 * l2_deaths / l2_beats, 4) if l2_beats else None),
        "l1_beats_total": l1_beats,
        "l1_kills_total": l1_kills,
        "l1_kills_per_1k_beats": (
            round(1000.0 * l1_kills / l1_beats, 3) if l1_beats else None),
        "death_dlvl_hist": _hist(d["dlvl"] for d in deaths),
        "death_beats_on_floor_median": _median(
            d["beats_on_current_floor"] for d in deaths),
        "death_belt_last_live_hist": _hist(
            d["belt_heals_last_live"] for d in deaths),
        "death_floor_potions_visible_mean": _mean(
            (d["floor_potions_visible"] for d in deaths), 2),
        "death_floor_potions_reachable_mean": _mean(
            (d["floor_potions_reachable"] for d in deaths), 2),
        "death_floor_potions_total_mean": _mean(
            (d["floor_potions_total"] for d in deaths), 2),
        "descent_ac_hist": _hist(d["armor_class"] for d in descents),
        "descent_ac_mean": _mean((d["armor_class"] for d in descents), 2),
        "descent_belt_hist": _hist(d["belt_heals"] for d in descents),
        "descent_belt_mean": _mean((d["belt_heals"] for d in descents), 2),
        "descent_clvl_hist": _hist(d["char_level"] for d in descents),
        "descent_hp_frac_mean": _mean(
            (d["hp"] / max(1, d["max_hp"]) for d in descents), 3),
        "descent_ratio_v2_mean": _mean(
            (d["ratio_v2"] for d in descents), 3),
        "descent_floor_beats_median": _median(
            d["floor_beats_before_descent"] for d in descents),
        "descent_monsters_left_prev_floor_median": _median(
            d["monsters_left_prev_floor"] for d in descents),
        "dive_windows_total": len(dive_windows),
        "dive_windows_descended": sum(
            1 for w in dive_windows if w["descended"]),
        "dive_window_end_reason_hist": _hist(
            w["end_reason"] for w in dive_windows),
        "dive_window_tau_median": _median(w["tau"] for w in dive_windows),
        "windows_total": sum(r["windows"]["total"] for r in rows),
        "mask_forced_decisions_total": sum(
            r["windows"]["mask_forced"] for r in rows),
        "forced_dive_windows_total": sum(
            r["windows"]["forced_dive"] for r in rows),
        # L1:教练 cleared 子句为真的决策数(v3 与 v3-strict 的唯一分歧来源;
        # 为 0 即「v3 ≡ strict」是经验巧合而非结构必然)
        "cleared_true_decisions": sum(
            r["windows"]["coach_cleared_true"] for r in rows),
    })
    return agg


def followup_status(first_descent_beat, final_beat, died):
    """Return (alive, censored); an unobserved survival horizon is unknown."""
    if first_descent_beat is None:
        return None, False
    horizon = first_descent_beat + ALIVE_AFTER_FIRST_DESCENT_BEATS
    censored = (not died) and final_beat < horizon
    return (None if censored else ((not died) or final_beat > horizon)), censored


def run_episode(env, cb, seed, stochastic, manager="readiness-v1"):
    """一局;返回 row(v3 列逐字同款 + R17 仪表)。"""
    snap = {"panel": None, "micro_step": None, "source": None}
    floor_state = {"dlvl": None, "kills_at_entry": 0}
    # ---- R17 仪表状态(只读 raw / env 状态) ----
    tele = {"descents": [], "floors": [], "dive_windows": []}
    windows = {"total": 0, "farm": 0, "dive": 0, "resupply": 0,
               "forced_dive": 0, "mask_forced": 0, "fallback": 0,
               "coach_dive_wants": 0, "coach_cleared_true": 0}
    cur_floor = {"dlvl": None, "entry_beat": 0, "kills_at_entry": 0}
    last_live = {"beat": None, "dlvl": None, "kills": 0, "alive_roster": None,
                 "potions": None, "belt": None, "hp": None, "max_hp": None}
    decision = {"window_id": None, "beat0": None, "want": None,
                "chosen": None, "trigger": None, "forced": False,
                "forced_reason": None, "ratio_v2": None, "cleared": None,
                "ready": None}

    def take(source):
        raw = env.env._raw
        if raw.get("dead"):
            return
        snap["panel"] = panel(raw)
        snap["micro_step"] = int(env.env._steps)
        snap["source"] = source

    def open_floor(dlvl, beat, kills):
        cur_floor["dlvl"] = dlvl
        cur_floor["entry_beat"] = beat
        cur_floor["kills_at_entry"] = kills

    def close_floor(exit_beat, kills_now, exit_reason):
        rec = {
            "dlvl": cur_floor["dlvl"],
            "entry_beat": cur_floor["entry_beat"],
            "exit_beat": exit_beat,
            "beats": exit_beat - cur_floor["entry_beat"],
            "kills": max(0, kills_now - cur_floor["kills_at_entry"]),
            "monsters_left_at_exit": last_live["alive_roster"],
            "potions_visible_at_exit": (
                last_live["potions"]["visible"]
                if last_live["potions"] else None),
            "exit_reason": exit_reason,
        }
        tele["floors"].append(rec)
        return rec

    def observe():
        """每个活体观测点(工人拍前 / 存活收窗后)刷新楼层与活体记录。"""
        raw = env.env._raw
        if raw.get("dead"):
            return
        beat = int(env.env._steps)
        dlvl = int(raw.get("dungeon_level", 1))
        kills = int(raw.get("monster_kill_total", 0))
        if cur_floor["dlvl"] is None:
            open_floor(dlvl, beat, kills)
        elif dlvl != cur_floor["dlvl"]:
            from_dlvl = cur_floor["dlvl"]
            prev = close_floor(
                beat, kills, "descend" if dlvl > from_dlvl else "ascend")
            if dlvl > from_dlvl:
                ratio = readiness_power_ratio_v2(raw, dlvl)
                tele["descents"].append({
                    "beat": beat, "from_dlvl": from_dlvl, "to_dlvl": dlvl,
                    "belt_heals": int(raw.get("belt_heals", 0)),
                    "hp": int(raw.get("hp", 0)),
                    "max_hp": int(raw.get("max_hp", 0)),
                    "armor_class": int(raw.get("armor_class", 0)),
                    "char_level": int(raw.get("char_level", 0)),
                    "hit_damage": (int(raw.get("item_max_damage", 0))
                                   + int(raw.get("damage_mod", 0))),
                    "ratio_v2": round(float(ratio), 4),
                    "ready": bool(ratio >= 1.0),
                    "forced": bool(decision["forced"]),
                    "trigger": decision["trigger"] or "fallback",
                    "forced_reason": decision["forced_reason"],
                    "window_id": decision["window_id"],
                    "window_opt": decision["chosen"],
                    "kills_so_far": kills,
                    "floor_kills_before_descent": prev["kills"],
                    "floor_beats_before_descent": prev["beats"],
                    "monsters_left_prev_floor": prev["monsters_left_at_exit"],
                    "potions_visible_prev_floor": prev[
                        "potions_visible_at_exit"],
                })
            open_floor(dlvl, beat, kills)
        last_live.update({
            "beat": beat, "dlvl": dlvl, "kills": kills,
            "alive_roster": roster_alive_count(raw),
            "potions": floor_potions(raw),
            "belt": int(raw.get("belt_heals", 0)),
            "hp": int(raw.get("hp", 0)),
            "max_hp": int(raw.get("max_hp", 0)),
        })

    def on_beat():
        take("beat")
        observe()

    cb.on_beat = on_beat
    try:
        obs, _ = env.reset(seed=seed)
        cb.episode_reseed(seed)
        take("reset")
        observe()
        done = trunc = False
        max_depth = 1
        while not (done or trunc):
            raw = env.env._raw
            mask = np.asarray(env.action_masks(), dtype=bool)
            if manager == "resource":
                # R17 T0:遥测用的 ready/cleared/ratio 沿用 v3-strict 口径,
                # want 由资源协议的脚本经理决定(coach-v03 下按六条法)。
                _w, ready, cleared, _r, ratio = coach_decide(
                    "readiness-v3-strict", raw, floor_state)
                want = int(env.resource_option_choice(mask))
                coach_reason = (("ready" if ready else
                                 ("cleared" if cleared else "const"))
                                if want == DIVE else "farm")
            else:
                want, ready, cleared, coach_reason, ratio = coach_decide(
                    manager, raw, floor_state)
            chosen = want
            fell_back = False
            if not mask[want]:
                for cand in (FARM, DIVE, RESUPPLY):
                    if mask[cand]:
                        chosen = cand
                        break
                fell_back = True
            forced = not bool(mask[FARM])
            if chosen == DIVE:
                if want == DIVE and not fell_back:
                    trigger = {"ready": "coach_ready",
                               "cleared": "coach_cleared",
                               "const": "const_dive"}[coach_reason]
                elif forced:
                    trigger = "mask_forced"
                else:
                    trigger = "fallback"
            else:
                trigger = None
            decision.update({
                "window_id": int(env._decisions) + 1,
                "beat0": int(env.env._steps),
                "want": int(want), "chosen": int(chosen),
                "trigger": trigger, "forced": forced,
                "forced_reason": forced_reason(env, raw) if forced else None,
                "ratio_v2": round(float(
                    readiness_power_ratio_v2(raw, int(raw["dungeon_level"]) + 1)
                ), 4),
                "cleared": cleared, "ready": ready,
            })
            windows["total"] += 1
            windows[("farm", "dive", "resupply")[int(chosen)]] += 1
            windows["mask_forced"] += int(forced)
            windows["fallback"] += int(fell_back)
            windows["coach_dive_wants"] += int(want == DIVE)
            windows["coach_cleared_true"] += int(cleared)
            windows["forced_dive"] += int(chosen == DIVE and forced)
            obs, r, done, trunc, info = env.step(int(chosen))
            max_depth = max(
                max_depth, int(env.env._raw.get("dungeon_level", 1)))
            take("window")
            observe()
            if chosen == DIVE:
                extra = info.get("option_extra") or {}
                tele["dive_windows"].append({
                    "window_id": decision["window_id"],
                    "beat0": decision["beat0"],
                    "beat1": int(env.env._steps),
                    "tau": int(extra.get("tau", env.env._steps
                                         - decision["beat0"])),
                    "end_reason": extra.get("reason"),
                    "descended": bool(
                        int(extra.get("dlvl_end", 0))
                        > int(extra.get("dlvl0", 0))),
                    "trigger": trigger, "forced": forced,
                    "forced_reason": decision["forced_reason"],
                    "ratio_v2": decision["ratio_v2"],
                    "cleared": cleared,
                })
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
    # ---- R17: 收尾楼层与死亡记录 ----
    if cur_floor["dlvl"] is not None:
        close_floor(
            micro_steps,
            (last_live["kills"] if died
             else int(raw.get("monster_kill_total", 0))),
            "death" if died else "end")
    death = None
    if died:
        pots = last_live["potions"] or {"total": None, "visible": None,
                                        "reachable": None}
        death = {
            "beat": micro_steps,
            "dlvl": (last_live["dlvl"] if last_live["dlvl"] is not None
                     else int(raw.get("dungeon_level", 1))),
            "dlvl_post_death": int(raw.get("dungeon_level", 0)),
            "belt_heals_last_live": last_live["belt"],
            "belt_heals_post_death": int(raw.get("belt_heals", 0)),
            "floor_potions_total": pots["total"],
            "floor_potions_visible": pots["visible"],
            "floor_potions_reachable": pots["reachable"],
            "beats_on_current_floor": micro_steps - cur_floor["entry_beat"],
            "hp_last_live": last_live["hp"],
            "max_hp_last_live": last_live["max_hp"],
            "last_live_beat": last_live["beat"],
            "window_opt": decision["chosen"],
            "window_trigger": decision["trigger"],
        }
    beats_by_dlvl = {}
    for f in tele["floors"]:
        key = str(f["dlvl"])
        beats_by_dlvl[key] = beats_by_dlvl.get(key, 0) + int(f["beats"])
    first_descent_beat = (
        tele["descents"][0]["beat"] if tele["descents"] else None)
    alive_at_fd, censored = followup_status(
        first_descent_beat, micro_steps, died)
    died_on_l2 = bool(died and death["dlvl"] == 2)
    # ---- R17 T0:资源通道遥测(协议关时为 None;不进 v3 行键) ----
    resource = None
    if getattr(env, "resource_protocol", "off") != "off":
        receipts = list(getattr(env.env, "_resource_transition_receipts", []))
        reasons = {}
        for rec in receipts:
            key = str(rec.get("reason"))
            reasons[key] = reasons.get(key, 0) + 1
        service = getattr(env, "resource_service", None)
        state = raw.get("resource_state") or {}
        resource = {
            "protocol": getattr(env, "resource_protocol", None),
            "purchase_mode": getattr(env, "resource_purchase_mode", None),
            "service_policy": getattr(env, "resource_service_policy", None),
            "readiness_law": getattr(env, "resource_readiness_law", None),
            "receipts": len(receipts),
            "receipt_reasons": reasons,
            "descents_ready_law": sum(
                1 for rec in receipts if rec.get("pretransition_ready_law")),
            "descents_forced_unready": sum(
                1 for rec in receipts
                if rec.get("accepted") and not rec.get("pretransition_ready_law")),
            "service_attempted": getattr(service, "attempted", None),
            "service_active_at_end": getattr(service, "active", None),
            "service_trigger": getattr(service, "trigger", None),
            "service_reason": getattr(service, "reason", None),
            "terminal_reason": getattr(env.env, "_resource_terminal_reason", None),
            "gold_final": int(raw.get("gold", 0)),
            "service_trip": state.get("service_trip"),
            "max_main_depth_reached": state.get("max_main_depth_reached"),
            # R18-A retreat-v1 (None when the flag is off; row keys unchanged otherwise)
            "retreats_started": state.get("retreats_started"),
            "retreat": (env.retreat_service.telemetry()
                        if getattr(env, "retreat_service", None) is not None else None),
            # R18-D/E telemetry (off -> "off" / 0)
            "aggro_cap": getattr(env.env, "aggro_cap", None),
            "aggro_cap_mask_hits": getattr(env.env, "_aggro_cap_fired", None),
            "engagement_priority": getattr(env.env, "engagement_priority", None),
            "engagement_decisions": getattr(env.env, "_engagement_decisions", None),
            "engagement_reordered": getattr(env.env, "_engagement_reordered", None),
            "hunt_scope": getattr(env.env, "hunt_scope", None),
            # R18-F portal-v1 telemetry (None when off)
            "portals_started": state.get("portals_started"),
            "portal": (env.portal_service.telemetry()
                       if getattr(env, "portal_service", None) is not None else None),
        }
    return {
        "seed": seed, "depth": max_depth,
        "resource": resource,
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
        # ---- R17.0 仪表 ----
        "descents": tele["descents"],
        "floors": tele["floors"],
        "dive_windows": tele["dive_windows"],
        "windows": windows,
        "death": death,
        "first_descent_beat": first_descent_beat,
        "alive_at_fd_plus_1800": alive_at_fd,
        "fd_plus_1800_censored": censored,
        "beats_by_dlvl": beats_by_dlvl,
        "l2_beats": beats_by_dlvl.get("2", 0),
        "died_on_l2": died_on_l2,
        "l2_death_beat": micro_steps if died_on_l2 else None,
    }


def main():
    worker_zip, seed_span, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    decoding_arg = sys.argv[5] if len(sys.argv) > 5 else "argmax"
    if decoding_arg not in ("argmax", "stochastic", "sample"):
        raise SystemExit(
            f"第 5 参数只允许 argmax|stochastic|sample,收到 {decoding_arg!r}")
    stochastic = decoding_arg in ("stochastic", "sample")
    overrides = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    unknown = set(overrides) - set(_R16_ENV_KEYS) - {"drink_sovereignty",
                                                     "manager"}
    if unknown:
        raise SystemExit(f"第 6 参数含未知键: {sorted(unknown)}")
    manager = str(overrides.get("manager", "readiness-v1"))
    if manager not in _R17_MANAGERS:
        raise SystemExit(f"manager 只允许 {_R17_MANAGERS},收到 {manager!r}")
    drink_sovereignty = bool(overrides.get("drink_sovereignty", False))
    env_overrides = {k: overrides[k] for k in _R16_ENV_KEYS if k in overrides}
    lo, hi = (int(x) for x in seed_span.split("-"))
    identity_before = source_identity(worker_zip)
    cb = load_zip_policy(worker_zip, stochastic=stochastic)
    if drink_sovereignty:
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
           "v3_compat": {"probe": V3_PROBE_VERSION,
                         "row_keys": list(V3_ROW_KEYS),
                         "rows_sha_v3": rows_sha_v3(rows)},
           "source_identity": identity_before,
           "source_changed_during_run": changed_during_run,
           "note": ("法证探针,非考卷;R17.0 仪表形态(v3 超集:逐次下楼遥测 "
                    "+ 脚本反事实经理)"),
           "row_semantics": (
               "died 行主列 = 死亡前最后存活快照(工人拍/存活收窗),"
               "post_death = 局末尸检值;存活行主列 = 局末值,post_death=None;"
               "descents/floors/dive_windows/death/windows 为 R17 仪表列"
               "(见模块 docstring)"),
           "trigger_semantics": {
               "coach_ready": "教练自愿 DIVE 且战备达标",
               "coach_cleared": "教练自愿 DIVE 因清场子句",
               "const_dive": "const-DIVE 无条件 DIVE(不 ready 不 cleared)",
               "mask_forced": "教练不想 DIVE,掩码 m[FARM]=False 回退到 DIVE",
               "fallback": "其余路径(非 DIVE 窗内换层)"},
           "forced_reason_semantics": {
               "handoff": "_farm_handoff(剧情目标交权)",
               "cap": "farm_scene_steps >= farm_scene_cap",
               "idle_clock": f"layer_clock >= KILL_PATIENCE({KILL_PATIENCE})",
               "exhausted_other": "exhausted 旗在位但上两者皆否"},
           "alive_at_fd_plus_1800_semantics": (
               "首降拍+1800 时存活(死亡拍 > 首降拍+1800 或存活到局末);"
               "存活但局末 < 首降拍+1800 者为右删失,记 "
               "fd_plus_1800_censored=True 且 alive_at_fd_plus_1800=None,"
               "不计存活、不进聚合分母(分母 = fd_plus_1800_observed_n)"),
           "worker_zip": worker_zip, "seeds": seed_span,
           "max_steps": max_steps,
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
