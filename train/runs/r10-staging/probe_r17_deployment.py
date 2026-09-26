# Review fixes (2026-09-06): findings H2/H4/M2/L1 of the adversarial review of R17-0 (not published)
"""R17.0 instrumentation probe (a superset of probe_r15_deployment.py v3; the v3 file's bytes are untouched).

The argv contract is identical to v3:
  probe_r17_deployment.py <worker.zip> <lo-hi> <out.json> [max_steps=3000]
                          [argmax|stochastic|sample] [json-form]
Argument 6 JSON keys are the same as v3 (_R16_ENV_KEYS + drink_sovereignty + manager);
manager is extended to readiness-v1 | readiness-v3 | readiness-v3-strict | const-FARM |
const-DIVE. Every manager goes through the same mask fallback ladder FARM->DIVE->RESUPPLY and forced
descents are recorded.

Bit-level contract: with manager=readiness-v3 (or readiness-v1), each row's v3 columns
(V3_ROW_KEYS) are bit-identical to probe_r15 v3 under the same zip/seed/form: the decision
sequence, the snapshot hook points (every worker beat + every surviving window close) and the sampling seeding (at episode start
model.set_random_seed(seed); single-threaded torch) are all the same; the new R17 instrumentation only reads
raw / OptionsEnv state and never touches the RNG. rows_sha_v3(rows) gives the sha256 of those restricted columns,
which the driver script uses for a bit-level regression against r16-deploy-arm.json.

New R17.0 instrumentation (per row):
  descents[]      every dungeon_level increase (observed at the window close with reason=descend):
                  {beat, from_dlvl, to_dlvl, belt_heals, hp, max_hp, armor_class,
                  char_level, hit_damage, ratio_v2(=readiness_power_ratio_v2(raw,
                  to_dlvl)), ready(ratio_v2>=1), forced(not mask[FARM] at decision time),
                  trigger, forced_reason, window_id, kills_so_far,
                  floor_kills_before_descent, floor_beats_before_descent,
                  monsters_left_prev_floor, potions_visible_prev_floor}
                  monsters_left_prev_floor / potions_* come from the last live
                  observation before the descent (at most one beat earlier within the same window).
  death           {beat, dlvl, belt_heals_last_live, belt_heals_post_death,
                  floor_potions_total/visible/reachable (healing potions on this floor's ground at the last live observation:
                  all / visible / visible and reachable),
                  beats_on_current_floor, hp_last_live, max_hp_last_live,
                  last_live_beat, window_opt, window_trigger}
  floors[]        every floor stay: {dlvl, entry_beat, exit_beat, beats, kills,
                  monsters_left_at_exit, potions_visible_at_exit, exit_reason}
  dive_windows[]  every DIVE window: {window_id, beat0, beat1, tau, end_reason,
                  descended, trigger, forced, forced_reason, ratio_v2, cleared}
  windows         {total, farm, dive, resupply, forced_dive, mask_forced,
                  fallback, coach_dive_wants, coach_cleared_true}
  first_descent_beat / alive_at_fd_plus_1800 (None = never descended or observation truncated) /
  fd_plus_1800_censored (survival truncated before first descent + 1800) / beats_by_dlvl / l2_beats /
  died_on_l2 / l2_death_beat
trigger definition (the manager decision of the window in which the descent happened):
  coach_ready    the coach chose DIVE voluntarily and ratio_v2>=1 (readiness-v1 is ready by the v0.1 table)
  coach_cleared  the coach chose DIVE voluntarily because of the cleared clause
  const_dive     the unconditional DIVE of const-DIVE (neither ready nor cleared)
  mask_forced    the coach did not want DIVE; the mask m[FARM]=False fell through the fallback ladder to DIVE
  fallback       everything else (e.g. a level change inside a non-DIVE window)
forced_reason (the cause of not mask[FARM] at decision time, inferred from the mask law at options_env.py:739-741):
handoff (story-goal handover _farm_handoff) / cap (farm_scene_steps>=cap) /
idle_clock (layer_clock>=KILL_PATIENCE) / exhausted_other / None.
Beyond all v3 keys, agg adds farm_masked_at_descent_share (share of descents with not mask[FARM],
regardless of the coach's intent) / mask_forced_descent_share (trigger=mask_forced,
i.e. the share where "the manager was really overruled") / descents_total / ready_descents /
cleared_true_decisions (number of decisions where the coach's cleared clause was true) /
mean_beat_first_descent / alive_at_first_descent_plus_1800 /
l2_hazard_per_1k (L2 deaths / sum of L2 stay beats x 1000) and the descent panel distributions.
"""
import hashlib
import json
import os
import pathlib
import statistics
import sys

import numpy as np

# R18-M2 (2026-09-07) conflict resolution K2b: k2b.patch authored the SAME
# override independently (k2b.patch, probe:66-69), so this is no longer a
# line of behaviour the merge invented -- M-REPORT.md S7.1 is closed by K2b.
# The M spelling is kept: `get(...) or <default>` also treats an EMPTY
# DIABLOGYM_ROOT as unset, where K2b's two-argument get() would resolve it
# to the process cwd.
# R18-M (2026-09-07): tree override so the merged copy can be probed without
# touching the main tree. Unset (the deployed default) = the frozen path, so the
# regression arm is byte-identical to every earlier run.
ROOT = pathlib.Path(os.environ.get("DIABLOGYM_ROOT")
                    or (pathlib.Path.home() / "AlphaDiablo" / "diablogym"))
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
                 # R17 T0 (2026-09-06): the environment flags of the resource-channel arm pass straight through to OptionsEnv;
                 # they are not v3 row/agg keys, so the readiness-v3 bit-level path is unaffected.
                 "resource_protocol", "resource_purchase_mode",
                 "resource_service_policy", "resource_readiness_law",
                 # R17 T0': sustain-loot-v1 requires an explicit completion-l2-v1 clock
                 "worker_time_protocol", "resource_retreat",
                 # R18-D/E (2026-09-07): aggro cap + engagement priority flags
                 "aggro_cap", "engagement_priority",
                 # R18-G: global-hunt scope (the pull mechanism)
                 "hunt_scope",
                 # R18-F: Scroll of Town Portal interface
                 "resource_portal",
                 # R18-H (2026-09-07): chest/barrel sweep on main L1
                 "resource_sweep",
                 # R18-M (2026-09-07) conflict resolution H1/H2: both patches
                 # extended _R16_ENV_KEYS at the same closing paren. The UNION is
                 # kept, in flag order (sweep leg then identify leg). Order is
                 # inert: the tuple is only used as a whitelist and to build the
                 # env kwargs dict at probe_r17_deployment.py env_overrides.
                 # R18-H: Cain identify leg of the loot economy town trip
                 "resource_identify",
                 # R18-M2 (2026-09-07) conflict resolution K2b: the third
                 # patch extended the same closing paren.  UNION again, in
                 # trip order (sweep leg, identify leg, weapon leg).
                 # R18-K: the surplus weapon upgrade leg at Griswold
                 "resource_weapon_upgrade")
def _identify_field(env, name, default):
    """R18-H identify-v1: read one identify counter, or the off-arm default.

    The probe must keep working with the flag absent (no service constructed and
    an older bridge that never heard of identify-v1)."""
    service = getattr(env, "identify_service", None)
    if service is None:
        return default
    return getattr(service, name, default)


_R17_MANAGERS = ("readiness-v1", "readiness-v3", "readiness-v3-strict",
                 "const-FARM", "const-DIVE",
                 # R17 T0: with the protocol on, the script manager = OptionsEnv.resource_option_choice
                 "resource")

# R18-M2 (2026-09-07) conflict resolution K2b: k2b.patch stamped
# "r17-deployment-v3-r18f-r18k" (the town-trip purchase ledger and the
# smith-v1 telemetry).  The merged probe now carries the sweep, identify
# AND weapon instrument sets, so the "one version names the merge" rule
# gives it ONE new string naming this pass.
PROBE_VERSION = "r17-deployment-v3-r18m2"
V3_PROBE_VERSION = "r15-deployment-v3"
# The per-row fields of probe_r15 v3 (the bit-level regression definition); R17 rows are a superset.
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
    """Row sha256 restricted to the v3 columns (the bit-level regression definition against r16-deploy-*.json rows)."""
    payload = json.dumps(
        [{k: r[k] for k in V3_ROW_KEYS} for r in rows],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_zip_policy(path, stochastic=False):
    """Same as v3 (sampling seeding / single thread / on_beat hook)."""
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
    """The six panel values + xp (all taken from the same raw snapshot). Verbatim the same as v3."""
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
    """Count of healing potions on this floor's ground (the heal flag of floor_items; visible/reachable are the bridge-side
    visibility/reachability flags, same definition as env._policy_floor_items(raw, "heal"))."""
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
    """The script manager's want. Returns (want, ready, cleared, coach_reason, ratio).

    The readiness-v1 / readiness-v3 branches are verbatim equivalent to the inline logic of probe_r15 v3 run_episode
    (the per-level baseline resets when dungeon_level changes)."""
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
            raise ValueError(f"unknown manager {manager!r}")
    if want == DIVE:
        coach_reason = ("ready" if ready
                        else ("cleared" if cleared else "const"))
    else:
        coach_reason = "farm"
    return want, bool(ready), bool(cleared), coach_reason, float(ratio)


def forced_reason(env, raw):
    """Cause of m[FARM]=False at decision time (inferred from options_env.py:739-741)."""
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
    """All v3 agg keys (computed verbatim the same) + R17 extension keys."""
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

    # ---- R17.0 extensions ----
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
        # FARM was masked at decision time (regardless of the coach's intent); old name forced_descent_share
        "farm_masked_at_descent_share": (
            round(sum(1 for d in descents if d["forced"]) / len(descents), 4)
            if descents else None),
        # "The manager was really overruled": the coach did not want DIVE and the fallback ladder landed on DIVE
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
        # L1: number of decisions where the coach's cleared clause was true (the only source of divergence between v3 and v3-strict;
        # 0 means "v3 == strict" is an empirical coincidence, not a structural necessity)
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
    """One episode; returns a row (v3 columns verbatim the same + R17 instrumentation)."""
    snap = {"panel": None, "micro_step": None, "source": None}
    floor_state = {"dlvl": None, "kills_at_entry": 0}
    # ---- R17 instrumentation state (reads only raw / env state) ----
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
        """At every live observation point (before a worker beat / after a surviving window close) refresh the floor and live records."""
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
                # R17 T0: the telemetry's ready/cleared/ratio keep the v3-strict definition;
                # want is decided by the resource protocol's script manager (by the six laws under coach-v03).
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
    # ---- R17: closing floor and death records ----
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
    # ---- R17 T0: resource-channel telemetry (None when the protocol is off; not a v3 row key) ----
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
            # R18-M (2026-09-07) conflict resolution H1/H2: both patches appended
            # to the same resource telemetry dict. The UNION is kept, sweep leg
            # first then identify leg; each half still strips itself to None/0
            # when its own flag is absent, so an off arm is byte-identical.
            # R18-H sweep-v1 telemetry (None when off; row keys otherwise unchanged)
            "sweep": (env.sweep_service.telemetry()
                      if getattr(env, "sweep_service", None) is not None else None),
            # R18-H identify-v1 telemetry (None/0 when off; row keys unchanged otherwise)
            "identify": (env.identify_service.telemetry()
                         if getattr(env, "identify_service", None) is not None else None),
            "identified_count": _identify_field(env, "identified_count", 0),
            "identify_gold_spent": _identify_field(env, "identify_gold_spent", 0),
            "identified_items": _identify_field(env, "identified_items", []),
            "sale_income_identified": _identify_field(env, "sale_income_identified", 0),
            # R18-K: the town-trip purchase ledger (needed for the K2 report;
            # not a v3 row column, so rows_sha_v3 is unaffected).
            "loot_cumulative": (service.telemetry().get("cumulative")
                                if service is not None and hasattr(service, "trip_count") else None),
            "loot_trip_count": getattr(service, "trip_count", None),
            # R18-K smith-v1 telemetry (None / 0 when the flag is off)
            "weapon_upgrade_flag": getattr(env, "resource_weapon_upgrade", None),
            "weapon_upgrade": (env.weapon_upgrade_service.telemetry()
                               if getattr(env, "weapon_upgrade_service", None) is not None
                               else None),
            "weapon_upgrades": (env.weapon_upgrade_service.weapon_upgrades
                                if getattr(env, "weapon_upgrade_service", None) is not None
                                else 0),
            "weapon_gold_spent": (env.weapon_upgrade_service.weapon_gold_spent
                                  if getattr(env, "weapon_upgrade_service", None) is not None
                                  else 0),
            "weapon_damage_before": ([p["weapon_damage_before"] for p in
                                      env.weapon_upgrade_service.purchases]
                                     if getattr(env, "weapon_upgrade_service", None) is not None
                                     else []),
            "weapon_seams": (env.weapon_upgrade_service.telemetry().get("seams")
                             if getattr(env, "weapon_upgrade_service", None) is not None
                             else None),
            "weapon_damage_after": ([p["weapon_damage_after"] for p in
                                     env.weapon_upgrade_service.purchases]
                                    if getattr(env, "weapon_upgrade_service", None) is not None
                                    else []),
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
        # ---- R17.0 instrumentation ----
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
            f"argument 5 only allows argmax|stochastic|sample, got {decoding_arg!r}")
    stochastic = decoding_arg in ("stochastic", "sample")
    overrides = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    unknown = set(overrides) - set(_R16_ENV_KEYS) - {"drink_sovereignty",
                                                     "manager"}
    if unknown:
        raise SystemExit(f"argument 6 has unknown keys: {sorted(unknown)}")
    manager = str(overrides.get("manager", "readiness-v1"))
    if manager not in _R17_MANAGERS:
        raise SystemExit(f"manager only allows {_R17_MANAGERS}, got {manager!r}")
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
           "note": ("forensic probe, not an exam; R17.0 instrumentation form (v3 superset: per-descent telemetry "
                    "+ scripted counterfactual managers)"),
           "row_semantics": (
               "died row main columns = last alive snapshot before death (worker beat/surviving window close), "
               "post_death = end-of-episode autopsy values; surviving row main columns = end-of-episode values, post_death=None; "
               "descents/floors/dive_windows/death/windows are R17 instrumentation columns"
               " (see the module docstring)"),
           "trigger_semantics": {
               "coach_ready": "the coach chose DIVE voluntarily and readiness was met",
               "coach_cleared": "the coach chose DIVE voluntarily because of the cleared clause",
               "const_dive": "const-DIVE unconditional DIVE (neither ready nor cleared)",
               "mask_forced": "the coach did not want DIVE; mask m[FARM]=False fell back to DIVE",
               "fallback": "other paths (level change inside a non-DIVE window)"},
           "forced_reason_semantics": {
               "handoff": "_farm_handoff (story-goal handover)",
               "cap": "farm_scene_steps >= farm_scene_cap",
               "idle_clock": f"layer_clock >= KILL_PATIENCE({KILL_PATIENCE})",
               "exhausted_other": "exhausted flag set but neither of the above"},
           "alive_at_fd_plus_1800_semantics": (
               "alive at first-descent beat + 1800 (death beat > first-descent beat + 1800, or alive at episode end); "
               "alive but episode end < first-descent beat + 1800 is right-censored, recorded as "
               "fd_plus_1800_censored=True and alive_at_fd_plus_1800=None, "
               "not counted as alive and not in the aggregate denominator (denominator = fd_plus_1800_observed_n)"),
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
