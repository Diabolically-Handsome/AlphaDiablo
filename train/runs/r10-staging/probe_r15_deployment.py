"""R15 deployment-form forensic probe (not an exam: it bypasses the archive identity system, results go only to the ledger forensics).

readiness-v1 script manager x worker zip (FARM+DIVE dual registration, autonomy on, economy v2),
per-seed statistics of deepest level/death/victory/micro-beats. Usage:
  probe_r15_deployment.py <worker.zip> <lo-hi seeds> <out.json> [max_steps=3000] [argmax|stochastic]
max_steps can be relaxed (long-horizon forensics: whether the readiness table can climb given an ample budget).

R16 amendment (v2):
  C4 autopsy contamination: on a single-player death the engine strips the gear and halves the gold, so reading the panel at episode end gives autopsy values
     (AC=4/dmg=1/gold=50). In this version the char_level/armor_class/max_hp/
     gold/xp/kills/hit_damage of died rows are all taken from the "last alive snapshot before death": the snapshot is refreshed once at every worker beat
     (callback entry; raw is the live state before that beat's decision) and once after every surviving window close;
     the post_death column separately stores the end-of-episode autopsy values for audit; snapshot_micro_step/
     snapshot_source record when the snapshot was taken. Surviving rows have post_death=None and the main columns are the end-of-episode values.
  agg adds survival strata: alive_n/alive_clvl_mean/alive_ac_mean/alive_kills_mean/
     dead_kills_mean/median_steps_dead etc.; xp_per_1k (raw["xp"]) normalized per thousand beats.
  C3 stochastic decoding now uses the same per-episode seeding as the exam (eval_assembled --worker-decoding sample):
     after each env.reset, model.set_random_seed(seed), single-threaded
     torch; the same zip and seed reproduce bit for bit across two runs. The argmax path is unchanged.
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

# R16 amendment (v3): argument 6 is a JSON environment override (new-law anchor/new-law arm deployment form), keys:
#   explore_global_hunt / explore_global_fallback / progress_far_tiles /
#   farm_scene_cap / reset_layer_clock_on_window / reward_economy /
#   drink_sovereignty(bool)/ manager("readiness-v1"|"readiness-v3").
# Omitted (no argument 6) = the old-law probe form, verbatim unchanged.
_R16_ENV_KEYS = ("explore_global_hunt", "explore_global_fallback",
                 "progress_far_tiles", "farm_scene_cap",
                 "reset_layer_clock_on_window", "reward_economy")
_R16_MANAGERS = ("readiness-v1", "readiness-v3")

PROBE_VERSION = "r15-deployment-v3"
# Forensic provenance (not the identity system): while several workstreams were changing the protocol in place in parallel, record the SHA-256 of the
# protocol sources and worker zip this probe actually imported, so results can be attributed to a specific source state.
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
    """stochastic=True: decode by sampling from the training distribution (used by the audit panel to tell "argmax freeze collapse" apart);
    the RNG is seeded per episode by the caller via choose.episode_reseed(seed); the default argmax matches the exam.
    If choose.on_beat is set, it is called once at every worker beat (before the decision); the probe uses it
    to take snapshots of the live state."""
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
    """The six panel values + xp (all taken from the same raw snapshot)."""
    return {
        "char_level": int(raw.get("char_level", 0)),
        "armor_class": int(raw.get("armor_class", 0)),
        "max_hp": int(raw.get("max_hp", 0)),
        "gold": int(raw.get("gold", 0)),
        "xp": int(raw.get("xp", 0)),
        # R15.2 extraction-rate forensics: kills/per-hit damage panel
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
        # Overall means (died rows now use the pre-death snapshot and are no longer contaminated by autopsy values)
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
        # Survival strata (None = that stratum is empty)
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
        # Audit comparison: mean autopsy values of died rows (the old probe v1 definition)
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
    """One episode; returns a row. See the module docstring for the snapshot strategy.
    manager: readiness-v1 (v0.1 table + drained-flag escape, old law) / readiness-v3 (R16:
    v0.2 table + escape on a per-level kill-count clear ratio >=1.0 (no live monsters), same as worker_env._mgr_choose)."""
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
        # At episode start seed with that episode's seed (same as eval_assembled sample mode);
        # a no-op under argmax.
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
                # v2 escape: a real clear; the free drained flag does not count
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
            f"argument 5 only allows argmax|stochastic|sample, got {decoding_arg!r}")
    stochastic = decoding_arg in ("stochastic", "sample")
    # R16 (v3): argument 6 is a JSON environment override; omitted = the old-law form, verbatim unchanged
    overrides = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    unknown = set(overrides) - set(_R16_ENV_KEYS) - {"drink_sovereignty",
                                                     "manager"}
    if unknown:
        raise SystemExit(f"argument 6 has unknown keys: {sorted(unknown)}")
    manager = str(overrides.get("manager", "readiness-v1"))
    if manager not in _R16_MANAGERS:
        raise SystemExit(f"manager only allows {_R16_MANAGERS}, got {manager!r}")
    drink_sovereignty = bool(overrides.get("drink_sovereignty", False))
    env_overrides = {k: overrides[k] for k in _R16_ENV_KEYS if k in overrides}
    lo, hi = (int(x) for x in seed_span.split("-"))
    identity_before = source_identity(worker_zip)
    cb = load_zip_policy(worker_zip, stochastic=stochastic)
    if drink_sovereignty:
        # Autonomy on: the worker callback's dual labels switch to environment-mask mode (OptionsEnv validates its own binding)
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
           # Non-empty means the source changed while the probe ran (parallel protocol changes); attribute results with care
           "source_changed_during_run": changed_during_run,
           "note": "forensic probe, not an exam; readiness-v1 script manager deployment form",
           "row_semantics": (
               "died row main columns = last alive snapshot before death (worker beat/surviving window close), "
               "post_death = end-of-episode autopsy values; surviving row main columns = end-of-episode values, post_death=None"),
           "worker_zip": worker_zip, "seeds": seed_span,
           "max_steps": max_steps,
           # R16 (v3): deployment-form declaration (omitted = old law: readiness-v1/autonomy off/v2/no override)
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
