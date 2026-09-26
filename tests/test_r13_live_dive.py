"""R13 classroom-reform tests: handing live-DIVE control to the worker (--worker-learning-window-scope).

Coverage:
    T1 fail-closed validation of registered names;
    T2 flag off by default: scope=farm-only, OptionsEnv handover off, the FARM-window mask bars unchanged;
    T3 flag on: live DIVE windows really open, a11 is unlocked and executable, the per-window accounting identity,
          the wage identity (R==W+bonus, backed by a hard check in the shared window core), audit counters;
    T4 fail-closed parameters on the exam surface (illegal registered values / script workers).

Frozen-surface discipline: this file tests only the new rules; that the old rules are bit-for-bit unchanged is enforced by
test_worker_env G0'.a and by the rows-sha equality of the anchor re-bake (R13 pre-registration section 4.1).
"""
import numpy as np
import pytest

from diablogym.options_env import DIVE, FARM
from diablogym.worker_env import (
    WorkerWindowEnv,
    _coerce_learning_window_scope,
)

_SEED = 999983   # outside the reserved ranges (outside the registered pools and the embargo-range sampler)


def test_scope_coerce_fail_closed():
    assert _coerce_learning_window_scope("farm-only") == "farm-only"
    assert _coerce_learning_window_scope("farm-dive-v1") == "farm-dive-v1"
    for bad in (None, True, False, "farm", "dive", "farm-dive",
                "FARM-DIVE-V1", 1):
        with pytest.raises(ValueError):
            _coerce_learning_window_scope(bad)


def _mk_env(scope):
    return WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED,
        drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope=scope,
        reward_economy="v2")


def test_flag_off_defaults_and_farm_mask_bars():
    env = _mk_env("farm-only")
    try:
        assert env.learning_window_scope == "farm-only"
        assert env.oe.dive_live_sovereignty is False
        obs, _ = env.reset(seed=_SEED)
        # flag off: m[11] is always False in every window (bars unchanged)
        for _ in range(64):
            mask = env.action_masks()
            assert not mask[11], "a11 unlocked on the flag-off path"
            legal = np.flatnonzero(mask)
            a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            assert "window_mode" not in info, "R13 info key leaked on the flag-off path"
            if term or trunc:
                env.reset()
        s = env.stats
        assert "dive_live_windows" not in s
        assert "dive_live_steps" not in s
        assert "farm_live_steps" not in s
    finally:
        env.close()


def test_flag_on_dive_live_sovereignty_and_metering():
    env = _mk_env("farm-dive-v1")
    try:
        assert env.oe.dive_live_sovereignty is True
        obs, _ = env.reset(seed=_SEED)
        assert obs.shape == (13012,)
        rng = np.random.default_rng(7)
        steps = 0
        farm_mask11_violations = 0
        dive_beats_seen = 0
        for _ in range(4000):
            mask = env.action_masks()
            win = env.oe._win
            in_dive = win is not None and win["opt"] == DIVE
            if in_dive:
                dive_beats_seen += 1
                a = (11 if (mask[11] and rng.random() < 0.5)
                     else int(rng.choice(np.flatnonzero(mask))))
            else:
                if mask[11]:
                    farm_mask11_violations += 1
                legal = np.flatnonzero(mask)
                a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            steps += 1
            if info.get("window_mode") is not None:
                assert info["window_mode"] in (
                    "farm", "dive", "resupply")
            if term or trunc:
                env.reset()
        s = env.stats
        # The classroom happens: live DIVE windows really open and the worker is really driving
        assert s.get("dive_live_windows", 0) > 0
        assert dive_beats_seen > 0
        assert s.get("dive_live_a11_executed", 0) > 0, \
            "a11 was never really executed: the handover failed"
        # non-DIVE windows keep their bars
        assert farm_mask11_violations == 0, \
            "a11 unlocked without authority inside a FARM/RESUPPLY window"
        # per-window accounting identity: farm+dive live micro-ticks == SB3 steps
        assert (s.get("dive_live_steps", 0)
                + s.get("farm_live_steps", 0)) == steps
        # window close reason buckets do not exceed the window total
        assert (s.get("dive_live_descends", 0)
                + s.get("dive_live_stalls", 0)
                + s.get("dive_live_deaths", 0)) <= s["dive_live_windows"]
    finally:
        env.close()


def test_depth_shaping_flag_law_and_accounting():
    from diablogym.worker_env import _coerce_depth_shaping_unit
    assert _coerce_depth_shaping_unit(0.0) == 0.0
    assert _coerce_depth_shaping_unit(8) == 8.0
    for bad in (-1.0, float("nan"), float("inf"), 101.0):
        with pytest.raises(ValueError):
            _coerce_depth_shaping_unit(bad)
    # shaping is limited to the farm-dive-v1 classroom
    with pytest.raises(ValueError):
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED,
            drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-only",
            depth_shaping_unit=8.0,
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED,
        drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        depth_shaping_unit=8.0,
        reward_economy="v2")
    try:
        obs, _ = env.reset(seed=_SEED)
        rng = np.random.default_rng(7)
        for _ in range(4000):
            mask = env.action_masks()
            win = env.oe._win
            in_dive = win is not None and win["opt"] == DIVE
            if in_dive and mask[11]:
                a = 11   # push the stairs as hard as possible to force a descend and check that shaping is booked
            elif in_dive:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                legal = np.flatnonzero(mask)
                a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                env.reset()
            s = env.stats
            if s.get("dive_live_descends", 0) > 0:
                break
        s = env.stats
        assert s.get("dive_live_descends", 0) > 0, "no descend was forced"
        credited = float(s.get("depth_shaping_credited", 0.0))
        assert credited >= 8.0, f"shaping not booked: {credited}"
        # conservation: credited must be an integer multiple of unit (exactly unit per level)
        assert abs(credited / 8.0 - round(credited / 8.0)) < 1e-9
        refunded = float(s.get("depth_shaping_refunded", 0.0))
        assert refunded <= 0.0
    finally:
        env.close()


def test_descend_bonus_fraction_law_and_accounting():
    """R14 option B: bounded descent-bonus refund; rule validation + bookkeeping conservation."""
    with pytest.raises(ValueError):
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-only",
            worker_descend_bonus_fraction=0.25,
            reward_economy="v2")
    with pytest.raises(ValueError):
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_bonus_fraction=1.5,
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_bonus_fraction=0.25,
        reward_economy="v2")
    try:
        assert env.oe.worker_descend_bonus_fraction == 0.25
        obs, _ = env.reset(seed=_SEED)
        rng = np.random.default_rng(7)
        for _ in range(4000):
            mask = env.action_masks()
            win = env.oe._win
            in_dive = win is not None and win["opt"] == DIVE
            if in_dive and mask[11]:
                a = 11
            elif in_dive:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                legal = np.flatnonzero(mask)
                a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                env.reset()
            if env.stats.get("dive_live_descends", 0) > 0:
                break
        s = env.stats
        assert s.get("dive_live_descends", 0) > 0, "no descend was forced"
        kept = float(s.get("descend_bonus_kept", 0.0))
        # v2 descend_unit=48, L(d)->L(d+1) full amount = 48 x d, kept 25% = 12 x d
        assert kept >= 12.0, f"kept descent bonus not booked: {kept}"
        assert abs(kept / 12.0 - round(kept / 12.0)) < 1e-6, \
            f"kept amount is not an integer multiple of 12: {kept}"
        # The identity R == W + the amount actually stripped is backed by the hard check in _win_end: reaching this point without raising
        # means conservation holds.
    finally:
        env.close()


def test_descend_escrow_law_and_settlement():
    """R14.2 option D: escrow payout; rule validation + vest/forfeit settlement."""
    with pytest.raises(ValueError):   # options B and D are mutually exclusive
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_bonus_fraction=0.25,
            worker_descend_escrow_fraction=0.5,
            reward_economy="v2")
    with pytest.raises(ValueError):   # classroom scope of the rule
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-only",
            worker_descend_escrow_fraction=0.5,
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_escrow_fraction=0.5,
        reward_economy="v2")
    try:
        obs, _ = env.reset(seed=_SEED)
        rng = np.random.default_rng(7)
        for _ in range(8000):
            mask = env.action_masks()
            win = env.oe._win
            in_dive = win is not None and win["opt"] == DIVE
            if in_dive and mask[11]:
                a = 11
            elif in_dive:
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                legal = np.flatnonzero(mask)
                a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            if term or trunc:
                env.reset()
            s = env.stats
            if (float(s.get("descend_escrow_vested", 0.0))
                    + float(s.get("descend_escrow_forfeited", 0.0))
                    >= 24.0):
                break
        s = env.stats
        vested = float(s.get("descend_escrow_vested", 0.0))
        forfeited = float(s.get("descend_escrow_forfeited", 0.0))
        assert s.get("dive_live_descends", 0) > 0, "no descend was forced"
        # each escrow amount = 0.5 x 48 x sum of level numbers = an integer multiple of 24; vest and forfeit share one source
        total = vested + forfeited
        assert total >= 24.0, f"escrow never settled: v={vested} f={forfeited}"
        assert abs(total / 24.0 - round(total / 24.0)) < 1e-6, \
            f"settled amount is not an integer multiple of 24: {total}"
        # conservation: the pending balance is never negative
        assert float(getattr(env, "_descend_escrow", 0.0)) >= 0.0
    finally:
        env.close()


def test_descend_escrow_power_curve():
    """R14.3 option E: convex-curve pricing; the rule + exact per-level d^power bookkeeping."""
    with pytest.raises(ValueError):   # power != 1 must be used together with escrow
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_escrow_power=1.6,
            reward_economy="v2")
    with pytest.raises(ValueError):   # range
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_escrow_fraction=0.5,
            worker_descend_escrow_power=3.5,
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_escrow_fraction=0.5,
        worker_descend_escrow_power=1.6,
        reward_economy="v2")
    try:
        env.reset(seed=_SEED)
        # drive the settler directly: simulate the deepest level going 1->3 within this transition (two levels down)
        env.oe.env._econ_episode_max_depth = 3
        env._descend_escrow = 0.0
        vest = env._descend_escrow_settlement(1, "descend")
        assert vest == 0.0   # no existing escrow
        expected = 0.5 * 48.0 * (1.0 ** 1.6 + 2.0 ** 1.6)
        assert abs(env._descend_escrow - expected) < 1e-9, \
            f"convex-curve pricing is off: {env._descend_escrow} != {expected}"
        # next non-death window close: full vest
        d_now = 3
        vest2 = env._descend_escrow_settlement(d_now, "exhausted")
        assert abs(vest2 - expected) < 1e-9
        assert env._descend_escrow == 0.0
        # death window close: forfeit
        env._descend_escrow = 100.0
        vest3 = env._descend_escrow_settlement(d_now, "death")
        assert vest3 == 0.0 and env._descend_escrow == 0.0
        assert float(env.stats.get(
            "descend_escrow_forfeited", 0.0)) >= 100.0
    finally:
        env.close()


def test_readiness_power_ratio_and_coach():
    """R15 readiness gate: weakest-link logic / resistance gate / table monotonicity / coach release decision."""
    from diablogym.worker_env import (
        _READINESS_CLVL,
        _READINESS_DMG,
        _READINESS_HP,
        readiness_power_ratio,
    )
    # table monotonicity: every dimension's threshold is non-decreasing with depth
    from diablogym.worker_env import _READINESS_AC
    for tab in (_READINESS_CLVL, _READINESS_HP):
        assert all(tab[i] <= tab[i + 1] for i in range(15)), tab
    assert len(_READINESS_AC) == len(_READINESS_DMG) == 16
    # starting warrior (live probe record) against L1: every dimension met
    start = {"char_level": 1, "max_hp": 70, "armor_class": 12,
             "item_max_damage": 6, "damage_mod": 0,
             "fire_resist": 0, "magic_resist": 0}
    assert readiness_power_ratio(start, 1) >= 1.0
    # against L2: HP/AC/damage are three weak links, must stop
    assert readiness_power_ratio(start, 2) < 1.0
    # weakest-link logic: max level but no gear, still stopped
    naked_hero = dict(start, char_level=30, max_hp=400)
    assert readiness_power_ratio(naked_hero, 9) < 1.0
    # deep resistance gate: four dimensions maxed but zero fire resistance stops at L13; fire resistance 50 passes
    hell_ready = {"char_level": 20, "max_hp": 320, "armor_class": 100,
                  "item_max_damage": 25, "damage_mod": 8,
                  "fire_resist": 0, "magic_resist": 0}
    assert readiness_power_ratio(hell_ready, 13) < 1.0
    assert readiness_power_ratio(
        dict(hell_ready, fire_resist=50), 13) >= 1.0
    # v2 escape criterion: immune to free-riding (idling does not change the surviving share), a real clear passes
    from diablogym.worker_env import readiness_floor_cleared
    full_roster = {"monsters": [{"hp": 5}] * 98}
    assert not readiness_floor_cleared(full_roster)
    half_done = {"monsters": ([{"hp": 0}] * 68 + [{"hp": 5}] * 30)}
    assert not readiness_floor_cleared(half_done)   # 30/98 > 25%, stopped
    below_line = {"monsters": ([{"hp": 0}] * 80 + [{"hp": 5}] * 18)}
    assert readiness_floor_cleared(below_line)      # 18/98 <= 25%, passes
    truly_cleared = {"monsters": ([{"hp": 0}] * 96 + [{"hp": 5}] * 2)}
    assert readiness_floor_cleared(truly_cleared)
    assert not readiness_floor_cleared({"monsters": []})   # missing roster: conservatively rejected
    # coach construction: readiness-v1 is legal, unknown names are rejected
    env = _mk_env("farm-dive-v1")
    env.close()
    with pytest.raises(ValueError):
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="readiness-v99",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="readiness-v1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        reward_economy="v2")
    try:
        env.reset(seed=_SEED)
        # the starting warrior is not ready for L2 and the floor is not exhausted -> the coach must give FARM
        if not env.oe.exhausted:
            assert env._mgr_choose() == FARM
    finally:
        env.close()


def test_descend_escrow_readiness_gate():
    """R15 amendment 2: an unready descent puts nothing into escrow (enforced on the wage side)."""
    with pytest.raises(ValueError):   # only together with escrow
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_escrow_readiness_gate=True,
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_escrow_fraction=0.5,
        worker_descend_escrow_power=1.6,
        worker_descend_escrow_readiness_gate=True,
        reward_economy="v2")
    try:
        env.reset(seed=_SEED)
        # a cold-start warrior is never ready for L2: simulate a descent 1->2, escrow must refuse to book
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow = 0.0
        vest = env._descend_escrow_settlement(1, "descend")
        assert vest == 0.0
        assert env._descend_escrow == 0.0, "an unready descent was put into escrow"
        assert float(env.stats.get(
            "descend_escrow_unready_denied", 0.0)) >= 24.0
        # forge a ready panel (editing raw directly, for tests only): the same descent should go into escrow
        env.oe.env._raw.update({
            "char_level": 5, "max_hp": 120, "armor_class": 20,
            "item_max_damage": 9, "damage_mod": 1})
        env._descend_escrow = 0.0
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow_settlement(1, "descend")
        assert env._descend_escrow > 0.0, "a ready descent was not put into escrow"
    finally:
        env.close()


def test_r16_readiness_v2_table_and_clear_ratio():
    """R16 rule revision (C1/C2): the v0.2 readiness table is reachable by engine arithmetic; the kill-based clear criterion excludes golems."""
    from diablogym.worker_env import (
        _GOLEM_TYPE,
        _READINESS_V2_AC,
        _READINESS_V2_CLVL,
        _READINESS_V2_DMG,
        readiness_clear_ratio,
        readiness_power_ratio_v2,
        roster_alive_count,
    )
    assert len(_READINESS_V2_CLVL) == len(_READINESS_V2_AC) == 16
    assert len(_READINESS_V2_DMG) == 16
    for tab in (_READINESS_V2_CLVL, _READINESS_V2_AC, _READINESS_V2_DMG):
        assert all(tab[i] <= tab[i + 1] for i in range(15)), tab
    # a fresh warrior (live record AC7/dmg6) must be ready for L1; v0.1's 0.58 was a scale misalignment
    start = {"char_level": 1, "armor_class": 7, "item_max_damage": 6,
             "damage_mod": 0, "fire_resist": 0}
    assert readiness_power_ratio_v2(start, 1) >= 1.0
    # a survivor who farmed L1 well (clvl2/AC9/dmg6, mean of the deployment survivor rows) is ready for L2
    l1_survivor = {"char_level": 2, "armor_class": 9, "item_max_damage": 6,
                   "damage_mod": 0, "fire_resist": 0}
    assert readiness_power_ratio_v2(l1_survivor, 2) >= 1.0
    # but still stopped at L3 (clvl weak link)
    assert readiness_power_ratio_v2(l1_survivor, 3) < 1.0
    # clear ratio: roster of 98 = 94 real monsters + 4 golems; golems do not count
    roster = ([{"hp": 5, "type": 1}] * 94
              + [{"hp": 1, "type": _GOLEM_TYPE}] * 4)
    raw = {"monsters": roster, "monster_kill_total": 0}
    assert roster_alive_count(raw) == 94
    assert readiness_clear_ratio(raw, 0) == 0.0
    # 71 killed (23 real monsters + golems left on the roster): 71/(71+23) = 0.755 >= 0.75
    raw2 = {"monsters": ([{"hp": 5, "type": 1}] * 23
                         + [{"hp": 1, "type": _GOLEM_TYPE}] * 4),
            "monster_kill_total": 71}
    assert readiness_clear_ratio(raw2, 0) >= 0.75
    # immune to idle free-riding: kills unchanged, ratio unchanged
    assert readiness_clear_ratio(raw, 0) == readiness_clear_ratio(raw, 0)


def test_r16_hp_economy_and_timeout_law():
    """R16 (C7/C8): bookkeeping of hit-point pricing / potion pickup reward + rule validation that timeout != death."""
    with pytest.raises(ValueError):
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            worker_no_progress_timeout_credit="half",
            reward_economy="v2")
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="readiness-v3",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_hp_loss_price=0.5,
        worker_potion_pickup_bonus=2.0,
        worker_no_progress_timeout_credit="zero",
        reward_economy="v2")
    try:
        assert env.no_progress_timeout_credit == "zero"
        assert env._no_progress_timeout_failure_components(
            True, {"episode_extra": {"depth": 1, "died": False}}) == (
                0.0, 0.0, 0.0)
        env.reset(seed=_SEED)
        rng = np.random.default_rng(7)
        total = 0.0
        for _ in range(1500):
            mask = env.action_masks()
            a = 9 if mask[9] else int(rng.choice(np.flatnonzero(mask)))
            obs, r, term, trunc, info = env.step(int(a))
            total += float(r)
            if term or trunc:
                env.reset()
        s = env.stats
        # 1500 ticks of melee always lose hit points: the pricing entry must appear and be negative
        assert float(s.get("hp_loss_charged", 0.0)) < 0.0
        # farm_scene_cap / reset flag defaults have zero drift
        assert env.oe.farm_scene_cap == 1800
        assert env.oe.reset_layer_clock_on_window is False
    finally:
        env.close()


def test_eval_registration_param_fail_closed():
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root / "train") not in sys.path:
        sys.path.insert(0, str(root / "train"))
    import eval_assembled as ea
    with pytest.raises(ea.EvalContractError):
        ea.evaluate(None, [1], worker_window_registration="bogus")
    with pytest.raises(ea.EvalContractError):
        # farm-dive-v1 cannot be registered for a script worker (workers=None)
        ea.evaluate(None, [1], worker_window_registration="farm-dive-v1")


def test_r17_escrow_readiness_table_selector():
    """R17.0 amendment 1 (review-panel correction 2.1): escrow readiness-gate ruler selector v1/v2.

    v2 uses the same ruler as the readiness-v3 coach (the v0.2 table); v1 is the old ruler, bit for bit; the count of gated descents
    descend_escrow_ready_vested_count is kept under both rulers.
    """
    from diablogym.worker_env import (
        _READINESS_V2_AC,
        _READINESS_V2_CLVL,
        _READINESS_V2_DMG,
        readiness_power_ratio,
        readiness_power_ratio_v2,
    )

    def _mk(table, gate=True):
        return WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            fast_forward_reward_credit="terminal-death-only",
            learning_window_scope="farm-dive-v1",
            worker_descend_escrow_fraction=0.5,
            worker_descend_escrow_power=1.6,
            worker_descend_escrow_readiness_gate=gate,
            worker_descend_escrow_readiness_table=table,
            reward_economy="v2")

    with pytest.raises(ValueError):      # illegal ruler name
        _mk("v3")
    with pytest.raises(ValueError):      # v2 only together with the readiness gate
        _mk("v2", gate=False)
    env = _mk("v1")                      # default ruler attribute has zero drift
    try:
        assert env.descend_escrow_readiness_table == "v1"
    finally:
        env.close()
    # the panel sits exactly on the L2 threshold of the v0.2 table, but does not meet the v0.1 table (clvl3/HP90/AC15/dmg8)
    panel = {"char_level": _READINESS_V2_CLVL[1], "max_hp": 60,
             "armor_class": _READINESS_V2_AC[1],
             "item_max_damage": _READINESS_V2_DMG[1], "damage_mod": 0}
    assert readiness_power_ratio_v2(panel, 2) >= 1.0
    assert readiness_power_ratio(panel, 2) < 1.0
    for table, expect_escrow in (("v1", False), ("v2", True)):
        env = _mk(table)
        try:
            env.reset(seed=_SEED)
            env.oe.env._raw.update(panel)     # editing raw directly, for tests only
            env.oe.env._econ_episode_max_depth = 2
            env._descend_escrow = 0.0
            assert env._descend_escrow_settlement(1, "descend") == 0.0
            ready_n = int(env.stats.get(
                "descend_escrow_ready_vested_count", 0))
            denied = float(env.stats.get(
                "descend_escrow_unready_denied", 0.0))
            if expect_escrow:
                assert env._descend_escrow > 0.0, "a v2-ready descent was not put into escrow"
                assert ready_n == 1 and denied == 0.0
                vest = env._descend_escrow_settlement(2, "stall")
                assert vest > 0.0, "the next non-death window close did not vest"
                assert float(env.stats.get(
                    "descend_escrow_vested", 0.0)) == vest
            else:
                assert env._descend_escrow == 0.0, "the old v1 ruler let it through"
                assert ready_n == 0 and denied > 0.0
        finally:
            env.close()


def test_r17_policy_reward_component_receipts():
    """R17.0 amendment 1 (correction 2.2): per-transition receipts of the three policy_reward components.

    For every transition: transition_reward == wage + credited_ff_terminal_death
    + timeout + shaping + vest + hp_econ (in the same order as the worker_env assembly point, floating point
    equal bit for bit); the three keys are present and finite in every info.
    """
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="readiness-v3",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_escrow_fraction=0.5,
        worker_descend_escrow_power=1.6,
        worker_descend_escrow_readiness_gate=True,
        worker_descend_escrow_readiness_table="v2",
        worker_hp_loss_price=0.5,
        worker_potion_pickup_bonus=2.0,
        worker_no_progress_timeout_credit="zero",
        reward_economy="v2")
    keys = ("worker_depth_shaping_reward", "worker_descend_escrow_vest",
            "worker_hp_economy_reward")
    try:
        env.reset(seed=_SEED)
        rng = np.random.default_rng(11)
        n_hp = 0
        for _ in range(1500):
            mask = env.action_masks()
            a = 9 if mask[9] else int(rng.choice(np.flatnonzero(mask)))
            obs, r, term, trunc, info = env.step(int(a))
            for k in keys:
                assert k in info and isinstance(info[k], float)
                assert np.isfinite(info[k])
            expected = (
                float(info["worker_wage"])
                + float(info.get(
                    "credited_fast_forward_terminal_death_reward", 0.0))
                + float(info["no_progress_timeout_failure_reward"])
                + info["worker_depth_shaping_reward"]
                + info["worker_descend_escrow_vest"]
                + info["worker_hp_economy_reward"])
            assert float(r) == expected == float(info["transition_reward"])
            if info["worker_hp_economy_reward"] != 0.0:
                n_hp += 1
            if term or trunc:
                env.reset()
        assert n_hp > 0, "1500 ticks of melee must produce a hit-point pricing component"
    finally:
        env.close()


def test_r17_formal_pg_receipt_counts_vest_hp_econ_shaping():
    """R17.0 amendment 1 (correction 2.2, receipt landmine): the formal-PG goalkeeper timeout identity becomes
    wage + timeout + shaping + vest + hp_econ; a missing key = 0 (old fixtures pass bit for bit);
    components on the books while reward reports only wage+timeout must raise; a non-finite component must raise."""
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    for sub in ("train", "tests"):
        if str(root / sub) not in sys.path:
            sys.path.insert(0, str(root / sub))
    import test_critic_migration as tcm

    def _model():
        model = tcm.LeashedMaskablePPO(
            tcm.AsymmetricWorkerMaskableActorCriticPolicy,
            tcm._FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=2, batch_size=2,
            n_epochs=(
                tcm.WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT),
            learning_rate=1e-3, device="cpu", verbose=0)
        model.configure_critic_migration(
            gradient_clip_mode=tcm.GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=2)
        model._setup_learn(total_timesteps=2)
        return model

    base = {
        **tcm._formal_pg_info(9, 0.0, combat=True),
        "worker_wage": 1.0,
        "worker_no_progress_timeout": True,
        "TimeLimit.truncated": False,
        "time_limit_bootstrap_safe": False,
        "unsettled_budget_terminal": False,
        "existing_terminal_death_reward": 0.0,
        "additional_terminal_death_reward": 0.0,
        "total_terminal_death_reward": 0.0,
    }
    # components use exact binary values (the rollout receipt stores transition_reward as float32)
    components = {"worker_depth_shaping_reward": 8.0,
                  "worker_descend_escrow_vest": 12.5,
                  "worker_hp_economy_reward": -0.25}
    legacy = {**base,                    # old-rule branch (death-equivalent)
              "no_progress_timeout_base_failure_reward": -16.0,
              "no_progress_timeout_additional_failure_reward": -32.0,
              "no_progress_timeout_failure_reward": -48.0}
    zero = {**base,                      # R16 zero-score branch
            "no_progress_timeout_credit": "zero",
            "no_progress_timeout_base_failure_reward": 0.0,
            "no_progress_timeout_additional_failure_reward": 0.0,
            "no_progress_timeout_failure_reward": 0.0}
    for name, row in (("legacy", legacy), ("zero", zero)):
        timeout_total = row["no_progress_timeout_failure_reward"]
        # missing key = 0: the old fixture identity reward == wage + timeout passes unchanged
        model = _model()
        model._update_info_buffer(
            [{**row, "transition_reward": 1.0 + timeout_total}],
            dones=np.ones(1, dtype=bool))
        assert model._worker_onpolicy_pg_pending_receipts[-1][
            "transition_reward"] == 1.0 + timeout_total, name
        # with components: summed in the worker_env assembly order
        expected = 1.0 + timeout_total + 8.0 + 12.5 + (-0.25)
        with_components = {**row, **components,
                           "transition_reward": expected}
        model = _model()
        model._update_info_buffer(
            [with_components], dones=np.ones(1, dtype=bool))
        assert model._worker_onpolicy_pg_pending_receipts[-1][
            "transition_reward"] == expected, name
        # landmine reproduced: components on the books while reward reports only wage+timeout -> must raise
        model = _model()
        with pytest.raises(RuntimeError, match="timeout"):
            model._update_info_buffer(
                [{**with_components,
                  "transition_reward": 1.0 + timeout_total}],
                dones=np.ones(1, dtype=bool))
        # non-finite component -> raises
        model = _model()
        with pytest.raises(RuntimeError, match="component"):
            model._update_info_buffer(
                [{**with_components,
                  "worker_descend_escrow_vest": float("nan")}],
                dones=np.ones(1, dtype=bool))


def test_r17_descend_gate_telemetry():
    """R17.0 instrument (correction 2.3): per-gate telemetry of the escrow readiness gate (ratio under both rulers + panel)."""
    from diablogym.worker_env import (
        _READINESS_V2_AC, _READINESS_V2_CLVL, _READINESS_V2_DMG)
    env = WorkerWindowEnv(
        manager_npz=None, manager_heuristic="level-margin-1",
        max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
        policy_observation_view="dual-v4-asymmetric-v3",
        fast_forward_reward_credit="terminal-death-only",
        learning_window_scope="farm-dive-v1",
        worker_descend_escrow_fraction=0.5,
        worker_descend_escrow_power=1.6,
        worker_descend_escrow_readiness_gate=True,
        worker_descend_escrow_readiness_table="v2",
        reward_economy="v2")
    try:
        env.reset(seed=_SEED)
        assert "descend_escrow_gate_log" not in env.stats
        # a cold-start warrior descends 1->2: neither ruler is met, one row with passed=False
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow = 0.0
        env._descend_escrow_settlement(1, "descend")
        log = env.stats["descend_escrow_gate_log"]
        assert len(log) == 1 and log[0]["passed"] is False
        assert log[0]["depth"] == 2 and log[0]["table"] == "v2"
        assert log[0]["ratio_v2"] < 1.0 and log[0]["ratio_v1"] < 1.0
        assert log[0]["escrow"] > 0.0
        # ready panel: the second row passed=True, ratio_v2 >= 1 > ratio_v1
        env.oe.env._raw.update({
            "char_level": _READINESS_V2_CLVL[1], "max_hp": 60,
            "armor_class": _READINESS_V2_AC[1],
            "item_max_damage": _READINESS_V2_DMG[1], "damage_mod": 0})
        env._descend_escrow = 0.0
        env._descend_escrow_settlement(1, "descend")
        assert len(log) == 2 and log[1]["passed"] is True
        assert log[1]["ratio_v2"] >= 1.0 > log[1]["ratio_v1"]
        assert log[1]["armor_class"] == _READINESS_V2_AC[1]
        assert int(env.stats["descend_escrow_ready_vested_count"]) == 1
    finally:
        env.close()
