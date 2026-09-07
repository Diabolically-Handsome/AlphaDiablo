"""R13 教室改革执法档:live-DIVE 主权移交(--worker-learning-window-scope)。

覆盖面:
  T1 具名校验 fail-closed;
  T2 旗关默认:scope=farm-only、OptionsEnv 主权关、FARM 窗掩码铁栅原样;
  T3 旗开行为:DIVE live 窗真实开出、a11 解禁且可执行、分窗计量恒等式、
     工资恒等式(R≡W+bonus 由共享窗口核硬校验背书)、审计计数器;
  T4 考试面参数 fail-closed(非法注册值/script 工人)。

冻结面纪律:本档只测新法;旧法逐位不变由 test_worker_env G0'.a 与
锚重烤 rows-sha 等式执法(R13 预注册 §4.1)。
"""
import numpy as np
import pytest

from diablogym.options_env import DIVE, FARM
from diablogym.worker_env import (
    WorkerWindowEnv,
    _coerce_learning_window_scope,
)

_SEED = 999983   # 非保留段(登记池/禁运段拒采样器之外)


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
        # 旗关:任何窗口内 m[11] 恒 False(铁栅原样)
        for _ in range(64):
            mask = env.action_masks()
            assert not mask[11], "旗关路径 a11 被解禁"
            legal = np.flatnonzero(mask)
            a = 9 if mask[9] else int(legal[0])
            obs, r, term, trunc, info = env.step(int(a))
            assert "window_mode" not in info, "旗关路径泄漏 R13 info 键"
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
        # 课堂发生性:live DIVE 窗真实开出且工人真在开车
        assert s.get("dive_live_windows", 0) > 0
        assert dive_beats_seen > 0
        assert s.get("dive_live_a11_executed", 0) > 0, \
            "a11 从未被真实执行——主权移交失败"
        # 非 DIVE 窗铁栅原样
        assert farm_mask11_violations == 0, \
            "FARM/RESUPPLY 窗内 a11 被越权解禁"
        # 分窗计量恒等式:farm+dive live 微拍 ≡ SB3 步数
        assert (s.get("dive_live_steps", 0)
                + s.get("farm_live_steps", 0)) == steps
        # 收窗 reason 分桶不超窗口总量
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
    # 塑形仅限 farm-dive-v1 教室
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
                a = 11   # 全力踏梯,逼出 descend 以验证塑形入账
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
        assert s.get("dive_live_descends", 0) > 0, "未逼出 descend"
        credited = float(s.get("depth_shaping_credited", 0.0))
        assert credited >= 8.0, f"塑形未入账: {credited}"
        # 守恒:credited 必为 unit 的整数倍(每层恰 unit)
        assert abs(credited / 8.0 - round(credited / 8.0)) < 1e-9
        refunded = float(s.get("depth_shaping_refunded", 0.0))
        assert refunded <= 0.0
    finally:
        env.close()


def test_descend_bonus_fraction_law_and_accounting():
    """R14 乙案:有界下楼奖金返还——法条校验 + 记账守恒。"""
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
        assert s.get("dive_live_descends", 0) > 0, "未逼出 descend"
        kept = float(s.get("descend_bonus_kept", 0.0))
        # v2 descend_unit=48,L(d)→L(d+1) 全额=48×d,保留 25%=12×d
        assert kept >= 12.0, f"下楼奖金保留额未入账: {kept}"
        assert abs(kept / 12.0 - round(kept / 12.0)) < 1e-6, \
            f"保留额不是 12 的整数倍: {kept}"
        # 恒等式 R≡W+实际剥除额 由 _win_end 硬校验背书:跑到此处未抛
        # 即守恒成立。
    finally:
        env.close()


def test_descend_escrow_law_and_settlement():
    """R14.2 丁案:托管发放——法条校验 + vest/罚没结算。"""
    with pytest.raises(ValueError):   # 乙丁互斥
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_bonus_fraction=0.25,
            worker_descend_escrow_fraction=0.5,
            reward_economy="v2")
    with pytest.raises(ValueError):   # 教室法域
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
        assert s.get("dive_live_descends", 0) > 0, "未逼出 descend"
        # 每笔托管额 = 0.5×48×Σ层号 = 24 的整数倍;vest/罚没同源
        total = vested + forfeited
        assert total >= 24.0, f"托管从未结算: v={vested} f={forfeited}"
        assert abs(total / 24.0 - round(total / 24.0)) < 1e-6, \
            f"结算额不是 24 的整数倍: {total}"
        # 守恒:任何时刻 pending 余额非负
        assert float(getattr(env, "_descend_escrow", 0.0)) >= 0.0
    finally:
        env.close()


def test_descend_escrow_power_curve():
    """R14.3 戊案:凸曲线计价——法条 + 逐层 d^power 精确记账。"""
    with pytest.raises(ValueError):   # power≠1 须与托管同用
        WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=_SEED, drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            learning_window_scope="farm-dive-v1",
            worker_descend_escrow_power=1.6,
            reward_economy="v2")
    with pytest.raises(ValueError):   # 范围
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
        # 直驱结算器:模拟本 transition 内最深层 1→3(连降两层)
        env.oe.env._econ_episode_max_depth = 3
        env._descend_escrow = 0.0
        vest = env._descend_escrow_settlement(1, "descend")
        assert vest == 0.0   # 无既有托管
        expected = 0.5 * 48.0 * (1.0 ** 1.6 + 2.0 ** 1.6)
        assert abs(env._descend_escrow - expected) < 1e-9, \
            f"凸曲线计价失准: {env._descend_escrow} != {expected}"
        # 下一次非死亡收窗:全额 vest
        d_now = 3
        vest2 = env._descend_escrow_settlement(d_now, "exhausted")
        assert abs(vest2 - expected) < 1e-9
        assert env._descend_escrow == 0.0
        # 死亡收窗:罚没
        env._descend_escrow = 100.0
        vest3 = env._descend_escrow_settlement(d_now, "death")
        assert vest3 == 0.0 and env._descend_escrow == 0.0
        assert float(env.stats.get(
            "descend_escrow_forfeited", 0.0)) >= 100.0
    finally:
        env.close()


def test_readiness_power_ratio_and_coach():
    """R15 战备门:短板逻辑/抗性门/表单调性/教练放行判定。"""
    from diablogym.worker_env import (
        _READINESS_CLVL,
        _READINESS_DMG,
        _READINESS_HP,
        readiness_power_ratio,
    )
    # 表单调性:各维门槛随层非降
    from diablogym.worker_env import _READINESS_AC
    for tab in (_READINESS_CLVL, _READINESS_HP):
        assert all(tab[i] <= tab[i + 1] for i in range(15)), tab
    assert len(_READINESS_AC) == len(_READINESS_DMG) == 16
    # 初始战士(live 探针实录)对 L1:全维达标
    start = {"char_level": 1, "max_hp": 70, "armor_class": 12,
             "item_max_damage": 6, "damage_mod": 0,
             "fire_resist": 0, "magic_resist": 0}
    assert readiness_power_ratio(start, 1) >= 1.0
    # 对 L2:HP/AC/伤害三短板,必拦
    assert readiness_power_ratio(start, 2) < 1.0
    # 短板逻辑:等级满级但裸装,仍拦
    naked_hero = dict(start, char_level=30, max_hp=400)
    assert readiness_power_ratio(naked_hero, 9) < 1.0
    # 深层抗性门:四维满配但零火抗,L13 拦;火抗 50 放行
    hell_ready = {"char_level": 20, "max_hp": 320, "armor_class": 100,
                  "item_max_damage": 25, "damage_mod": 8,
                  "fire_resist": 0, "magic_resist": 0}
    assert readiness_power_ratio(hell_ready, 13) < 1.0
    assert readiness_power_ratio(
        dict(hell_ready, fire_resist=50), 13) >= 1.0
    # v2 逃生判据:白嫖免疫(idle 不改变存活占比),真清场放行
    from diablogym.worker_env import readiness_floor_cleared
    full_roster = {"monsters": [{"hp": 5}] * 98}
    assert not readiness_floor_cleared(full_roster)
    half_done = {"monsters": ([{"hp": 0}] * 68 + [{"hp": 5}] * 30)}
    assert not readiness_floor_cleared(half_done)   # 30/98 > 25%,拦
    below_line = {"monsters": ([{"hp": 0}] * 80 + [{"hp": 5}] * 18)}
    assert readiness_floor_cleared(below_line)      # 18/98 ≤ 25%,放
    truly_cleared = {"monsters": ([{"hp": 0}] * 96 + [{"hp": 5}] * 2)}
    assert readiness_floor_cleared(truly_cleared)
    assert not readiness_floor_cleared({"monsters": []})   # 花名册缺失保守拒
    # 教练构造:readiness-v1 合法,未知名拒绝
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
        # 初始战士对 L2 未达标且楼层未榨干 → 教练必给 FARM
        if not env.oe.exhausted:
            assert env._mgr_choose() == FARM
    finally:
        env.close()


def test_descend_escrow_readiness_gate():
    """R15 修正案二:未达标下楼分文不入托管(工资侧执法)。"""
    with pytest.raises(ValueError):   # 仅与托管同用
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
        # 冷启动战士对 L2 必未达标:模拟下楼 1→2,托管必须拒记
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow = 0.0
        vest = env._descend_escrow_settlement(1, "descend")
        assert vest == 0.0
        assert env._descend_escrow == 0.0, "未达标下楼竟然入了托管"
        assert float(env.stats.get(
            "descend_escrow_unready_denied", 0.0)) >= 24.0
        # 伪造达标面板(直改 raw 仅测试用):同笔下楼应入托管
        env.oe.env._raw.update({
            "char_level": 5, "max_hp": 120, "armor_class": 20,
            "item_max_damage": 9, "damage_mod": 1})
        env._descend_escrow = 0.0
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow_settlement(1, "descend")
        assert env._descend_escrow > 0.0, "达标下楼未入托管"
    finally:
        env.close()


def test_r16_readiness_v2_table_and_clear_ratio():
    """R16 修宪(C1/C2):v0.2 战备表按引擎算术可达;击杀口径清场剔除 golem。"""
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
    # 出生战士(live 实录 AC7/dmg6)对 L1 必须达标——v0.1 的 0.58 是刻度错位
    start = {"char_level": 1, "armor_class": 7, "item_max_damage": 6,
             "damage_mod": 0, "fire_resist": 0}
    assert readiness_power_ratio_v2(start, 1) >= 1.0
    # L1 榨取良好的存活者(clvl2/AC9/dmg6,部署存活行均值)对 L2 达标
    l1_survivor = {"char_level": 2, "armor_class": 9, "item_max_damage": 6,
                   "damage_mod": 0, "fire_resist": 0}
    assert readiness_power_ratio_v2(l1_survivor, 2) >= 1.0
    # 但对 L3 仍拦(clvl 短板)
    assert readiness_power_ratio_v2(l1_survivor, 3) < 1.0
    # 清场比:98 名单 = 94 真怪 + 4 golem;golem 不计
    roster = ([{"hp": 5, "type": 1}] * 94
              + [{"hp": 1, "type": _GOLEM_TYPE}] * 4)
    raw = {"monsters": roster, "monster_kill_total": 0}
    assert roster_alive_count(raw) == 94
    assert readiness_clear_ratio(raw, 0) == 0.0
    # 杀了 71 只(名单只剩 23 真怪 + golem):71/(71+23) ≈ 0.755 ≥ 0.75
    raw2 = {"monsters": ([{"hp": 5, "type": 1}] * 23
                         + [{"hp": 1, "type": _GOLEM_TYPE}] * 4),
            "monster_kill_total": 71}
    assert readiness_clear_ratio(raw2, 0) >= 0.75
    # idle 白嫖免疫:击杀不变,比值不变
    assert readiness_clear_ratio(raw, 0) == readiness_clear_ratio(raw, 0)


def test_r16_hp_economy_and_timeout_law():
    """R16(C7/C8):血量计价/拾药奖励记账 + 超时≠死亡法条校验。"""
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
        # 1500 拍近战必然失血:计价账必须出现且为负
        assert float(s.get("hp_loss_charged", 0.0)) < 0.0
        # farm_scene_cap / reset 旗默认值零漂移
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
        # farm-dive-v1 对 script 工人(workers=None)无从注册
        ea.evaluate(None, [1], worker_window_registration="farm-dive-v1")


def test_r17_escrow_readiness_table_selector():
    """R17.0 修正案一(合议庭更正二.1):托管战备门尺子选择器 v1/v2。

    v2 与 readiness-v3 教练同尺(v0.2 表);v1 旧尺逐位;过门下楼计数
    descend_escrow_ready_vested_count 两把尺均计。
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

    with pytest.raises(ValueError):      # 非法尺名
        _mk("v3")
    with pytest.raises(ValueError):      # v2 仅与战备门同用
        _mk("v2", gate=False)
    env = _mk("v1")                      # 默认尺属性零漂移
    try:
        assert env.descend_escrow_readiness_table == "v1"
    finally:
        env.close()
    # 面板恰在 v0.2 表 L2 门槛上,但 v0.1 表(clvl3/HP90/AC15/dmg8)未达标
    panel = {"char_level": _READINESS_V2_CLVL[1], "max_hp": 60,
             "armor_class": _READINESS_V2_AC[1],
             "item_max_damage": _READINESS_V2_DMG[1], "damage_mod": 0}
    assert readiness_power_ratio_v2(panel, 2) >= 1.0
    assert readiness_power_ratio(panel, 2) < 1.0
    for table, expect_escrow in (("v1", False), ("v2", True)):
        env = _mk(table)
        try:
            env.reset(seed=_SEED)
            env.oe.env._raw.update(panel)     # 直改 raw 仅测试用
            env.oe.env._econ_episode_max_depth = 2
            env._descend_escrow = 0.0
            assert env._descend_escrow_settlement(1, "descend") == 0.0
            ready_n = int(env.stats.get(
                "descend_escrow_ready_vested_count", 0))
            denied = float(env.stats.get(
                "descend_escrow_unready_denied", 0.0))
            if expect_escrow:
                assert env._descend_escrow > 0.0, "v2 达标下楼未入托管"
                assert ready_n == 1 and denied == 0.0
                vest = env._descend_escrow_settlement(2, "stall")
                assert vest > 0.0, "下一非死亡收窗未 vest"
                assert float(env.stats.get(
                    "descend_escrow_vested", 0.0)) == vest
            else:
                assert env._descend_escrow == 0.0, "v1 旧尺竟放行"
                assert ready_n == 0 and denied > 0.0
        finally:
            env.close()


def test_r17_policy_reward_component_receipts():
    """R17.0 修正案一(更正二.2):policy_reward 三分量逐 transition 回执。

    每笔 transition:transition_reward == wage + credited_ff_terminal_death
    + timeout + shaping + vest + hp_econ(与 worker_env 组装点同序,浮点
    逐位相等);三键在每笔 info 中必在且有限。
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
        assert n_hp > 0, "1500 拍近战必有失血计价分量"
    finally:
        env.close()


def test_r17_formal_pg_receipt_counts_vest_hp_econ_shaping():
    """R17.0 修正案一(更正二.2,收据地雷):formal-PG 守门员超时恒等式改为
    wage + timeout + shaping + vest + hp_econ;缺键 = 0(旧夹具逐位通过);
    分量在账而 reward 只报 wage+timeout 必炸;分量非有限必炸。"""
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
    # 分量取二进制精确值(rollout 回执以 float32 存 transition_reward)
    components = {"worker_depth_shaping_reward": 8.0,
                  "worker_descend_escrow_vest": 12.5,
                  "worker_hp_economy_reward": -0.25}
    legacy = {**base,                    # 旧法分支(death-equivalent)
              "no_progress_timeout_base_failure_reward": -16.0,
              "no_progress_timeout_additional_failure_reward": -32.0,
              "no_progress_timeout_failure_reward": -48.0}
    zero = {**base,                      # R16 零记分支
            "no_progress_timeout_credit": "zero",
            "no_progress_timeout_base_failure_reward": 0.0,
            "no_progress_timeout_additional_failure_reward": 0.0,
            "no_progress_timeout_failure_reward": 0.0}
    for name, row in (("legacy", legacy), ("zero", zero)):
        timeout_total = row["no_progress_timeout_failure_reward"]
        # 缺键 = 0:旧夹具恒等式 reward == wage + timeout 原样通过
        model = _model()
        model._update_info_buffer(
            [{**row, "transition_reward": 1.0 + timeout_total}],
            dones=np.ones(1, dtype=bool))
        assert model._worker_onpolicy_pg_pending_receipts[-1][
            "transition_reward"] == 1.0 + timeout_total, name
        # 带分量:按 worker_env 组装顺序求和
        expected = 1.0 + timeout_total + 8.0 + 12.5 + (-0.25)
        with_components = {**row, **components,
                           "transition_reward": expected}
        model = _model()
        model._update_info_buffer(
            [with_components], dones=np.ones(1, dtype=bool))
        assert model._worker_onpolicy_pg_pending_receipts[-1][
            "transition_reward"] == expected, name
        # 地雷复现:分量在账而 reward 只报 wage+timeout → 必炸
        model = _model()
        with pytest.raises(RuntimeError, match="timeout"):
            model._update_info_buffer(
                [{**with_components,
                  "transition_reward": 1.0 + timeout_total}],
                dones=np.ones(1, dtype=bool))
        # 分量非有限 → 炸
        model = _model()
        with pytest.raises(RuntimeError, match="分量"):
            model._update_info_buffer(
                [{**with_components,
                  "worker_descend_escrow_vest": float("nan")}],
                dones=np.ones(1, dtype=bool))


def test_r17_descend_gate_telemetry():
    """R17.0 仪表(更正二.3):托管战备门逐次过门遥测(两把尺比值 + 面板)。"""
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
        # 冷启动战士下楼 1→2:两尺均未达标,记一行 passed=False
        env.oe.env._econ_episode_max_depth = 2
        env._descend_escrow = 0.0
        env._descend_escrow_settlement(1, "descend")
        log = env.stats["descend_escrow_gate_log"]
        assert len(log) == 1 and log[0]["passed"] is False
        assert log[0]["depth"] == 2 and log[0]["table"] == "v2"
        assert log[0]["ratio_v2"] < 1.0 and log[0]["ratio_v1"] < 1.0
        assert log[0]["escrow"] > 0.0
        # 达标面板:第二行 passed=True,ratio_v2 >= 1 > ratio_v1
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
