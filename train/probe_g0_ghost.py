"""G0-幽灵 v12 病灶回归探针(PREREG-v32 W-G0;E1 施工后运行)。

断言清单:
  A. 主权开:m[12] 传递基础环境掩码(belt>0 合法);主动按 12 正常饮
     (belt 递减、拍账照常),饮至 belt=0 后 m[12] 转非法(防空饮幽灵)。
  B. 主权关(对照腿旋钮):m[12] 恒 False(旧协议逐字复现)。
  C. 反射谓词逐字:_reflex 语义(hp<0.5∧belt>0)未被 E1 触碰(合成 raw
     单元断言;真实分布上的排水行为由 G0-恒等的位级工资哈希兜底——
     排水拍的动作/工资全在哈希内)。
  D. 终止纪律:窗口终结后再 step 必拒(死后不可饮之包装器层保证)。
退出码 0 = 全过。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.options_env import OptionsEnv, _reflex  # noqa: E402
from diablogym.worker_env import WorkerWindowEnv       # noqa: E402

H_NPZ = str(ROOT / "train" / "models" / "v22-h-manager" / "policy.npz")


def check(cond, msg):
    if not cond:
        print(f"G0-幽灵 失败:{msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main():
    # C. 反射谓词单元断言(E1 前后语义逐字)
    check(_reflex({"hp": 49, "max_hp": 100, "belt_heals": 1}) is True,
          "反射谓词:半血以下有药 → True")
    check(_reflex({"hp": 50, "max_hp": 100, "belt_heals": 3}) is False,
          "反射谓词:恰半血 → False(阈值语义未动)")
    check(_reflex({"hp": 10, "max_hp": 100, "belt_heals": 0}) is False,
          "反射谓词:无药 → False")

    # B. 主权关:旧协议逐字复现
    env_off = OptionsEnv(max_steps=3000, drink_sovereignty=False)
    env_off.reset(seed=424242)
    m_off = env_off._worker_masks()
    check(bool(m_off[12]) is False, "主权关:m[12] 恒 False")
    check(bool(m_off[11]) is False, "主权关:m[11] 恒 False(职权不变)")
    env_off.close()

    # A/D. 主权开:活体走一窗
    env = WorkerWindowEnv(H_NPZ, max_steps=3000, rng_seed=0,
                          seed_scope="replay",
                          drink_sovereignty=True)
    obs, _ = env.reset(seed=424242)
    check(obs is not None, "主权开:reset 得到首窗观测")
    raw = env.oe.env._raw
    m = env.oe._worker_masks()
    check(bool(m[11]) is False, "主权开:m[11] 仍恒掩(DIVE 归经理)")
    belt0 = raw.get("belt_heals", 0)
    check(bool(m[12]) == (belt0 > 0),
          f"主权开:m[12] 系 belt 前置(belt={belt0} → {bool(m[12])})")
    drinks = 0
    reason_ended = False
    # 先谋一瓶:600 拍预算内捡药(13 合法即按,否则清怪/探索),拿到再饮测
    for _ in range(600):
        raw = env.oe.env._raw
        if raw.get("belt_heals", 0) > 0:
            break
        m = env.oe._worker_masks()
        a = 13 if m[13] else (9 if m[9] else 10)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            nxt = env.next_window()
            if nxt is None:
                reason_ended = True
                break
    for _ in range(40):                       # 有药就喝,喝到见底
        if reason_ended:
            break
        m = env.oe._worker_masks()
        raw = env.oe.env._raw
        belt_before = raw.get("belt_heals", 0)
        if not m[12]:
            check(belt_before == 0,
                  f"m[12] 非法 ⟺ belt=0(belt={belt_before})")
            break
        beats_b = env.oe._win["beats"]
        ov_b = env.oe._win["overrides"]
        steps_b = env.oe.env._steps
        obs2, w, term, trunc, info = env.step(12)
        raw2 = env.oe.env._raw
        check(raw2.get("belt_heals", 0) < belt_before,
              f"主动饮:belt {belt_before}→{raw2.get('belt_heals', 0)} 严格递减")
        check(np.isfinite(w), "主动饮:工资拍有限")
        check(env.oe.env._steps >= steps_b + 1, "主动饮:微步推进(时钟语义)")
        if not (term or trunc):
            check(env.oe._win["beats"] >= beats_b + 1, "主动饮:拍账递增")
            check(env.oe._win["overrides"] == ov_b, "主动饮:无保险丝改写拍")
        drinks += 1
        if term or trunc:
            reason_ended = True
            break
    check(drinks > 0, "实饮达成(fail-closed:空转即 FAIL,v30 空探针纪律)")
    print(f"  主动饮 {drinks} 拍(belt 见底或窗口自然终结)")
    if not reason_ended:
        # D. 终止纪律:强行走完窗口后 step 必拒
        for _ in range(5000):
            m = env.oe._worker_masks()
            a = 9 if m[9] else int(np.flatnonzero(m)[0])
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                reason_ended = True
                break
    check(reason_ended, "窗口可自然终结")
    nxt = env.next_window()
    while nxt is not None:                    # 无条件驱动至局尽,再断拒
        m = env.oe._worker_masks()
        a = 9 if m[9] else int(np.flatnonzero(m)[0])
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            nxt = env.next_window()
    try:
        env.step(9)
        check(False, "局尽后 step 应当拒绝")
    except Exception:
        check(True, "局尽/死后 step 拒绝(不可饮之包装器层保证)")
    env.close()

    # E. 桥证③实弹:脚本模式主权开/关位级等价 + dispatch 永不返 12
    from diablogym import options_env as oe_mod
    orig = oe_mod.dispatch
    seen12 = {"n": 0}

    def spy(mode, raw, gear):
        a = orig(mode, raw, gear)
        if a == 12:
            seen12["n"] += 1
        return a

    traces = {}
    for sv in (True, False):
        oe_mod.dispatch = spy
        try:
            e = oe_mod.OptionsEnv(max_steps=3000, drink_sovereignty=sv)
            e.reset(seed=424242)
            tr = []
            done = trunc = False
            while not (done or trunc):
                _, r, done, trunc, info = e.step(0)   # FARM 保底位
                ex = info["option_extra"]
                tr.append((ex["tau"], round(ex["R"], 6), round(ex["W"], 6),
                           ex["beats"], ex["overrides"], ex["reason"]))
            traces[sv] = tr
            e.close()
        finally:
            oe_mod.dispatch = orig
    check(traces[True] == traces[False],
          "断言E:脚本模式主权开/关整局逐窗位级相等(dispatch 不消费掩码)")
    check(seen12["n"] == 0,
          "断言E:dispatch 全程未返 12(反射先饮,12 支不可达之动态证明)")

    print("G0-幽灵 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
