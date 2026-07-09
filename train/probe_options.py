"""v22 发车前探针:G1b 机械保真 / G2 词表充分性 / G3 评估段参考行 / G4 预算校准。

闸门(预注册):
  G1b wrapper-rush(恒 DIVE)对神谕 rush 臂逐种子 |Δ| ≤ max(5%, 1.0)
  G2  教师(榨干旗或 clvl≥dlvl+2 → DIVE)均值 ≥36 且 配对胜 wrapper-retire ≥24/32
  G3  三参考臂 9000-9031 成绩写入 leaderboard-hier.md(主指标及格线挂评估段)
  G4  τ̄ 与吞吐实测 → 双币种停车规则定数
"""
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import OptionsEnv
from diablogym.options_env import DIVE, FARM

S = pathlib.Path("/private/tmp/claude-501/-Users-lawrencegrey-Desktop-Shut-Up-Refusal-Letters/9aaec51c-def0-4f5a-9368-4212826ec2d3/scratchpad")
ORACLE = json.load(open(S / "oracle_mountain.json"))

PROBE_SEEDS = list(range(7000, 7032))
EVAL_SEEDS = list(range(9000, 9032))


def run_policy(env, choose, seed):
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    R, taus, info = 0.0, [], {}
    while not (done or trunc):
        m = env.action_masks()
        opt = choose(env, m)
        if not m[opt]:
            opt = FARM
        obs, r, done, trunc, info = env.step(opt)
        R += r
        taus.append(info["option_extra"]["tau"])
    ex = info.get("episode_extra", {})
    return {"ret": round(R, 2), "depth": ex.get("depth", 1),
            "died": bool(ex.get("died", False)), "kills": ex.get("kills", 0),
            "decisions": len(taus), "tau_mean": round(sum(taus) / max(1, len(taus)), 1),
            "mode_seq": info["option_extra"]["mode_seq"]}


POLICIES = {
    "wrapper-retire": lambda env, m: FARM,
    "wrapper-rush": lambda env, m: DIVE,
    "teacher": lambda env, m: DIVE if (env.exhausted or
                                       env.env._raw["char_level"] >= env.env._raw["dungeon_level"] + 2)
                              else FARM,
}


def main():
    env = OptionsEnv(max_steps=3000)
    out = {"probe": {}, "eval_refs": {}}
    t_wall0, micro0 = time.time(), 0

    # ---- 探针段 7000-7031 ----
    for name, pol in POLICIES.items():
        eps = []
        for seed in PROBE_SEEDS:
            eps.append({"seed": seed, **run_policy(env, pol, seed)})
            micro0 += eps[-1]["decisions"] * eps[-1]["tau_mean"]
        out["probe"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[probe] {name}: mean {sum(rs)/32:.1f} med {sorted(rs)[16]:.1f} "
              f"died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)} "
              f"decisions_med {statistics.median(e['decisions'] for e in eps)}", flush=True)

    # G1b:wrapper-rush vs 神谕 rush(3000 快照)
    oracle_rush = {e["seed"]: e["snaps"]["3000"]["ret"] for e in ORACLE["arms"]["rush"] if "error" not in e}
    fails = []
    for e in out["probe"]["wrapper-rush"]:
        ref = oracle_rush[e["seed"]]
        if abs(e["ret"] - ref) > max(0.05 * abs(ref), 1.0):
            fails.append((e["seed"], e["ret"], ref))
    g1b = len(fails) == 0
    print(f"G1b {'PASS' if g1b else 'FAIL'}: wrapper-rush 对神谕 rush 逐种子偏差超限 {len(fails)}/32 "
          + (f"首例 {fails[0]}" if fails else ""), flush=True)

    # G2:教师充分性
    t_rets = [e["ret"] for e in out["probe"]["teacher"]]
    r_rets = [e["ret"] for e in out["probe"]["wrapper-retire"]]
    t_mean = sum(t_rets) / 32
    wins = sum(1 for a, b in zip(t_rets, r_rets) if a > b)
    g2 = t_mean >= 36 and wins >= 24
    grey = 34 <= t_mean < 36
    print(f"G2 {'PASS' if g2 else ('GREY' if grey else 'FAIL')}: 教师均值 {t_mean:.1f}(线 36,灰带 [34,36)) "
          f"配对胜 retire {wins}/32(线 24)", flush=True)

    # ---- G3:评估段参考行 9000-9031 ----
    for name, pol in POLICIES.items():
        eps = [{"seed": s, **run_policy(env, pol, s)} for s in EVAL_SEEDS]
        out["eval_refs"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[eval] {name}: mean {sum(rs)/32:.1f} died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)}", flush=True)

    # G4:预算校准
    wall = time.time() - t_wall0
    all_probe = [e for eps in out["probe"].values() for e in eps]
    taus = [e["tau_mean"] for e in all_probe]
    tau_bar = sum(taus) / len(taus)
    total_micro = sum(e["decisions"] * e["tau_mean"] for e in all_probe) \
        + sum(e["decisions"] * e["tau_mean"] for eps in out["eval_refs"].values() for e in eps)
    micro_per_s = total_micro / wall
    print(f"G4: τ̄≈{tau_bar:.0f} 微拍/选项,吞吐≈{micro_per_s:.0f} micro/s(单 env),"
          f"40k 管理器步 ≈ {40_000 * tau_bar / 1e6:.1f}M 微步;"
          f"4-env 预计墙钟 ≈ {40_000 * tau_bar / (micro_per_s * 2.5) / 60:.0f} 分钟", flush=True)

    (S / "probe_v22.json").write_text(json.dumps(out, default=float))
    # 评估段参考行入新表
    lb = ROOT / "train" / "leaderboard-hier.md"
    if not lb.exists():
        lb.write_text(
            "# Hierarchy board — manager over frozen options, 32 fixed seeds\n\n"
            "Protocol: 3000 micro-steps, argmax + option masks, seeds 9000-9031,\n"
            "idle machine, engine pinned, world = v20 rules (ladder + death price\n"
            "+ auto stat-spend). Return = UNDISCOUNTED episode reward (the oracle\n"
            "ledger). Reference rows are scripted policies via the same wrapper.\n\n"
            "| run | ret mean | ret med | died | depth med | notes |\n|---|---|---|---|---|---|\n")
    lines = lb.read_text().splitlines(keepends=True)
    last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
    for name in POLICIES:
        eps = out["eval_refs"][name]
        rs = sorted(e["ret"] for e in eps)
        row = (f"| {name} (scripted ref) | {sum(rs)/32:.1f} | {rs[16]:.1f} | "
               f"{sum(e['died'] for e in eps)}/32 | "
               f"{statistics.median(e['depth'] for e in eps)} | G3 reference |\n")
        lines.insert(last_row + 1, row)
        last_row += 1
    lb.write_text("".join(lines))
    print("G3: 参考行已写入 leaderboard-hier.md", flush=True)
    print(f"GATES: G1b={'PASS' if g1b else 'FAIL'} G2={'PASS' if g2 else ('GREY' if grey else 'FAIL')}", flush=True)


if __name__ == "__main__":
    main()
