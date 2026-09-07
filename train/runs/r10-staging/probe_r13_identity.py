"""R13 门 D 探针(D1/D2,零种子消耗,考场前置 fail-closed)。

用法: probe_r13_identity.py <old_worker.zip> <new_worker.zip> <out.json>

采集:旧工人驾驶 farm-dive-v1 教室(SMART 教练)N 微拍,逐拍记录
(观测, 掩码, 窗口模式, 正典脚本本拍动作)。
判据:
  D1 = FARM 态上 新旧工人 masked-argmax 一致率,要求 < 0.99(学习发生);
  D2 = DIVE 态上 新工人 masked-argmax 对正典脚本动作 一致率,要求 < 0.99
       (改革触及 DIVE 决策面——R12 结构定律的直接反证探针)。
任一 >= 0.99 即结构性失败,退出码 1,禁止烧考试场。
探针种子取自训练采样域(非保留段),不触任何登记评测池。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

from diablogym.options_env import DIVE, FARM, dispatch  # noqa: E402
from diablogym.worker_env import WorkerWindowEnv  # noqa: E402

PROBE_SEEDS = (999983, 999979, 999961, 999959)   # 非保留段,拒采样器之外
TARGET_BEATS = 4000
THRESHOLD = 0.99


def load_zip_policy(path):
    from leashed_ppo import LeashedMaskablePPO
    model = LeashedMaskablePPO.load(
        path, env=None, device="cpu",
        teacher_path=None, teacher_sha256=None)

    def choose(obs, mask):
        a, _ = model.predict(
            obs, action_masks=mask, deterministic=True)
        return int(a)
    return choose


def main():
    old_zip, new_zip, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    old = load_zip_policy(old_zip)
    new = load_zip_policy(new_zip)

    records = []   # (obs, mask, in_dive, old_action, script_action)
    for seed in PROBE_SEEDS:
        env = WorkerWindowEnv(
            manager_npz=None, manager_heuristic="level-margin-1",
            max_steps=3000, rng_seed=seed,
            drink_sovereignty=False,
            policy_observation_view="dual-v4-asymmetric-v3",
            fast_forward_reward_credit="terminal-death-only",
            learning_window_scope="farm-dive-v1",
            reward_economy="v2")
        try:
            obs, _ = env.reset(seed=seed)
            for _ in range(TARGET_BEATS // len(PROBE_SEEDS)):
                mask = env.action_masks()
                win = env.oe._win
                in_dive = win is not None and win["opt"] == DIVE
                a_old = old(obs, mask)
                script_mask, nearest = (
                    env.oe.env.controller_action_context())
                script_mask = np.asarray(script_mask, dtype=bool)
                mode = ("farm", "dive", "resupply")[
                    int(win["opt"]) if win is not None else FARM]
                s = dispatch(
                    mode, env.oe.env._raw, bool(script_mask[14]),
                    action_mask=script_mask,
                    nearest_engageable_distance=nearest)
                records.append((obs.copy(), np.asarray(mask, bool),
                                in_dive, a_old, int(s)))
                obs, r, term, trunc, _ = env.step(a_old)
                if term or trunc:
                    obs, _ = env.reset()
        finally:
            env.close()

    farm_states = [r for r in records if not r[2]]
    dive_states = [r for r in records if r[2]]
    d1_hits = sum(1 for (o, m, _, a_old, _s) in farm_states
                  if new(o, m) == a_old)
    d2_hits = sum(1 for (o, m, _, _a, s) in dive_states
                  if new(o, m) == s)
    d1 = d1_hits / max(1, len(farm_states))
    d2 = d2_hits / max(1, len(dive_states))
    result = {
        "probe": "r13-gateD-v1",
        "beats": len(records),
        "farm_states": len(farm_states), "dive_states": len(dive_states),
        "D1_farm_argmax_agreement_old_vs_new": round(d1, 6),
        "D2_dive_new_vs_script_agreement": round(d2, 6),
        "threshold": THRESHOLD,
        "D1_pass": bool(d1 < THRESHOLD),
        "D2_pass": bool(d2 < THRESHOLD),
    }
    pathlib.Path(out_path).write_text(
        json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    if not (result["D1_pass"] and result["D2_pass"]):
        print("GATE-D-FAIL: 行为同一性未破,禁止烧考试场", file=sys.stderr)
        raise SystemExit(1)
    print("GATE-D-PASS")


if __name__ == "__main__":
    main()
