"""R16 台账追记:B/C 两队交付 + 主刀裁决(max_steps 硬门放宽)。"""
import json
import pathlib
import time

LEDGER = (pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs"
          / "r10-staging" / "r13_ledger.jsonl")

EVENTS = [
    {"event": "R16_TEAM_DELIVERY", "team": "C",
     "files": ["train/leashed_ppo.py", "train/train_ppo.py"],
     "summary": "a13 logit 先验(a11 先例克隆);8 个 None-off 契约键;白名单 +max_steps/"
                "drink_sovereignty;CLI --reward-economy v4 / --manager-heuristic readiness-v3 / "
                "--explore-global-fallback / --progress-far-tiles / --farm-scene-cap / "
                "--reset-layer-clock-on-window / --worker-potion-action13-logit-bonus / "
                "--worker-hp-loss-price / --worker-potion-pickup-bonus / "
                "--worker-no-progress-timeout-credit;141 passed"},
    {"event": "R16_TEAM_DELIVERY", "team": "B",
     "files": ["train/eval_assembled.py", "train/eval_contract.py",
               "train/runs/r10-staging/probe_r15_deployment.py"],
     "summary": "--worker-decoding sample(逐局 set_random_seed,单线程,两遍逐位复现 "
                "f397eae58a340f54/de57d8d47c15959c);meta.protocol.worker_decoding 仅 sample 时写;"
                "默认旗迷你 diff r16-minidiff-b1 vs r10-cand-a 与 r16-minidiff-m29-snap vs "
                "r10-anchor-a-rep1 rows 逐位 PASS;探针 v2 死亡前快照 + 存活分层(C4 坐实:"
                "died 行 AC 7-8/gold 100 vs 尸检 AC 4/gold 50/hit 1);62 passed"},
    {"event": "R16_TEAM_DELIVERY", "team": "主刀",
     "files": ["python/diablogym/worker_env.py", "python/diablogym/options_env.py"],
     "summary": "战备表 v0.2 + readiness-v3 教练 + 清场比(golem 剔除)+ hp 记账 + 超时零记 + "
                "farm_scene_cap/reset_layer_clock_on_window;test_r13_live_dive 12/12;"
                "旧法执法档(worker_env/terminal_bootstrap/options_env/dual_worker_observation/"
                "scene_semantics)exit 0"},
    {"event": "R16_RULING", "by": "主刀",
     "ruling": "train_ppo.py PREREG-v23/v25 硬门 max_steps==3000 放宽为 3000 或 >=6000"
               "(审计 C6/C11,白名单已放行);py_compile OK",
     "lines": ["train/train_ppo.py:6412", "train/train_ppo.py:6428"]},
]

with open(LEDGER, "a") as fh:
    for ev in EVENTS:
        ev = {"t": time.strftime("%H:%M:%S"), **ev}
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print("[ledger]", ev["event"], ev.get("team", ev.get("by")))
