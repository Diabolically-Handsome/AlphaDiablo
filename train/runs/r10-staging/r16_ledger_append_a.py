"""R16 台账追记:A 队交付(env.py)+ 主刀裁决(hunt 为 R16 主力,补接线)。"""
import json
import pathlib
import time

LEDGER = (pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs"
          / "r10-staging" / "r13_ledger.jsonl")

EVENTS = [
    {"event": "R16_TEAM_DELIVERY", "team": "A",
     "files": ["python/diablogym/env.py"], "env_md5": "5ac9ca74cc07583b8bfc24790ae8f400",
     "summary": "经济 v4(dataclasses.replace(v2, idle_counts_micro_beats, idle_reset_on_kill);"
                "先按清零前钟结算再清零);progress_far_tiles(锚点集语义);explore_global_fallback"
                "(规格内,8 种子 375→375 杀,2/8 触发);附加 explore_global_hunt(窗内无可见怪→"
                "全图 BFS 朝最近存活怪,375→670 杀,clvl 2→3);test_env_v4_semantics+test_options_env "
                "72 passed;未做重烤(留给发射闸门)",
     "probe_8seeds_kills": {"off": 375, "fallback": 375, "hunt": 670},
     "risks": ["hunt/fallback 二次调 bridge.local_map(radius=112) 为工人观测外越权信息(a11 同款),"
               "预注册已明写(5′ 条)",
               "v4 反躺平阈值 300 改微拍单位未校准(G0 定标项)",
               "/tmp 会被外部周期清空——锚文件禁放 /tmp"]},
    {"event": "R16_RULING", "by": "主刀",
     "ruling": "explore_global_hunt 列为 R16 视野条主力(fallback 保留但非主力);"
               "追加 C 队(train_ppo 接线 --explore-global-hunt)与 B 队(考卷 R16 环境旗 + "
               "meta.protocol.r16_environment 身份 + v4 choices);重烤推迟到两队收工后的最终字节"},
]

with open(LEDGER, "a") as fh:
    for ev in EVENTS:
        ev = {"t": time.strftime("%H:%M:%S"), **ev}
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print("[ledger]", ev["event"], ev.get("team", ev.get("by")))
