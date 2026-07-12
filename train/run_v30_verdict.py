"""v30 判决续跑器(重启协议执行件;PREREG-v30 残余/重启条款 + 06:32 尸检)。

背景:主驱动 06:32:24 在满 32 仪表段死于 metrics() 的 JSON 字典键 bug
(worker_action_hist 键系字符串,sum 炸 TypeError)。全部训练干净完好
(4 腿 nt 链逐位;bc 腿 2 绊线止训系条款正常动作),king 满 32 档案已合法
落盘(v30-king-full32.json,32 行全)。本件复用全部干净资产、只执行剩余
判决条款:bc 满 32 → r30_6 → 资格/递补 → 科学主判 → 地板 → 双锚发射 →
判词。逻辑与主驱动逐字同源(import 复用修复后的库函数)。
用法:.venv/bin/python train/run_v30_verdict.py
"""
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_contract import (PROTOCOL_VERSION, OperationalFailure,  # noqa: E402
                           OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           sha256_file, verify_eval_identity)
from run_v30_relay import (D2DEATH_LAUNCH_MAX, EVAL, FLOOR, LAUNCH_ANCHOR,  # noqa: E402
                           LAUNCH_SHA, M29_NPZ, M29_SHA, PD112_LINE, PRE140_LINE,
                           ROOT, RUNS, SCI_ANCHOR, SCI_SHA, PY, WINS112_LINE,
                           attention, by_seed, log, metrics, qual_of,
                           read_comparable_launch, read_comparable_science,
                           require, run, sha16)
from run_v30_relay import OperationalFailure as RelayOperationalFailure  # noqa: E402

ARM_MODELS = {"king": str(RUNS / "v30-king-leg2" / "model_final.zip"),
              "bc": str(RUNS / "v30-bc-leg2" / "model_final.zip")}   # 终腿 = 最后干净收官腿(条款)
ARM_MODEL_SHA256 = {
    "king": "179b5ede444601279b79790be477246ecb6546eb1d4e6de7d923702f8634e24f",
    "bc": "37acee85b3450168deed267bd8e0797522e4af19bb7b8c83858ccde97aad09f7",
}
KING_ARCHIVE_SHA256 = "6ce157a36703aca65554489c5742a96c75a4585d7212bf7247492403db4bfac6"
CALIBRATED_PROTOCOL_VERSION = 2


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v30 verdict 的资格/R4/科学与发射静态阈值仅在 pre-v3 环境语义标定；"
            "必须先重跑 protocol-v3 基线并人工更新预注册，禁止混用旧阈值"
        )


def read_comparable_king_archive() -> dict:
    """续判复用的 king 结果也是活动选模输入，禁止 legacy 特赦。"""
    path = EVAL / "v30-king-full32.json"
    try:
        snapshot = freeze_eval_identity(ROOT, ARM_MODELS["king"], M29_NPZ)
        expected = expected_eval_identity(
            snapshot, tag="v30-king-full32", seeds=range(7000, 7032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            "v30-king-full32 不满足当前 schema-v2 可比性契约；"
            "环境语义变更后须用固定 king leg2 worker + M29 manager 重跑该臂"
        ) from exc


def exam(model_path, tag, seeds):
    """续判专用评测：发车前/回收后双重锁定完整输入身份。"""
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    require(seed_values and lo >= 0, f"非法 seed 范围:{seeds}")
    snapshot = freeze_eval_identity(ROOT, model_path, M29_NPZ)
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    cmd = [PY, "train/eval_assembled.py",
           "--worker", snapshot["worker"]["path"],
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        document = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        document["agg"]["_sha"] = sha16(out)
        return document
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None


def exam_retry(model_path, tag, seeds):
    document = exam(model_path, tag, seeds)
    if document is None:
        log({"event": "exam_crash", "tag": tag,
             "note": "评测失败,按崩溃条款重考一次"})
        document = exam(model_path, tag, seeds)
    return document


def main():
    try:
        with exclusive_lock(RUNS / "v30" / ".verdict.lock", "v30 续判驱动"):
            _main()
    except (OperationalFailure, RelayOperationalFailure,
            OutputReservationError) as exc:
        log({"event": "OPERATIONAL_FAILURE", "why": str(exc)})
        attention("续判运维失败:\n" + str(exc))
        raise SystemExit(2) from exc
    except Exception as exc:
        log({"event": "VERDICT_EXCEPTION", "why": repr(exc),
             "traceback": traceback.format_exc()})
        attention("续判异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    require_calibrated_protocol()
    king_archive = EVAL / "v30-king-full32.json"
    require(M29_NPZ.is_file() and sha256_file(M29_NPZ) == M29_SHA,
            "M29 经理 npz 漂移/缺失")
    for arm, model in ARM_MODELS.items():
        path = pathlib.Path(model)
        require(path.is_file() and sha256_file(path) == ARM_MODEL_SHA256[arm],
                f"{arm} 续判模型缺失或 sha 漂移:{model}")
    read_comparable_science()
    read_comparable_launch()
    read_comparable_king_archive()
    log({"event": "restart", "why": "主驱动 06:32:24 DRIVER_EXCEPTION(metrics 字典键 bug,"
         "已修 run_v30_relay.metrics);训练资产全干净,king 满 32 档案合法在盘",
         "reused": {"v30-king-full32.json": sha16(EVAL / "v30-king-full32.json"),
                    "king_leg2": sha16(ARM_MODELS["king"]), "bc_leg2": sha16(ARM_MODELS["bc"])},
         "note": "重启协议:仅续判决段,不动训练与既有档案"})
    science_doc = read_comparable_science()
    ref_sci = by_seed(science_doc["rows"])

    full, mets = {}, {}
    d = read_comparable_king_archive()
    d["agg"]["_sha"] = sha16(king_archive)
    full["king"] = d
    mets["king"] = metrics(d)
    log({"event": "full32", "arm": "king", "reused_archive": True, **mets["king"]})
    d = exam_retry(pathlib.Path(ARM_MODELS["bc"]), "v30-bc-full32", "7000-7031")
    if d is None:
        why = "bc 满 32 连败——人工验尸"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    full["bc"] = d
    mets["bc"] = metrics(d)
    log({"event": "full32", "arm": "bc", **mets["bc"]})

    fk, fb = by_seed(full["king"]["rows"]), by_seed(full["bc"]["rows"])
    r30_6 = sum(fk[s]["ret"] - fb[s]["ret"] for s in fk) / 32
    r30_6w = sum(fk[s]["ret"] > fb[s]["ret"] for s in fk)
    log({"event": "r30_6", "king_minus_bc_mean": round(r30_6, 2), "king_wins": r30_6w,
         "note": "|均差|<2 判方向未判定(预注册判读规则)"})

    quals = {a: qual_of(full[a]) for a in full}
    log({"event": "quals", **{a: quals[a] for a in full}})
    pool = [a for a in full if quals[a]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双臂资格失败——无胜者,接力命题未答(功效外)",
             "arms": {a: mets[a] for a in full}})
        attention("判决:双臂资格失败")
        return
    prelim = max(full, key=lambda a: full[a]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim]})
    ms = {a: full[a]["agg"]["ret_mean"] for a in pool}
    band = [a for a in pool if max(ms.values()) - ms[a] <= 0.05]
    if len(band) > 1:
        dmin = min(full[a]["agg"]["died"] for a in band)
        band = [a for a in band if full[a]["agg"]["died"] == dmin]
        winner = "king" if "king" in band else band[0]
    else:
        winner = band[0]
    W, wm = full[winner], mets[winner]
    wrows = by_seed(W["rows"])
    log({"event": "winner", "arm": winner, "mean": wm["mean"], "died": wm["died"],
         "substituted": winner != prelim})

    science_doc = read_comparable_science()
    launch_doc = read_comparable_launch()
    ref_sci = by_seed(science_doc["rows"])
    ref_launch = by_seed(launch_doc["rows"])
    science_base = science_doc["agg"]["ret_mean"]
    launch_base = launch_doc["agg"]["ret_mean"]
    floor = round(0.92 * science_base, 1)
    d2death_launch_max = sum(
        r["died"] and r["depth"] >= 2 for r in science_doc["rows"])
    d112 = [wrows[s]["ret"] - ref_launch[s]["ret"] for s in sorted(ref_launch)]
    pd112, wins112 = sum(d112) / 32, sum(x > 0 for x in d112)
    d140 = [wrows[s]["ret"] - ref_sci[s]["ret"] for s in sorted(ref_sci)]
    pd140, wins140 = sum(d140) / 32, sum(x > 0 for x in d140)
    log({"event": "paired", "vs112_mean": round(pd112, 2), "vs112_wins": wins112,
         "vs140_mean": round(pd140, 2), "vs140_wins": wins140})
    log({"event": "draw_ledger", "note": "同池 18/32 线第 4 次挑战者开奖(11→16→17→本案);"
         "台账只记不裁;P(赢≥18|p=.5)≈43% 注记;发射线新证据合取已加"})

    d2, d2d = wm["depth2_seeds"], wm["d2_deaths"]
    if d2 >= 12 and d2d <= 1 and pd140 >= 2.0:
        sci = f"接力有效(曝露≥12 ∧ 深层死伤≤1 ∧ 对{science_base}≥+2)"
    elif d2d >= 3 and pd140 < 0:
        sci = "接力无效(短板非本处方可修——课③④线索)"
    elif d2d in (2, 3) and abs(pd140) < 2.0:
        sci = "信号稀释/学不动档(命题未判定)"
    elif d2 < 12:
        sci = (f"带外(曝露塌缩:depth2={d2}<12;d2死={d2d},"
               f"对{science_base}={pd140:+.2f})入册不叙事")
    else:
        sci = f"带外(d2死={d2d},对{science_base}={pd140:+.2f})入册不叙事"
    log({"event": "science_verdict", "verdict": sci, "depth2": d2, "d2_deaths": d2d,
         "pd140": round(pd140, 2), "a13": wm["a13_share"], "dive": wm["dive_per_ep"],
         "note": "科学主判与发射/王座/Mark-I 互不改写;判词携带双臂仪表(quals/full32 事件)"})

    if wm["mean"] < floor:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": f"胜者 {wm['mean']} < 地板 {floor}(=0.92×{science_base})——重训未复现"
                        f"起点水平,发射流程终止;科学主判:{sci}"})
        attention(f"判决:未及地板;科学主判:{sci}")
        return

    launch = (pd140 >= PRE140_LINE and d2d <= d2death_launch_max
              and pd112 >= PD112_LINE and wins112 >= WINS112_LINE)
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(ARM_MODELS[winner]).with_suffix('')} --manager-npz {M29_NPZ} "
                      f"--seeds 9000-9031 --tag v30-golden --board")
        dual_note = ("【双归因未裁,先裁后烧:回写 dual_attr_ruling 后方可手启】"
                     if quals[winner]["dual_attr"] else "")
        sci_note = ("" if sci.startswith("接力有效")
                    else f"【科学主判非'有效'({sci}),判词禁用接力成功措辞】")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wm["mean"],
             "died": wm["died"], "vs112": [round(pd112, 2), wins112],
             "vs140": [round(pd140, 2), wins140], "model": ARM_MODELS[winner],
             "model_sha": sha16(pathlib.Path(ARM_MODELS[winner])), "full32_sha": wm["sha"],
             "golden_cmd": golden_cmd,
             "p_line": "按序:死>6回退;≥101.2且死≤4登基;(97.2,101.2)且死≤4点估;"
                       ">97.2且死5-6持平安全性;[93.9,97.2]持平;<93.9回退",
             "note": dual_note + sci_note
                     + "金池史上第4次实开;单臂一次;开牌后回写 golden_result"})
        attention(dual_note + f"金牌待手启:{winner};科学主判:{sci}")
        return

    quad = (f"(对{launch_base} {pd112:+.2f}/赢{wins112},"
            f"对{science_base} {pd140:+.2f},d2死{d2d})")
    wins_note = f"(宽度移动注记:赢 {wins112}/32 ≥14,不改判档)" if wins112 >= 14 else ""
    if pd140 < PRE140_LINE:
        verdict = (f"新证据不足档:对{science_base} {pd140:+.2f} < +2"
                   f"——存量垫烧被拦,不烧牌{quad}")
    elif d2d > d2death_launch_max:
        verdict = (f"深层死伤未改善拦截档:d2死 {d2d} > 基线 "
                   f"{d2death_launch_max}——不烧牌{quad}")
    elif pd112 >= PD112_LINE and wins112 < WINS112_LINE:
        verdict = f"均值增益而宽度未达——点估增益,不烧牌{quad}{wins_note}"
    elif pd112 >= 2.0:
        verdict = f"探针级改进,不烧牌{quad}{wins_note}"
    else:
        verdict = f"现任组装体连任,接力无发射级增益(功效限定){quad}"
    log({"event": "VERDICT_PATH", "golden_authorized": False, "verdict": verdict,
         "science_verdict": sci, "winner": winner, "winner_mean": wm["mean"]})
    attention(f"判决(不发射):{verdict};科学主判:{sci}")


if __name__ == "__main__":
    main()
