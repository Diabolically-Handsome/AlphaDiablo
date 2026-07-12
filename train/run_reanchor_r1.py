"""重锚R1:protocol-v3 开元 · BC 基底重生成驱动器(docs/PREREG-重锚R1.md 终稿)。

三阶段 worker→manager→flat,BC 脚本逐字执行零改动。判词唯一来源 =
bc_report.json 判词键 + R-S 回执闸(退出码不作判词依据);科学 FAIL 是
合法结局,RUNNING 残留/缺键/漂移才是 OPERATIONAL(每阶段累计 1 次重跑
额度,跨续跑记账)。金种子零接触,环境侧零改动,不训练任何 PPO。
"""
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (DEFAULT_MANAGER_SHA256, PROTOCOL_VERSION,  # noqa: E402
                           SCHEMA_VERSION, default_game_data_dir,
                           engine_binary_path, runtime_versions_identity,
                           sha256_file)

RUN = ROOT / "train" / "runs" / "reanchor-r1"
LOGS = RUN / "logs"
LEDGER = RUN / "gate_ledger.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")
PREREG = "docs/PREREG-重锚R1.md"
SELF = "train/run_reanchor_r1.py"
ENGINE_REF_LINE = 'ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"'
STAGE_TIMEOUT_S = 4 * 3600
IDLE_PATTERN = r"train/(bc_|train_ppo|eval_assembled|run_v[0-9]+)"

STAGES = [
    # (阶段, 脚本, canonical 目录, 判词键)
    ("S1-worker", "bc_worker.py", "bc-worker", "data_gate"),
    ("S2-manager", "bc_manager.py", "bc-manager", "hypothesis"),
    ("S3-flat", "bc_flat.py", "bc-flat", "memoryless_hypothesis"),
]
SCIENCE_KEYS = {
    "data_gate": ("pairs", "held_out_top1", "class_recalls",
                  "class_weighted_retry"),
    "hypothesis": ("pairs", "bc_replay_7000", "teacher_7000", "ratio"),
    "memoryless_hypothesis": ("pairs", "bc_replay_mean_7000s",
                              "teacher_replay_mean_7000s", "ratio"),
}
RECEIPT_KEYS = ("policy_sha256", "demos_sha256", "manager_npz_sha256",
                "implementation_sha256", "generator_sha256")


def log(event: dict):
    event = {"t": time.strftime("%F %T"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(RUN / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def impl_sha() -> str:
    from train_ppo import _implementation_bundle_sha256
    return _implementation_bundle_sha256()


def read_ledger() -> list[dict]:
    """续跑先读旧台账并断言逐行可解析(D5);不可解析 → 停机呈报。"""
    if not LEDGER.is_file():
        return []
    events = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"台账第 {i} 行不可解析,停机呈报: {exc}") from exc
    return events


def preflight(events: list[dict]) -> str:
    # W1 冻结公证三步:clean 树;HEAD 即冻结 commit;预注册与驱动器的
    # 最后触碰 commit 都必须就是 HEAD(正文/驱动器零自指 sha)。
    require(git("status", "--porcelain") == "", "W1: 工作树不净")
    head = git("rev-parse", "HEAD")
    for path in (PREREG, SELF):
        touch = git("log", "-1", "--format=%H", "--", path)
        require(touch == head, f"W1: {path} 最后触碰 {touch[:12]} != HEAD "
                               f"{head[:12]}(冻结 commit 必须收录两者)")
    prior_freeze = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if prior_freeze:
        require(prior_freeze[0]["sha"] == head,
                "W1: 续跑 HEAD 与台账 FREEZE_SHA 不一致")
    else:
        log({"event": "FREEZE_SHA", "sha": head})
    # W2 双常数分别断言(评测契约 schema ≠ BC 报告 schema)
    require(PROTOCOL_VERSION == 3 and SCHEMA_VERSION == 2,
            f"W2: 契约版本漂移 protocol={PROTOCOL_VERSION} schema={SCHEMA_VERSION}")
    import train_ppo
    require(train_ppo._BC_REPORT_SCHEMA_VERSION == 1,
            "W2: BC 报告 schema 常数漂移")
    # W3 引擎唯一定位 + 钉死 REF(逐字含 env 覆盖语法)+ 覆盖口封死
    engine = engine_binary_path(ROOT)
    require("DEVILUTIONX_REF" not in os.environ, "W3: DEVILUTIONX_REF 被设置")
    require(ENGINE_REF_LINE in (ROOT / "bootstrap.sh").read_text(),
            "W3: bootstrap.sh ENGINE_REF 行与预注册不符")
    # W4 全内容世界
    mpq = default_game_data_dir() / "diabdat.mpq"
    require(mpq.is_file(), f"W4: 全内容世界缺失 {mpq}")
    # W7 采集夹具(v2 冻结经理,v3 契约钦定默认经理)
    fixture = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
    fixture_sha = sha256_file(fixture)
    require(fixture_sha == DEFAULT_MANAGER_SHA256,
            f"W7: 采集夹具漂移 {fixture_sha[:16]}")
    case_impl = impl_sha()                                   # W5 案级首取
    if not any(e.get("event") == "PREFLIGHT_OK" for e in events):
        # W8 前科封存(仅首次发车;逐文件 sha256+mtime 原样入册)
        for _, _, report_dir, _ in STAGES:
            d = ROOT / "train" / "runs" / report_dir
            for f in sorted(p for p in d.iterdir() if p.is_file()):
                log({"event": "PREV_ARTIFACT", "dir": report_dir,
                     "name": f.name, "sha256": sha256_file(f),
                     "mtime": time.strftime("%F %T",
                                            time.localtime(f.stat().st_mtime))})
    log({"event": "PREFLIGHT_OK", "freeze_sha": head,
         "engine_binary": str(engine),
         "engine_binary_sha256": sha256_file(engine),
         "diabdat_sha256": sha256_file(mpq),
         "implementation_sha256": case_impl,
         "manager_fixture_sha256": fixture_sha,
         "runtime": runtime_versions_identity()})            # W6 完整落账
    return case_impl


def read_report(report_dir: str):
    path = ROOT / "train" / "runs" / report_dir / "bc_report.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "UNPARSEABLE"
    return report if isinstance(report, dict) else "UNPARSEABLE"


def fail_receipt_problems(report: dict, gate: str, script: str,
                          case_impl: str, fixture_sha: str) -> list[str]:
    """R-S FAIL 分支手检(验证器设计上拒收 FAIL 判词,按同一标准逐条)。"""
    from train_ppo import _BC_PASS_KEYS
    expected = set(_BC_PASS_KEYS[gate]) - {"policy_sha256"}
    if gate == "data_gate":
        expected -= {"demos_sha256"}          # manager_npz_sha256 恒在,不去
    problems = []
    if set(report) != expected:
        problems.append(f"键集差 {sorted(set(report) ^ expected)}")
    if report.get("schema_version") != 1:
        problems.append("schema_version")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        problems.append("protocol_version")
    if report.get("implementation_sha256") != case_impl:
        problems.append("implementation_sha256")
    gen_sha = hashlib.sha256((ROOT / "train" / script).read_bytes()).hexdigest()
    if report.get("generator_sha256") != gen_sha:
        problems.append("generator_sha256(须等值,非在场)")
    if gate == "data_gate" and report.get("manager_npz_sha256") != fixture_sha:
        problems.append("manager_npz_sha256 != W7 台账值")
    return problems


def stage_state(stage, script, report_dir, gate, case_impl, fixture_sha):
    """三分判词:PASS/FAIL(回执过闸的科学终局)/ INCOMPLETE(其余一切)。"""
    report = read_report(report_dir)
    if report is None:
        return "INCOMPLETE", "无报告", {}
    if report == "UNPARSEABLE":
        return "INCOMPLETE", "报告不可解析", {}
    verdict = report.get(gate)
    science = {k: report.get(k) for k in SCIENCE_KEYS[gate] if k in report}
    receipts = {k: report.get(k) for k in RECEIPT_KEYS if k in report}
    extra = {"science": science, "receipts": receipts}
    if verdict == "PASS":
        import train_ppo
        try:
            train_ppo._validate_bc_report(
                ROOT / "train" / "runs" / report_dir / "policy_sd.pt",
                gate, case_impl, verify_replay=False)
            return "PASS", "", extra
        except Exception as exc:                    # noqa: BLE001 判词降档
            return "INCOMPLETE", f"R-S PASS 分支未过:{exc}", extra
    if verdict == "FAIL":
        problems = fail_receipt_problems(report, gate, script,
                                         case_impl, fixture_sha)
        if problems:
            return "INCOMPLETE", f"R-S FAIL 分支未过:{problems}", extra
        return "FAIL", "", extra
    return "INCOMPLETE", f"判词键 {gate}={verdict!r}", extra


def prior_operational(events: list[dict], stage: str) -> int:
    """跨续跑的重跑额度记账:OPERATIONAL 收车 + 未收车的发车各计一次。"""
    ends = [e for e in events
            if e.get("event") == "STAGE_END" and e.get("stage") == stage]
    starts = [e for e in events
              if e.get("event") == "STAGE_START" and e.get("stage") == stage]
    ops = sum(1 for e in ends if e.get("verdict") == "OPERATIONAL")
    ops += max(0, len(starts) - len(ends))
    return ops


def machine_idle() -> bool:
    return subprocess.run(["pgrep", "-f", IDLE_PATTERN],
                          capture_output=True).returncode != 0


def run_stage(stage: str, script: str) -> tuple[int, pathlib.Path]:
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOGS / f"{stage}-{ts}.log"
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen([PY, str(ROOT / "train" / script)],
                                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=STAGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()                       # 回收僵尸并确认组内无存活
            code = -9
            log({"event": "STAGE_TIMEOUT", "stage": stage,
                 "log": str(log_path)})
    return code, log_path


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    events = read_ledger()
    case_impl = preflight(events)
    fixture_sha = DEFAULT_MANAGER_SHA256
    results = {}
    for stage, script, report_dir, gate in STAGES:
        state, why, extra = stage_state(stage, script, report_dir, gate,
                                        case_impl, fixture_sha)
        if state == "PASS":
            log({"event": "STAGE_SKIP_COMPLETE", "stage": stage,
                 "verdict": "PASS", **extra})
            results[stage] = "PASS"
            continue
        if state == "FAIL":
            if stage == "S1-worker":
                # R-W FAIL:无论首跑续跑一律停机(幂等跳过不得吞停机)
                attention("R-W FAIL:v3 工人教师缺位,全案停机(D3)")
                log({"event": "CASE_HALT_RW_FAIL", **extra})
                return 3
            log({"event": "STAGE_SKIP_FAIL", "stage": stage,
                 "verdict": "FAIL", "note": "科学终局,禁止加试", **extra})
            results[stage] = "FAIL"
            continue
        ops = prior_operational(events, stage)
        while True:
            if ops >= 2:
                attention(f"{stage} OPERATIONAL 额度耗尽:{why}")
                log({"event": "CASE_HALT_OPERATIONAL", "stage": stage,
                     "why": why})
                return 2
            if not machine_idle():
                attention(f"{stage} 发车前机器不空闲(P6)")
                log({"event": "CASE_HALT_BUSY", "stage": stage})
                return 4
            impl_before = impl_sha()
            if impl_before != case_impl:
                why = f"W5 发车前实现漂移 {impl_before[:12]}"
                log({"event": "STAGE_END", "stage": stage, "exit_code": None,
                     "verdict": "OPERATIONAL", "why": why,
                     "impl_before": impl_before, "impl_after": None})
                ops += 1
                continue
            log({"event": "STAGE_START", "stage": stage,
                 "impl_before": impl_before})
            code, log_path = run_stage(stage, script)
            impl_after = impl_sha()
            if "比值闸无定义" in log_path.read_text(errors="replace"):
                # ANOMALY:教师同池均回报 ≤0,科学异常,不占额度、不重跑
                attention(f"{stage} ANOMALY:同池教师均回报 ≤0,停机呈报")
                log({"event": "ANOMALY", "stage": stage,
                     "log": str(log_path)})
                return 5
            if impl_after != case_impl:
                verdict, why, extra = "OPERATIONAL", \
                    f"W5 收车后实现漂移 {impl_after[:12]}", {}
            else:
                state, why, extra = stage_state(stage, script, report_dir,
                                                gate, case_impl, fixture_sha)
                verdict = state if state in ("PASS", "FAIL") else "OPERATIONAL"
            log({"event": "STAGE_END", "stage": stage, "exit_code": code,
                 "verdict": verdict, "why": why, "log": str(log_path),
                 "impl_before": impl_before, "impl_after": impl_after,
                 **extra})
            if verdict == "PASS":
                results[stage] = "PASS"
                break
            if verdict == "FAIL":
                if stage == "S1-worker":
                    attention("R-W FAIL:v3 工人教师缺位,全案停机(D3)")
                    log({"event": "CASE_HALT_RW_FAIL", **extra})
                    return 3
                results[stage] = "FAIL"
                break
            ops += 1
    log({"event": "CASE_CLOSED", "results": results,
         "note": "R-M/R-F 的 PASS/FAIL 均为合法科学结局,如实入册"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        RUN.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        try:
            log({"event": "DRIVER_EXCEPTION", "traceback": tb})
        finally:
            attention("DRIVER_EXCEPTION:\n" + tb)
        raise
