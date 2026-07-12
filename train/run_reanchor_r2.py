"""定锚R2:protocol-v3 开元 · 金种子定锚重测驱动器(docs/PREREG-定锚R2.md 终稿)。

五发(四锚测量 + A5 组装考核闸)各限台账制 2 次点火;定锚是测量不是
竞赛。R-V 第三方复验不信任子进程自报;科学读数一经有效落档即终局。
退出码:0 案结/幂等;2 额度耗尽;3 预检不过;4 不空闲/锁冲突;
5 发车前实现漂移;6 案中 runtime 漂移;其余 = DRIVER_EXCEPTION。
"""
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

from eval_contract import (DEFAULT_MANAGER_SHA256, OutputReservationError,  # noqa: E402
                           PROTOCOL_VERSION, SCHEMA_VERSION,
                           default_game_data_dir, engine_binary_path,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           runtime_versions_identity, sha256_file,
                           verify_eval_identity)

RUN = ROOT / "train" / "runs" / "reanchor-r2"
LOGS = RUN / "logs"
LEDGER = RUN / "gate_ledger.jsonl"
OUTDIR = ROOT / "train" / "runs" / "eval-assembled"
PY = str(ROOT / ".venv" / "bin" / "python")
PREREG = "docs/PREREG-定锚R2.md"
SELF = "train/run_reanchor_r2.py"
ENGINE_REF_LINE = 'ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"'
BOARD_EXEMPT = "?? train/leaderboard-assembled-v3.md"
FIRING_TIMEOUT_S = 3600
SEEDS = range(9000, 9032)
IDLE_PATTERN = r"train/(bc_|train_ppo|eval_assembled|run_v[0-9]+|run_reanchor)"

W28 = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
W24 = ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz"
M29 = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
H22 = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
BCSD = ROOT / "train" / "runs" / "bc-worker" / "policy_sd.pt"
PINNED = {
    W28: "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a",
    W24: "a31fa7c6b18b5c3593f4e1753d97aac9386689aa6ad8b158c526b673c57fbc2a",
    M29: "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6",
    H22: DEFAULT_MANAGER_SHA256,
    BCSD: "f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6",
}
PRIORS = {
    OUTDIR / "v29-mfresh-full32.json":
        "08633101c010a2975b9001a71660bb50d43e681842e3fe1befdfdfd48f99ce63",
    OUTDIR / "v28-G3-leg1.json":
        "6fc6a44c7862424ab5f71ff3a5031adfd34a3e33f9f4f2f8aee781a07711e59d",
    OUTDIR / "v24-golden.json":
        "d9387dcb1a392d62b68d4c22297045b99f4712c044d4a54db6ebc3a7e37a7dac",
}
# (发名, tag, worker_spec, manager npz, v2 前科, 前科池别)
FIRINGS = [
    ("A1-science", "r2-science", str(W28), str(M29), 140.3, "7000-pool"),
    ("A2-launch", "r2-launch", str(W28), str(H22), 112.4, "7000-pool"),
    ("A3-throne", "r2-throne", str(W24), str(H22), 97.2, "golden"),
    ("A4-script", "r2-script", "script", str(H22), 93.9, "golden-board"),
    ("A5-bcworker", "r2-bcworker", "bc", str(H22), None, "gate-not-anchor"),
]
WORKER_PIN = {str(W28): PINNED[W28], str(W24): PINNED[W24],
              "bc": PINNED[BCSD]}


class RuntimeDrift(Exception):
    """台账已 FIRING_VALID 而现快照 R-V 不过:案中 runtime 漂移。"""


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
    if not LEDGER.is_file():
        return []
    events = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"台账第 {i} 行不可解析,停机呈报: {exc}") from exc
    return events


def firing_cmd(worker_spec: str, manager: str, tag: str) -> list[str]:
    return [PY, str(ROOT / "train" / "eval_assembled.py"),
            "--worker", worker_spec, "--manager-npz", manager,
            "--seeds", "9000-9031", "--board", "--tag", tag]


def git_assert(events: list[dict]):
    """W1 口径(含榜面豁免行);W11 每发重申时复用。"""
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != BOARD_EXEMPT]
    require(not dirty, f"W1: 工作树不净 {dirty}")
    head = git("rev-parse", "HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes:
        require(head == freezes[-1]["sha"],
                "W1: HEAD != 台账最后一条 FREEZE_SHA(链式重冻结须走 P6)")
    return head


def preflight(events: list[dict]) -> tuple[str, str]:
    head = git_assert(events)
    for path in (PREREG, SELF):
        touch = git("log", "-1", "--format=%H", "--", path)
        require(touch == head, f"W1: {path} 最后触碰 {touch[:12]} != HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    pending_freeze = None
    if not freezes:
        pending_freeze = {"sha": head}
    elif freezes[-1]["sha"] != head:
        # git_assert 已拦;此分支唯 REFREEZE_REASON 在场时可达(P6 链式)
        reason = (RUN / "REFREEZE_REASON")
        require(reason.is_file() and reason.read_text().strip() != "",
                "P6: 重冻结须 REFREEZE_REASON 非空")
        pending_freeze = {"sha": head, "prev_sha": freezes[-1]["sha"],
                          "reason": reason.read_text().strip()}
    require(PROTOCOL_VERSION == 3 and SCHEMA_VERSION == 2, "W2: 契约版本漂移")
    engine = engine_binary_path(ROOT)
    require("DEVILUTIONX_REF" not in os.environ, "W3: DEVILUTIONX_REF 被设置")
    require(ENGINE_REF_LINE in (ROOT / "bootstrap.sh").read_text(),
            "W3: ENGINE_REF 行不符")
    mpq = default_game_data_dir() / "diabdat.mpq"
    require(mpq.is_file(), f"W4: 缺 {mpq}")
    for path, expected in {**PINNED, **PRIORS}.items():       # W7 + PRIORS
        actual = sha256_file(path)
        require(actual == expected, f"W7/P4: {path.name} sha 漂移 {actual[:16]}")
    for name, tag, *_ in FIRINGS:                             # W9 首燃先决
        if not any(e.get("event") == "FIRING_START" and e.get("firing") == name
                   for e in events):
            for t in (tag, f"{tag}-b"):
                require(not (OUTDIR / f"{t}.json").exists(),
                        f"W9: 案前残档 {t}.json 在位")
    case_impl = impl_sha()
    engine_sha = sha256_file(engine)
    diab_sha = sha256_file(mpq)
    prior_ok = [e for e in events if e.get("event") == "PREFLIGHT_OK"]
    if prior_ok:                                              # W10 续跑对账
        for key, value in (("implementation_sha256", case_impl),
                           ("engine_binary_sha256", engine_sha),
                           ("diabdat_sha256", diab_sha)):
            require(prior_ok[0].get(key) == value,
                    f"W10 续跑对账: {key} 与首条 PREFLIGHT_OK 不一致")
    # 全部 W 线通过后方落账(面板 blocker:预检失败不留 FREEZE_SHA)
    if pending_freeze:
        log({"event": "FREEZE_SHA", **pending_freeze})
        events.append({"event": "FREEZE_SHA", **pending_freeze})
    if not any(e.get("event") == "GOLDEN_AUTHORIZED" for e in events):
        log({"event": "GOLDEN_AUTHORIZED", "seeds": "9000-9031",
             "firings": [" ".join(firing_cmd(w, m, t))
                         for _, t, w, m, *_ in FIRINGS],
             "知会出处": "PREREG-定锚R2 案由·金种子授权段(知情知会,"
                        "v29/v30 先例裁量入册)",
             "freeze_sha": head})
    log({"event": "PREFLIGHT_OK", "freeze_sha": head,
         "engine_binary_sha256": engine_sha, "diabdat_sha256": diab_sha,
         "implementation_sha256": case_impl,
         "runtime": runtime_versions_identity()})
    return case_impl, head


def machine_idle() -> bool:
    result = subprocess.run(["pgrep", "-f", IDLE_PATTERN],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return True
    pids = {int(x) for x in result.stdout.split()}
    pids -= {os.getpid(), os.getppid()}
    return not pids


def validate_archive(tag, worker_spec, manager):
    """R-V:第三方复核 + W7 常量断言。返回 (agg, 档案sha, 快照)。"""
    path = OUTDIR / f"{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"档案缺失: {path}")
    snapshot = freeze_eval_identity(ROOT, worker_spec, manager)
    if worker_spec in WORKER_PIN:
        require(snapshot["worker"]["sha256"] == WORKER_PIN[worker_spec],
                f"R-V: worker 快照 sha != W7 冻结常量({tag})")
    require(snapshot["manager"]["sha256"] == PINNED[pathlib.Path(manager)],
            f"R-V: manager 快照 sha != W7 冻结常量({tag})")
    expected = expected_eval_identity(snapshot, tag=tag, seeds=SEEDS)
    document = read_eval_archive(path, **expected)
    verify_eval_identity(snapshot, ROOT)
    return document["agg"], sha256_file(path), snapshot


def seal_residue(events, tag, why):
    path = OUTDIR / f"{tag}.json"
    if not path.is_file():
        return
    if any(e.get("event") == "RESIDUE_SEALED" and e.get("tag") == tag
           for e in events):
        return
    ev = {"event": "RESIDUE_SEALED", "tag": tag,
          "sha256": sha256_file(path), "why": why}
    log(ev)
    events.append(ev)


def firing_starts(events, name) -> int:
    return sum(1 for e in events
               if e.get("event") == "FIRING_START" and e.get("firing") == name)


def adopt_from_resume(events, name, tag, worker_spec, manager):
    """P1 双条件采信;FIRING_VALID 在册而 R-V 不过 → RuntimeDrift。"""
    for t in (tag, f"{tag}-b"):
        started = any(e.get("event") == "FIRING_START" and e.get("tag") == t
                      for e in events)
        valid_in_ledger = any(e.get("event") == "FIRING_VALID"
                              and e.get("tag") == t for e in events)
        condemned = any(e.get("event") in ("FIRING_INVALID", "RESIDUE_SEALED")
                        and e.get("tag") == t for e in events)
        if not started:
            continue
        try:
            agg, sha, snapshot = validate_archive(t, worker_spec, manager)
        except Exception as exc:                              # noqa: BLE001
            if valid_in_ledger:
                raise RuntimeDrift(
                    f"{t} 台账 FIRING_VALID 在册而现快照 R-V 不过:{exc}"
                ) from exc
            continue
        if condemned and not valid_in_ledger:
            continue                    # 已判 invalid 的档案不得静默翻案
        return t, agg, sha, snapshot
    return None


def fire(name, tag, worker_spec, manager):
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOGS / f"{tag}-{ts}.log"
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(firing_cmd(worker_spec, manager, tag),
                                cwd=ROOT, stdout=lf,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=FIRING_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass          # 竞态:超时判定与击杀之间子进程已自然退出
            proc.wait()
            code = -9
            log({"event": "FIRING_TIMEOUT", "firing": name,
                 "log": str(log_path)})
    return code, log_path


def identity_fields(snapshot):
    return {"worker": {k: snapshot["worker"][k]
                       for k in ("kind", "path", "sha256")},
            "manager": {k: snapshot["manager"][k]
                        for k in ("path", "sha256")}}


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    events = read_ledger()
    if any(e.get("event") == "ANCHOR_GRANT" for e in events):
        print("案已结:ANCHOR_GRANT 在册,幂等退出", flush=True)
        return 0
    try:
        case_impl, head = preflight(events)
    except Exception as exc:                                  # noqa: BLE001
        log({"event": "PREFLIGHT_FAIL",
             "why": (str(exc).splitlines() or ["?"])[0]})
        attention(f"P4 预检不过:{exc}")
        return 3
    results = {}
    for name, tag, worker_spec, manager, prior, pool in FIRINGS:
        try:
            found = adopt_from_resume(events, name, tag, worker_spec, manager)
        except RuntimeDrift as exc:
            attention(f"P2/R-V:{exc}")
            log({"event": "CASE_HALT_RUNTIME_DRIFT", "firing": name,
                 "why": (str(exc).splitlines() or ["?"])[0]})
            return 6
        while found is None:
            if firing_starts(events, name) >= 2:              # P2 台账额度
                attention(f"{name} 点火额度耗尽(台账制),全案停机(P5 甲案)")
                log({"event": "CASE_HALT_OPERATIONAL", "firing": name})
                return 2
            attempt_tag = next(
                (t for t in (tag, f"{tag}-b")
                 if not (OUTDIR / f"{t}.json").exists()), None)
            if attempt_tag is None:
                attention(f"{name} 首发与 -b 档均在位且不可采信,全案停机")
                log({"event": "CASE_HALT_OPERATIONAL", "firing": name})
                return 2
            git_assert(events)                                # W11 重申
            w_path = (BCSD if worker_spec == "bc"
                      else None if worker_spec == "script"
                      else pathlib.Path(worker_spec))
            for p in (w_path, pathlib.Path(manager)):
                if p is not None:
                    require(sha256_file(p) == PINNED[p],
                            f"W11: {p.name} 案中漂移")
            if not machine_idle():
                attention(f"{name} 发车前机器不空闲(P3)")
                log({"event": "CASE_HALT_BUSY", "firing": name})
                return 4
            before = impl_sha()
            if before != case_impl:                           # W5 前漂移
                attention(f"{name} 发车前实现漂移(W5),全案停机")
                log({"event": "CASE_HALT_IMPL_DRIFT", "firing": name,
                     "impl_before": before})
                return 5
            ev = {"event": "FIRING_START", "firing": name,
                  "tag": attempt_tag, "impl_before": before}
            log(ev)
            events.append(ev)
            code, log_path = fire(name, attempt_tag, worker_spec, manager)
            after = impl_sha()
            log({"event": "FIRING_EXIT", "firing": name, "tag": attempt_tag,
                 "exit_code": code, "log": str(log_path),
                 "impl_after": after})
            if after != case_impl:                            # W5 后漂移
                ev = {"event": "FIRING_INVALID", "firing": name,
                      "tag": attempt_tag,
                      "why": f"W5 收车后实现漂移 {after[:12]}"}
                log(ev)
                events.append(ev)
                seal_residue(events, attempt_tag, ev["why"])
                continue                    # P2(c):-b 前须漂移复原(W5 前检)
            try:
                agg, sha, snapshot = validate_archive(
                    attempt_tag, worker_spec, manager)
            except Exception as exc:                          # noqa: BLE001
                why = (str(exc).splitlines() or ["?"])[0]
                ev = {"event": "FIRING_INVALID", "firing": name,
                      "tag": attempt_tag, "why": why}
                log(ev)
                events.append(ev)
                seal_residue(events, attempt_tag, why)
                if code == 0 and (OUTDIR / f"{attempt_tag}.json").is_file():
                    # P2 封闭枚举:出档且退出 0 且实现稳定而 R-V 不过 =
                    # 成因不明,重烧零信息增量 → 停机呈报,不得 -b
                    attention(f"{name} R-V 失败且成因不明(实现稳定、"
                              f"档案在位、退出 0):{why}")
                    log({"event": "CASE_HALT_UNEXPLAINED", "firing": name})
                    return 2
                continue                                      # (a)/(b)/(d)
            found = (attempt_tag, agg, sha, snapshot)
        tag_used, agg, sha, snapshot = found
        results[name] = {
            "firing": name, "tag": tag_used,
            "archive_path": f"train/runs/eval-assembled/{tag_used}.json",
            "archive_sha256": sha, "seeds": "9000-9031",
            "ret_mean": agg.get("ret_mean"), "v2_prior": prior,
            "prior_pool": pool, "freeze_sha": head,
            **identity_fields(snapshot)}
        log({"event": "FIRING_VALID", "firing": name, "tag": tag_used,
             "archive_sha256": sha, "agg": agg, **identity_fields(snapshot)})
    ro_held = (results["A1-science"]["ret_mean"]
               >= results["A2-launch"]["ret_mean"]
               >= results["A4-script"]["ret_mean"])
    log({"event": "R_O_OBSERVATION",
         "predicted": "r2-science >= r2-launch >= r2-script",
         "observed": {n: results[n]["ret_mean"] for n in results},
         "held": ro_held})
    log({"event": "R_DELTA",
         "same_pool_cross_world": {
             n: {"v2": results[n]["v2_prior"],
                 "v3": results[n]["ret_mean"],
                 "delta": round(results[n]["ret_mean"]
                                - results[n]["v2_prior"], 1)}
             for n in ("A3-throne", "A4-script")},
         "cross_pool_composite": {
             n: {"v2_7000pool": results[n]["v2_prior"],
                 "v3_golden": results[n]["ret_mean"],
                 "note": "跨世界×跨池复合量,禁止以位移名义解读"}
             for n in ("A1-science", "A2-launch")}})
    g1_line = 0.85 * results["A4-script"]["ret_mean"]
    log({"event": "R_G1",
         "bcworker": results["A5-bcworker"]["ret_mean"],
         "line": round(g1_line, 2),
         "passed": results["A5-bcworker"]["ret_mean"] >= g1_line,
         "note": "闸门发,读数永不称锚(R1 顺延义务清偿)"})
    log({"event": "ANCHOR_GRANT",
         "v3_science_anchor": results["A1-science"],
         "v3_launch_anchor": results["A2-launch"],
         "throne_reading": results["A3-throne"],
         "script_reference": results["A4-script"],
         "bcworker_gate": results["A5-bcworker"],
         "note": "五发全部有效;课程战役解锁以判决附录 commit 为准"})
    return 0


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_lock(RUN / ".driver.lock", "R2 定锚驱动"):
            code = main()
    except OutputReservationError as exc:
        log({"event": "CASE_HALT_LOCK", "why": str(exc)})
        attention(f"P3 驱动器锁冲突:{exc}")
        code = 4
    except BaseException:
        tb = traceback.format_exc()
        try:
            log({"event": "DRIVER_EXCEPTION", "traceback": tb})
        finally:
            attention("DRIVER_EXCEPTION:\n" + tb)
        raise
    sys.exit(code)
