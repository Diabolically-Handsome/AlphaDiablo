"""v23:FARM 操作脑 BC 热启动(docs/PREREG-v23.md D4)。

在位采集:冻结 H 经理 + 脚本教师(dispatch farm 分支;反射拍是包装器所有,
天然不入集;保险丝强制拍整拍剔除)。示范种子 100-227,只录 FARM 窗口。
产出 train/runs/bc-worker/policy_sd.pt(SB3 键名,--bc-init 用)+ bc_report.json。
闸门 G1(数据侧):held-out top-1 ≥0.95;样本 ≥300 的类召回 ≥0.85
(不达标 → 类加权 CE 重训一次,BC 唯一重试)。

E2 乙1′(PREREG-内容案-课⑤x④乙 E2):增教师 v2(bc-worker-v2)——
dispatch("farm") 前置预防饮分支 hp∈[0.5,0.65)∧belt>0→12,每窗预防饮 ≤1
迟滞保护;世代旗 teacher_generation 1/2;v2 产物独立目录 runs/bc-worker-v2/
(v1 canonical 路径一字不动,规避 _previous 归档互斥);v2 demos 增逐样本
masks(env.action_masks() 现场捕获,唯一 on-manifold 真源);v2 回执独立
文件名 bc_report_v2.json + 独立 schema 标识 + 专用验证器(v1 验证器对
v2 件天然 fail-loud);n₁₂ 闸与 recall 门读数入回执(fail-closed)。
`python train/bc_worker.py` = v1(原样);`--v2 [--preventive-threshold 0.7]` = v2。
"""
import json
import hashlib
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import WorkerWindowEnv
from diablogym.options_env import dispatch
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import (_BC_REPORT_SCHEMA_VERSION, _WORKER_BC_DEMO_SEEDS,
                       _WORKER_BC_FORBIDDEN_ACTIONS,
                       _implementation_bundle_sha256)

OUT = ROOT / "train" / "runs" / "bc-worker"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
DEMO_SEEDS = list(_WORKER_BC_DEMO_SEEDS)

# ---- E2 乙1′ 教师 v2(bc-worker-v2)常量注册(PREREG-内容案 E2/D7)----
OUT_V2 = ROOT / "train" / "runs" / "bc-worker-v2"  # v2 独立产物目录;v1 canonical 原封
TEACHER_GENERATION_V1 = 1
TEACHER_GENERATION_V2 = 2
_PREVENTIVE_HP_LOW = 0.5                # 预防带下界(闭;恰在 0.5 脑干反射之上,
                                        # 反射态归排水,教师原理上不可见)
_PREVENTIVE_THRESHOLD_MAIN = 0.65       # 主案预防阈(D7"预防阈"行)
_PREVENTIVE_THRESHOLD_OC = 0.70         # 唯一注册 OC 之重采旋钮(D7 n₁₂ 闸行);
                                        # 其余取值系未注册旋钮,一律拒绝
_REGISTERED_PREVENTIVE_THRESHOLDS = (_PREVENTIVE_THRESHOLD_MAIN,
                                     _PREVENTIVE_THRESHOLD_OC)
_V2_FORBIDDEN_ACTIONS = (11,)           # v2 路径禁 11 允 12(守卫面不弱化;
                                        # v1 路径 _WORKER_BC_FORBIDDEN_ACTIONS 原封)
_N12_GATE_MIN = 122                     # n₁₂ 闸 ≥122(=审计点估 244 之半,D7)
_RECALL12_GATE_MIN = 0.5                # held-out 12 类 recall 门 ≥0.5(D7)
_BC_V2_REPORT_NAME = "bc_report_v2.json"  # v2 回执独立文件名(schema 隔离)
# 独立 schema 标识:取非 int 字符串——v1 验证器(train_ppo._validate_bc_report)
# 之键集合精确等断言 + _is_plain_int(schema_version)==1 断言对 v2 件双重必炸。
_BC_V2_REPORT_SCHEMA_VERSION = "bc-worker-v2/1"
_BC_V2_PASS_KEYS = frozenset({
    "schema_version", "teacher_generation", "preventive_threshold",
    "pairs", "held_out_top1", "held_out_pairs", "held_out_episodes",
    "class_recalls", "class_weighted_retry",
    "n12", "n12_gate_min", "n12_by_episode",
    "recall_12", "recall_12_denominator", "recall_12_gate_min",
    "class_share_12", "class_share_13", "belt_economy",
    "data_gate", "protocol_version", "implementation_sha256",
    "generator_sha256", "manager_npz_sha256", "policy_sha256",
    "demos_sha256",
})


def artifact_provenance():
    return {
        "schema_version": _BC_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
    }


def begin_output_attempt():
    """去掉 canonical 旧产物，防止本次 FAIL 后下游误吃上次权重。"""
    old = [OUT / name for name in ("policy_sd.pt", "bc_report.json", "demos.npz")
           if (OUT / name).exists()]
    if old:
        archive = OUT / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT / "bc_report.json").write_text(json.dumps({"data_gate": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT / "bc_report.json")


def teacher_action(env: WorkerWindowEnv) -> int:
    raw = env.oe.env._raw
    return dispatch("farm", raw, bool(env.oe.env.action_masks()[14]))


def collect():
    env = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0)
    X, Y, groups = [], [], []
    dropped = 0
    for i, seed in enumerate(DEMO_SEEDS):
        obs, _ = env.reset(seed=seed)
        while obs is not None:
            a = teacher_action(env)
            pair = (np.asarray(obs, dtype=np.float32), a)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1          # 保险丝改写过的步整拍剔除
            else:
                X.append(pair[0]); Y.append(pair[1]); groups.append(seed)
            # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
            obs = env.next_window() if (term or trunc) else obs2
        if (i + 1) % 16 == 0:
            print(f"  采集 {i+1}/{len(DEMO_SEEDS)} 局,{len(Y)} 对(剔除 {dropped})",
                  flush=True)
    # 示范池纪律断言:每个示范种子恰好一局,零兜底滚局(否则数据混入未知种子)
    if env.stats["episodes"] != len(DEMO_SEEDS) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"示范池种子纪律破坏: {env.stats}")
    print(f"示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS)):
        raise RuntimeError("示范集没有精确覆盖固定种子 100..227")
    if np.isin(labels, _WORKER_BC_FORBIDDEN_ACTIONS).any():
        raise RuntimeError("示范集含禁采动作 11/12(11 恒掩归经理;12 系脚本教师"
                           "排水后不采——主权世代示范池纪律)")
    return np.stack(X), labels, groups_array


class PiHead(nn.Module):
    """与 SB3 MlpPolicy(64,64) 策略侧同构。"""

    def __init__(self, obs_dim=298, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def split_by_episode(groups):
    """整局切分，避免同一条确定性轨迹的相邻帧泄漏到 held-out。"""
    episodes = np.unique(groups)
    if len(episodes) < 2:
        raise ValueError("BC held-out 至少需要 2 个独立 episode")
    order = np.random.default_rng(23).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    holdout_episodes = order[:n_holdout]
    ho_mask = np.isin(groups, holdout_episodes)
    tr, ho = np.flatnonzero(~ho_mask), np.flatnonzero(ho_mask)
    if len(tr) == 0 or len(ho) == 0:
        raise ValueError("BC episode split 产生了空训练集或空 held-out")
    return tr, ho, holdout_episodes


def train_bc(X, Y, groups, class_weights=None):
    torch.manual_seed(23)
    tr, ho, _ = split_by_episode(groups)
    model = PiHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    wt = None
    if class_weights is not None:
        wt = torch.as_tensor(class_weights, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(23))
    for epoch in range(8):
        tot = cnt = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); cnt += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/cnt:.4f} acc {correct/cnt:.3f}", flush=True)
    # held-out 评分 + 逐类召回(门槛类 = 全集样本 ≥300 的类,召回在 held-out 上量——
    # 审查团修正:若按 held-out 内 ≥300 筛类,门槛被 10% 切片稀释十倍)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[ho])).argmax(1).numpy()
    yh = Y[ho]
    top1 = float((pred == yh).mean())
    full_counts = Counter(Y.tolist())
    recalls = {}
    for c in sorted(k for k, v in full_counts.items() if v >= 300):
        m = yh == c
        # 门槛类若在 held-out 中零覆盖，必须 fail closed，不能从
        # recalls 字典中消失后被 all([]) 当成 PASS。
        recalls[int(c)] = (round(float((pred[m] == c).mean()), 3)
                           if m.sum() > 0 else 0.0)
    return model, top1, recalls


def export_sb3_sd(model):
    sd = {
        "mlp_extractor.policy_net.0.weight": model.net[0].weight,
        "mlp_extractor.policy_net.0.bias": model.net[0].bias,
        "mlp_extractor.policy_net.2.weight": model.net[2].weight,
        "mlp_extractor.policy_net.2.bias": model.net[2].bias,
        "action_net.weight": model.head.weight,
        "action_net.bias": model.head.bias,
    }
    return {k: v.detach().clone() for k, v in sd.items()}


def main():
    provenance = artifact_provenance()
    begin_output_attempt()
    X, Y, groups = collect()
    demos_tmp = OUT / "demos.tmp.npz"
    np.savez_compressed(demos_tmp, X=X, Y=Y, episode_id=groups)
    demos_tmp.replace(OUT / "demos.npz")
    tr, ho, holdout_episodes = split_by_episode(groups)
    model, top1, recalls = train_bc(X, Y, groups)
    retrained = False
    if top1 < 0.95 or any(r < 0.85 for r in recalls.values()):
        print(f"首训未达标(top1 {top1:.3f} 召回 {recalls})→ 类加权重训(唯一重试)",
              flush=True)
        counts = np.bincount(Y[tr], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(X, Y, groups, class_weights=weights)
        retrained = True
    ok = top1 >= 0.95 and all(r >= 0.85 for r in recalls.values())
    report = {
        "pairs": len(Y), "held_out_top1": round(top1, 4),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"BC 数据闸 FAIL(top1={top1:.3f}, recalls={recalls});"
            "拒绝覆写 policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("BC worker 运行期间实现/引擎/内容/经理发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        (OUT / "demos.npz").read_bytes()).hexdigest()
    write_report(report)
    print(f"held-out top-1 {top1:.3f} 召回 {recalls} retry={retrained} "
          f"→ 数据闸 PASS;已存 {OUT}/policy_sd.pt", flush=True)


# ==== E2 乙1′ v2 件(以下全部新增;v1 面(上文)一字不动)====

def forbidden_actions_for_generation(generation: int) -> tuple[int, ...]:
    """禁采断言世代条件化(E2):v1 禁 (11,12) 原封;v2 禁 11 允 12。"""
    if generation == TEACHER_GENERATION_V1:
        return tuple(_WORKER_BC_FORBIDDEN_ACTIONS)   # (11, 12) 原封
    if generation == TEACHER_GENERATION_V2:
        return _V2_FORBIDDEN_ACTIONS                 # 禁 11 允 12(守卫面不弱化)
    raise ValueError(f"未知教师世代: {generation!r}(只有 1/2)")


def teacher_v2_preventive_trigger(raw, preventive_threshold: float
                                  = _PREVENTIVE_THRESHOLD_MAIN) -> bool:
    """E2 教师 v2 前置分支谓词:hp∈[0.5, 阈)∧belt>0(预防饮)。

    边界语义:下界闭(hp==0.5 触发,恰在脑干反射 hp<0.5 之上),上界开
    (hp==阈 不触发)。hp 口径 = raw["hp"]/max(1, raw["max_hp"]),与
    dispatch/_reflex 逐字同款。纯函数不含迟滞——每窗 ≤1 由 TeacherV2 闩承担。
    """
    hp = raw["hp"] / max(1, raw["max_hp"])
    belt = raw.get("belt_heals", 0)
    return _PREVENTIVE_HP_LOW <= hp < preventive_threshold and belt > 0


class TeacherV2:
    """E2 乙1′ 教师 v2:dispatch("farm") 前置预防饮分支 + 每窗 ≤1 迟滞保护。

    迟滞闩以"提案"计非"执行"计(裁量注记:提案即闩,故 overridden 剔除拍
    亦耗闩)——从构造上保证每窗预防饮提案 ≤1,防排水出口 0.5 与预防阈间
    连锁喝药畸变(主案已定条款;n₁₂=244 系此设计下之保真点估)。
    begin_window() 系闩之唯一复位口:构造后未开窗即不触发,漏接线 fail-visible
    (n₁₂=0 → 闸必拦)。
    """

    def __init__(self, preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN):
        if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
            raise ValueError(
                f"预防阈 {preventive_threshold!r} 未注册:仅 0.65(主案)与 "
                "0.70(唯一注册 OC 重采旋钮)合法(PREREG-内容案 D7)")
        self.preventive_threshold = float(preventive_threshold)
        self._window_preventive_used = True

    def begin_window(self) -> None:
        """新 FARM 窗:迟滞闩清零(每窗预防饮 ≤1 之唯一复位口)。"""
        self._window_preventive_used = False

    def action(self, env) -> int:
        raw = env.oe.env._raw
        if (not self._window_preventive_used
                and teacher_v2_preventive_trigger(raw, self.preventive_threshold)):
            self._window_preventive_used = True
            return 12                   # 预防饮:每窗首个带内态实标 a12(窗首触发态)
        a = dispatch("farm", raw, bool(env.oe.env.action_masks()[14]))
        if a == 12:
            # dispatch 内嵌 0.5 反射分支对教师应为死代码:开窗排水 + 反射尾部
            # 排水保证工人观测永为无反射态;走到此处即示范池纪律破坏,禁静默采。
            raise RuntimeError("教师 v2 见反射态(hp<0.5∧belt>0):排水失守,"
                               "a12 实标只许出自前置预防分支")
        return a


def collect_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN):
    """E2:主权开采集环真实执行、真实入池(on-manifold,禁反事实标签)。

    与 v1 collect() 平行成文而非改写(v1 路径原封):
      - 逐样本 masks 系决策态 env.action_masks() 现场捕获(唯一真源;
        a12 之 belt 位自 obs belt 维反推系第二真源,禁用);
      - overridden 整拍剔除断言原封(保险丝改写拍不入池);
      - 禁采断言世代条件化:v2 禁 11 允 12。
    返回 (X, labels, groups, masks, belts);belts 系逐样本决策态腰带读数
    (腰带经济回执供源)。
    """
    teacher = TeacherV2(preventive_threshold)
    env = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0)
    X, Y, M, groups, belts = [], [], [], [], []
    dropped = 0
    for i, seed in enumerate(DEMO_SEEDS):
        obs, _ = env.reset(seed=seed)
        teacher.begin_window()            # reset 即首窗,开闩
        while obs is not None:
            masks = np.asarray(env.action_masks(), dtype=bool)  # 现场捕获
            a = teacher.action(env)
            if not masks[a]:
                raise RuntimeError(
                    f"教师 v2 提案 {a} 不在现场掩码内(a12 须 m[12]=True):"
                    "on-manifold 示范纪律破坏")
            belt = int(env.oe.env._raw.get("belt_heals", 0))
            pair = (np.asarray(obs, dtype=np.float32), a, masks, belt)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1              # 保险丝改写过的步整拍剔除(v1 条款原封)
            else:
                X.append(pair[0]); Y.append(pair[1]); M.append(pair[2])
                groups.append(seed); belts.append(pair[3])
            if term or trunc:
                # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
                obs = env.next_window()
                teacher.begin_window()    # 新窗开闩(局尽 None 时无害)
            else:
                obs = obs2
        if (i + 1) % 16 == 0:
            print(f"  v2 采集 {i+1}/{len(DEMO_SEEDS)} 局,{len(Y)} 对"
                  f"(剔除 {dropped})", flush=True)
    if env.stats["episodes"] != len(DEMO_SEEDS) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"示范池种子纪律破坏: {env.stats}")
    print(f"v2 示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS)):
        raise RuntimeError("示范集没有精确覆盖固定种子 100..227")
    if np.isin(labels, forbidden_actions_for_generation(
            TEACHER_GENERATION_V2)).any():
        raise RuntimeError("v2 示范集含禁采动作 11(11 恒掩归经理;"
                           "12 系 v2 预防饮实标,允采)")
    masks_array = np.stack(M).astype(bool)
    belts_array = np.asarray(belts, dtype=np.int64)
    return np.stack(X), labels, groups_array, masks_array, belts_array


def _save_demos_v2(out_dir: pathlib.Path, X, labels, groups, masks) -> pathlib.Path:
    """v2 demos schema:X/Y/episode_id(v1 同构)+ 逐样本 masks(E2 新增)。"""
    n = len(labels)
    if masks.shape != (n, 15) or masks.dtype != np.bool_:
        raise RuntimeError(f"v2 masks 形状/dtype 非法: {masks.shape}/{masks.dtype}"
                           "(须 (pairs, 15) bool)")
    if not masks[np.arange(n), labels].all():
        raise RuntimeError("v2 demos 存在提案不在掩码内的样本"
                           "(含 a12 须 m[12]=True;on-manifold 破坏)")
    if masks[:, 11].any():
        raise RuntimeError("v2 demos m[11] 必须恒 False(11 恒掩归经理)")
    tmp = out_dir / "demos.tmp.npz"
    np.savez_compressed(tmp, X=X, Y=labels, episode_id=groups, masks=masks)
    tmp.replace(out_dir / "demos.npz")
    return out_dir / "demos.npz"


def _n12_readings(labels, groups, belts) -> dict:
    """n₁₂ 闸读数(E2/D7):n₁₂ = 迟滞采集下实标 a12 态总数(每窗首触发态,
    overridden 整拍剔除后之入池计数)+ 逐局分解 + 12/13 类占比 + 腰带经济。"""
    n = int(len(labels))
    a12 = labels == 12
    a13 = labels == 13
    n12 = int(a12.sum())
    by_episode = {str(int(seed)): int((a12 & (groups == seed)).sum())
                  for seed in np.unique(groups[a12])}   # 零计局省略,余局恒 0
    return {
        "n12": n12,
        "n12_by_episode": by_episode,
        "class_share_12": round(n12 / n, 6) if n else 0.0,
        "class_share_13": round(int(a13.sum()) / n, 6) if n else 0.0,
        "belt_economy": {
            # 裁量注记:图纸"腰带经济读数"未钉字段,注册为三项——a12 实标态
            # 腰带均值 / 全集腰带均值 / a13(拾取补给)对数;零覆盖记 0.0 不消失。
            "belt_mean_at_a12": (round(float(belts[a12].mean()), 4)
                                 if n12 > 0 else 0.0),
            "belt_mean_overall": round(float(belts.mean()), 4) if n else 0.0,
            "a13_pairs": int(a13.sum()),
        },
    }


def _recall12_from_model(model, X, labels, groups) -> tuple[float, int]:
    """recall 门(E2/D7 钉死):分母 = 现切分下 held-out 之实标 a12 态
    (教师 v2 迟滞下每窗首触发态,非全部带内态——1,975 态口径废止);
    度量 = held-out argmax 命中;fail-closed 承 v1 类召回先例
    (train_bc 零覆盖记 0.0 不消失)。返回 (recall_12, 分母计数)。"""
    _, ho, _ = split_by_episode(groups)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[ho])).argmax(1).numpy()
    yh = labels[ho]
    m = yh == 12
    denominator = int(m.sum())
    recall = (round(float((pred[m] == 12).mean()), 4) if denominator > 0
              else 0.0)                                 # fail-closed
    return recall, denominator


def artifact_provenance_v2(preventive_threshold: float) -> dict:
    return {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "teacher_generation": TEACHER_GENERATION_V2,
        "preventive_threshold": preventive_threshold,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
    }


def begin_output_attempt_v2():
    """v2 目录去旧防误吃(v1 同型;目录独立,规避 _previous 归档互斥)。"""
    OUT_V2.mkdir(parents=True, exist_ok=True)
    old = [OUT_V2 / name
           for name in ("policy_sd.pt", _BC_V2_REPORT_NAME, "demos.npz")
           if (OUT_V2 / name).exists()]
    if old:
        archive = OUT_V2 / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT_V2 / _BC_V2_REPORT_NAME).write_text(json.dumps(
        {"data_gate": "RUNNING", "teacher_generation": TEACHER_GENERATION_V2}))


def write_report_v2(record):
    tmp = OUT_V2 / "bc_report_v2.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT_V2 / _BC_V2_REPORT_NAME)


def _validate_bc_v2_report(p: pathlib.Path,
                           expected_implementation_sha256: str | None = None,
                           *, policy_payload: bytes | None = None,
                           report_payload: bytes | None = None,
                           demos_payload: bytes | None = None) -> dict:
    """BC-v2 专用验证器(E2 schema 隔离,形制承 v1 验证器)。

    键集合精确等断言 + 独立 schema 标识:对 v1 件必炸(键集合不等);
    v1 验证器(train_ppo._validate_bc_report)对 v2 件亦必炸(键集合
    精确等 + schema_version 非 int 双重不相容)——v1/v2 验证器互斥。
    demos 实测字节断言(rev4 十二附二④ 补铸):demos.npz 字节 sha256 ≡
    回执 demos_sha256,镜像 policy 侧真字节断言形制。
    """
    from eval_contract import EvalContractError, strict_json_loads

    def _fail(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _plain_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _plain_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    report = p.with_name(_BC_V2_REPORT_NAME)
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC-v2 回执缺失/不可读: {report}") from exc
    _fail(isinstance(rec, dict), f"BC-v2 回执必须是 JSON 对象: {report}")
    _fail(set(rec) == set(_BC_V2_PASS_KEYS),
          f"BC-v2 回执字段/schema 不匹配(v1/v2 件互斥): {report}")
    _fail(rec["schema_version"] == _BC_V2_REPORT_SCHEMA_VERSION,
          f"BC-v2 回执 schema 标识不符: {rec['schema_version']!r}")
    _fail(rec["teacher_generation"] == TEACHER_GENERATION_V2,
          f"BC-v2 回执 teacher_generation 必须为 2: {rec['teacher_generation']!r}")
    _fail(rec["preventive_threshold"] in _REGISTERED_PREVENTIVE_THRESHOLDS,
          f"BC-v2 回执预防阈未注册: {rec['preventive_threshold']!r}")
    _fail(rec["data_gate"] == "PASS",
          f"拒绝采信未过 data_gate 闸的 BC-v2 件: {rec['data_gate']!r}")
    _fail(_plain_int(rec["n12"]) and rec["n12"] >= _N12_GATE_MIN,
          f"BC-v2 n₁₂ 闸不满足(≥{_N12_GATE_MIN}): {rec['n12']!r}")
    _fail(_plain_num(rec["recall_12"])
          and _RECALL12_GATE_MIN <= float(rec["recall_12"]) <= 1.0,
          f"BC-v2 recall 门不满足(≥{_RECALL12_GATE_MIN}): {rec['recall_12']!r}")
    _fail(rec["protocol_version"] == PROTOCOL_VERSION,
          f"BC-v2 回执协议过期: {rec['protocol_version']!r}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _fail(rec["implementation_sha256"] == expected_impl,
          "BC-v2 回执的实现/引擎/游戏内容身份与当前运行时不一致")
    generator_sha = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()
    _fail(rec["generator_sha256"] == generator_sha,
          "BC-v2 回执生成器已漂移: train/bc_worker.py")
    expected_sha = rec["policy_sha256"]
    _fail(isinstance(expected_sha, str) and len(expected_sha) == 64,
          f"BC-v2 回执缺少 policy_sha256 绑定: {report}")
    try:
        frozen_policy = (p.read_bytes() if policy_payload is None
                         else policy_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 权重缺失/不可读: {p}") from exc
    _fail(hashlib.sha256(frozen_policy).hexdigest() == expected_sha,
          "BC-v2 权重与回执 SHA 不匹配")
    _fail(isinstance(rec["demos_sha256"], str) and len(rec["demos_sha256"]) == 64,
          f"BC-v2 回执缺少 demos_sha256 绑定: {report}")
    demos = p.with_name("demos.npz")
    try:
        frozen_demos = (demos.read_bytes() if demos_payload is None
                        else demos_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 demos 缺失/不可读: {demos}") from exc
    _fail(hashlib.sha256(frozen_demos).hexdigest() == rec["demos_sha256"],
          "BC-v2 demos 与回执 SHA 不匹配")
    return rec


def main_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN):
    if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
        raise ValueError(f"预防阈 {preventive_threshold!r} 未注册"
                         "(仅 0.65 主案 / 0.70 唯一注册 OC)")
    provenance = artifact_provenance_v2(preventive_threshold)
    begin_output_attempt_v2()
    X, labels, groups, masks, belts = collect_v2(preventive_threshold)
    _save_demos_v2(OUT_V2, X, labels, groups, masks)
    tr, ho, holdout_episodes = split_by_episode(groups)
    model, top1, recalls = train_bc(X, labels, groups)
    retrained = False
    if top1 < 0.95 or any(r < 0.85 for r in recalls.values()):
        # v1 同款唯一重试,触发条件限 v1 面质量闸(top1/≥300 类召回)——
        # n₁₂/recall_12 永不触发类加权重训:类加权系 N12 已除名预案,
        # 须另行亲批(PREREG-内容案 D2-5/D5 P-N12/D7)。
        print(f"v2 首训未达标(top1 {top1:.3f} 召回 {recalls})→ 类加权重训"
              "(唯一重试;n₁₂/recall_12 永不触发此重试)", flush=True)
        counts = np.bincount(labels[tr], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(X, labels, groups, class_weights=weights)
        retrained = True
    recall_12, recall_12_denominator = _recall12_from_model(
        model, X, labels, groups)
    readings = _n12_readings(labels, groups, belts)
    ok = (top1 >= 0.95 and all(r >= 0.85 for r in recalls.values())
          and readings["n12"] >= _N12_GATE_MIN
          and recall_12 >= _RECALL12_GATE_MIN)
    report = {
        "pairs": len(labels), "held_out_top1": round(top1, 4),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "recall_12": recall_12,
        "recall_12_denominator": recall_12_denominator,
        "recall_12_gate_min": _RECALL12_GATE_MIN,
        "n12_gate_min": _N12_GATE_MIN,
        **readings,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report_v2(report)
        raise RuntimeError(
            f"BC-v2 数据闸 FAIL(top1={top1:.3f}, n12={readings['n12']}, "
            f"recall_12={recall_12});拒绝覆写 policy_sd.pt。n₁₂ 不足之"
            "唯一注册 OC = 预防阈 0.65→0.70 重采一次(D5 P-N12);仍不足 → "
            "不冻结呈报,禁带闸伤发车")
    policy_tmp = OUT_V2 / "policy_sd.tmp.pt"
    if artifact_provenance_v2(preventive_threshold) != provenance:
        raise RuntimeError("BC-v2 运行期间实现/引擎/内容/经理发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT_V2 / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT_V2 / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        (OUT_V2 / "demos.npz").read_bytes()).hexdigest()
    write_report_v2(report)
    print(f"v2 held-out top-1 {top1:.3f} n12 {readings['n12']} "
          f"recall_12 {recall_12} retry={retrained} → 数据闸 PASS;"
          f"已存 {OUT_V2}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BC worker 采集/训练(默认 v1 面原样;--v2 系 E2 乙1′ 件)")
    parser.add_argument("--v2", action="store_true",
                        help="教师 v2 采集+训练 → runs/bc-worker-v2/"
                             "(PREREG-内容案 E2)")
    parser.add_argument("--preventive-threshold", type=float, default=None,
                        choices=_REGISTERED_PREVENTIVE_THRESHOLDS,
                        help="v2 预防阈:0.65 主案 / 0.70 唯一注册 OC 重采旋钮"
                             "(仅与 --v2 同用)")
    cli = parser.parse_args()
    if cli.v2:
        threshold = (cli.preventive_threshold
                     if cli.preventive_threshold is not None
                     else _PREVENTIVE_THRESHOLD_MAIN)
        OUT_V2.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(OUT_V2 / ".bc.lock", "BC worker v2 产物"):
            main_v2(threshold)
    else:
        if cli.preventive_threshold is not None:
            parser.error("--preventive-threshold 仅与 --v2 同用(v1 面一字不动)")
        with exclusive_lock(OUT / ".bc.lock", "BC worker 产物"):
            main()
