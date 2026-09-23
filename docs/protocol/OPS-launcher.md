# OPS-launcher:案级驱动器点火与值守细则(B1-E6 成文)

**出处**:PREREG-B1 E6(承 v32 OPS 事故——v32 系临场手作 nohup,仓内无
committed launcher;本文把事故补丁升格为驱动器标准条款,随冻结 commit 入库)。

## 交付件

| 件 | 路径 | 说明 |
|---|---|---|
| 标准点火器 | `train/launch_case.sh` | nohup 孤儿化 + `caffeinate -is` + 日志重定向 + launch receipt + caffeinate 断言 |
| launchd 模板 | `train/ops/com.alphadiablo.case-driver.plist.template` | 孤儿化直挂(PPID=1),`__CASE__/__REPO__/__DRIVER__` 三占位符 |
| 值守细则 | 本文 | 心跳阈值成文 |

## 标准点火

```bash
# B1 案(示例;金牌类手启发射永不走 launcher,照旧值守手启)
train/launch_case.sh train/runs/infra-b1 train/run_b1_infra.py
```

点火后立即核验三项:

1. **孤儿化**:`ps -o ppid= -p <PID>`(壳退出后应为 `1`);
2. **caffeinate 在树**:`pgrep -f "caffeinate -is"` 非空;
3. **日志在写**:`tail -f <case-dir>/driver.<stamp>.log`。

## 心跳检查细则(阈值成文,值守照表执行)

| 阶段 | 心跳文件 | WARN | DEAD |
|---|---|---|---|
| 训练腿在跑 | `train/runs/<leg>/progress.jsonl` mtime | > 120 s | > 600 s |
| 评测/导出/重放阶段 | `<case-dir>/driver.<stamp>.log` mtime | > 900 s | > 1800 s |
| 全程 | `pgrep -f "caffeinate -is"` | 空 = WARN(睡眠风险) | — |

- **WARN 处置**:只观察不干预,15 分钟内复查一次;连续两次 WARN 升级为 DEAD 处置。
- **DEAD 处置**:按案 P 线走(B1:P1 顶层异常/中断条款)——先取证
  (`ps`、日志尾、`status.json`、台账尾),后 `kill`;禁在未取证前重启;
  重启走驱动器幂等续跑(exam_or_adopt/续跑对账条款),禁手改台账续命。
- **辅助读数**:`train/runs/<leg>/status.json` 之 `sps`(v32 腿约 180 sps;
  低于 60 sps 持续 5 分钟按 WARN 记)与 `updated_at`(与 progress mtime 同义)。

## launchd 直挂(可选,系统级看护)

```bash
sed -e "s/__CASE__/infra-b1/g" \
    -e "s#__REPO__#$HOME/Desktop/AlphaDiablo/diablogym#g" \
    -e "s#__DRIVER__#train/run_b1_infra.py#g" \
    train/ops/com.alphadiablo.case-driver.plist.template \
    > ~/Library/LaunchAgents/com.alphadiablo.infra-b1.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphadiablo.infra-b1.plist
# 卸载
launchctl bootout gui/$(id -u)/com.alphadiablo.infra-b1
```

`KeepAlive=false` 系有意为之:台账制额度(腿点火 2 次/评测发 2 次)是
**人裁的**,自动复活会在无人值守时烧穿额度;崩溃 → 值守按 P 线取证后手动重启。

## 与驱动器互斥的关系

驱动器自带 `.driver.lock` flock 互斥(W8);launcher 不做第二套互斥,
重复点火的第二个进程会以退出码 4(不空闲/锁冲突)自然死亡并留日志。
