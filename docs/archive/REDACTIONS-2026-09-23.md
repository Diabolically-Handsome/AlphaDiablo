# 2026-09-23 去除个人信息记录

从 2026-09-23 起，仓库当前文件里不再出现所有者的真实姓名，也不再出现本机用户名：
既没有带姓名的 macOS 用户名路径，也没有 Linux/Windows 的本机用户名路径。
历史提交没有改写，旧提交保持原样。

## 内容改变的文件

下面的文件只把本机路径里的用户名文件夹换成了 `/Users/user/`，其余字节不变。SHA-256 因此改变。
改之前的原文件在提交 `c1ffced` 里。

| 文件 | 改了什么 | 原 SHA-256 | 新 SHA-256 |
|---|---|---|---|
| `train/models/v22-h-manager/model_final.zip` | zip 内 `data` 元数据里的 1 条路径 | `f3b579d2b0c9b613045692435a46702d1a9e8de8fc62e155c651f565d8bd6f1a` | `9dcf40b061bcb40b5548f30587d726758f2ea27df745dd7fb418c44adaa81c38` |
| `train/models/v23-worker-1M/model.zip` | zip 内 `data` 元数据里的 1 条路径 | `b6e1cbdd0137feca5253fe8ca1deec6ae468bb7018d8f798a0a41b8f519a9777` | `104f72fd8368bc23886396ba5069cdcd5e24f7424d0d43ea6510c1d0bcde8661` |
| `train/models/v24-worker-leg7/model.zip` | zip 内 `data` 元数据里的 2 条路径 | `ac65d4eb91fdb678f38ba7ea502812353c7c83afa0551e89ea4a12b13d55e781` | `fb9cd6f58c5e212202e1579457234288acdfe156b292c5971bebd1bac8b63d9b` |
| `train/models/v28-worker-leg1/model_final.zip` | zip 内 `data` 元数据里的 3 条路径 | `2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42` | `0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434` |
| `train/runs/eval-assembled/r2-launch.json` | `meta` 里 6 条路径 | `8ab6b51065105a6719c1348d561c5fe4b15f674f27869e6870c0bf813642dff1` | `bf9a543759bde261eed92e0beccc3b9422c21c10876c07e7da81ec441ec5a47b` |
| `train/runs/eval-assembled/r2-science.json` | `meta` 里 6 条路径 | `8d6a05d3517a6481c756109158b446b3eb7bae8858d02841b42c43a972e273f5` | `37fab311e0a07a1ea22b35fd13d23448291b1d461d2f3b81f2dbe0e9d9391511` |
| `train/runs/eval-assembled/r2-script.json` | `meta` 里 5 条路径 | `71c298e05b6bf19ea94ced26c68b67b9e05303f22637357b610b15c6fb21a7f7` | `ede6c0507e4010649f426de43748b27c2117faff373341f4b39b827b5e4645f5` |
| `train/runs/eval-assembled/r2-throne.json` | `meta` 里 6 条路径 | `2324648a416cbc0cb858858b006b5f797ee9c63422f13cb86c86462948f28c63` | `1390a72db6a527a90238d37b2935adc0872c13bc0cc609e356bd72a86752fc41` |

模型 zip 是重新打包的：`policy.pth`、`policy.optimizer.pth`、`pytorch_variables.pth` 等其余成员逐字节不变，
权重和优化器状态没有变化，`MaskablePPO.load` 照常加载。

`train/leaderboard-assembled-v3.md` 的 6 段 base64 出处注释里也有同样的路径，已解码、替换、再编码。
全局合同的 SHA-256 因此从 `ba9e54d2932ec962…` 变为 `7a23c3d29c363b6e…`，5 行的 `contract_sha256` 同步更新。
可见表格、行键和各行记录的 `archive_sha256` 保留原值，它们记录的是评测当时的档案。

## 本机用户名路径

另有 25 个文件的本机路径里带着 Linux/Windows 的本机用户名，共 309 处。用户名文件夹统一换成 `user`：
`/home/<用户名>/` 换成 `/home/user/`，`C:\Users\<用户名>\` 换成 `C:\Users\user\`。
5 个测试文件例外：代码里的路径改成 `Path.home()` 或仓库内的相对路径，文档字符串里的改成 `~/`。
其余字节不变。改之前的原文件在提交 `c1ffced` 里。
当前树里没有代码、测试或文书引用这些文件的 SHA-256。已移出主干的 r13 总账（在 `c1ffced` 里）记录过其中两份报告的原始
SHA-256：`r18-B-GATES-REPORT-20260908.md` 和 `r19-M2-COMBINED-PROBE-REPORT-20260909.md`，与下表的“原 SHA-256”一致。

| 文件 | 处数 | 原 SHA-256（前 16 位） | 新 SHA-256（前 16 位） |
|---|---|---|---|
| `docs/FORENSICS-F3-why-no-progress.md` | 1 | `53f83f51ade703cf` | `3893c1106bdb6ba8` |
| `docs/LOOT-ECONOMY-R21.md` | 1 | `842303ef2b11d23d` | `c347fb933a913961` |
| `docs/RESOURCE-CALIBRATION-R19.md` | 1 | `51c89b5e5a7e58f7` | `81caf916bb048d75` |
| `docs/RESOURCE-PROTOCOL-L2.md` | 6 | `e4cb892b636ff67f` | `3dd8293e7716b360` |
| `docs/SUSTAIN-PROTOCOL-R20.md` | 3 | `56fa7b8709342982` | `770cb05830836584` |
| `tests/test_completion_migration.py` | 1 | `9b75b5ef1ead230b` | `8ce37eae900151bc` |
| `tests/test_completion_r18c.py` | 1 | `446628e0050aa64f` | `8c005912a704007a` |
| `tests/test_hunt_scope.py` | 1 | `4c9ddefc777c3853` | `f8a69f2e95d9939c` |
| `tests/test_r18b6_training_wiring.py` | 1 | `0547c3d62e349079` | `d93905b3fe31a08b` |
| `tests/test_training_completion_diagnostics.py` | 1 | `5ca171f6494b84b3` | `7c92b43df0dfeaf8` |
| `train/runs/r10-staging/r16-deploy-arm.json` | 1 | `58d1192678d159a8` | `7ee06d6809109deb` |
| `train/runs/r10-staging/r17-0/G0-0a-REPORT.md` | 24 | `e90fb598275d4bb8` | `14badb45a00d3387` |
| `train/runs/r17-DIRECTION-PANEL-20260902.md` | 1 | `e9d997bb7775d4cd` | `44f94f7d67a37534` |
| `train/runs/r18-B-GATES-REPORT-20260908.md` | 8 | `12696c15f9f7ab9e` | `2ac104ed9766a775` |
| `train/runs/r19-A-PREREG-DRAFT-REV2-20260910.md` | 1 | `652e2d722d487a3e` | `84ab58508d7a1165` |
| `train/runs/r19-A1-PROBE-REPORT-20260908.md` | 31 | `daab8027a058e862` | `da8d1d78d702dc74` |
| `train/runs/r19-B1-PROBE-REPORT-20260908.md` | 23 | `d71767792b344065` | `4888dc15f8a15d55` |
| `train/runs/r19-GOLD-GRAB-PROBE-REPORT-20260909.md` | 23 | `ba4989ef73d16426` | `3b179cb142964a0a` |
| `train/runs/r19-M1-COMBINED-PROBE-REPORT-20260909.md` | 40 | `9c51a020c0dd2bee` | `2bc41aa3b940dfff` |
| `train/runs/r19-M2-COMBINED-PROBE-REPORT-20260909.md` | 51 | `90fded7597367c33` | `e68a2c8f32afbf00` |
| `train/runs/r19-SPEND-V2-PROBE-REPORT-20260909.md` | 36 | `99f3fd1dc8064bf9` | `04be9b383c02589c` |
| `train/runs/r19-STEP2-DESIGN-PANEL-20260910.md` | 1 | `22089d88c67cd261` | `dd2d07a186167f97` |
| `train/runs/r19-SWEEP-V2-PROBE-REPORT-20260909.md` | 21 | `4e275877c4ec46c9` | `e96f7d5e836a8d85` |
| `train/runs/r19-TEACHER-ROUND2-PROBE-REPORT-20260911.md` | 10 | `d22c8d896a66f4f6` | `dfff20eb62110201` |
| `train/runs/r19-TEACHER-V1-PROBE-REPORT-20260910.md` | 21 | `01445123dbbd99fa` | `9f748f6217f8e1f8` |

## 跟着改的 SHA 钉

代码里校验上面这些文件的 SHA 钉改成了新值，每处上方注明了原值的前 16 位：
`train/run_b1_infra.py`、`run_r7_combat_recovery.py`、`run_r8_certification.py`、`run_v4_combat_recovery.py`、
`run_v25_election.py`、`run_v29_relection.py`、`run_v30_relay.py`、`run_v31_neweducation.py`、
`run_v32_sovereign.py`、`run_v33_content.py`。四张 MODEL_CARD 同时写明新旧两个值。

没有改的：

- `train/run_r9_reeducation.py` 里 R8 终考 `worker.zip` 的 SHA 钉。它校验的是 `train/runs/` 下的本地副本，不是仓库里的这个文件。
- 历史文书（PREREG、`docs/DESIGN.md`、台账 `gate_ledger.jsonl`、排行榜可见行）里记下的原始 SHA-256。它们描述的是当时的文件。
- 已结案战役留在本地 `train/runs/` 下的证据（从未入库）记录的是原始 SHA-256。重新续跑这些战役前要先知道这一点。

## 署名

LICENSE 和 `pyproject.toml` 的署名、以及代码和设计文书里批准人一栏的签名，统一改为 `Diabolically-Handsome`：
`docs/DESIGN.md` 3 处，`python/diablogym/options_env.py` 1 处，`train/run_r7_combat_recovery.py` 2 处，
`train/run_r8_certification.py` 1 处。LICENSE 改成标准 MIT 文本，原来的附加说明移到 `NOTICE`。
