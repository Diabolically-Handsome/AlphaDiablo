# 文档目录

2026-09-23 整理后，文档按用途分到下面这些目录。

| 目录 | 内容 |
|---|---|
| [`design/`](design/) | 设计笔记。主文档是 [`design/DESIGN.md`](design/DESIGN.md)（二十轮迭代、十七课教训） |
| [`prereg/`](prereg/) | 各案预注册（`PREREG-*.md`，v23 到 R9、B1、G1、E-fix、定锚/重锚） |
| [`forensics/`](forensics/) | 法证、尸检和档案审计报告 |
| [`protocol/`](protocol/) | R19–R21 的协议规格，以及案级驱动器的值守细则 `OPS-launcher.md` |
| [`rounds/`](rounds/) | R9–R19 的轮次文书：预注册、发车令、判决书、探针报告、评审记录 |
| [`archive/`](archive/) | 内部备忘，以及 [`REDACTIONS-2026-09-23.md`](archive/REDACTIONS-2026-09-23.md)（本次去除个人信息的记录） |
| [`assets/`](assets/) | 图表和基线档，代码按路径读取，不要搬动 |
| [`milestones/`](milestones/) | 里程碑证据包，由各自的 `manifest.json` 和 `verify.py` 封存，内容不能改动 |

## 旧路径怎么换算

2026-09-23 之前写成的文书正文里用的是旧路径。这些历史文书是原样搬过来的，因为很多文书的 SHA-256
被别的文书或台账记录过，改了就对不上。唯一的例外是本机路径里的用户名：`docs/` 下有 17 份文书把用户名文件夹换成了 `user`，
其余字节不变，新旧 SHA-256 见 [`archive/REDACTIONS-2026-09-23.md`](archive/REDACTIONS-2026-09-23.md)。按下表把旧路径换成新路径：

| 旧路径 | 新路径 |
|---|---|
| `docs/PREREG-*.md` | `docs/prereg/` |
| `docs/DESIGN.md`、`docs/DESIGN-*.md` | `docs/design/` |
| `docs/FORENSICS-*.md`、`docs/AUTOPSY-*.md`、`docs/AUDIT-*.md` | `docs/forensics/` |
| `docs/*-R19.md`、`docs/*-R20.md`、`docs/*-R21.md`、`docs/RESOURCE-PROTOCOL-L2.md`、`docs/OPS-launcher.md` | `docs/protocol/` |
| `docs/呈报-内容案发车审阅.md`、`docs/COURT-wf_5fa772de.md`、`docs/UPSTREAM-PR-候选清单.md`、`docs/ROADMAP-完全体.md`、`docs/OPS-windows-feasibility.md` | `docs/archive/` |
| `train/runs/r*.md`（轮次文书） | `docs/rounds/` |

代码、README、MODEL_CARD 和 `design/DESIGN.md` 里的路径已经改成新路径。

下面 10 份草稿和对应的 FROZEN 版逐字节相同，已经删掉。文书里提到草稿时，看同一轮的 FROZEN 版：
`r10-PREREG-DRAFT-v0.1-20260827`、`r12-PREREG-DRAFT-v0.1-20260830`、`r13-PREREG-DRAFT-v0.1-20260830`、
`r17-PREREG-DRAFT-20260906`、`r18-A-PREREG-20260906`、`r18-B-PREREG-DRAFT-20260907`、`r18-C-PREREG-20260907`、
`r18-DE-PREREG-20260907`、`r18-F-PREREG-20260907`、`r18-G-PREREG-20260907`。

## 不在主干上的文件

下面这些 2026-09-23 起不在主干上。它们还在 git 历史里，最后一次出现在提交 `c1ffced`
（标签 `skeleton-king-first-kill-20260921`）。文书里引用它们的地方，例如 `r13_ledger.jsonl` 的行号，请到那个提交里查。

- 运行日志（`*.log`）和原始运行输出：`train/runs/r10-staging/r17-0/` 下的原始 JSON（包括 `r17-0-summary.json`），
  `train/runs/r10-staging/` 下 8 份探针汇总 `*-SUMMARY.json`，`train/runs/eval-assembled/` 里代码不读的 40 份原始评测档案，
  `train/runs/probe-gear-value/` 的探针输出。
- 原始事件台账：`train/runs/r10-staging/` 下的 r10、r12、r13 三本总账，
  `train/models/v24-worker-leg7/gate_ledger.jsonl`，`train/models/v23-worker-1M/sentinel.jsonl`。
- 两份值班和交接日志：仓库根目录的 2026-07 值班日志，`train/runs/r17-0-HANDOFF-20260902.md`。
- 只跑过一次的脚本：`train/runs/r10-staging/` 下 48 个，`train/` 下 8 个，`tests/` 下 3 个没人调用的探针。

`train/runs/` 下仍然入库的少量文件，都是代码或测试要读的回执，或者是人写的报告：

- 6 份评测锚档：`eval-assembled/` 下的 4 份 `r2-*.json`、`v24-G3-leg7.json` 和 `v26-G3-leg6.json`（v25 到 v32 的驱动和
  `tests/test_active_reference_contracts.py` 会读）；
- `probe-zeroflip/report.json`（`run_v32_sovereign.py` 会读）；
- `recal-g1/` 和 `v32/` 的 `gate_ledger.jsonl`（`run_v33_content.py` 的 W-PIN 和 `tests/test_content_case_driver.py` 会读）；
- `r10-staging/probe_r17_deployment.py`（`tests/test_resource_protocol.py` 会读），以及它记录源码指纹时要读的
  `r10-staging/probe_r15_deployment.py`；
- `r10-staging/r16-deploy-arm.json`（`probe_r17_deployment.py` 做逐位回归时比对的部署行）；
- 人写的报告：`r10-staging/r11-SEEDS.md` 和 `r10-staging/r17-0/G0-0a-REPORT.md`。
