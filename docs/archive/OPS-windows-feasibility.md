# OPS 备忘:Windows 新机上复活训练/评测的可行性勘察

2026-07-27 值守勘察(纯只读)。结论先行:**推荐 WSL2 路线,预计半天内可复活;
原生 Windows 移植不推荐;无论哪条路,本机运行时都是一个"新世界",与 Mac 档案的
位级锚全部失效——这是登记项,不是障碍。**

## 现状盘点

- 在手:全部源码(src/diablogym.cpp、python/、train/)、全部权重与档案、
  GOG 安装包 `setup_diablo_1.09_hellfire_v4_(78466).exe`(850MB,根目录,
  可用 innoextract 抽出 DIABDAT.MPQ);引擎钉死版本 ENGINE_REF=34c4cfc2(bootstrap.sh)。
- 缺失:.venv(Mac 二进制)、build/(引擎 dylib+桥 .so)、diablogym/.git。

## 三条路线

1. **WSL2(推荐)**:Ubuntu 下 fcntl 原生可用(train_ppo/eval_contract/env/
   三个 R 系驱动器共 6 文件 import fcntl,零改动);bootstrap.sh/build.sh 仅需
   小改(Homebrew→apt,sysctl -n hw.physicalcpu→nproc,.app 资产路径→Linux 布局,
   数据目录 ~/Library/...→~/.local/share/diasurgical/devilution);DevilutionX
   官方支持 Linux 构建;torch/sb3 CPU wheel 齐全。预计工作量:脚本改造 1-2 小时
   + 编译 + 冒烟。风险:WSL2 IO 性能(档案在 NTFS 侧时 9p 慢,应把整树搬进
   ext4 侧)。
2. **原生 Windows(不推荐)**:fcntl 无对应物,需给 6 个核心文件写锁 shim——
   其中 eval_contract.py/env.py 属评测协议束,手术面大;引擎需 vcpkg/MSVC 工具链;
   bash 驱动脚本(launch_case.sh)需重写。工作量数天,且协议文件改动会使
   protocol sha 变更叠加平台变更,法证面最脏。
3. **回 Mac 点火(零工作量)**:若 Mac 仍可用,R7 直接在原世界发车,历史位级锚
   全部有效。唯一条件是机器在手。

## 新世界登记(WSL2/原生共同适用)

- 引擎/桥二进制 sha、runtime_versions(python 3.13.5→新版本、numpy/torch 版本)
  必变 → CASE_RUNTIME 五 sha 全新;Mac 档案的 REF_BITEQ(113.0/140.9)在本机
  **预期不可复现**(跨编译器浮点),B1 残余⑥早有此设防:破防即运维事实,非科学
  异常。
- **R7 的科学效力不受影响**:其 final 门是同运行时内 baseline/candidate 配对
  (R6 fresh-ledger 先例:基线在开池时现场重评),处女池 2,110,000-2,129,999
  未被任何世界触碰。需要的只是:发车预检把"新运行时世界"作为案级事实落账,
  且 baseline 参照一律本机现评、禁与 Mac 档案跨世界配对。
- 王座/经理权重(npz/zip)是纯数值工件,跨平台可用;BC 件同理(回执按新 impl
  sha 刷新,循 B1 勘正 2 先例)。

## 执行结果(2026-07-27,总设计师口令「方案1」,当日完成)

- 环境:WSL2 Ubuntu 24.04,64 核/62GB,~12k tick/s(约为 MacBook 的引擎口径同量级)。
- 全链路复活:MPQ(sha 与档案钉死值同一)→ 引擎 @34c4cfc2 + **8 补丁**(新增
  0008-headless-hardware-cursor:上游 SetHardwareCursor 缺 HeadlessMode 守卫,
  Linux 下 SDL 无窗口建光标段错误;0001-0007 零冲突)→ 桥构建 → 冒烟三件套全绿
  → 测试套件 **776 过 / 13 败**(余账见下)。
- 钉死教训:torch 必须用 **+cpu 轮子**(cu130 轮在 WSL 下 import 间歇段错误;
  本机虽有 CUDA GPU 亦不用);venv 须在 diablogym/.venv(软链 ../.venv,Mac 原状);
  版本门三处(train_ppo/eval_contract 采集门+校验门)已按公开版本段比对修订。
- **协议时代认知(最重要)**:工作树是 07-25 傍晚未提交的协议演进(rev22/R7 语义),
  Mac 全部档案/demos 系 R6 时代(≤07-25 00:37)实现产物,跨版本不可比——一切旧参照
  数字失效,新世界数字只能 R7 案内自比。wsl-world-king-{h,m29}-7000 两档已登记
  (rev22 语义,ret 49.5/died 28;两经理逐种子同分,悬案待查)。
- 余账 13 败 = 三件裁断项:①test_run_v4(rev10 归档 launcher vs 07-25 抬到 rev11
  的 train_ppo,修复两路线待总设计师裁);②v33 W-PIN 两份 AUDIT 文档只存在于 Mac
  未 push 的 git(GBK 转码丢文件;GitHub 远端停在 07-12,已从其恢复 3 份 07-12 前
  文档);③content_case_aux 1-ULP 直通残差(代码侧/测试侧两案待裁)。
- Mac 侧待办:接到 Mac 时先 push 全部未提交/未推送 commit(≥12 个,含两份 AUDIT
  文档与 07-24/25 的协议演进本身——当前 Windows/WSL 树无 git,历史全在 Mac)。

## 建议执行序(若批 WSL2)

①装 WSL2+Ubuntu,整树 rsync 进 ext4;②innoextract 抽 DIABDAT.MPQ 至
~/.local/share/diasurgical/devilution/;③bootstrap.sh/build.sh Linux 化
(diff 面≈20 行);④.venv 重建(pyproject [train,build]);⑤冒烟三件套
(smoke_random_agent/确定性/种子分化);⑥本机重评 king+H/king+M29 @7000 各一发
作新世界基线登记(不与 Mac 档案配对);⑦R7 按 rev21(若批)发车。
