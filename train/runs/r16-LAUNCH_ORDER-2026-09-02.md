# R16 发射令(2026-09-02)

依据:r16-PREREG-FROZEN-20260902.md(sha256 见台账 prereg_frozen_r16);主席批文
2026-09-01「没问题 立刻进行修复!」。

## 发射前置(全部满足方可点火)
1. `train/runs/r10-staging/r16_gauntlet.sh` 在最终字节上 PASS:py_compile ×8、
   全套件、4×128 旧法锚逐位重烤(r13-rebake-* rows sha == r10-cand/r10-anchor)。
2. `run_r16_anchors.py` 六卷新法锚铸完(r16-anchor-{xdevil,xm29,xm29full}-{a,b},
   各 128 种子,meta.protocol.r16_environment 立碑)。
3. 冻结后协议源码(PROTOCOL_SOURCE_FILES)零改动;如有改动,前置 1、2 作废重来。

## 点火命令(挂载后台,不用 `&`)
```
cd ~/AlphaDiablo/diablogym && .venv/bin/python train/runs/r10-staging/run_r16_arm_a.py 325632 0.1 2.0 \
  > train/runs/r10-staging/r16-arm-a.log 2>&1
```
链:训练 T=325,632(≈5 个检查点)→ 门 D 探针(fail-closed)→ 六卷 R16 考卷 →
`analyze_arm_gates.py r16-arm-a-constitution r16-anchor` → 部署探针 v3(臂 / 认证工人,
16 种子 2114000-2114015,6000 拍,sample,readiness-v3 形态)。

## 判据
- 主指标(部署探针,臂 vs 新法部署锚):存活率、alive 分层 clvl/AC/kills、xp_per_1k、
  depth_hist;臂须在存活率不劣(McNemar 同款口径)的前提下 clvl/AC/kills 任一显著优于锚,
  或深度均值优于锚且存活率不劣。
- 四门(对 r16-anchor):A 存活优越(UCB<0 且 died≤110);B 深度保持(LCB>−0.10);
  C 反遗忘(ret/kills ≥0.95×);D 行为同一性(argmax 探针 <99%)。
- 台账 r13_ledger.jsonl 全程留痕;判决书 r16-VERDICT-<date>.md 新立文件。

## 禁令
处女池 2_117-119 / 2_126-129 零接触;2_116 仅凭主席明令;禁烧段种子不消耗;
考卷 3000 拍协议不动;冻结后配方改动须以修正案另立文件。
