# G0-0a 报告:未改动桥的重建管线证明(R17.0 · Team B · 2026-09-02)

**裁定:PASS。** 未改动的 `src/diablogym.cpp` + 引擎 `34c4cfc2`(8 补丁在位)用同一工具链在全新目录
`~/r17_work/B/build-g0a` 重建成功(configure 33 s,build 54 s,零编译错误),重建桥在 **64 种子
2114000-2114063** 旧法卷(认证工人 × r10-econ-mgr × raw-v4 × economy v2)上 rows **逐位等于**冻结锚
`r10-cand-a` 与 `r13-rebake-devil-a`(`rows_sha16 d8ae4204338ee127` == 锚前 64 行;`n_mismatch 0`)。
二进制层面:桥 `.so` 的 `.text/.rodata/.data/.eh_frame/...` 全部逐字节相同,仅 `.dynstr`(RUNPATH 路径)、
`.dynamic`、build-id 不同;引擎 `.so` 唯一语义差异是版本横幅字符串的 git 后缀(来自顶层 diablogym 仓 HEAD),
钉 `-DVERSION_SUFFIX=-Release` 后规范化反汇编逐行相同,残差仅 20 处源码路径字符串。
**工具链漂移 = 0;代码漂移 = 0。128 种子证明留给主会话。**

零训练、零 C++ 改动、零 `src/`/`build/`/已安装扩展/引擎检出写操作(§6 有 mtime/git 证据)。

---

## 1. 已安装桥的身份(只读勘察)

| 项 | 值 |
|---|---|
| Python 加载点 | `python/diablogym/__init__.py:10` `_build_dir = Path(__file__).resolve().parents[2] / "build"`;`:14-27` `spec_from_file_location("_diablogym", build/_diablogym<EXT_SUFFIX>)`。**不在** site-packages;`.venv/.../__editable__.diablogym-0.1.0.pth` 仅含 `/home/laure/AlphaDiablo/diablogym/python` |
| 桥 | `/home/laure/AlphaDiablo/diablogym/build/_diablogym.cpython-312-x86_64-linux-gnu.so` — sha256 `b9be56d3780512d6b0625e1f75f9da92c8e77f67d3965042f58d97290ed6145e`,402744 B,mtime **2026-07-27 07:38:25.200 -0400**,RUNPATH `/home/laure/AlphaDiablo/diablogym/build/engine` |
| 引擎 | `build/engine/liblibdevilutionx_so.so` — sha256 `5da1594bfb3669b774461040669de7a1d59a53666ed43dede1035091f8e4e5d8`,8285336 B,mtime 2026-07-27 07:38:23.656 -0400 |
| 档案一致性 | `r10-cand-a.json meta.runtime.bridge.sha256 = b9be56d3…`,`engine.sha256 = 5da1594b…`(与磁盘一致) |
| 旧 CMakeCache | `DEVILUTIONX_SRC=/tmp/alphadiablo-dev/devilutionX`(已消失);`CMAKE_C_COMPILER=/usr/bin/cc`,`CMAKE_CXX_COMPILER=/usr/bin/c++`;`CMAKE_BUILD_TYPE=Release`;`Unix Makefiles`;cmake 3.28.3;`-flto=auto`(DISABLE_LTO=OFF);Python `/home/laure/AlphaDiablo/.venv/bin/python` 3.12.3;pybind11 3.0.4;`BUILD_TESTING=ON`(取 `libdevilutionx_so` 目标) |
| 引擎检出 | `~/alphadiablo-dev/devilutionX` HEAD `34c4cfc2e733240ac717f23bba2def887c793008`("Add missing <fmt/format.h> includes");工作树 12 个修改文件 |
| 补丁核对 | 复现 build.sh:51-64 的临时 index 法(`read-tree HEAD` + `apply --cached patches/000{1..8}` → `diff --stat` 为空;`ls-files --others` 为空):**工作树 = HEAD + 恰好 8 个登记补丁**,已在位,本次无需(也未)写共享检出 |

识别系统(`train/eval_contract.py`):`runtime_identity :429-459` 记 `bridge.sha256`/`engine.sha256`/`content`/`python_protocol`;
`expected_eval_identity :586-589` 出 `expected_bridge_sha256`/`expected_engine_sha256`;`loaded_engine_binary_path :379-426`
从 `/proc/self/maps` 核对实际映射引擎路径 == `root/build/engine/liblibdevilutionx_so.so`;`bridge_binary_path :461-466`
钉 `root/build/_diablogym<EXT>`;`eval_assembled.py:34-35` `ROOT = parents[1]` 且 `sys.path.insert(0, ROOT/python)`。
→ 仅靠 PYTHONPATH 换不掉桥,必须换 ROOT(§5 的做法)。

## 2. 构建配方(等价 build.sh,只改 `-B`;未跑 build.sh,因其硬编码 `-B build`)

`~/r17_work/B/configure.sh`(verbatim 关键行):
```
cmake -S /home/laure/AlphaDiablo/diablogym -B /home/laure/r17_work/B/build-g0a -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DDEVILUTIONX_SRC=/home/laure/alphadiablo-dev/devilutionX \
  -Dpybind11_DIR=/home/laure/AlphaDiablo/.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 \
  -DPython_EXECUTABLE=/home/laure/AlphaDiablo/.venv/bin/python \
  -DALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE=/home/laure/AlphaDiablo/.venv/bin/python \
  -DALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX=.cpython-312-x86_64-linux-gnu.so \
  -DALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR=/usr/include/python3.12
```
`~/r17_work/B/build.sh`:`nice -n 10 cmake --build build-g0a -j 40 --target devilutionx` → `--target _diablogym`
(与 build.sh:129-130 同两目标);随后 `layout_assets.sh` 复现 build.sh:150-156 Linux 分支
(`cp -a engine/assets/. engine/devilutionx.app/Contents/Resources/`)。

configure.log 摘录:
```
-- Found Python: /home/laure/AlphaDiablo/.venv/bin/python (found version "3.12.3") found components: Interpreter Development.Module
-- Found pybind11: /home/laure/AlphaDiablo/.venv/lib/python3.12/site-packages/pybind11/include (found version "3.0.4")
-- Configuring done (31.8s)  -- Generating done (0.4s)
CONFIGURE_RC=0
executable=/home/laure/AlphaDiablo/.venv/bin/python
include_dirs=/usr/include/python3.12
ext_suffix=.cpython-312-x86_64-linux-gnu.so
real 0m33.380s
```
依赖:FetchContent 由网络重新拉取(tarball:asio 4bcf552f、libsmackerdec 0aaaf8c9、Lua 3ed55a56、magic_enum v0.9.7、
SDL_audiolib cc1bb6af、SDL_image release-2.0.5、SheenBidi v2.9.0、sol2 832ac772、unordered_dense v4.4.0;git:libzt、mpqfs),
与旧 `build/_deps` 同一组 11 个源;系统库(SDL2/libsodium/zlib/bz2/libfmt 9/libpng16/GTest 1.14)与旧缓存同。

CMakeCache 变量对比(去路径/INTERNAL 后 diff):仅 `CMAKE_C(XX)_COMPILER` 的 `FILEPATH→STRING` 类型标签
(命令行显式传入所致)与旧缓存里 9 条 `GTEST_*/GMOCK_*-NOTFOUND` 残留(新配置走 GTestConfig.cmake),**无实质差异**。
`compile_commands.json`:两边各 586 条,路径规范化后 `src/diablogym.cpp` 编译命令逐字相同(`-O3 -DNDEBUG -flto=auto -std=gnu++20 ...`)。

## 3. 构建结果

```
=== build start 2026-09-02T13:55:18-04:00 jobs=40 ===
=== devilutionx target rc=0 2026-09-02T13:55:56-04:00 ===
=== _diablogym target rc=0 2026-09-02T13:56:12-04:00 ===
BUILD_DONE rc1=0 rc2=0
-rwxr-xr-x 402744  2026-09-02 13:56:12  build-g0a/_diablogym.cpython-312-x86_64-linux-gnu.so
-rwxr-xr-x 8285336 2026-09-02 13:56:05  build-g0a/engine/liblibdevilutionx_so.so
-rwxr-xr-x 5742704 2026-09-02 13:55:56  build-g0a/engine/devilutionx
assets laid out: build-g0a/engine/devilutionx.app/Contents/Resources (209 files); symlinks 0; diff -r vs installed Resources: RESOURCES_IDENTICAL
```
编译警告仅来自 `_deps/mpqfs-src/src/mpq_archive.c`(`-Waddress-of-packed-member`、`-Wpedantic` typedef 重定义),与上游一致;零错误。
工具链:`c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`,`GNU ld 2.42`,`cmake 3.28.3`。

## 4. 二进制对比(重建 vs 已安装)

| 文件 | 已安装 sha256 | 重建 sha256 | 大小 |
|---|---|---|---|
| 桥 `_diablogym…so` | `b9be56d3780512d6…6145e` | `8a33725668fe9ff460f58989cbb84bd9a36ea07cedaf9f5dd5c822d84d19b293` | 402744 = 402744 |
| 引擎 `liblibdevilutionx_so.so` | `5da1594bfb3669b7…e5d8` | `3324e294463689c173c002bed6861bc76cd39e0dcac36f91f79003e49c4431f0` | 8285336 = 8285336 |

**桥(逐节 `objcopy -j` + `cmp`)**:`.text`(279490 B,sha256 `4909fd22857fe52f…`)、`.rodata`(21787)、`.data`、`.data.rel.ro`、
`.eh_frame`、`.eh_frame_hdr`、`.gcc_except_table`、`.dynsym`、`.gnu.hash`、`.gnu.version(_r)`、`.rela.dyn/.plt`、`.plt*`、`.got*`、
`.init/.fini(_array)`、`.note.gnu.property` **全部 identical**;仅 `.dynstr`(13939 vs 13932 B:RUNPATH 字符串
`/home/laure/AlphaDiablo/diablogym/build/engine` → `/home/laure/r17_work/B/build-g0a/engine`)、`.dynamic`(同因)、
`.note.gnu.build-id`(2610eb3b… → a71da7d3…)不同。规范化反汇编 diff = 0 行(仅文件名头)。
`nm -D` 464/464 动态符号逐条相同(`BRIDGE_DYNSYM_IDENTICAL`,`BRIDGE_EXPORTS_IDENTICAL`,54 个已定义导出);NEEDED 列表相同。

**引擎**:`nm -D` 7360/7360 相同(`ENGINE_DYNSYM_IDENTICAL`,6831 已定义);NEEDED 相同。
`.text` 4867970 → 4867986 B(+16),`.rodata` 441480 → 441672 B(+192)。规范化反汇编(去地址/立即数/符号)diff:
**1,137,504 行中 24 行 / 9 hunk**,全部在同一函数——内联拷贝版本横幅字符串的 `movabs/movups/movb` 序列 + 一处对齐 nop:
```
< 1.6.0-dev-Release                      (已安装)
> 1.6.0-dev-Release-3d9aafa              (重建;3d9aafa = ~/AlphaDiablo/diablogym 仓 HEAD)
```
成因:引擎 `CMakeLists.txt:57-67` `if(NOT VERSION_SUFFIX)` → `get_git_commit_hash`,而 `CMake/functions/git.cmake:3-4`
在 **`${CMAKE_SOURCE_DIR}`**(= 顶层 diablogym 项目,非引擎目录)执行 `git log -1 --format=%h`。2026-07-27 构建时
diablogym/.git 不存在(docs/OPS-windows-feasibility.md「缺失:…diablogym/.git」)故无后缀;现在存在故带 `-3d9aafa`。
`.rodata` 字符串路径规范化后仅剩该横幅两处不同;源码路径字符串两边各 20 处(`/tmp/alphadiablo-dev/…` vs
`/home/laure/alphadiablo-dev/…`,`__FILE__`/断言文本),长度 +7 解释 `.rodata` 其余增量。

**补充(§7)钉 `-DVERSION_SUFFIX=-Release` 的第二次构建**(`build-g0a-pin`,未用于行为证明):引擎 `.text` 大小回到 4867970,
横幅恰为 `1.6.0-dev-Release`,**规范化反汇编与已安装逐行相同(diff 仅文件名头)**,`.rodata` 规范化字符串 diff 为空,
`.data/.init_array` 逐字节相同;`.text/.eh_frame/.gcc_except_table/.data.rel.ro` 的剩余字节差全部来自 `.rodata` 中 20 处
路径字符串长度变化(+160 B 含填充)引起的 RIP 相对位移/指针平移。桥在 pin 构建中同样 `.text/.rodata` sha 相同。
→ **除路径与顶层 git 横幅外,编译产物无差异;工具链零漂移。**

## 5. 行为证明(不安装、不动 build/)

做法:镜像 ROOT `~/r17_work/B/root/{python,train/*.py}`(rsync 自仓库;9 个协议文件 sha 与仓库逐一相同),
`root/build -> ../build-g0a` 符号链接;从 `root` 目录以 `PYTHONPATH=root/python` 运行 `train/eval_assembled.py`,
输出档案落在 `root/train/runs/eval-assembled/`(**不进** 仓库 `train/runs/eval-assembled/`)。

加载路径核验(verbatim):
```
$ cd ~/r17_work/B && PYTHONPATH=/home/laure/r17_work/B/root/python .venv/bin/python -c "import diablogym, sys; print(diablogym.__file__); print(diablogym.bridge.__file__)"
/home/laure/r17_work/B/root/python/diablogym/__init__.py
/home/laure/r17_work/B/root/build/_diablogym.cpython-312-x86_64-linux-gnu.so
identity_check.py: bridge.path=/home/laure/r17_work/B/build-g0a/_diablogym…so sha256 8a337256…;engine.path=/home/laure/r17_work/B/build-g0a/engine/liblibdevilutionx_so.so sha256 3324e294…
  assets sha256 661adc715d6b4ccdb3e87cb6373b745e134a08677af3e78d82e396ec67ff155c file_count 209(== 锚)
  python_protocol sha256 14e2bbc7790d7c552ee13030b48fd6a35695a5dff603664bd5009033769a0e4b(== r13-rebake-devil-a 当前冻结 bundle)
  loaded_engine_binary_path OK -> /home/laure/r17_work/B/build-g0a/engine/liblibdevilutionx_so.so
  mapped native images = [build-g0a/_diablogym…so, build-g0a/engine/liblibdevilutionx_so.so]   (无仓库 build/ 映像)
  identity stable after import = True
```
评测命令(两次,`run_eval.sh`):
```
train/eval_assembled.py --worker /home/laure/AlphaDiablo/diablogym/train/runs/r9-reeducation/staging/worker.zip \
  --manager-npz /home/laure/AlphaDiablo/diablogym/train/runs/r10-econ-mgr/policy.npz \
  --manager-policy-observation-view raw-v4 --reward-economy v2 --seeds 2114000-2114007 --tag r17-g0a-rebuilt-8
  … --seeds 2114008-2114063 --tag r17-g0a-rebuilt-56
```
结果(verbatim):
```
eval8 : 13:57:18 → 13:57:29  EVAL_RC=0  已存并复验 …/r17-g0a-rebuilt-8.json
        r17-g0a-rebuilt-8: ret 146.6 (med 139.94) died 8/8 depth_med 2.5 | R4: 换层率 0.0000 override 0.0000 cap 0.1304 farmτ̄ 0.0
eval56: 13:58:17 → 13:59:26  EVAL_RC=0  已存并复验 …/r17-g0a-rebuilt-56.json
        r17-g0a-rebuilt-56: ret 150.6 (med 130.91) died 55/56 depth_med 2.0 | R4: 换层率 0.0000 override 0.0163 cap 0.0391 farmτ̄ 0.0
```
逐位对照(`compare_rows.py`/`compare64.py`,rows_sha16 口径 = run_r13_rebake.py:45-48):

| 段 | 新 rows_sha16 | 锚切片 rows_sha16 | n_mismatch | 裁定 |
|---|---|---|---|---|
| 2114000-2114007(8) | `8c51e596339ef7c7` | `8c51e596339ef7c7` | 0 | PASS(vs r10-cand-a 与 r13-rebake-devil-a) |
| 2114008-2114063(56) | `80b7c098505b8a71` | `80b7c098505b8a71` | 0 | PASS |
| **2114000-2114063(64 合并)** | **`d8ae4204338ee127`** | **`d8ae4204338ee127`** | **0** | **PASS**(vs r10-cand-a 与 r13-rebake-devil-a;锚全 128 行 `00f4672db9c3ac69`) |

聚合复核(新 64 vs 锚前 64):`(ret_mean, died, depth_med, kills_sum, micro_steps_sum) = (150.1367, 63, 2.0, 900, 41362)` 两边相同。
档案身份:worker sha `2837288d…`、manager sha `ea506f4f…`、DIABDAT sha `63fb47d9…`、assets sha `661adc71…` 与锚一致;
bridge/engine sha 按预期不同(§4);`validate_eval_archive` 写后复验通过(「已存并复验」)。

## 6. 未触碰证据

```
files under ~/AlphaDiablo/diablogym/build newer than marker: 0
files under ~/alphadiablo-dev/devilutionX (excl .git) newer than marker: 0
git status --porcelain before == after (12 modified = 8 patches)   ENGINE_STATUS_UNCHANGED
installed: b9be56d3…  build/_diablogym…so (mtime 2026-07-27 07:38:25 不变);5da1594b… build/engine/liblibdevilutionx_so.so 不变
```
(引擎 `.git` 目录 mtime 变化仅为本队 `git status` 的 index.lock 创建/删除;`.git/index` mtime 13:21 早于本队开工。)
仓库 `train/runs/eval-assembled/` 未新增/修改任何文件;新增仅 `train/runs/r10-staging/r17-0/G0-0a-REPORT.md`(本文)与台账一行。
未消耗任何处女种子(2114000-2114063 属已焚 2_114 池)。

## 7. 备注与建议(供 R17.1 桥施工)

1. **重建管线已验证可用**:`bootstrap.sh` 默认仍指向 `${TMPDIR:-/tmp}/alphadiablo-dev`(/tmp 会被清),而真实检出在 `~/alphadiablo-dev`;
   build.sh 需 `DEVILUTIONX_SRC=~/alphadiablo-dev/devilutionX ./build.sh`(其 HEAD/补丁漂移审计会通过:本报告 §1 已复现该审计为空)。
   注意 build.sh 硬编码 `-B build`,会覆盖已安装桥——R17.1 动 C++ 前先归档 `build/_diablogym…so` + `build/engine/liblibdevilutionx_so.so`
   (sha b9be56d3 / 5da1594b)。
2. **档案身份对"同码重建"的敏感度**:`expected_bridge_sha256/expected_engine_sha256` 会随 RUNPATH/build-id/横幅/源码路径变化,
   即使代码零改动。这是本次(以及 R17.1 之后每次)`.so` sha 轮换的预期;**尺子应是 rows 逐位(4×128 重烤)**,不是 .so sha。
   若要横幅稳定,配置时加 `-DVERSION_SUFFIX=-Release`;源码路径字符串只能靠同路径检出消除(不建议为此改动)。
3. 64 种子用时 ≈ 80 s;主会话补 2114064-2114127 及 4×128 重烤成本极低(可直接用 `~/r17_work/B/run_eval.sh <lo-hi> <tag>`,
   或把 `root/build` 软链指向任意候选 build 目录)。
4. 产物清单(`~/r17_work/B/`):`configure.sh/log`、`build.sh/log`、`layout_assets.sh`、`identity_check.py/log`、`run_eval.sh`、
   `eval8.log`、`eval56.log`、`nm_diff.sh/log`、`nm/*.asm|*.diff`、`compare_rows.py`、`compare64.py`、`compare*.json`、
   `build-g0a/`(474 MB)、`build-g0a-pin/`(463 MB,补充实验)、`root/`(镜像 ROOT + 两份档案)。
