# R19 spend-v2 报告：`resource_purchase_mode = "full-v2"`（把钱花出去）

- 日期：2026-09-08（夜）
- 实现者标签：`r19-spend-v2-implementer`
- 工作树：`/home/laure/r17_work/r19/spend-tree`（Python only；`build -> /home/laure/r17_work/r17-1/build-res`，与主树同一条活桥）
- 主树 `/home/laure/AlphaDiablo/diablogym` 未被修改（唯一例外：账本 append）。
  校验：`find <主树>/python <主树>/src <主树>/tests -newermt "2026-09-08 20:00" -name '*.py'` 为空。
- 补丁：`/home/laure/r17_work/r19/spend2.patch`（11 个改动文件 + 1 个新测试文件，+966 / −49 行）
- C++ / 引擎：**一字未改**。`src/resource_protocol.hpp` 只被**读**，用来确认哪些原生购买路径存在。

---

## 零、结论先说

1. **回归通过。** 法条关闭（`resource_purchase_mode = "full"`，即认证控制行的 16 键覆盖）时，
   本工作树在 pool 2_133 的 48 个种子上复现
   `rows_sha_v3 = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，
   与 `/home/laure/r17_work/r18/merge-probe/ctl-v1-world-193557.json` **逐位相同**，
   `arm_stats` 也逐字段相同（alive 20、l2 34、l3 4、hazard 0.222、
   belt/AC 直方图、gold_final median 78、weapon_upgrades 29 / 6620 金）。运行时错误 0。
2. **药水停在 4 不是 bug，是另一条法的阈值。** 认证臂跑的是
   `SustainLootService → SustainResourceService`，它的 `minimum_potions` 相位买到
   **`native_readiness.required_belt_heals`（原生 `RequiredResourceBeltHeals = 4`）** 就收手；
   传统 `ResourceService` 那条 `< 8` 的行程**根本到不了**。腰带 5/6/7 是捡来的药水叠在这 4 瓶之上。
   **不是 Pepin 缺货**：ON 臂 39 次行程里 `stock_limited` 恒为 0，`potion_stop`
   只有 `belt_full`(25) 与 `unaffordable`(14)，**没有一次** `stock_empty` 或 `no_free_slot`。
3. **法条开着以后钱确实花出去了**，而且**上限不再是脚本**：
   富启动臂（每局 reset 后 +3000 金）里 38 次行程**全部**以
   `potion_stop = belt_full` 且 `armor_stop = no_upgrade` 结束——腰带买满 8、
   铁匠柜台上再没有一件能提高护甲的货。**格里斯沃德的存货成了天花板，脚本不是。**
4. **代价要说清楚：正常金钱下，护甲把武器挤掉了。** ON 臂武器升级 29 → 4 次
   （6620 → 710 金），首降伤害分布从 `{7:11, 9:18, 11:5}` 退到 `{7:19, 9:16}`；
   配对 McNemar `saved 3 / lost 6 / net −3`，单侧 95% UCB `0.16424`——**排除不了变坏**。
   富臂（钱够买全部）则 `saved 11 / lost 8 / net +3`，UCB `0.08614`，L2 hazard 0.222 → 0.1599。

---

## 一、先诊断：为什么 "full" 的腰带停在 4–7

`python/diablogym/resource_sustain.py`，`SustainResourceService._command`，相位 `minimum_potions`：

```python
if int(ready["belt_heals"]) < int(ready["required_belt_heals"]) and int(ready["belt_free_slots"]) > 0:
```

- `required_belt_heals` 由原生导出：`src/resource_protocol.hpp:514`
  `ready["required_belt_heals"] = RequiredResourceBeltHeals;`，而
  `src/resource_protocol.hpp:67` `constexpr int RequiredResourceBeltHeals = 4;`。
  同一个 4 也是六条件备战法里 `potions` 那一条的门槛（hpp:129）。
- 传统 `ResourceService` 的行程确实写着 8
  （`python/diablogym/resource_protocol.py:488` 原文 `< 8`），
  但 `SustainResourceService._command` 在
  `if self.phase not in ("survey_smith", "survey_healer", "joint_gear", "minimum_potions")`
  这一句之前就把相位链**截胡**了：认证臂永远走不到那条 `< 8`。
- 所以 **"full" 的药水目标就是 4**，不是 8。腰带 4（14 局）/5（7）/6（7）/7（4）
  里，4 是买的，5–7 是捡的。

**第二道墙是腰带空位，不是货。** 两点证据，都不是猜的：

- 引擎侧：`items.cpp SpawnHealer` 把 `IDI_HEAL` 钉在 `HealerItems[0]`，单人模式下
  这两个 pinned 行**不因购买而消失**（这正是 R18-K2b 把 `HEALER_MIN_HEAL_PRICE = 50`
  当作“Pepin 永远报得出的最便宜即时治疗”的依据，见
  `python/diablogym/resource_weapon_upgrade.py` 模块文档）。
- 探针侧：ON 臂 39 次 v2 行程，`healer_rows_seen` 每次都 ≥ 4，
  `stock_limited` 恒 false，`potion_stop` 直方图 `{belt_full: 25, unaffordable: 14}`。
  **一次都没有**因为“Pepin 没货”或“腰带没空位”停下。

一句话：**钱和腰带容量是墙，货不是**——与主席的裁定一致，缺陷在采购脚本。

---

## 二、法条与登记（"full" 逐字节不动）

新值是既有开关 `resource_purchase_mode` 的第六个词：
`("none", "heal", "potions", "armor", "full", "full-v2")`。

登记处（每一处原来只认 `"full"` 的判词都改成认这一族，
`FULL_PURCHASE_MODES = ("full", "full-v2")` / `is_full_purchase_mode()`）：

| 文件 | 内容 |
|---|---|
| `python/diablogym/resource_protocol.py` | `RESOURCE_PURCHASE_MODES` 加词；`SPEND_V2_MODE` / `FULL_PURCHASE_MODES` / `BELT_TARGET_HEALS=8` / 两个 fail-closed 阀门；`validate_ordinary_armor_scope`；传统行程 4 处 `mode` 判词；`ResourceService` 的 spend-v2 账页与遥测出口 |
| `python/diablogym/resource_sustain.py` | `validate_service_policy`；新函数 `plan_armor_upgrade`；新相位 `v2_potions` / `v2_armor` |
| `python/diablogym/resource_sustain_loot.py` | `__post_init__` 的 `full` 判词 |
| `python/diablogym/resource_weapon_upgrade.py` | `validate_weapon_upgrade`；`WeaponUpgradeService.purchase_mode`；`_settle` 的续买；`book_weapon_purchase` 把武器记进同一张行程账页；`SPEND_V2_WEAPON_CAP` |
| `python/diablogym/env.py` | 战利品经济 / smith-v1 的构造器判词；`unequip` 与 `repair` 两条 script 命令的 `full` 限制 |
| `python/diablogym/options_env.py` | completion-l2 时钟判词；把 `purchase_mode` 传给 `WeaponUpgradeService` |
| `python/diablogym/worker_env.py` | `earned-dive-suffix-v1` 判词 |
| `train/eval_contract.py` | `_RESOURCE_SERVICE_STAGES` 新增 `full-v2` 配方；`validate_resource_service_config`；`resource_service_recipe` 的 `potion_target`；`validate_r16_environment` 的枚举（`full-v2` 是**非缺省值**，必须写进档案身份）与 sustain / weapon-upgrade 两处 `== "full"` |
| `train/eval_assembled.py` | CLI `choices` 与 weapon-upgrade 判词 |
| `train/train_ppo.py` | CLI `choices`、两处 `_require(... in (...))`、两处 weapon-upgrade 判词、warm-start 与 make_env 判词 |
| `train/runs/r10-staging/probe_r17_deployment.py` | 每行资源遥测新增 `spend_v2`（**非 full-v2 时恒为 `None`**） |

**其他法条的阈值一个没动。** 特别是：`required_belt_heals` 仍是 4，
`meets_armor_gate` 仍是原生判词，`WEAPON_UPGRADE_MIN_GAIN` 仍是 1，
`RESOURCE_SERVICE_CAP` / `SUSTAIN_SERVICE_CAP` / `WEAPON_UPGRADE_SERVICE_CAP` 原样。

**"full" 的逐字节承诺怎么保证。** 每一条新分支的守卫都是
`mode == SPEND_V2_MODE`（或 `purchase_mode == "full-v2"`），**不是**族成员资格；
族成员资格只出现在 validator 里（那里 `full` 本来就通过）。遥测同理：
`spend_v2` 这个键只在 `mode == "full-v2"` 时才出现在
`ResourceService.telemetry()` 和探针行里，`weapon_repeats` / `purchase_mode`
只在 `purchase_mode == "full-v2"` 时才出现在武器腿遥测里。
测试 `FullIsUnchangedTests::test_full_telemetry_has_no_new_key` 把这条钉死：
`set(v2_telemetry) - set(full_telemetry) == {"spend_v2"}`，其余键值逐个相等（除 `mode` 自身）。

---

## 三、三条腿的设计（行程内顺序：药水 → 护甲 → 武器 → 修理照旧）

### 腿 1：`v2_potions` —— 买到腰带**满**

接在冻结的 `minimum_potions` 之后（那条把腰带填到备战法要的 4 瓶）。
判词只用原生数字：`belt_heals < belt_capacity`（原生 `MaxBeltItems`，hpp:529）
且 `belt_free_slots > 0`。候选沿用既有的 `medicine_candidates()`
（`heal_kind in (1,2,3,4)`、`can_use`、`can_fit`，按 (price, index) 排序），
每次买**最便宜**的那行——单位金币买到的瓶数最多。
停下时必须说出**是哪一堵墙**：`belt_full` / `no_free_slot` / `unaffordable` /
`stock_empty`（后者才置 `stock_limited = true`）。**不发明库存**。

### 腿 2：`v2_armor` —— 每个槽位挑**护甲增益最大**的那行

冻结的 `plan_basket` 只在**备战失败**时提护甲，而且在能解决失败的行里挑**最便宜**的
（`resource_sustain.py` 的 `buy_chest`）；人一旦达标就再也不提了。新的
`plan_armor_upgrade()` 是可自由裁量的另一半：

- **门是原生的**：`meets_armor_gate`。在 ordinary armor scope 打开时
  （认证臂必然打开——`configure_resource_weapon_purchase` 要求它，见 hpp:427），
  `AppendOrdinaryResourceArmorState`（hpp:747）把这个键**改写**成与槽位无关的版本：
  `IsOrdinaryResourceArmor`（= `isArmor() || isHelm() || isShield()`，普通品质，hpp:455-457）
  + 耐久 + `PlanResourceArmor` 的 valid/safe/canFit + 换上以后
  `projected_readiness` 里没有 armor/damage/weapon 失败。
  **所以身甲、盾、头盔三个槽位原生都买得到**——这是本次直接从 `resource_protocol.hpp`
  读出来的事实，不是推测（`ActBuyStoreItem` hpp:1137 那条判词在 scope 打开时就是
  `!(IsOrdinaryResourceArmor(item) || upgradeWeapon)`）。
- **序是原生的**：增益 = `projected_armor_class`（`plan.gear.nextArmorClass`）
  − 备战裁决自己的 `armor_class`。本法**不复刻任何护甲公式**。
  并列时先便宜、再低 index（确定性）。
- **预算 = `gold − reserve`**，`reserve` 直接调用**既有**的
  `resource_weapon_upgrade.weapon_upgrade_reserve(raw)["reserve"]`，
  即 R18-K2b/H2 已经定义的药水底线
  `(required_belt_heals − belt_heals) × HEALER_MIN_HEAL_PRICE(=50)`。
  **没有新阈值。** 腿 1 跑完腰带通常已满，于是 reserve = 0，全部余额可用——
  这正是主席要的“钱要花出去”。
- 循环：买一件 → 用既有的 `_armor_pending` 机制装上 → 重新在铁匠处取**新报价**
  重算增益 → 直到没有增益、买不起、或阀门 `SPEND_V2_ARMOR_MUTATION_CAP = 8`。
  被买走的行会从柜台消失，所以循环天然收敛。

### 腿 3：武器 —— 解除“每趟一件”

`WeaponUpgradeService` 原样复用（同一条 K2b 原生购买路径、同一个
`plan_weapon_upgrade`、同一个 `require_native=True` 门、同一个 reserve）。
唯一改动在 `_settle`：`purchase_mode == "full-v2"` 时**不结束这条腿**，
而是交回 `None`，让 `_command` 在**同一拍**回到 `shop` 相位、用**新报价**再规划一次。
没有多花一个微步在交接上；结束条件是 `weapon_no_candidate`（没有获批的、
买得起的、伤害更高的单手武器了），外加 400 微步的腿预算与阀门
`SPEND_V2_WEAPON_CAP = 8`。

**K2b 的两条限制是原生的，报告如实说：**
- **普通品质**：`IsResourceUpgradeWeapon`（hpp:466-472）要求
  `item._iMagical == ITEM_QUALITY_NORMAL`；`ActBuyStoreItem` 在 scope 打开时
  用它当门（hpp:1137）。**不是 Python 的保守，是原生柜台的判词**——
  放宽它要改 C++，今晚禁止。
- **单手**：同一函数要求 `_iLoc == ILOC_ONEHAND` 且 `player.GetItemLocation(item) == ILOC_ONEHAND`。
  双手武器会顶掉盾，那是护甲决定。

### 顺序为什么是这样（一处刻意的说明）

主席要的次序是「药水买满 → 护甲按 AC → 武器 → 修理照旧」。
本实现把**三条可自由裁量的腿**按这个次序排在冻结的
`joint_gear`（备战篮子 + 修理）**之后**：

> `joint_gear` 是**下楼的门槛**，而且它自己已经把 4 瓶药钱
> （`plan_basket` 的 `med_cost`）先扣下来才肯买装备。把“买满腰带 / 买最好护甲”
> 排到它前面，就等于让奢侈品吃掉解决**真备战失败**的那件装备。
> 所以：修理照旧在原位（= 在药水之前），三条新腿之间才是「药水 → 护甲 → 武器」。

每一笔都走既有收据：`ResourceService.receipt`（药水、护甲）与
`book_weapon_purchase`（武器）——`gold_spent` / `purchases` 与
`SustainLootService._seal_trip` 的现金对账因此保持一致（探针里 0 次
“Completed loot trip has unexplained cash movement”）。

---

## 四、遥测（只增不改；关时是严格子集）

每趟行程一条账页（`ResourceService.spend_v2_trips`）：

```
{trip, gold_before, gold_after, reserve,
 potions_bought, potions_gold, belt_heals_before/after, belt_capacity,
 armor_bought: [{slot, ac_before, ac_after, ac_after_projected, price, index}],
 armor_gold, armor_class_before/after,
 weapons_bought: [{dmg_before, dmg_after, dmg_after_projected, price, index}],
 weapons_gold, healer_rows_seen, armor_rows_seen,
 potion_stop, armor_stop, stock_limited}
```

外加整局合计（`potions_bought/gold`、`armor_bought/gold`、`weapons_bought/gold`、
`trips_served`、`stock_limited_trips`、`potion_stops`、`armor_stops`）。
探针行的 `resource.spend_v2` 在任何非 `full-v2` 臂上恒为 `None`。

**一条可核对的自证**：ON 臂 28 次护甲购买里，实现的 AC 增益合计 **110**，
原生投影 `projected_armor_class` 的增益合计也是 **110**（富臂 43 次：**171 = 171**）。
投影**次次精确**，说明这条腿用的是引擎自己的数，没有猜。

---

## 五、测试

- 新增 `tests/test_r19_spend_v2.py`：**26 tests 全绿**
  （`/home/laure/r17_work/r19/spend-tree`，`PYTHONPATH=python:train`）。覆盖：
  词表登记与 validator 收/拒（`full-v3`/`fullv2`/`FULL-V2`/`full_v2` 全拒）；
  下游三条法（service policy / weapon upgrade / ordinary armor scope）只认这一族；
  **"full" 逐字节**（`minimum_potions` 仍在 4 瓶收手、仍直接转 `return`、
  遥测无新键、武器腿仍买一件就结束）；
  买到腰带满 8、`no_free_slot` / `unaffordable` / `stock_empty` 三堵墙各自被**如实报出**、
  容量取自原生 `belt_capacity`；
  护甲按原生增益择优、并列先便宜再低 index、预算 = `gold − reserve`、
  零增益不买、贪心循环逐槽位买到无增益为止、阀门被报出；
  reserve 就是既有药水底线（腰带缺 3 瓶 → 150；腰带满 → 0）、
  **同一件武器在腰带没满时因 reserve 被拒、腰带满后又买得起**；
  full-v2 的续买交回 `None` 并开新 attempt、阀门关腿；
  武器购买记进同一张行程账页且实现伤害在装备落定时回填。
- 邻域回归（resource_protocol / sustain / sustain_loot / sustain_armor(+contract) /
  sustain_combinations / weapon_upgrade / loot_protocol_integration / identify /
  retreat / portal / sweep / probe_sustain / gold_memory / sustain_gold(+v4) /
  sustain_protection + 新卷）：**687 passed / 1 skipped / 359 subtests passed，0 失败**。
- 全量 `tests/`（跳过 3 个**在纯净副本上同样收集失败**的文件：
  `test_build_contracts.py` / `test_content_case_aux.py` / `test_worker_env.py`，
  它们要读本副本没有拷贝的启动脚本与资产；固定顺序 `-p no:randomly`）：

  | 树 | 结果 |
  |---|---|
  | 改动树 `spend-tree` | **81 failed / 2430 passed / 99 skipped / 3 errors / 1481 subtests passed** |
  | 纯净副本 `spend-pristine` | 82 failed / 2403 passed / 99 skipped / 3 errors / 1481 subtests passed |

  失败集合**双向 `comm` 对照**：
  **改动树独有 0 条**；纯净副本独有 1 条
  （`test_resource_identify::IdentifyServiceContractTests::test_the_engine_price_constant_matches_the_python_one`
  ——原生价格常量对照，与本改动无关，两树共用同一条 build 软链，疑似不稳定用例）。
  多出的 27 条 passed = 本卷 26 条 + 上面那条。

  **对照基准是今晚现做的纯净副本** `/home/laure/r17_work/r19/spend-pristine`
  （从主树 rsync，未打任何补丁）。A1 留下的 `base-tree` 已经过期：先用它比对时
  凭空报出 24 条“新失败”（`test_critic_migration` / `test_content_case_bc` /
  `test_training_core` / `test_run_r7_combat_recovery` / `test_eval_pipeline` /
  `test_r13_live_dive`），在纯净副本上**逐条相同**地失败，说明那是副本资产问题，
  不是本改动。这条弯路记在这里，免得下一位再踩。

---

## 六、探针（pool 2_133，48 种子，3 分片，worker 7e31dc54）

驱动：`/home/laure/r17_work/r19/spend2_probe_driver.py`；
原始产物：`/home/laure/r17_work/r19/spend-probe/`
（`spend2-{off,on,rich}-{a,b,c}.json|log|cmd`、三份 `spend2-*-merged.json`、
`spend2-summary-all.json`）。
探针版本 `r17-deployment-v3-r18m2`，`max_steps 6000`，decoding `sample`，
覆盖 = `/home/laure/r17_work/r18/gates/v1world-overrides.json` 的 16 键。
三臂九分片同时起，总耗时 656 s，**运行时错误 0**。

### 6.1 OFF（回归）—— 通过

| | 值 |
|---|---|
| `rows_sha_v3` | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` |
| 期望 | 同上 |
| 相等 | **True** |
| 行数 | 48 |
| 运行时错误 | 0 |

`arm_stats` 与认证控制行**逐字段相同**（alive 20 / l2 34 / l3 4 / clvl≥4 25 /
deaths_by_floor {1:7, 2:19, 3:2} / l2_beats 85578 / hazard 0.222 / l2_kills 894 /
belt 直方图 {4:18,5:6,6:4,7:4,8:2} / AC 直方图 {9:13,10:5,11:4,12:4,13:1,14:5,15:2} /
damage {7:11,9:18,11:5} / gold_final median 78 max 623 total 6855 /
weapon_upgrades 29 / weapon_gold 6620 / identify_gold 1600）。

### 6.2 ON-normal（`resource_purchase_mode = "full-v2"`，钱来自世界本身）

| 指标 | 控制 | ON-normal |
|---|---|---|
| alive / 48 | 20 | **17** |
| 配对 McNemar | — | saved 3 / lost 6 / **net −3** / discordant 9 / **UCB95 0.16424** |
| deaths by floor | {1:7, 2:19, 3:2} | {1:7, **2:22**, 3:2} |
| L2 hazard /1k | 0.222 | **0.281** |
| L2 beats | 85578 | 78301 |
| L2 kills | 894 | 738 |
| clvl≥4 | 25 | 24 |
| 到 L2 / L3 | 34 / 4 | 35 / 4 |
| **首降腰带** | {4:18, 5:6, 6:4, 7:4, **8:2**} | {4:1, 5:4, 6:5, 7:3, **8:22**} |
| **首降 AC** | {9:13,10:5,11:4,12:4,13:1,14:5,15:2}（≤15） | {9:5,10:6,11:3,12:4,13:1,14:3,15:4,16:1,17:3,18:3,19:1,**20:1**} |
| **首降武器伤害** | {7:11, 9:18, **11:5**} | {7:**19**, 9:16, 11:**0**} |
| gold_final median / max / total | 78 / 623 / 6855 | **28.5** / 335 / 2480 |
| 武器腿升级次数 / 金额 | 29 / 6620 | **4 / 710** |
| 鉴定花费 | 1600 | 1900 |

spend-v2 账本（ON，48 局共 39 次到达新腿的行程）：

| 类别 | 次数 | 金额 |
|---|---|---|
| 药水 | 39 | 1950 |
| 护甲 | 28 | 1895（实现 AC 增益合计 110 = 原生投影 110） |
| 武器（记在行程账页里的） | 2 | 370 |
| 每趟药水 | 1.000 | |
| 每趟采购（三类合计） | 1.769（最多一趟 7 笔） | |
| `stock_limited` 占比 | **0.0** | |
| `potion_stop` | belt_full 25 / unaffordable 14 | |
| `armor_stop` | over_budget 28 / no_upgrade 10 / 未触及 1 | |

**读法（不粉饰）**：
- 药水法**成了**：首降腰带 8 的局数 2 → 22。
- 护甲法**成了**：首降 AC 上限从 15 抬到 20。
- **但武器被挤掉了**：`armor_stop` 有 28 次是 `over_budget`，说明护甲把余额吃光，
  轮到武器腿时已无预算——武器升级 29 → 4，首降伤害 11 的局数 5 → 0。
- 净效果 `net −3`、UCB95 `0.16424 > 0`：**这一臂排除不了变坏**。
  在正常金钱下，「护甲优先于武器」这个次序是有代价的。

### 6.3 ON-rich（诊断臂：同 ON，外加 reset 后 +3000 金）

标注：**诊断，不是法条**。金币由 `bridge.probe_resource_add_gold` 在每局 reset 后注入
（`train/runs/r10-staging/spend2_rich_probe.py`，抄自
`/home/laure/r17_work/r19/rich-probe/rich_probe.py` 并改用本工作树 + `full-v2`）。

| 指标 | 控制 | ON-rich |
|---|---|---|
| alive / 48 | 20 | **23** |
| 配对 McNemar | — | saved 11 / lost 8 / **net +3** / discordant 19 / **UCB95 0.08614** |
| deaths by floor | {1:7, 2:19, 3:2} | {1:6, **2:15**, 3:3, 4:1} |
| L2 hazard /1k | 0.222 | **0.1599** |
| L2 beats / kills | 85578 / 894 | 93798 / 924 |
| clvl≥4 | 25 | 24 |
| 到 L2 / L3 | 34 / 4 | 35 / **6** |
| **首降腰带** | {4:18,5:6,6:4,7:4,8:2} | {7:1, **8:34**} |
| **首降 AC** | ≤15 | {10:1,15:5,16:4,17:6,18:3,19:2,20:2,21:4,22:2,23:2,24:2,25:1,**26:1**} |
| **首降武器伤害** | {7:11, 9:18, 11:5} | {7:4, 9:16, **11:15**} |
| gold_final median / max / total | 78 / 623 / 6855 | **1550** / 3142 / 87040 |
| 武器腿升级次数 / 金额 | 29 / 6620 | **52 / 14920** |

spend-v2 账本（rich，38 次行程）：药水 73 次 / 3650 金；护甲 43 次 / 5050 金
（实现增益 171 = 投影 171）；武器 18 次 / 5230 金；每趟药水 1.921、每趟采购 3.526；
`stock_limited` 占比 **0.0**；
**`potion_stop` 38/38 全是 `belt_full`，`armor_stop` 38/38 全是 `no_upgrade`。**

**读法**：钱管够时，这份采购脚本会把柜台上**每一件**有提升的货买光，然后**真的没东西可买了**。
`gold_final` 中位数仍有 1550，不是因为脚本舍不得花，而是因为
格里斯沃德那一趟的存货已经被买空——**天花板从脚本移到了存货**。
这正是对主席「只要有提升就买，不限制购买次数」的直接回答，也是对
「144000 金只多花了 9000」那份证据的修复：同样的富启动，
现在护甲 5050 + 药水 3650 + 武器 5230 = **13930 金**走的是新腿，
武器腿总额 6620 → 14920。

### 6.4 两个点名种子

| | 控制 | ON-normal | ON-rich |
|---|---|---|---|
| **2133010** | 死于 L2；clvl 4；kills 148；dmg 11；gold_final 22；武器腿买了 1 件 300 金 | 死于 L2；clvl 4；kills 147；**AC 20**；**dmg 7**；gold_final 80；v2 账页 1 趟 **0 笔**（进城时腰带已 8、AC 已 20，铁匠柜台 15 行**无一提升**）；武器腿 **0 件**（余额 161 买不起） | 死于 L2；clvl 4；kills 153；**AC 25**；**dmg 13**；gold_final 611；v2：药水 2、护甲 1 |
| **2133002** | 死于 L1，第 559 拍，clvl 1，kills 21 | **完全相同**（559 拍、clvl 1、kills 21、gold 50）——这局在**第一次进城之前**就死了，法条是空操作 | **活到终局**：depth 1、clvl 3、kills 133、gold_final 2718；v2：药水 4、武器 1 |

2133010 的 ON 一行值得单独说：**v2 账页 0 笔采购，这一局仍然与控制发散**。
原因是诚实的开销——`v2_armor` 会从 Pepin 走回格里斯沃德**取新报价**，
即使最后一件也不买，这趟往返也要花微步。**新腿不是免费的**，
只要这一趟进了城就会与控制分叉。（可优化：若 `v2_potions` 一件没买且
备战篮子刚在铁匠处收工，可以跳过这次回访；本轮没做。）

---

## 七、已知缺口（说清楚，不掩盖）

1. **ON-normal 排除不了变坏。** `net −3`、UCB95 `0.16424`。护甲优先于武器的次序
   在正常金钱下把武器腿饿死（29 → 4 次）。**建议**：下一轮把武器腿的
   reserve 从「药水底线」扩成「药水底线 + 一件最便宜的已获批武器」，
   或把三条腿改成按**原生投影收益/金币**统一排序，而不是固定次序。本轮**没做**——
   那会引入一个新阈值，越出今晚的授权。
2. **`v2_armor` 的回访开销。** 见 6.4；`v2_potions` 空手时也会走一趟铁匠。
3. **`train/eval_contract.py` 的配方还在说旧话。**
   `resource_service_recipe("l2-town-v1", "full-v2", "sustain-loot-v1", ...)`
   仍返回 `potion_target: 4`，也没列出三条新阶段。
   这是**元数据**：`eval_contract` 在一局里从不被调用（它只出现在探针的
   `_SOURCE_FILES` 指纹里），而 sustain-loot-v1 的评测档案在 R18-B3 下**本来就被
   `validate_r16_environment` 一律拒铸**，所以今晚没有任何产物能带上这个数字。
   之所以**不顺手改**：改它会改掉探针指纹，而三臂的行都是在改之前跑出来的——
   宁可留一条被测试**钉住**的缺口（`test_eval_contract_admits_the_value` 里有
   一段说明为什么这里断言的是 4），也不要让报告里的 sha 与树里的字节对不上。
4. **CLI 未打通**：`train_ppo.py` 的 `--resource-purchase-mode` 已经接受 `full-v2`，
   但没有为它做过一次训练 smoke（**not verified**）。
5. **原生限制两条**（不是本法的保守）：武器只能是**普通品质**、只能**单手**
   （`IsResourceUpgradeWeapon`，hpp:466-472）。放宽要改 C++。
6. **护甲只买柜台上的**：`v2_armor` 不看背包里已有的护甲（那是
   `plan_basket` 的 `equip_owned_chest` 分支的地盘），也不卖被顶下来的旧件
   （沿用 K2b「同趟不卖」的做法）。
7. 全量 10 分钟套件里被跳过的 3 个文件（见 §五）在纯净副本上同样收集失败，
   与本改动无关；但它们**没有被跑过**（**not verified**）。
8. 富启动臂是**诊断**：它给的钱不来自世界，任何“它更安全”的读法都不能当作
   本法在正常世界里的效果。

---

## 八、产物清单

| 用途 | 路径 |
|---|---|
| 工作树 | `/home/laure/r17_work/r19/spend-tree` |
| 补丁 | `/home/laure/r17_work/r19/spend2.patch` |
| 探针驱动 | `/home/laure/r17_work/r19/spend2_probe_driver.py` |
| 富启动探针（诊断） | `/home/laure/r17_work/r19/spend-tree/train/runs/r10-staging/spend2_rich_probe.py` |
| 三臂原始行 | `/home/laure/r17_work/r19/spend-probe/spend2-{off,on,rich}-merged.json` |
| 三臂汇总 | `/home/laure/r17_work/r19/spend-probe/spend2-summary-all.json` |
| 新测试 | `/home/laure/r17_work/r19/spend-tree/tests/test_r19_spend_v2.py` |
| 全量测试日志（改动树 / 纯净副本） | `/home/laure/r17_work/r19/spend-tests-full.log` / `pristine-tests-full.log` |
| 失败集合对照 | `/home/laure/r17_work/r19/spend-fails.txt` / `pristine-fails.txt` |
| 纯净对照副本 | `/home/laure/r17_work/r19/spend-pristine` |
| 本报告 | `/home/laure/r17_work/r19/SPEND2-REPORT.md` |
| 账本事件 | `R19_SPEND_V2_IMPLEMENTED` / `R19_SPEND_V2_PROBE_RESULT`（`train/runs/r10-staging/r13_ledger.jsonl`） |
