"""DiabloGymEnv —— Gymnasium 包装(v0:结构化向量观测 + 离散动作)。

观测向量(float32,长度 12 + K*4 + 2*(2R+1)² + 8,R=5 时共 294):
  [hp/maxhp, mana/maxmana, xp(log1p/10), gold/1000, char_level/50,
   dungeon_level/16, player_x/112, player_y/112,
   存活怪数/50, 最近怪距离/30(无怪=1),
   最近下行楼梯方向 dx/56, dy/56(本层无则 0,0)]
  + K 个最近怪物的 (dx/20, dy/20, hp/max_hp, 1存在标志)
  + 11×11 局部地图两通道(可走性、怪物占位)——run4 教训:没有空间感知,
    奖励再好也是"盲人拿完美账本"(隔墙锁定、穿墙塑形、找不到房门)
  + [腰带治疗药数/8, 最近地面治疗药 dx/20, dy/20(截断至 ±1), 存在标志]
    (v13,治"瓶盲"——教训十一:动作的前置条件必须可观测,否则策略学不会
    按键纪律)
  + [护甲值/50(截断至 1), 最近可穿装备 dx/20, dy/20(截断至 ±1), 存在标志]
    (v14,装备章:存在标志已预判"槽位为空+属性达标",=1 即值得按)

动作(Discrete(15)):
  0      原地不动
  1-8    朝八方向走一格(寻路)
  9      交战宏:锁定最近怪物持续追击,直到它死/自己死/换层/超时(≤10 拍)
         (v2 教训:单拍攻击会被下一个走位动作打断,策略学不会"坚持进攻")
  10     探索宏:走向 25×25 视野内最近的"可走且未踏足"边疆点;发现猎物
         (最近怪 ≤6 格)立即交还控制权;无边疆点时朝下行楼梯走
         (run5 教训:出生区无可达怪时,反应式策略不会"换个房间找")
  11     下楼宏(v11):接力寻路走向本层最近的下行楼梯并站上去等触发。
         与探索宏不同,发现猎物**不**打断——这是策略主动选择的撤离/换层键
         (困局的逃生舱 + 清层后的下一章按钮);12 拍后控制权自然归还。
         (v10 教训:困局是死的 0,多给时间没用——得给一扇门)
  12     喝药键(v12):腰带有治疗类药水就喝一瓶(引擎手柄快捷键同路),
         没有则为空拍。v12 曾刻意不把腰带药数放进观测(保 286 维历代
         可复评),结果 99.5% 的按键落在空腰带上(教训十一"瓶盲"),
         v13 起腰带药数与最近地面药方向入观测。
  13     捡药宏(v13):与下楼宏同款门/桶感知 BFS 走向最近的地面治疗药,
         遇关门先开门;进入 2 格内交给引擎原生拾取(CMD_GOTOAGETITEM,
         自动走近+拾取+入腰带)。引擎自带的 MakePlrPath 是门盲的(关门
         =墙,寻路失败即静默弃疗——9003 号种子实锤:药在关门后,原生
         命令原地罚站),所以跨房间接近必须由宏承担。无地面药则空拍。
         供给侧=怪物掉落+地面固定刷新(32 评估种子出生层全部有药,1-5 瓶)。
  14     捡装备宏(v14):同款门感知 BFS 走向最近的"值得穿"装备(空槽+属性
         达标,桥侧预判——引擎 AutoEquip 失败会把装备落进背包黑洞,所以
         预判必须在按键前);到位后引擎自动上身(EngineInit 已开盔甲/头盔/
         首饰的自动装备选项)。武器/盾牌不碰(出厂双手已满,换装留给 v15)。

奖励(v2,逐刀致密化,Lawrence 提案 + 防磨刀修正):
  +0.5 * (本刀伤害/目标最大血) * 残血系数     每刀即时到账;系数 1.0→1.5,
        残血系数 = 1 + 0.5*(1 - 击后血量比)   越残血越值钱(补刀激励),
                                              挂在伤害占比上→无磨刀/秒杀漏洞
  +1.0 * 击杀                                  收头奖励
  +0.01 * ΔXP                                  真实目标(升级)
  +8.0  * Δ地牢层                               ≈4 只怪的价值,清完才值得下楼
  +0.5  * ΔAC(穿甲时,奖励 v3/v15)             教训十三的自举塑形:守恒存量、
                                                不可刷;负 Δ(死亡掉装)不罚
  +0.005 * 自己走近最近怪的格数(远离同额扣)
  -0.002  原地不动(含撞墙)
  -2.0 死亡   +10.0 通关
  历史教训:v0 的掉血惩罚→面壁塌缩;v1 的"怪贴脸也计分"→站桩钓鱼。
"""

from __future__ import annotations

import math
import pathlib
import tempfile
from collections import deque

import gymnasium as gym
import numpy as np

from . import bridge, nav

# 八方向(等距地牢的 tile 坐标系)
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
_K_MONSTERS = 8
_MAP_RADIUS = 5  # 11×11 局部地图

_DEFAULT_ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "engine" / "devilutionx.app" / "Contents" / "Resources"
)


class DiabloGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        assets_dir: str | None = None,
        save_dir: str | None = None,
        data_dir: str | None = None,
        ticks_per_step: int = 4,
        max_steps: int = 5000,
        start_in_dungeon: bool = False,
        include_raw: bool = True,
        descend_ladder: bool = False,
        death_ladder: bool = False,
    ):
        super().__init__()
        assets = str(assets_dir or _DEFAULT_ASSETS)
        saves = save_dir or tempfile.mkdtemp(prefix="diablogym-saves-")
        data = str(
            data_dir
            or pathlib.Path.home() / "Library/Application Support/diasurgical/devilution"
        )
        bridge.init(assets_dir=assets, save_dir=saves, data_dir=data, hero_class=0)

        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.start_in_dungeon = start_in_dungeon
        self.include_raw = include_raw
        # v17 深水区:下楼奖金层数递进(N→N+1 付 8×N;False = v6-v16 的扁平 8.0,
        # 旧章金标准的世界规则不动)
        self.descend_ladder = descend_ladder
        # v18:死亡成本与阶梯同步定价(死在 N 层罚 8×N;False = 恒 -2.0)。
        # 教训十六:阶梯 8/16/24 对上死亡 -2,冲刺期望值稳赚(+5.8),
        # "活着抵达"必须在拍卖行里赢过"摸到深度"
        self.death_ladder = death_ladder
        side = 2 * _MAP_RADIUS + 1
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12 + _K_MONSTERS * 4 + 2 * side * side + 8,), dtype=np.float32,
        )  # +8 = v13 药 4 维(腰带数+最近地面药)+ v14 装备 4 维(AC+最近可穿装备)
        self._raw = None
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = 0
        self._visited: set[tuple[int, int]] = set()

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        self._raw = bridge.reset(seed=actual_seed)
        if self.start_in_dungeon:
            # 城镇布局固定,脚本化走到教堂楼梯(约 500-900 tick,~0.05s)
            self._raw = nav.descend_to_dungeon(bridge)
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = int(self._raw["xp"])
        self._visited = {(self._raw["player_x"], self._raw["player_y"])}
        return self._vectorize(self._raw), self._info(self._raw)

    def action_masks(self) -> np.ndarray:
        """v16:无效动作掩码(MaskablePPO 协议方法;SubprocVecEnv 经
        env_method("action_masks") 跨进程调用,Monitor 包装经 __getattr__ 透传)。

        只掩码 14 号键:视野内没有可穿装备(floor_items 无 gear 标志,与
        _vectorize 的第 294 维同源)时,键不在动作分布里——空按的负样本从此
        不进梯度(教训十四:塑形只放大不召唤;掩码让键只在机会到场时存在)。
        12/13 号键保持自由:"何时不按"是 v13 已学会的真本事(尽管是风格
        彩票),掩掉等于换考卷,四手牌基线全作废。掩码不保证宏走得完——
        它消灭空按,不消灭白按(路径受阻/12 拍超时/半路挨打仍会失败)。"""
        mask = np.ones(15, dtype=bool)
        mask[14] = any(it.get("gear") for it in self._raw.get("floor_items", []))
        return mask

    def step(self, action: int):
        prev = self._raw
        action = int(action)
        if action == 9:
            self._raw, micro = self._macro_engage()
        elif action == 10:
            self._raw, micro = self._macro_explore()
        elif action == 11:
            self._raw, micro = self._macro_descend()
        elif action == 12:
            bridge.act_drink()  # 无药时引擎侧为空操作;站桩惩罚由奖励函数自然覆盖
            self._raw = bridge.step(ticks=self.ticks_per_step)
            micro = 1
        elif action == 13:
            self._raw, micro = self._macro_pickup("heal")
        elif action == 14:
            self._raw, micro = self._macro_pickup("gear")
            if bridge.sweep_backpack_gear():
                # PM_GOTHIT 时序窗(审查确认):拾取执行前挨硬直会让装备静默
                # 沉入背包;打捞穿上后刷新观测(无 tick 成本)
                self._raw = bridge.observe()
        else:
            self._apply_action(action)
            self._raw = bridge.step(ticks=self.ticks_per_step)
            micro = 1
        self._steps += micro
        if self._raw["dungeon_level"] != prev["dungeon_level"]:
            # 新一层:足迹清零。各层共用同一坐标系,不清的话探索宏在新层
            # 会把旧层足迹当"已踏足",边疆逻辑整层失效
            self._visited = set()
        self._visited.add((self._raw["player_x"], self._raw["player_y"]))

        # 击杀统计:同层内 id 消失即击杀(换层时基线失效,跳过)
        if self._raw["dungeon_level"] == prev["dungeon_level"]:
            cur_ids = {m["id"] for m in self._raw["monsters"]}
            self._ep_kills += sum(1 for m in prev["monsters"] if m["id"] not in cur_ids)

        reward = self._reward(prev, self._raw)
        terminated = bool(self._raw["dead"] or self._raw["game_over"] or self._raw["victory"])
        truncated = self._steps >= self.max_steps

        info = self._info(self._raw)
        if terminated or truncated:
            info["episode_extra"] = {
                "xp": int(self._raw["xp"]) - self._ep_start_xp,
                "kills": self._ep_kills,
                "char_level": self._raw["char_level"],
                "depth": self._raw["dungeon_level"],
                "died": bool(self._raw["dead"]),
                "gold": self._raw["gold"],
            }
        return self._vectorize(self._raw), reward, terminated, truncated, info

    def _info(self, raw):
        return {"raw": raw} if self.include_raw else {}

    # ---------- 内部 ----------

    def _apply_action(self, action: int) -> None:
        obs = self._raw
        px, py = obs["player_x"], obs["player_y"]
        if 1 <= action <= 8:
            dx, dy = _DIRS[action - 1]
            bridge.act_walk(px + dx, py + dy)

    def _macro_engage(self, max_beats: int = 10):
        """交战宏:锁定最近怪物,持续下追击指令直到分出结果或超时。"""
        target = self._nearest_monster(self._raw)
        if target is None:
            return bridge.step(ticks=self.ticks_per_step), 1
        tid = target["id"]
        start_level = self._raw["dungeon_level"]
        raw = prev = self._raw
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_attack_monster(tid)
            raw = bridge.step(ticks=self.ticks_per_step)
            cur_target = next((m for m in raw["monsters"] if m["id"] == tid), None)
            if cur_target is None or raw["dead"] or raw["dungeon_level"] != start_level:
                break
            # 止损:连续 2 拍既没接近目标也没造成伤害(多半隔墙不可达)→ 提前放弃,
            # 把决策权还给策略,避免 run3 式"对着墙白烧 10 拍"
            if beats >= 2:
                prev_target = next((m for m in prev["monsters"] if m["id"] == tid), None)
                if prev_target is not None and cur_target["hp"] >= prev_target["hp"]:
                    d_prev = max(abs(prev_target["x"] - prev["player_x"]), abs(prev_target["y"] - prev["player_y"]))
                    d_cur = max(abs(cur_target["x"] - raw["player_x"]), abs(cur_target["y"] - raw["player_y"]))
                    if d_cur >= d_prev:
                        break
            prev = raw
        return raw, beats

    _EXPLORE_RADIUS = 12  # 25×25 搜索窗

    def _macro_explore(self, max_beats: int = 12):
        """探索宏:走向最近的未踏足可走边疆点;发现猎物立即交还控制权。"""
        raw = self._raw
        px, py = raw["player_x"], raw["player_y"]
        r = self._EXPLORE_RADIUS
        side = 2 * r + 1
        lm = bridge.local_map(radius=r)
        walk = lm["walkable"]

        # 候选:可走、离玩家 ≥5 格、且不在足迹邻域(±1)内的边疆点
        near_visited = self._visited | {
            (x + dx, y + dy) for x, y in self._visited for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        candidates = []
        for i, w in enumerate(walk):
            if not w:
                continue
            tx, ty = px + (i % side) - r, py + (i // side) - r
            d_player = max(abs(tx - px), abs(ty - py))
            if d_player >= 5 and (tx, ty) not in near_visited:
                candidates.append((d_player, tx, ty))
        if candidates:
            _, tx, ty = min(candidates)  # 最近的边疆点(便宜且稳)
        else:
            # 本窗内已探明:朝下行楼梯推进(层级目标),没有就原地一拍
            stairs = [t for t in raw.get("triggers", []) if t["msg"] == 0]
            if not stairs:
                return bridge.step(ticks=self.ticks_per_step), 1
            tx, ty = stairs[0]["x"], stairs[0]["y"]

        start_level = raw["dungeon_level"]
        last_pos = (px, py)
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_walk(tx, ty)
            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            nd = self._nearest_dist(raw)
            if (raw["dead"] or raw["dungeon_level"] != start_level
                    or (nd is not None and nd <= 6)          # 发现猎物,交还控制权
                    or max(abs(pos[0] - tx), abs(pos[1] - ty)) <= 1):  # 到达
                break
            stall = stall + 1 if pos == last_pos else 0
            if stall >= 2:  # 目标不可达,止损
                break
            last_pos = pos
        return raw, beats

    _DESCEND_RADIUS = 112  # 规划窗覆盖全图(地牢 112×112):有的层联通回廊会绕大圈,
                           # 40 格窗曾在 seed 9005 上漏掉西侧绕行路线。每次按键只规划一次,
                           # C++ 端一次调用出图,开销在毫秒级,换全局最优值得

    def _plan_descend_path(self, raw, sx, sy, avoid_monsters: bool = False):
        """全局窗 4 向 BFS(关着的门视为可通行),返回去往"可达且离楼梯最近的格"
        的路径 [(x, y, 是否关门), ...](不含起点)。None = 可达域内没有比脚下
        更接近楼梯的格子(真·被困)。4 向保证引擎寻路必然接受每段(斜穿墙角
        引擎会拒绝);贪心"只挑更近的格"会死在凹形迷宫里,BFS 允许先绕远。

        avoid_monsters=True 时把怪物占位格视为墙(v14 修复:引擎寻路拒绝穿怪,
        规划器若怪物盲,遇到闲置怪堵走廊会陷入"重规划出同一条路"的失速死循环
        ——9024 号种子的 1 血骷髅当场抓获;调用方应在返回 None 时退回
        avoid_monsters=False 保底,行为最坏退化为旧版失速交还)。"""
        px, py = raw["player_x"], raw["player_y"]
        r = self._DESCEND_RADIUS
        side = 2 * r + 1
        lm = bridge.local_map(radius=r)
        walk, door = lm["walkable"], lm["door"]
        mon = lm["monster"] if avoid_monsters else None

        def idx(tx, ty):
            return (ty - py + r) * side + (tx - px + r)

        start = (px, py)
        prev = {start: None}
        depth = {start: 0}
        best = (max(abs(sx - px), abs(sy - py)), 0, start)
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + ddx, cy + ddy
                if abs(nx - px) > r or abs(ny - py) > r or (nx, ny) in prev:
                    continue
                i = idx(nx, ny)
                if not walk[i] and not door[i]:
                    continue
                if mon is not None and mon[i]:
                    continue  # 怪物占位=墙(引擎寻路拒绝穿怪;见 docstring)
                prev[(nx, ny)] = (cx, cy)
                depth[(nx, ny)] = depth[(cx, cy)] + 1
                d_stairs = max(abs(sx - nx), abs(sy - ny))
                if (d_stairs, depth[(nx, ny)]) < best[:2]:
                    best = (d_stairs, depth[(nx, ny)], (nx, ny))
                queue.append((nx, ny))
        if best[2] == start:
            return None
        path = []
        cur = best[2]
        while cur != start:
            path.append((cur[0], cur[1], bool(door[idx(*cur)])))
            cur = prev[cur]
        path.reverse()
        return path

    def _macro_descend(self, max_beats: int = 12):
        """下楼宏:全局 BFS 规划一次,沿路径逐路点走向下行楼梯,遇关门先开门
        (CMD_OPOBJXY 引擎自动走近再操作;地牢房间靠门连通,而关着的门在
        walkable 通道里长得和墙一样——这是宏必须自带门感知的原因)。

        发现猎物不打断(这是主动撤离键);换层/阵亡/持续失速提前结束;
        12 拍耗尽自然归还控制权,下次按键重新规划。全程无随机数,确定性。
        """
        raw = self._raw
        stairs = [t for t in raw.get("triggers", []) if t["msg"] == 0]
        if not stairs:
            return bridge.step(ticks=self.ticks_per_step), 1
        px, py = raw["player_x"], raw["player_y"]
        st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
        sx, sy = st["x"], st["y"]
        start_level = raw["dungeon_level"]

        path = self._plan_descend_path(raw, sx, sy, avoid_monsters=True)
        if path is None:
            path = self._plan_descend_path(raw, sx, sy)  # 怪物封死唯一通路:退回旧行为
        if path is None:
            return bridge.step(ticks=self.ticks_per_step), 1  # 真被困:原地一拍,交还控制权

        pi = 0            # 路径消费指针
        target = None     # (kind, x, y, path_index)
        stall = 0
        beats = 0
        last_pos = (px, py)
        for beats in range(1, max_beats + 1):
            if target is None:
                if pi >= len(path):
                    break  # 路径走完(最近可达格≠楼梯时会发生),交还控制权
                # 先处理前方 8 格内的第一扇关门,否则取 ~8 格外的路点
                nxt = None
                for j in range(pi, min(pi + 8, len(path))):
                    if path[j][2]:
                        nxt = ("open", path[j][0], path[j][1], j)
                        break
                if nxt is None:
                    j = min(pi + 7, len(path) - 1)
                    nxt = ("walk", path[j][0], path[j][1], j)
                target = nxt
                if target[0] == "open":
                    bridge.act_operate(target[1], target[2])
                else:
                    bridge.act_walk(target[1], target[2])
            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            if raw["dead"] or raw["dungeon_level"] != start_level:
                break  # 换层成功(或阵亡);足迹由 step() 统一按层重置
            if pos == (sx, sy):
                continue  # 已站上楼梯格,等触发换层——站桩不算失速
            if target[0] == "open":
                # 开门型目标:门格真的变可走才算完成(贴脸≠已开,动画要几拍)
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[target[3]] = (target[1], target[2], False)
                    pi = target[3]  # 从门所在格继续消费路径
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif max(abs(pos[0] - target[1]), abs(pos[1] - target[2])) <= 1:
                pi = target[3] + 1  # 到达路点,继续下一段
                target = None
                stall = 0
                last_pos = pos
                continue
            if pos == last_pos:
                stall += 1
                if stall == 3 and target is not None:
                    # 命令可能被打断(被怪撞开路径等):原地重发一次
                    if target[0] == "open":
                        bridge.act_operate(target[1], target[2])
                    else:
                        bridge.act_walk(target[1], target[2])
                if stall >= 6:
                    break  # 重发后仍无进展 → 交还控制权,下次按键重新规划
            else:
                stall = 0
            last_pos = pos
        return raw, beats

    def _macro_pickup(self, kind: str = "heal", max_beats: int = 12):
        """捡取宏(v13 药 / v14 装备):复用下楼宏的规划器(_plan_descend_path
        本就目标参数化,门/桶=可操作软墙),沿路径开门走向最近的目标物;
        进入 2 格内改用引擎原生拾取命令收尾(此时无门阻隔,MakePlrPath 必成)。

        成功判据:目标从地面消失(药:进腰带;装备:引擎 AutoEquip 自动上身,
        桥侧 gear 标志已预判空槽+属性达标);药另有腰带数上涨的快速判据。
        阵亡/换层/路径耗尽/持续失速提前结束;12 拍耗尽自然归还控制权,
        下次按键重新规划。全程无随机数,确定性。
        """
        raw = self._raw
        flag = "heal" if kind == "heal" else "gear"
        act = bridge.act_pickup if kind == "heal" else bridge.act_pickup_gear
        targets = [it for it in raw.get("floor_items", []) if it.get(flag)]
        if not targets or (kind == "heal" and raw["belt_heals"] >= 8):
            # 无目标,或腰带已满(捡了直落背包=喝药键看不见的黑洞):空拍交还
            return bridge.step(ticks=self.ticks_per_step), 1
        px, py = raw["player_x"], raw["player_y"]
        h = min(targets, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
        hx, hy = h["x"], h["y"]
        start_belt = raw["belt_heals"]
        start_level = raw["dungeon_level"]

        near0 = max(abs(hx - px), abs(hy - py)) <= 2
        path = self._plan_descend_path(raw, hx, hy, avoid_monsters=True)
        if path is None:
            path = self._plan_descend_path(raw, hx, hy)  # 怪物封死唯一通路:退回旧行为
        if path is None and not near0:
            return bridge.step(ticks=self.ticks_per_step), 1  # 真不可达:空拍交还

        pi = 0            # 路径消费指针
        target = None     # (kind, x, y, path_index)
        stall = 0
        beats = 0
        last_pos = (px, py)
        for beats in range(1, max_beats + 1):
            if target is None:
                cur = (raw["player_x"], raw["player_y"])
                near = max(abs(hx - cur[0]), abs(hy - cur[1])) <= 2
                door_pending = bool(path) and any(
                    p[2] for p in path[pi:min(pi + 3, len(path))])
                if near and not door_pending:
                    # 近旁且无门阻隔:引擎原生拾取收尾(审查角落:贴门站位时
                    # 必须先走开门分支,否则原生寻路对门失败=白按)
                    target = ("pick", hx, hy, pi)
                    act()
                elif path is None or pi >= len(path):
                    break  # 路径耗尽仍未进入近旁,交还控制权
                else:
                    # 与下楼宏同款:先处理前方 8 格内的第一扇关门,否则取 ~8 格外路点
                    nxt = None
                    for j in range(pi, min(pi + 8, len(path))):
                        if path[j][2]:
                            nxt = ("open", path[j][0], path[j][1], j)
                            break
                    if nxt is None:
                        j = min(pi + 7, len(path) - 1)
                        nxt = ("walk", path[j][0], path[j][1], j)
                    target = nxt
                    if target[0] == "open":
                        bridge.act_operate(target[1], target[2])
                    else:
                        bridge.act_walk(target[1], target[2])
            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            if raw["dead"] or raw["dungeon_level"] != start_level:
                break
            if kind == "heal" and raw["belt_heals"] > start_belt:
                break  # 到手
            if target[0] == "pick":
                still_there = any(
                    it["x"] == hx and it["y"] == hy
                    for it in raw.get("floor_items", []) if it.get(flag))
                if not still_there:
                    break  # 目标离地:药进腰带 / 装备上身,均算完成
            elif target[0] == "open":
                # 开门型目标:门格真的变可走才算完成(贴脸≠已开,动画要几拍)
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[target[3]] = (target[1], target[2], False)
                    pi = target[3]
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif max(abs(pos[0] - target[1]), abs(pos[1] - target[2])) <= 1:
                pi = target[3] + 1  # 到达路点,继续下一段
                target = None
                stall = 0
                last_pos = pos
                continue
            if pos == last_pos:
                stall += 1
                if stall == 3 and target is not None:
                    # 命令可能被打断:原地重发一次
                    if target[0] == "open":
                        bridge.act_operate(target[1], target[2])
                    elif target[0] == "pick":
                        act()
                    else:
                        bridge.act_walk(target[1], target[2])
                if stall >= 6:
                    break  # 重发后仍无进展 → 交还控制权,下次按键重新规划
            else:
                stall = 0
            last_pos = pos
        return raw, beats

    @staticmethod
    def _nearest_monster(obs):
        px, py = obs["player_x"], obs["player_y"]
        best, best_d = None, None
        for m in obs["monsters"]:
            d = abs(m["x"] - px) + abs(m["y"] - py)
            if best_d is None or d < best_d:
                best, best_d = m, d
        return best

    @staticmethod
    def _nearest_dist(raw):
        px, py = raw["player_x"], raw["player_y"]
        dists = [max(abs(m["x"] - px), abs(m["y"] - py)) for m in raw["monsters"]]
        return min(dists) if dists else None

    @staticmethod
    def _combat_reward(prev, cur) -> float:
        """逐刀伤害奖励 + 击杀奖励(id 匹配的血量差分;1 层无怪物互殴,归因安全)。"""
        cur_hp = {m["id"]: m["hp"] for m in cur["monsters"]}
        r = 0.0
        for m in prev["monsters"]:
            hp_after = cur_hp.get(m["id"], 0)  # id 消失 = 已死,击后血量按 0 计
            damage = m["hp"] - hp_after
            if damage <= 0:
                continue
            hp_after_frac = hp_after / max(1, m["max_hp"])
            finish_mult = 1.0 + 0.5 * (1.0 - hp_after_frac)  # 残血系数 1.0→1.5
            r += 0.5 * (damage / max(1, m["max_hp"])) * finish_mult
            if m["id"] not in cur_hp:
                r += 1.0  # 击杀收头
        return r

    def _reward(self, prev, cur) -> float:
        cls = type(self)
        r = 0.01 * (cur["xp"] - prev["xp"])
        dl = cur["dungeon_level"] - prev["dungeon_level"]
        if self.descend_ladder and dl > 0:
            # v17:深度递进——每个 N→N+1 付 8×N(L1→2 仍是 8,锚定旧章;
            # L2→3 付 16、L3→4 付 24……越深越值钱,给"往下活着"一个未来)
            r += 8.0 * sum(range(prev["dungeon_level"], cur["dungeon_level"]))
        else:
            r += 8.0 * dl
        if cur["armor_class"] > prev["armor_class"]:
            # v15(奖励 v3——自 v6 冻结以来首次修订):穿甲一次性入账,自举塑形。
            # 动机=教训十三:护甲收益(每击少几点血,摊几百步)对 3M 步视界统计
            # 不可见,0/32 穿甲——行为必须先发生,真实回报才有机会被观测。
            # Goodhart 预审:AutoEquip 只填空槽、无卸装/丢弃动作、属性点不自动
            # 分配、未鉴定魔法加成不生效 → AC 是守恒存量,ΔAC>0 ⟺ 穿上装备,
            # 不可刷。近似势函数塑形,取正半边(死亡掉装的负 Δ 不罚,死亡已有
            # -2.0)。v15b 计划:学会后拆塑形微调,检验行为是否内化。
            r += 0.5 * (cur["armor_class"] - prev["armor_class"])
        if cur["dungeon_level"] == prev["dungeon_level"]:
            r += cls._combat_reward(prev, cur)
        # 接近塑形:仅当是"自己走近"才有奖励(v2 教训:怪主动贴脸也计分,
        # 会训出"站桩钓鱼却不开打"的白嫖策略)
        if cur["dungeon_level"] == prev["dungeon_level"]:
            moved = (cur["player_x"], cur["player_y"]) != (prev["player_x"], prev["player_y"])
            d0, d1 = cls._nearest_dist(prev), cls._nearest_dist(cur)
            if moved and d0 is not None and d1 is not None:
                r += 0.005 * (d0 - d1)
            if not moved:
                r -= 0.002  # 反面壁/站桩
        if cur["dead"]:
            r -= 8.0 * cur["dungeon_level"] if self.death_ladder else 2.0
        if cur["victory"]:
            r += 10.0
        return float(r)

    @classmethod
    def _vectorize(cls, obs) -> np.ndarray:
        px, py = obs["player_x"], obs["player_y"]
        nearest = cls._nearest_dist(obs)
        stairs = [t for t in obs.get("triggers", []) if t["msg"] == 0]  # WM_DIABNEXTLVL
        if stairs:
            st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
            stair_dx, stair_dy = (st["x"] - px) / 56.0, (st["y"] - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(obs["monsters"]) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            obs["monsters"], key=lambda m: abs(m["x"] - px) + abs(m["y"] - py)
        )[:_K_MONSTERS]
        for m in monsters:
            vec += [(m["x"] - px) / 20.0, (m["y"] - py) / 20.0, m["hp"] / max(1, m["max_hp"]), 1.0]
        vec += [0.0, 0.0, 0.0, 0.0] * (_K_MONSTERS - len(monsters))
        lm = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(v) for v in lm["walkable"]]
        vec += [float(v) for v in lm["monster"]]
        heals = [it for it in obs.get("floor_items", []) if it.get("heal")]
        if heals:  # v13:瓶盲修复——喝药/捡药两个键的前置条件入观测
            h = min(heals, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [obs.get("belt_heals", 0) / 8.0,
                    max(-1.0, min(1.0, (h["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (h["y"] - py) / 20.0)), 1.0]
        else:
            vec += [obs.get("belt_heals", 0) / 8.0, 0.0, 0.0, 0.0]
        gears = [it for it in obs.get("floor_items", []) if it.get("gear")]
        ac = min(1.0, obs.get("armor_class", 0) / 50.0)
        if gears:  # v14:装备章——捡装备键的前置条件入观测(教训十一验收单)
            g = min(gears, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [ac,
                    max(-1.0, min(1.0, (g["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (g["y"] - py) / 20.0)), 1.0]
        else:
            vec += [ac, 0.0, 0.0, 0.0]
        return np.asarray(vec, dtype=np.float32)
