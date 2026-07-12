"""DiabloGymEnv —— Gymnasium 包装(v0:结构化向量观测 + 离散动作)。

观测向量(float32,长度 12 + K*4 + 2*(2R+1)² + 9,R=5 时共 295):
  [hp/maxhp, mana/maxmana, xp(log1p/10), gold/1000, char_level/50,
   dungeon_level/16, player_x/112, player_y/112,
   存活怪数/50, 最近怪距离/30(无怪=1),
   下一项必需主线目标方向 dx/56, dy/56(普通层即下行楼梯;无则 0,0)]
  + K 个最近怪物的 (dx/20, dy/20, hp/max_hp, 1存在标志)
  + 11×11 局部地图两通道(可走性、怪物占位)——run4 教训:没有空间感知,
    奖励再好也是"盲人拿完美账本"(隔墙锁定、穿墙塑形、找不到房门)
  + [腰带治疗药数/8, 最近地面治疗药 dx/20, dy/20(截断至 ±1), 存在标志]
    (v13,治"瓶盲"——教训十一:动作的前置条件必须可观测,否则策略学不会
    按键纪律)
  + [护甲值/50(截断至 1), 最近可穿装备 dx/20, dy/20(截断至 ±1), 存在标志]
    (v14,装备章:存在标志已预判"槽位为空+属性达标",=1 即值得按)
  + [min(2, 角色等级/max(1,地牢层数))/2]——v19 强弱仪表

动作(Discrete(15)):
  0      原地不动
  1-8    朝八方向走一格(寻路)
  9      交战宏:锁定最近怪物持续追击,直到它死/自己死/换层/超时(≤10 拍)
         (v2 教训:单拍攻击会被下一个走位动作打断,策略学不会"坚持进攻")
  10     探索宏:若存在通关必需剧情目标,先走严格白名单剧情宏;否则走向
         25×25 视野内最近的"可走且未踏足"边疆点;发现猎物
         (最近怪 ≤6 格)立即交还控制权;无边疆点时朝下行楼梯走
         (run5 教训:出生区无可达怪时,反应式策略不会"换个房间找")
  11     主线推进宏(v11+):优先完成法杖台/法杖/L15 任务入口/Vile 书与
         法阵/L16 机关的严格白名单,无待办时再走下行或任务返回触发点。
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
import os
import pathlib
import shutil
import tempfile
from collections import deque
from contextlib import contextmanager

import gymnasium as gym
import numpy as np

from . import bridge, nav

# 八方向(等距地牢的 tile 坐标系)
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
_K_MONSTERS = 8
_MAP_RADIUS = 5  # 11×11 局部地图

# 换层奖金单价(_reward 的 Δ地牢层项;v23 起为具名常量——工人工资剥薪
# 需要在包装器侧按同一公式反算,数字只许存在一份)
DESCEND_UNIT = 8.0

_DEFAULT_ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "engine" / "devilutionx.app" / "Contents" / "Resources"
)
_TEMP_SAVE_LEGACY_PREFIX = "diablogym-saves-"
_TEMP_SAVE_PREFIX = "diablogym-saves-v2-"
_TEMP_SAVE_LOCK = ".owner.lock"
_TEMP_SAVE_REGISTRY_LOCK = (
    f".diablogym-saves-v2.{os.getuid() if hasattr(os, 'getuid') else 0}.global.lock"
)


def _scene_identity(raw) -> tuple[int, bool, int]:
    """主线深度相同的任务副本仍是另一张地图，不能跨图做差分。"""
    depth = int(raw["dungeon_level"])
    is_set = bool(raw.get("is_set_level", False))
    set_id = int(raw.get("set_level_id", 0)) if is_set else 0
    return depth, is_set, set_id


@contextmanager
def _temp_save_registry_lock(root: pathlib.Path):
    """Serialize scratch publication and reclamation across spawn workers.

    The registry file is intentionally persistent and outside the scratch glob.
    Unlinking it would let waiters retain the old inode while newcomers lock a
    new inode, splitting the critical section in two.
    """
    import fcntl

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(root / _TEMP_SAVE_REGISTRY_LOCK, flags, 0o600)
    try:
        registry = os.fdopen(fd, "a+", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise
    acquired = False
    try:
        fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(registry.fileno(), fcntl.LOCK_UN)
        registry.close()


def _cleanup_stale_temp_save_dirs_locked(base: pathlib.Path) -> int:
    """Reclaim stale scratch while the caller holds the registry lock."""
    import fcntl

    removed = 0
    for candidate in base.glob(f"{_TEMP_SAVE_LEGACY_PREFIX}*"):
        if not candidate.is_dir():
            continue
        marker = candidate / _TEMP_SAVE_LOCK
        if not marker.is_file():
            # A v2 creator publishes the directory and owner marker while
            # holding the same registry lock.  Therefore a visible markerless
            # v2 directory can only be debris from a crashed creator.  Older
            # directories predate that invariant and remain fail-closed.
            if candidate.name.startswith(_TEMP_SAVE_PREFIX):
                try:
                    shutil.rmtree(candidate)
                    removed += 1
                except OSError:
                    pass
            continue
        try:
            owner = open(marker, "a+", encoding="utf-8")
        except OSError:
            continue
        acquired = False
        try:
            try:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                continue
            try:
                shutil.rmtree(candidate)
                removed += 1
            except OSError:
                # The owner may have completed cleanup after our path lookup;
                # genuine I/O failures remain for a later startup to retry.
                pass
        finally:
            if acquired:
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
            owner.close()
    return removed


def _cleanup_stale_temp_save_dirs(root: pathlib.Path | None = None) -> int:
    """回收被 SIGKILL 遗留、且已没有进程持锁的新式 scratch 目录。

    v2 的创建/清理由全局事务锁串行化，因此可回收崩溃留下的无 marker
    半成品；旧版本没有这项所有权证据，宁可保留也不猜。flock 由内核在
    进程死亡时释放，因此 PID 复用不会导致误删或漏删。
    """
    base = (pathlib.Path(root) if root is not None
            else pathlib.Path(tempfile.gettempdir()))
    with _temp_save_registry_lock(base):
        return _cleanup_stale_temp_save_dirs_locked(base)


def _create_locked_temp_save_dir():
    import fcntl

    base = pathlib.Path(tempfile.gettempdir())
    with _temp_save_registry_lock(base):
        _cleanup_stale_temp_save_dirs_locked(base)
        directory = tempfile.TemporaryDirectory(
            prefix=_TEMP_SAVE_PREFIX, dir=base)
        owner = open(pathlib.Path(directory.name) / _TEMP_SAVE_LOCK,
                     "a+", encoding="utf-8")
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner.write(f"pid={os.getpid()}\n")
            owner.flush()
            os.fsync(owner.fileno())
        except Exception:
            owner.close()
            directory.cleanup()
            raise
    return directory, owner


class DiabloGymEnv(gym.Env):
    metadata = {"render_modes": []}

    # DevilutionX 是进程内全局单例，不是可重入的多实例引擎。同一
    # 进程可以顺序复用多个 wrapper，但不能交错 step；多环境必须用
    # SubprocVecEnv 之类的多进程方案。在这里显式记账，把静默串状态
    # 变成响亮的异常。
    _engine_initialized = False
    _engine_pid: int | None = None
    _engine_config: tuple[str, str, str, int] | None = None
    _active_token = None
    _temp_save_dir: tempfile.TemporaryDirectory | None = None
    _temp_save_lock = None
    _atfork_registered = False

    @classmethod
    def _after_fork_child(cls) -> None:
        """子进程不得析构父进程仍在使用的 scratch 或原生引擎。"""
        directory, cls._temp_save_dir = cls._temp_save_dir, None
        owner, cls._temp_save_lock = cls._temp_save_lock, None
        if directory is not None:
            finalizer = getattr(directory, "_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()  # 只取消子进程副本；不能 rmtree 父进程目录
        if owner is not None:
            owner.close()

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
        hero_class: int = 0,
    ):
        super().__init__()
        if (isinstance(ticks_per_step, bool)
                or not isinstance(ticks_per_step, (int, np.integer))
                or int(ticks_per_step) <= 0):
            raise ValueError(f"ticks_per_step 必须是正整数，收到 {ticks_per_step!r}")
        if (isinstance(max_steps, bool)
                or not isinstance(max_steps, (int, np.integer))
                or int(max_steps) <= 0):
            raise ValueError(f"max_steps 必须是正整数，收到 {max_steps!r}")
        if (isinstance(hero_class, bool)
                or not isinstance(hero_class, (int, np.integer))
                or int(hero_class) != 0):
            raise ValueError(
                "当前动作/自动加点契约只支持 hero_class=0(战士)；"
                f"收到 {hero_class!r}")

        assets = str(pathlib.Path(assets_dir or _DEFAULT_ASSETS).expanduser().resolve())
        data = str(pathlib.Path(
            data_dir
            or pathlib.Path.home() / "Library/Application Support/diasurgical/devilution"
        ).expanduser().resolve())
        cls = DiabloGymEnv
        pid = os.getpid()
        if not cls._atfork_registered and hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=cls._after_fork_child)
            cls._atfork_registered = True
        if cls._engine_initialized and cls._engine_pid != pid:
            raise RuntimeError(
                "DiabloGym 引擎已在父进程初始化，不能 fork 后复用；"
                "fork 子进程只能立即 exec/os._exit，多环境训练请使用 spawn")

        # bridge.init() 是一个长 C++ 调用，SIGINT 可能恰在它成功返回、
        # Python 尚未来得及写三个 class 属性时转成 KeyboardInterrupt。
        # 原生配置是提交事实源；每次构造都先据此修复可能被异步异常撕裂的
        # Python 账本，不能误删原生仍在使用的临时存档目录。
        native_config = bridge.engine_config()
        if native_config is not None:
            recovered = (str(native_config[0]), str(native_config[1]),
                         str(native_config[2]), int(native_config[3]))
            cls._engine_config = recovered
            cls._engine_pid = pid
            cls._engine_initialized = True
        elif cls._engine_initialized:
            raise RuntimeError(
                "DiabloGym Python/原生单例账本不一致：Python 标为已初始化，"
                "原生桥却未初始化")
        if cls._engine_initialized:
            if cls._engine_config is None:
                raise RuntimeError("DiabloGym 引擎单例状态损坏: 已初始化但配置缺失")
            _, old_saves, _, _ = cls._engine_config
            requested_save = (str(pathlib.Path(save_dir).expanduser().resolve())
                              if save_dir is not None else old_saves)
            requested = (assets, requested_save, data, int(hero_class))
            if requested != cls._engine_config:
                raise RuntimeError(
                    "DevilutionX 是进程内单例，不能用不同的 assets/save/data/"
                    f"hero_class 重复初始化；已有={cls._engine_config!r}, 请求={requested!r}")
            saves = old_saves
        else:
            if save_dir is not None:
                saves = str(pathlib.Path(save_dir).expanduser().resolve())
            else:
                # 存档 scratch 要活到进程内引擎退出，但不应像
                # mkdtemp 那样在每次多进程训练结束后永久遗留磁盘垃圾。
                cls._temp_save_dir, cls._temp_save_lock = _create_locked_temp_save_dir()
                saves = cls._temp_save_dir.name
            try:
                bridge.init(assets_dir=assets, save_dir=saves, data_dir=data,
                            hero_class=int(hero_class))
            except BaseException:
                # 若异步异常发生在原生提交之后，保留 native 与 scratch，
                # 并把 Python 账本补齐；只有原生确实未提交时才回滚磁盘。
                committed = bridge.engine_config()
                if committed is not None:
                    cls._engine_config = (
                        str(committed[0]), str(committed[1]),
                        str(committed[2]), int(committed[3]))
                    cls._engine_pid = pid
                    cls._engine_initialized = True
                elif save_dir is None and cls._temp_save_dir is not None:
                    cls._temp_save_dir.cleanup()
                    cls._temp_save_dir = None
                    if cls._temp_save_lock is not None:
                        cls._temp_save_lock.close()
                        cls._temp_save_lock = None
                raise
            cls._engine_config = (assets, saves, data, int(hero_class))
            cls._engine_pid = pid
            # 提交位必须最后写：若 SIGINT 落在配置/PID 两次赋值之间，下一次
            # 构造会从原生 engine_config 恢复；反过来先置 True 会把缺失 PID
            # 误判成 fork，甚至进不到恢复逻辑。
            cls._engine_initialized = True

        self.ticks_per_step = int(ticks_per_step)
        self.max_steps = int(max_steps)
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
            shape=(12 + _K_MONSTERS * 4 + 2 * side * side + 9,), dtype=np.float32,
        )  # +9 = v13 药 4 维 + v14 装备 4 维 + v19 强弱仪表 1 维
        self._token = object()
        self._raw = None
        self._native_generation: int | None = None
        self._episode_seed: int | None = None
        self._episode_ended = True
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = 0
        self._visited: set[tuple[int, int]] = set()

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options=None):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "禁止在 fork 子进程 reset 父进程已初始化的 DevilutionX；"
                "多环境训练必须使用 spawn")
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        actual_seed = int(actual_seed)
        if not 0 <= actual_seed <= np.iinfo(np.uint32).max:
            raise ValueError(f"seed 必须在 uint32 范围 [0, 2**32-1] 内，收到 {actual_seed}")
        try:
            DiabloGymEnv._active_token = self._token
            self._raw = bridge.reset(seed=actual_seed)
            self._native_generation = int(bridge.episode_generation())
            if self.start_in_dungeon:
                # 城镇布局固定,脚本化走到教堂楼梯(约 500-900 tick,~0.05s)
                self._raw = nav.descend_to_dungeon(bridge)
            self._steps = 0
            self._episode_seed = actual_seed
            self._episode_ended = False
            self._ep_kills = 0
            self._ep_start_xp = int(self._raw["xp"])
            self._visited = {(self._raw["player_x"], self._raw["player_y"])}
            obs = self._vectorize(self._raw)
            info = self._info(self._raw)
        except BaseException:
            # 导航/观测构造也是 reset 事务的一部分；中途失败时不得
            # 留下一个看似可 step 的半初始化 episode。
            try:
                bridge.end_game()
            except Exception:
                pass
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_seed = None
            self._episode_ended = True
            raise
        return obs, info

    def action_masks(self) -> np.ndarray:
        """v16:无效动作掩码(MaskablePPO 协议方法;SubprocVecEnv 经
        env_method("action_masks") 跨进程调用,Monitor 包装经 __getattr__ 透传)。

        只掩码 14 号键:视野内没有可穿装备(floor_items 无 gear 标志,与
        _vectorize 的第 294 维同源)时,键不在动作分布里——空按的负样本从此
        不进梯度(教训十四:塑形只放大不召唤;掩码让键只在机会到场时存在)。
        12/13 号键保持自由:"何时不按"是 v13 已学会的真本事(尽管是风格
        彩票),掩掉等于换考卷,四手牌基线全作废。掩码不保证宏走得完——
        它消灭空按,不消灭白按(路径受阻/12 拍超时/半路挨打仍会失败)。"""
        self._ensure_active(allow_ended=True)
        mask = np.ones(15, dtype=bool)
        mask[14] = any(it.get("gear") for it in self._raw.get("floor_items", []))
        return mask

    def step(self, action: int):
        self._ensure_active()
        if not self.action_space.contains(action):
            raise ValueError(f"动作必须是 {self.action_space}中的整数，收到 {action!r}")
        prev = self._raw
        action = int(action)
        remaining = self.max_steps - self._steps
        if action == 9:
            self._raw, micro = self._macro_engage(max_beats=min(10, remaining))
        elif action == 10:
            self._raw, micro = self._macro_explore(max_beats=min(12, remaining))
        elif action == 11:
            self._raw, micro = self._macro_descend(max_beats=min(12, remaining))
        elif action == 12:
            bridge.act_drink()  # 无药时引擎侧为空操作;站桩惩罚由奖励函数自然覆盖
            self._raw = bridge.step(ticks=self.ticks_per_step)
            micro = 1
        elif action == 13:
            self._raw, micro = self._macro_pickup("heal", max_beats=min(12, remaining))
        elif action == 14:
            self._raw, micro = self._macro_pickup("gear", max_beats=min(12, remaining))
            if bridge.sweep_backpack_gear():
                # PM_GOTHIT 时序窗(审查确认):拾取执行前挨硬直会让装备静默
                # 沉入背包;打捞穿上后刷新观测(无 tick 成本)
                self._raw = bridge.observe()
        else:
            self._apply_action(action)
            self._raw = bridge.step(ticks=self.ticks_per_step)
            micro = 1
        if (self.start_in_dungeon
                and self._raw["dungeon_level"] < prev["dungeon_level"]):
            # 未来若出现新的回城/向上传送路径，宁可终止训练也
            # 不能把 depth=0 空耗轨迹静默喂给 PPO。原生触发层已封住
            # 常见上楼与回城楼梯，这里是第二道 fail-closed 不变量。
            bad_transition = (prev["dungeon_level"], self._raw["dungeon_level"])
            try:
                bridge.end_game()
            finally:
                if DiabloGymEnv._active_token is self._token:
                    DiabloGymEnv._active_token = None
                self._raw = None
                self._native_generation = None
                self._episode_ended = True
            raise RuntimeError(
                f"DiabloGym 禁止地牢层级回退: {bad_transition[0]}→{bad_transition[1]}")
        self._steps += micro
        same_scene = _scene_identity(self._raw) == _scene_identity(prev)
        if not same_scene:
            # 新主层或任务副本:足迹清零。不同地图共用同一坐标系,不清的话
            # 探索宏会把旧图足迹当"已踏足",边疆逻辑整层失效。
            self._visited = set()
        self._visited.add((self._raw["player_x"], self._raw["player_y"]))

        # 击杀统计:同层内 id 消失即击杀(换层时基线失效,跳过)
        if same_scene:
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
        obs = self._vectorize(self._raw)
        if terminated or truncated:
            self._episode_ended = True
        return obs, reward, terminated, truncated, info

    def _info(self, raw):
        info = {"episode_seed": self._episode_seed}
        if self.include_raw:
            # info 属于调用方；不得把内部奖励/宏状态依赖的可变 raw
            # 直接泄露出去，否则回调或调试代码修改 info["raw"] 会篡改下一拍奖励。
            # raw 的嵌套结构只有这些 list[dict]；定向拷贝比泛化
            # deepcopy 快两个数量级，避免 include_raw 默认路径吞掉训练吞吐。
            snapshot = dict(raw)
            for key in ("monsters", "floor_items", "triggers", "progression_targets"):
                snapshot[key] = [dict(entry) for entry in raw.get(key, ())]
            info["raw"] = snapshot
        return info

    def _ensure_active(self, *, allow_ended: bool = False) -> None:
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "禁止在 fork 子进程使用父进程已初始化的 DevilutionX；"
                "多环境训练必须使用 spawn")
        if self._raw is None:
            raise gym.error.ResetNeeded("step/action_masks 前必须先调用 reset()")
        if DiabloGymEnv._active_token is not self._token:
            raise RuntimeError(
                "检测到同进程多个 DiabloGymEnv 交错使用；引擎是全局单例。"
                "请顺序 reset/使用，或改用 SubprocVecEnv")
        if int(bridge.episode_generation()) != self._native_generation:
            raise RuntimeError(
                "引擎已被直接 bridge.reset() 或其他 wrapper 重置，"
                "当前环境缓存已失效；请对本环境重新 reset()")
        if self._episode_ended and not allow_ended:
            raise gym.error.ResetNeeded("episode 已终止/截断，继续 step() 前必须 reset()")

    def close(self):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            # 子进程继承的是父进程多线程引擎的一份不安全快照；绝不能
            # 在这里进入 SDL/NetClose/Lua。OS 会回收子进程地址空间。
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_ended = True
            return
        if DiabloGymEnv._active_token is self._token:
            bridge.end_game()
            DiabloGymEnv._active_token = None
        self._raw = None
        self._native_generation = None
        self._episode_ended = True

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
        start_scene = _scene_identity(self._raw)
        raw = prev = self._raw
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_attack_monster(tid)
            raw = bridge.step(ticks=self.ticks_per_step)
            cur_target = next((m for m in raw["monsters"] if m["id"] == tid), None)
            if cur_target is None or raw["dead"] or _scene_identity(raw) != start_scene:
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

    _PROGRESSION_PRIORITY = {
        "lazarus_stand": 0,
        "lazarus_staff": 1,
        "vile_entrance": 2,
        "vile_book": 3,
        "vile_center_circle": 4,
        "diablo_switch": 5,
    }

    @staticmethod
    def _progression_present(raw, target) -> bool:
        """坐标+种类是单场景内稳定身份；目标消失即本次交互已提交。"""
        return any(
            p.get("kind") == target.get("kind")
            and int(p.get("x", -1)) == int(target["x"])
            and int(p.get("y", -1)) == int(target["y"])
            for p in raw.get("progression_targets", ())
        )

    @staticmethod
    def _progression_ready(raw, target) -> bool:
        px, py = int(raw["player_x"]), int(raw["player_y"])
        tx, ty = int(target["x"]), int(target["y"])
        gx, gy = int(target["goal_x"]), int(target["goal_y"])
        action = target["action"]
        if action == "walk":
            return (px, py) == (gx, gy)
        if action == "pickup":
            return max(abs(px - tx), abs(py - ty)) <= 2
        if action == "operate" and bool(target.get("exact")):
            return (px, py) == (gx, gy)
        if action == "operate":
            return max(abs(px - tx), abs(py - ty)) <= 1
        raise RuntimeError(f"未知剧情目标动作: {action!r}")

    @staticmethod
    def _issue_progression(target) -> None:
        action = target["action"]
        if action == "operate":
            bridge.act_operate(int(target["x"]), int(target["y"]))
        elif action == "pickup":
            bridge.act_pickup_progression(int(target["x"]), int(target["y"]))
        elif action == "walk":
            bridge.act_walk(int(target["goal_x"]), int(target["goal_y"]))
        else:
            raise RuntimeError(f"未知剧情目标动作: {action!r}")

    def _macro_progression(self, max_beats: int = 12):
        """推进严格白名单中的下一项通关必需交互。

        action 10/11 都调用此宏：平坦策略、FARM 工人与 DIVE 经理因此不会
        在不同控制路径上得到两套可达性。Vile 书要求精确站圈；普通机关只
        要相邻。全局 BFS 仍只把门/桶当软墙，不传送、不穿墙。
        """
        raw = self._raw
        targets = [dict(p) for p in raw.get("progression_targets", ())]
        if not targets:
            return bridge.step(ticks=self.ticks_per_step), 1

        candidates = []
        for target in targets:
            required = {"kind", "action", "x", "y", "goal_x", "goal_y", "exact"}
            if set(target) != required:
                raise RuntimeError(f"剧情目标 schema 异常: {target!r}")
            ready = self._progression_ready(raw, target)
            path = [] if ready else self._plan_descend_path(
                raw, int(target["goal_x"]), int(target["goal_y"]),
                avoid_monsters=True)
            px, py = int(raw["player_x"]), int(raw["player_y"])

            def assess(candidate_path):
                ex, ey = ((candidate_path[-1][0], candidate_path[-1][1])
                          if candidate_path else (px, py))
                if target["action"] == "operate" and not bool(target["exact"]):
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 1
                elif target["action"] == "pickup":
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 2
                else:
                    remaining_ = max(abs(ex - int(target["goal_x"])),
                                     abs(ey - int(target["goal_y"])))
                    limit = 0
                reachable_ = ready or (
                    candidate_path is not None and remaining_ <= limit)
                return remaining_, reachable_

            remaining, reachable = assess(path)
            if not ready and not reachable:
                # “避怪 BFS”即使只能走到怪物墙前也会返回一条 partial path，
                # 不是 None。若只在 None 时回退，L16 会把真正可达的第二个
                # switch 错判为远目标，反复走向另一扇尚未开放的墙。
                fallback = self._plan_descend_path(
                    raw, int(target["goal_x"]), int(target["goal_y"]))
                fallback_remaining, fallback_reachable = assess(fallback)
                old_rank = (not reachable, remaining,
                            len(path) if path is not None else 10**9)
                new_rank = (not fallback_reachable, fallback_remaining,
                            len(fallback) if fallback is not None else 10**9)
                if new_rank < old_rank:
                    path = fallback
                    remaining, reachable = fallback_remaining, fallback_reachable
            priority = self._PROGRESSION_PRIORITY.get(target["kind"], 999)
            candidates.append((not reachable, priority, remaining,
                               len(path) if path is not None else 10**9,
                               target, path))

        _, _, _, _, target, path = min(candidates, key=lambda c: c[:4])
        if path is None and not self._progression_ready(raw, target):
            return bridge.step(ticks=self.ticks_per_step), 1

        start_scene = _scene_identity(raw)
        pi = 0
        command = None  # ("open"/"walk"/"progress", x, y, path_index)
        last_pos = (raw["player_x"], raw["player_y"])
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            if not self._progression_present(raw, target):
                break
            if command is None:
                if self._progression_ready(raw, target):
                    self._issue_progression(target)
                    command = ("progress", int(target["x"]), int(target["y"]), pi)
                elif pi >= len(path):
                    break
                else:
                    nxt = None
                    for j in range(pi, min(pi + 8, len(path))):
                        if path[j][2]:
                            nxt = ("open", path[j][0], path[j][1], j)
                            break
                    if nxt is None:
                        j = min(pi + 7, len(path) - 1)
                        nxt = ("walk", path[j][0], path[j][1], j)
                    command = nxt
                    if command[0] == "open":
                        bridge.act_operate(command[1], command[2])
                    else:
                        bridge.act_walk(command[1], command[2])

            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break
            if not self._progression_present(raw, target):
                break

            if command[0] == "open":
                if bridge.probe_tile(command[1], command[2])["walkable"]:
                    path[command[3]] = (command[1], command[2], False)
                    pi = command[3]
                    command = None
                    stall = 0
                    last_pos = pos
                    continue
            elif command[0] == "walk":
                if max(abs(pos[0] - command[1]), abs(pos[1] - command[2])) <= 1:
                    pi = command[3] + 1
                    command = None
                    stall = 0
                    last_pos = pos
                    continue

            if pos == last_pos:
                stall += 1
                if stall == 3:
                    if command[0] == "open":
                        bridge.act_operate(command[1], command[2])
                    elif command[0] == "walk":
                        bridge.act_walk(command[1], command[2])
                    else:
                        self._issue_progression(target)
                if stall >= 6:
                    break
            else:
                stall = 0
            last_pos = pos
        return raw, beats

    def _macro_explore(self, max_beats: int = 12):
        """探索宏:走向最近的未踏足可走边疆点;发现猎物立即交还控制权。"""
        if self._raw.get("progression_targets"):
            return self._macro_progression(max_beats=max_beats)
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
            transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in raw.get("triggers", [])
                      if t["msg"] == transition]
            if not stairs:
                return bridge.step(ticks=self.ticks_per_step), 1
            tx, ty = stairs[0]["x"], stairs[0]["y"]

        start_scene = _scene_identity(raw)
        last_pos = (px, py)
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_walk(tx, ty)
            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            nd = self._nearest_dist(raw)
            if (raw["dead"] or _scene_identity(raw) != start_scene
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
        if self._raw.get("progression_targets"):
            return self._macro_progression(max_beats=max_beats)
        raw = self._raw
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        stairs = [t for t in raw.get("triggers", [])
                  if t["msg"] == transition]
        if not stairs:
            return bridge.step(ticks=self.ticks_per_step), 1
        px, py = raw["player_x"], raw["player_y"]
        st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
        sx, sy = st["x"], st["y"]
        start_scene = _scene_identity(raw)

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
            if raw["dead"] or _scene_identity(raw) != start_scene:
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
        start_scene = _scene_identity(raw)

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
            if raw["dead"] or _scene_identity(raw) != start_scene:
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
            r += DESCEND_UNIT * sum(range(prev["dungeon_level"], cur["dungeon_level"]))
        else:
            r += DESCEND_UNIT * dl
        if cur["armor_class"] > prev["armor_class"]:
            # v15(奖励 v3——自 v6 冻结以来首次修订):穿甲一次性入账,自举塑形。
            # 动机=教训十三:护甲收益(每击少几点血,摊几百步)对 3M 步视界统计
            # 不可见,0/32 穿甲——行为必须先发生,真实回报才有机会被观测。
            # Goodhart 预审:AutoEquip 只填空槽、无卸装/丢弃动作、属性点不自动
            # 分配、未鉴定魔法加成不生效 → AC 是守恒存量,ΔAC>0 ⟺ 穿上装备,
            # 不可刷。近似势函数塑形,取正半边(死亡掉装的负 Δ 不罚,死亡已有
            # -2.0)。v15b 计划:学会后拆塑形微调,检验行为是否内化。
            r += 0.5 * (cur["armor_class"] - prev["armor_class"])
        same_scene = _scene_identity(cur) == _scene_identity(prev)
        if same_scene:
            r += cls._combat_reward(prev, cur)
        # 接近塑形:仅当是"自己走近"才有奖励(v2 教训:怪主动贴脸也计分,
        # 会训出"站桩钓鱼却不开打"的白嫖策略)
        if same_scene:
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
        advance = list(obs.get("progression_targets", ()))
        if advance:
            st = min(advance, key=lambda t: max(
                abs(t["goal_x"] - px), abs(t["goal_y"] - py)))
            sx, sy = st["goal_x"], st["goal_y"]
        else:
            transition = (bridge.WM_DIABRTNLVL if obs.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in obs.get("triggers", [])
                      if t["msg"] == transition]
            st = min(stairs, key=lambda t: max(
                abs(t["x"] - px), abs(t["y"] - py))) if stairs else None
            sx, sy = (st["x"], st["y"]) if st is not None else (px, py)
        if advance or st is not None:
            stair_dx, stair_dy = (sx - px) / 56.0, (sy - py) / 56.0
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
        ac = max(0.0, min(1.0, obs.get("armor_class", 0) / 50.0))
        if gears:  # v14:装备章——捡装备键的前置条件入观测(教训十一验收单)
            g = min(gears, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [ac,
                    max(-1.0, min(1.0, (g["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (g["y"] - py) / 20.0)), 1.0]
        else:
            vec += [ac, 0.0, 0.0, 0.0]
        # v19:强弱仪表(教训五族)。"够不够强、该不该下"的决策变量是
        # 等级/层数之比,但 dim5 的 char_level/50 让 1 级和 3 级只差 0.04,
        # 对策略近乎不可见——农到多强才下楼,得先看得见"多强"。
        # 比值 1.0 = 等级与层数持平,>1 越级碾压,<1 越级送死;封顶 2 归一。
        vec += [min(2.0, obs["char_level"] / max(1, obs["dungeon_level"])) / 2.0]
        return np.asarray(vec, dtype=np.float32)
