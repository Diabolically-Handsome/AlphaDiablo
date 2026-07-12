/**
 * @file diablogym.cpp
 *
 * DiabloGym v0 —— DevilutionX 无头嵌入桥(pybind11)。
 *
 * 嵌入方式与上游 test/timedemo_test.cpp 同源:HeadlessMode + loopback 单机,
 * 由 Python 侧逐 tick 驱动主循环(复刻 RunGameLoop 循环体,去掉墙钟限速与绘制),
 * 动作走网络命令层(NetSendCmd*)—— 与多人协议同一条路,天然支持日后联机部署。
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_SDL3
#include <SDL3/SDL.h>
#else
#include <SDL.h>
#endif

#include "DiabloUI/diabloui.h" // _uiheroinfo
#include "control/control.hpp" // FreeControlPan
#include "controls/control_mode.hpp"
#include "controls/plrctrls.h" // UseBeltItem(喝药键 v12)
#include "cursor.h"
#include "diablo.h"
#include "engine/render/scrollrt.h" // CalcViewportGeometry
#include "gmenu.h"
#include "inv.h"     // v14:AutoEquip(背包打捞,PM_GOTHIT 时序窗修复)
#include "options.h" // v14:自动穿装备选项(盔甲/头盔/首饰默认关)
#include "qol/monhealthbar.h"
#include "qol/xpbar.h"
#include "engine/assets.hpp"
#include "engine/backbuffer_state.hpp"
#include "engine/demomode.h" // FetchMessage
#include "engine/events.hpp" // SetEventHandler
#include "engine/palette.h"
#include "engine/random.hpp"
#include "engine/sound.h"
#include "game_mode.hpp"
#include "headless_mode.hpp"
#include "init.hpp"
#include "interfac.h"
#include "items.h"
#include "levels/gendung.h"
#include "levels/tile_properties.hpp"
#include "levels/trigs.h"
#include "loadsave.h" // giNumberOfLevels
#include "lua/lua_event.hpp"
#include "lua/lua_global.hpp"
#include "menu.h" // gSaveNumber
#include "minitext.h"
#include "monster.h"
#include "msg.h"
#include "multi.h"
#include "nthread.h"
#include "objects.h" // FindObjectAtPosition / isDoor(下楼宏的门感知)
#include "options.h"
#include "pfile.h"
#include "player.h"
#include "panels/info_box.hpp"
#include "portal.h"
#include "qol/chatlog.h"
#include "quests.h"
#include "tables/monstdat.h"
#include "tables/playerdat.hpp"
#include "stores.h"
#include "utils/display.h"
#include "utils/paths.h"

#ifndef SDL_EVENT_QUIT
#define SDL_EVENT_QUIT SDL_QUIT
#endif

namespace py = pybind11;
using namespace devilution;

namespace {

bool gEngineInited = false;
bool gInGame = false;
bool gStartupTick = true; // 对应 RunGameLoop 里的 gbGameLoopStartup
int gHeroClass = 0;       // HeroClass::Warrior
int gStallPrints = 0;     // 逻辑失速诊断打印限额
std::string gAssetsDir;
std::string gSaveDir;
std::string gDataDir;
uint64_t gEpisodeGeneration = 0;
bool gLuaInitialized = false;
bool gExitCleanupRegistered = false;
int64_t gEnginePid = -1;
bool gMonotonicQuestTurnInUsed = false;

void EndGame();
void EngineShutdownAtExit() noexcept;

int64_t CurrentProcessId()
{
#ifdef _WIN32
	return static_cast<int64_t>(_getpid());
#else
	return static_cast<int64_t>(getpid());
#endif
}

void EnsureEngineProcess(const char *operation)
{
	if (gEngineInited && gEnginePid != CurrentProcessId())
		throw std::runtime_error(
		    std::string(operation)
		    + ": 禁止在 fork 子进程复用父进程已初始化的 DevilutionX；请使用 spawn");
}

void CleanupFailedEngineInit() noexcept
{
	if (gLuaInitialized) {
		try {
			LuaShutdown();
		} catch (...) {
		}
		gLuaInitialized = false;
	}
	try {
		FreeItemGFX();
	} catch (...) {
	}
	MpqArchives.clear();
	if (SDL_WasInit((~0U) & ~SDL_INIT_HAPTIC) != 0)
		SDL_Quit();
}

void EnsureInGame(const char *operation)
{
	EnsureEngineProcess(operation);
	if (!gInGame || MyPlayer == nullptr)
		throw std::runtime_error(std::string(operation) + ": 先调用 reset()");
}

bool CanAcceptPlayerAction(const char *operation)
{
	EnsureInGame(operation);
	// Step 正常会在返回 Python 前结算拍尾换层；这里仍做纵深防御，覆盖
	// probe 直接排队、异常中断或未来新增入口留下的 PM_NEWLVL 窗口。
	// 若此时向网络命令队列塞动作，SyncLoad 会先切到新地图，随后
	// ProcessGameMessagePackets 就会把旧场景命令施加到新场景。
	return MyPlayer->_pmode != PM_NEWLVL && !MyPlayer->_pLvlChanging;
}

int ConceptualDungeonDepth()
{
	if (!setlevel)
		return static_cast<int>(currlevel);
	const int returnLevel = GetMapReturnLevel();
	if (returnLevel <= 0)
		throw std::runtime_error(
		    "DiabloGym 遇到无法映射到主地牢深度的 set-level: "
		    + std::to_string(static_cast<int>(setlvlnum)));
	return returnLevel;
}

void DiscardPendingEvents()
{
	// 换层事件在 game_loop 的拍尾入 SDL 队列。若该拍恰好命中
	// episode 截断，下一局 reset 会早于下一次 PumpSdlEvents；不清队列
	// 就会把上局的 WM_DIABNEXTLVL 施加给新英雄，造成跨局状态泄漏。
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
	}
}

bool DummyGetHeroInfo(_uiheroinfo * /*info*/)
{
	return true;
}

int CountBeltHeals();            // 定义在动作区(v12);Observe 的 raw 字段也要用
bool IsHealItem(const Item &);   // 定义在动作区(v13);floor_items 的 heal 标志也要用
bool IsWantedGear(Item &);       // 定义在动作区(v14);floor_items 的 gear 标志也要用

std::string ExpectedMainArchivePath(const std::string &dataDir)
{
	for (const char *name : { "DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq" }) {
		const std::filesystem::path candidate = std::filesystem::path(dataDir) / name;
		std::error_code error;
		if (std::filesystem::is_regular_file(candidate, error) && !error)
			return candidate.string();
	}
	throw std::runtime_error("data_dir 缺少 DIABDAT.MPQ/diabdat.mpq/spawn.mpq: " + dataDir);
}

// 空事件处理器:demo::FetchMessage 在 CurrentEventHandler==DisableInputEventHandler
// 时拒绝吐出事件(demomode.cpp:727),必须装一个"游戏中"处理器才能解锁事件流。
// 事件的实际分发在 PumpSdlEvents 里完成,这里无需处理。
void GymEventHandler(const SDL_Event & /*event*/, uint16_t /*modState*/)
{
}

void CreateFreshHeroSave()
{
	// 每个 episode 重建 0 号存档槽 → 每局都是全新 1 级英雄,可复现
	Players.resize(1);
	MyPlayerId = 0;
	MyPlayer = &Players[MyPlayerId];
	*MyPlayer = {};

	_uiheroinfo heroInfo = {};
	heroInfo.saveNumber = 0;
	std::snprintf(heroInfo.name, sizeof(heroInfo.name), "Gym");
	heroInfo.heroclass = static_cast<HeroClass>(gHeroClass);
	if (!pfile_ui_save_create(&heroInfo))
		throw std::runtime_error("pfile_ui_save_create failed");
	gSaveNumber = 0;
}

// 同步版关卡加载:复刻 interfac.cpp DoLoad 的各分支。
// 无头模式不需要进度动画,因此绕开上游线程化的 ShowProgress,单线程完成加载。
void SyncLoad(interface_mode uMsg)
{
	Player &myPlayer = *MyPlayer;
	tl::expected<void, std::string> loadResult;

	switch (uMsg) {
	case WM_DIABNEWGAME:
		myPlayer.pOriginalCathedral = !gbIsHellfire;
		FreeGameMem();
		pfile_remove_temp_files();
		loadResult = LoadGameLevel(true, ENTRY_MAIN);
		break;
	case WM_DIABNEXTLVL:
		pfile_save_level();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_MAIN);
		break;
	case WM_DIABPREVLVL:
		pfile_save_level();
		FreeGameMem();
		currlevel--;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_PREV);
		break;
	case WM_DIABSETLVL:
		pfile_save_level();
		setlevel = true;
		leveltype = setlvltype;
		currlevel = static_cast<uint8_t>(setlvlnum);
		FreeGameMem();
		loadResult = LoadGameLevel(false, ENTRY_SETLVL);
		break;
	case WM_DIABRTNLVL:
		pfile_save_level();
		setlevel = false;
		FreeGameMem();
		currlevel = GetMapReturnLevel();
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_RTNLVL);
		break;
	case WM_DIABWARPLVL:
		pfile_save_level();
		FreeGameMem();
		GetPortalLevel();
		loadResult = LoadGameLevel(false, ENTRY_WARPLVL);
		break;
	case WM_DIABTOWNWARP:
		pfile_save_level();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_TWARPDN);
		break;
	case WM_DIABTWARPUP:
		pfile_save_level();
		FreeGameMem();
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_TWARPUP);
		break;
	case WM_DIABRETOWN:
		pfile_save_level();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_MAIN);
		break;
	default:
		throw std::runtime_error("SyncLoad: 未支持的 interface_mode " + std::to_string(static_cast<int>(uMsg)));
	}

	if (!loadResult.has_value())
		throw std::runtime_error("关卡加载失败: " + loadResult.error());

	// ProgressEventHandler WM_DONE 分支的无头必需部分:宣告加入关卡
	NetSendCmdLocParam2(true, CMD_PLAYER_JOINLEVEL, myPlayer.position.tile, myPlayer.plrlevel,
	    myPlayer.plrIsOnSetLevel ? 1 : 0);
	// 本桥固定是 loopback 单机；OnPlayerJoinLevel 对本地已激活玩家的
	// 唯一同步效果就是清掉 _pLvlChanging。拍尾同步加载后会立即
	// Observe，不能等到下一个 Python 动作后才处理 join 包，否则动作
	// guard 会多吞一次新场景的合法动作。队列中的 join 包下拍再清一次
	// 是幂等的，且仍保留与上游相同的消息路径。
	if (!gbIsMultiplayer)
		myPlayer._pLvlChanging = false;
	// 复刻上游 WM_DONE 分支的 NewCursor(CURSOR_HAND)(interfac.cpp,无头下被
	// skipRendering 跳过):拾取的到位判定要求 pcurs==CURSOR_HAND(player.cpp),
	// 每次换层都重申,把这个隐性不变量钉死(v13 审查发现)
	NewCursor(CURSOR_HAND);
	gStartupTick = true;
}

// 复刻 GameEventHandler 的自定义事件分支(关卡切换等),改走同步加载
void PumpSdlEvents()
{
	SDL_Event event;
	uint16_t modState;
	// 注意必须是 devilution::FetchMessage(events.hpp 的真实事件泵);
	// demo::FetchMessage 在非 demo 模式下会吞掉除 QUIT 外的一切事件
	while (FetchMessage(&event, &modState)) {
		if (event.type == SDL_EVENT_QUIT) {
			gbRunGame = false;
			break;
		}
		if (IsCustomEvent(event.type)) {
			nthread_ignore_mutex(true);
			try {
				SyncLoad(GetCustomEvent(event));
			} catch (...) {
				nthread_ignore_mutex(false);
				throw;
			}
			nthread_ignore_mutex(false);
			continue;
		}
		// 无头环境不产生键鼠事件;动作全部经由网络命令层注入
	}
}

py::dict Observe()
{
	EnsureInGame("observe");
	py::dict obs;
	const Player &player = *MyPlayer;

	obs["player_x"] = static_cast<int>(player.position.tile.x);
	obs["player_y"] = static_cast<int>(player.position.tile.y);
	obs["hp"] = player._pHitPoints >> 6;
	obs["max_hp"] = player._pMaxHP >> 6;
	obs["mana"] = player._pMana >> 6;
	obs["max_mana"] = player._pMaxMana >> 6;
	obs["xp"] = static_cast<uint64_t>(player._pExperience);
	obs["gold"] = player._pGold;
	obs["char_level"] = static_cast<int>(player.getCharacterLevel());
	// `currlevel` 在任务副本中会被复用为 `_setlevels` 枚举值（例如
	// Vile Betrayer=5），绝不是主线深度。训练奖励、死亡定价和排行榜
	// 深度必须使用该副本的返回主层；同时保留场景身份，避免进入/退出
	// 任务副本时把整张旧地图的怪物消失误算成击杀。
	obs["dungeon_level"] = ConceptualDungeonDepth();
	obs["engine_level"] = static_cast<int>(currlevel);
	obs["is_set_level"] = setlevel;
	obs["set_level_id"] = setlevel ? static_cast<int>(setlvlnum) : 0;
	obs["level_type"] = static_cast<int>(leveltype);
	obs["player_mode"] = static_cast<int>(player._pmode);
	obs["walkpath0"] = static_cast<int>(player.walkpath[0]);
	obs["future_x"] = static_cast<int>(player.position.future.x);
	obs["future_y"] = static_cast<int>(player.position.future.y);
	obs["dest_action"] = static_cast<int>(player.destAction);
	obs["dead"] = player._pmode == PM_DEATH || (player._pHitPoints >> 6) <= 0;
	obs["game_over"] = !gbRunGame;
	obs["victory"] = !IsDiabloAlive(false);
	obs["belt_heals"] = CountBeltHeals(); // v12 起入 raw;v13 起由 env 写进观测向量(瓶盲修复)
	obs["armor_class"] = player.GetArmor(); // v14:护甲值(_pIBonusAC + _pIAC + 敏捷/5)
	// 单向训练任务不能回城找 Cain，但原版单机的 Lazarus 主线硬性要求
	// “捡法杖→回城交给 Cain→再下 L15”。桥在拾取法杖后只自动执行这一次
	// 等价交付（见 Step）；把任务状态与是否用过适配器留在 raw，便于探针和
	// 轨迹审计。它们不进入策略向量，也不伪装成原版自然流程。
	const Quest &betrayerQuest = Quests[Q_BETRAYER];
	obs["betrayer_quest_active"] = static_cast<int>(betrayerQuest._qactive);
	obs["betrayer_quest_stage"] = static_cast<int>(betrayerQuest._qvar1);
	obs["betrayer_portal_stage"] = static_cast<int>(betrayerQuest._qvar2);
	obs["monotonic_quest_turn_in_used"] = gMonotonicQuestTurnInUsed;

	py::list monsters;
	for (size_t i = 0; i < ActiveMonsterCount; i++) {
		const unsigned monsterId = ActiveMonsters[i];
		const Monster &monster = Monsters[monsterId];
		// 引擎以定点 HP 的整数部分判死(hasNoLife)。只比较 raw
		// hitPoints<=0 会把 0<HP<1 的已死怪以 hp=0 多暴露一拍，使 Python
		// 侧在它下拍消失时丢掉击杀奖励。
		if (monster.hasNoLife())
			continue;
		py::dict m;
		m["id"] = monsterId;
		m["type"] = static_cast<int>(monster.type().type);
		m["x"] = static_cast<int>(monster.position.tile.x);
		m["y"] = static_cast<int>(monster.position.tile.y);
		m["hp"] = monster.hitPoints >> 6;
		m["max_hp"] = monster.maxHitPoints >> 6;
		monsters.append(m);
	}
	obs["monsters"] = monsters;

	py::list items;
	for (int i = 0; i < ActiveItemCount; i++) {
		Item &item = Items[ActiveItems[i]];
		py::dict it;
		it["x"] = static_cast<int>(item.position.x);
		it["y"] = static_cast<int>(item.position.y);
		it["heal"] = IsHealItem(item);   // v13:捡药宏的目标标志
		it["gear"] = IsWantedGear(item); // v14:捡装备宏的目标标志(空槽+属性达标)
		items.append(it);
	}
	obs["floor_items"] = items;

	// action 10/11 共用的“下一项必需剧情目标”。白名单只包含不完成便
	// 无法抵达 Diablo 的交互，绝不把普通箱子/神龛/支线物体变成全知
	// 自动操作。goal 是实际应抵达的格；Vile 两本书尤其要求玩家精确站在
	// 书西南方的法阵上，直接从任意相邻格操作会被上游静默拒绝。
	py::list progressionTargets;
	auto appendProgression = [&progressionTargets](const char *kind, const char *action,
	                              Point target, Point goal, bool exact) {
		py::dict p;
		p["kind"] = kind;
		p["action"] = action;
		p["x"] = static_cast<int>(target.x);
		p["y"] = static_cast<int>(target.y);
		p["goal_x"] = static_cast<int>(goal.x);
		p["goal_y"] = static_cast<int>(goal.y);
		p["exact"] = exact;
		progressionTargets.append(p);
	};

	if (!gbIsSpawn) {
		if (!setlevel && betrayerQuest._qactive == QUEST_INIT) {
			for (int i = 0; i < ActiveObjectCount; i++) {
				const Object &object = Objects[ActiveObjects[i]];
				if (object._otype == OBJ_LAZSTAND && object.canInteractWith())
					appendProgression("lazarus_stand", "operate", object.position, object.position, false);
			}
			for (int i = 0; i < ActiveItemCount; i++) {
				const Item &item = Items[ActiveItems[i]];
				if (item.IDidx == IDI_LAZSTAFF)
					appendProgression("lazarus_staff", "pickup", item.position, item.position, true);
			}
		}

		if (!setlevel
		    && currlevel == betrayerQuest._qlevel
		    && betrayerQuest._qactive == QUEST_ACTIVE
		    && betrayerQuest._qvar1 >= 2
		    && betrayerQuest._qvar1 <= 3) {
			appendProgression("vile_entrance", "walk", betrayerQuest.position,
			    betrayerQuest.position, true);
		}

		if (setlevel && setlvlnum == SL_VILEBETRAYER
		    && betrayerQuest._qactive == QUEST_ACTIVE) {
			for (int i = 0; i < ActiveObjectCount; i++) {
				const Object &object = Objects[ActiveObjects[i]];
				if (object._otype == OBJ_BOOK2L && object.canInteractWith()) {
					const Point circle = object.position + Direction::SouthWest;
					appendProgression("vile_book", "operate", object.position, circle, true);
				}
				if (object.position == Point { 35, 36 }
				    && IsAnyOf(object._otype, OBJ_MCIRCLE1, OBJ_MCIRCLE2)
				    && object._oVar5 == 3
				    && betrayerQuest._qvar1 <= 4) {
					appendProgression("vile_center_circle", "walk", object.position,
					    object.position, true);
				}
			}
		}

		if (!setlevel && currlevel == 16) {
			for (int i = 0; i < ActiveObjectCount; i++) {
				const Object &object = Objects[ActiveObjects[i]];
				if (IsAnyOf(object._otype, OBJ_LEVER, OBJ_SWITCHSKL)
				    && object.canInteractWith()) {
					appendProgression("diablo_switch", "operate", object.position,
					    object.position, false);
				}
			}
		}
	}
	obs["progression_targets"] = progressionTargets;

	// 关卡出入口(楼梯/传送点)—— agent 的导航目标
	py::list triggers;
	for (int i = 0; i < numtrigs; i++) {
		py::dict t;
		t["x"] = static_cast<int>(trigs[i].position.x);
		t["y"] = static_cast<int>(trigs[i].position.y);
		t["msg"] = static_cast<int>(trigs[i]._tmsg);
		triggers.append(t);
	}
	obs["triggers"] = triggers;

	return obs;
}

void EngineInit(const std::string &assetsDir, const std::string &saveDir, const std::string &dataDir, int heroClass, bool verbose)
{
	if (heroClass != static_cast<int>(HeroClass::Warrior))
		throw std::invalid_argument("当前动作/自动加点契约只支持 hero_class=0(战士)");
	if (gEngineInited) {
		EnsureEngineProcess("init");
		if (assetsDir != gAssetsDir || saveDir != gSaveDir || dataDir != gDataDir || heroClass != gHeroClass)
			throw std::runtime_error("DevilutionX 是进程内单例，不能用不同配置重复 init()");
		return;
	}
	gHeroClass = heroClass;
	const std::string expectedMainArchive = ExpectedMainArchivePath(dataDir);

	// 最先置无头,任何后续错误路径都不得弹 GUI 对话框(对齐 test/main.cpp:84)
	HeadlessMode = true;
	if (verbose) {
#ifdef USE_SDL3
		SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
#else
		SDL_LogSetAllPriority(SDL_LOG_PRIORITY_VERBOSE);
#endif
	}

	if (
#ifdef USE_SDL3
	    !SDL_Init(SDL_INIT_EVENTS)
#else
	    SDL_Init(SDL_INIT_EVENTS) < 0
#endif
	)
		throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());
	struct InitGuard {
		bool committed = false;
		~InitGuard()
		{
			if (!committed)
				CleanupFailedEngineInit();
		}
	} initGuard;

	// 上游只在创建窗口时注册自定义 SDL 事件(display.cpp);无头嵌入必须自己注册,
	// 否则关卡切换事件(WM_DIABNEXTLVL 等)推送后无法被识别,玩家会卡死在 PM_NEWLVL
	RegisterCustomEvents();

	// MPQ 搜索顺序:BasePath → PrefPath → ConfigPath(assets.cpp GetMPQSearchPaths)。
	// BasePath 指向游戏数据目录;Pref/Config 指 scratch,存档与用户真实游戏隔离
	paths::SetBasePath(dataDir + "/");
	paths::SetAssetsPath(assetsDir + "/");
	paths::SetPrefPath(saveDir + "/");
	paths::SetConfigPath(saveDir + "/");

	LoadCoreArchives();
	LoadGameArchives(); // 找不到 diabdat.mpq 时自动回落 spawn.mpq 并置 gbIsSpawn
	if (!HaveMainData())
		throw std::runtime_error("diabdat.mpq / spawn.mpq 均未找到(默认搜索含 "
		                         "~/Library/Application Support/diasurgical/devilution/)");
#ifndef UNPACKED_MPQS
	// LoadGameArchives 还会搜索 scratch、系统目录和当前工作目录。若那里
	// 恰有更高优先级/同名 MPQ，单纯哈希 data_dir 会把评测身份绑到错误
	// 文件。嵌入模式必须只接受调用方显式 data_dir 中按引擎优先级选中的
	// 主档案，使训练/评测的 content SHA 与真正加载的字节一一对应。
	const auto mainArchive = MpqArchives.find(MainMpqPriority);
	if (mainArchive == MpqArchives.end()
	    || mainArchive->second.path() != expectedMainArchive) {
		const std::string actual = mainArchive == MpqArchives.end()
		    ? "<missing>"
		    : mainArchive->second.path();
		MpqArchives.clear();
		if (SDL_WasInit((~0U) & ~SDL_INIT_HAPTIC) != 0)
			SDL_Quit();
		throw std::runtime_error("实际主 MPQ 未来自 data_dir: actual=" + actual
		                         + ", expected=" + expectedMainArchive);
	}
#endif

	InitKeymapActions();
	LoadOptions();
	gLuaInitialized = true;
	try {
		LuaInitialize();
	} catch (...) {
		// LuaInitialize may already have emplaced CurrentLuaState and lazily
		// constructed sol usertype-name statics. Tear it down while unwinding;
		// otherwise the same cross-TU exit-order UAF reappears on init errors.
		try {
			LuaShutdown();
		} catch (...) {
		}
		gLuaInitialized = false;
		throw;
	}
	// The embedded bridge has no application main() that can call
	// DiabloDeinit().  Register after LuaInitialize(): sol's lazily-created
	// usertype-name statics have already registered their destructors, so the
	// LIFO atexit order destroys the Lua state while those strings are still
	// alive.  Leaving CurrentLuaState to cross-translation-unit static
	// destruction causes a deterministic heap-use-after-free under ASan.
	if (!gExitCleanupRegistered && std::atexit(EngineShutdownAtExit) != 0) {
		LuaShutdown();
		gLuaInitialized = false;
		throw std::runtime_error("无法注册 DevilutionX 进程退出清理");
	}
	gExitCleanupRegistered = true;

	gbIsHellfire = false;
	gbMusicOn = false;
	gbSoundOn = false;

	// 无头下永远没有鼠标事件来把 ControlMode 设成键鼠模式;若停留在 None,
	// plrctrls 的 WalkInDir 会把"摇杆无输入"理解为松开手柄,每 tick 给寻路发刹车
	// (plrctrls.cpp:1744),导致走路命令只能执行一步。
	ControlMode = ControlTypes::KeyboardAndMouse;
	ControlDevice = ControlTypes::KeyboardAndMouse;
	// DiabloGym 的任务空间只允许向下推进。否则 FARM/工人的普通走位
	// 会偶然踩中上楼触发格，回到城镇后把余下 3000 步空耗掉。
	DisableLevelBacktracking = true;

	// v14:装备自动上身。引擎拾取链(AutoGetItem)会先试 AutoEquip,但盔甲/
	// 头盔/首饰的自动装备选项默认是关的(options.cpp)——不开的话,捡到的
	// 装备会直落背包,成为对观测与动作都不可见的价值黑洞(v13 审查教训的
	// 装备版)。只填空槽 + _iStatFlag 属性需求由引擎把关(inv.cpp CanEquip)。
	GetOptions().Gameplay.autoEquipArmor.SetValue(true);
	GetOptions().Gameplay.autoEquipHelms.SetValue(true);
	GetOptions().Gameplay.autoEquipJewelry.SetValue(true);

	LoadSpellData();
	LoadPlayerDataFiles();
	LoadMissileData();
	LoadMonsterData();
	LoadItemData();
	LoadObjectData();
	pfile_ui_set_hero_infos(DummyGetHeroInfo);
	AdjustToScreenGeometry(forceResolution);

	gAssetsDir = assetsDir;
	gSaveDir = saveDir;
	gDataDir = dataDir;
	gEngineInited = true;
	gEnginePid = CurrentProcessId();
	initGuard.committed = true;
}

void CleanupGameResources()
{
	FreeMonsterHealthBar();
	FreeXPBar();
	FreeControlPan();
	FreeInvGFX();
	FreeGMenu();
	FreeQuestText();
	FreeInfoBoxGfx();
	FreeStoreMem();
	// NetInit appends two global chat-history entries every episode. Upstream's
	// chat log is process-lifetime state and is otherwise never cleared, so a
	// long vectorized training worker retains memory linearly with reset count.
	ClearChatLog();
	for (Player &player : Players)
		ResetPlayerGFX(player);
	FreeCursor();
	FreeGameMem();
	stream_stop();
	music_stop();
}

void EndGame()
{
	EnsureEngineProcess("end_game");
	if (!gInGame) {
		if (gEngineInited)
			DiscardPendingEvents();
		return;
	}
	gbRunGame = false;
	// 复刻上游 RunGameLoop 尾声的 FreeGame()。它在 diablo.cpp 匿名
	// 命名空间中无法直接调用，但不能省略：InitCursor 明确要求
	// 上一局已 FreeCursor，任务字幕/面板/玩家图形缓存也不得跨 episode。
	CleanupGameResources();
	NetClose(); // 外层 StartGame 尾声(会清空 Players)
	gInGame = false;
	gEpisodeGeneration++; // 使直接 end_game() 立即作废 Python wrapper 的 raw 缓存
	DiscardPendingEvents();
}

void EngineShutdownAtExit() noexcept
{
	// fork 后只有调用线程存活；继承来的 SDL/network/Lua 锁与线程状态不能
	// 在子进程析构。仅仅 return 仍会继续执行上游 CurrentLuaState 等 C++
	// 全局静态析构，重新暴露跨翻译单元析构顺序 UAF。fork child 的合法
	// 终点只有 exec/os._exit；若误走普通 exit/SystemExit，这里 fail-closed
	// 直接终止且返回失败，跳过其余 atexit 与所有静态析构。
	if (gEngineInited && gEnginePid != CurrentProcessId())
		std::_Exit(EXIT_FAILURE);
	// atexit callbacks must never unwind through the C runtime.  Each phase is
	// deliberately independent so a best-effort game cleanup cannot prevent
	// the ordering-critical Lua shutdown.
	try {
		EndGame();
	} catch (...) {
	}
	try {
		FreeItemGFX();
	} catch (...) {
	}
	if (gLuaInitialized) {
		try {
			LuaShutdown();
		} catch (...) {
		}
		gLuaInitialized = false;
	}
	try {
		init_cleanup();
	} catch (...) {
	}
	if (SDL_WasInit((~0U) & ~SDL_INIT_HAPTIC) != 0)
		SDL_Quit();
}

py::dict Reset(uint32_t seed)
{
	if (!gEngineInited)
		throw std::runtime_error("先调用 init()");
	EndGame();
	gStallPrints = 0;
	gMonotonicQuestTurnInUsed = false;

	CreateFreshHeroSave();
	gbLoadGame = false;

	if (!NetInit(/*bSinglePlayer=*/true))
		throw std::runtime_error("NetInit failed");
	struct ResetGuard {
		bool committed = false;
		~ResetGuard()
		{
			if (committed)
				return;
			gbRunGame = false;
			CleanupGameResources();
			NetClose();
			gInGame = false;
			DiscardPendingEvents();
		}
	} resetGuard;

	// 确定性:用用户种子覆写全部地牢种子(引擎在 NetInit 里刚按熵源填过一遍)
	std::mt19937 rng(seed);
	for (int i = 0; i < NUMLEVELS; i++) {
		DungeonSeeds[i] = static_cast<uint32_t>(rng());
		LevelSeeds[i] = std::nullopt;
	}
	// 防御性接管全局 RNG:CreatePlayer(经 pfile_ui_save_create)刚用墙钟毫秒
	// SetRndSeed 过(player.cpp)。钉死版引擎里任务抽选不受其影响(InitQuests 走
	// InitialiseQuestPools(DungeonSeeds[15]),局部 RNG,种子已在上面循环里被接管),
	// 关卡加载时也会按层种子重播;此覆写是把"全局 RNG 归 episode 种子管"钉成
	// 不随上游演化失效的不变量。实测修复前后 32 种子评估指纹位级一致。
	SetRndSeed(static_cast<uint32_t>(rng()));

	// 外层 StartGame(bNewGame=true) 的新开局初始化
	InitLevels();
	InitQuests();
	InitPortals();
	InitDungMsgs(*MyPlayer);
	DeltaSyncJunk();
	giNumberOfLevels = gbIsHellfire ? 25 : 17;

	// RunGameLoop 进入 while 前的序幕(无头版,略绘制/渐变/discord)。
	// 其中内层 StartGame(uMsg) 在匿名命名空间,以下为其公开 API 复刻
	SetEventHandler(GymEventHandler);
	nthread_ignore_mutex(true);
	try {
		CalcViewportGeometry();
		cineflag = false;
		InitCursor();
		music_stop();
		InitMonsterHealthBar();
		InitXPBar();
		SyncLoad(WM_DIABNEWGAME);
		gmenu_init_menu();
		InitLevelCursor();
		sgbMouseDown = CLICK_NONE;
		LastPlayerAction = PlayerActionType::None;
		run_delta_info();
		gbRunGame = true;
		gbProcessPlayers = true;
		gbRunGameResult = true;
		LoadPWaterPalette();
		InitBackbufferState();
		RedrawEverything();
	} catch (...) {
		nthread_ignore_mutex(false);
		throw;
	}
	nthread_ignore_mutex(false);
	lua::GameStart();
	gStartupTick = true;

	gInGame = true;
	gEpisodeGeneration++;
	py::dict result = Observe();
	resetGuard.committed = true;
	return result;
}

// v20:属性点自动分配——修复"属性点黑洞"。引擎每级发 5 属性点
// (NextPlrLevel 只累积 _pStatPts,花点历来靠人类点 UI),本桥十九代从未
// 调用过花点路径,等级→生存力的兑换链断裂,"先农后潜"在力学上不成立
// (clvl3 裸身打 L3 中位怪包负期望;唯一翻盘线需要 +10 体力 + 穿甲)。
// 战士口径:每批点数按 3体:2力 分配(体力主导低等级生存,力量喂伤害与
// 负重)。ModifyPlr* 自带属性封顶与派生量重算(HP/命中/负重);封顶后的
// 溢出点与人类玩家一致地原地作废。挂在 Step 尾部:升级发生在 tick 内,
// 最迟一个 env step(4 tick)后点数落袋,对策略等效即时。
static void AutoSpendStatPoints()
{
	Player &p = *MyPlayer;
	const int pts = p._pStatPts;
	if (pts <= 0)
		return;
	const int vit = (pts * 3 + 4) / 5; // 3/5 向上取整给体力
	ModifyPlrVit(p, vit);
	ModifyPlrStr(p, pts - vit);
	p._pStatPts = 0;
}

// 原版单机主线要求把 Staff of Lazarus 带回城交给 Cain。DiabloGym 的
// 训练任务自 L1 起严格单向下潜，既没有回城动作也不允许层级回退；若仍
// 保留该 UI 前置，完整通关在动作图上就是不可达的。这里只复刻
// TalkToStoryteller 中该物品的一次性交付状态迁移，不跳过法杖台、拾取、
// L15 入口、Vile 机关或战斗，并在 raw 中永久标记本局用过适配器。
static void AutoTurnInBetrayerStaffForMonotonicTask()
{
	if (gbIsSpawn || UseMultiplayerQuests())
		return;
	Quest &quest = Quests[Q_BETRAYER];
	if (quest._qactive != QUEST_INIT)
		return;
	if (!RemoveInventoryItemById(*MyPlayer, IDI_LAZSTAFF))
		return;
	quest._qlog = true;
	quest._qactive = QUEST_ACTIVE;
	quest._qvar1 = 2;
	NetSendCmdQuest(true, quest);
	gMonotonicQuestTurnInUsed = true;
}

py::dict Step(int ticks)
{
	EnsureInGame("step");
	if (ticks <= 0)
		throw std::invalid_argument("ticks 必须是正整数");
	for (int i = 0; i < ticks && gbRunGame; i++) {
		PumpSdlEvents();
		if (!gbRunGame)
			break;
		ProcessGameMessagePackets();
		if (!game_loop(gStartupTick) && gStallPrints < 8) {
			std::fprintf(stderr, "[diablogym] game_loop 失速(multi_handle_delta 拿不到 turn), destroyed=%d\n",
			    gbGameDestroyed ? 1 : 0);
			gStallPrints++;
		}
		gStartupTick = false;
	}
	// StartNewLvl 可在最后一个 game_loop 拍尾才把自定义事件
	// 放入 SDL 队列。返回 Python 前再泵一次，使换层场景与奖励
	// 归属于真正触发楼梯的这一次 env step，而不是下一个被
	// 迫空拍的策略动作。这里只加载，不额外消耗游戏逻辑 tick。
	if (gbRunGame)
		PumpSdlEvents();
	AutoSpendStatPoints();
	AutoTurnInBetrayerStaffForMonotonicTask();
	return Observe();
}

void ActWalk(int x, int y)
{
	if (!CanAcceptPlayerAction("act_walk"))
		return;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
		return; // 地图边缘的越界走格按 Gym 无效动作处理
	NetSendCmdLoc(MyPlayerId, true, CMD_WALKXY, { x, y });
}

void ActAttackMonster(uint16_t monsterId)
{
	if (!CanAcceptPlayerAction("act_attack_monster"))
		return;
	if (monsterId >= MaxMonsters)
		throw std::out_of_range("monster_id 越界");
	NetSendCmdParam1(true, CMD_ATTACKID, monsterId);
}

void ActAttackTile(int x, int y)
{
	if (!CanAcceptPlayerAction("act_attack_tile"))
		return;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
		return;
	NetSendCmdLoc(MyPlayerId, true, CMD_SATTACKXY, { x, y });
}

void ActOperate(int x, int y)
{
	if (!CanAcceptPlayerAction("act_operate"))
		return;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
		return;
	// 操作目标格上的物体(门/箱子/杠杆):引擎自动走过去再操作——与鼠标点击同路
	NetSendCmdLoc(MyPlayerId, true, CMD_OPOBJXY, { x, y });
}

bool IsHealItem(const Item &item)
{
	if (item.isEmpty())
		return false;
	return IsAnyOf(item._iMiscId, IMISC_HEAL, IMISC_FULLHEAL, IMISC_REJUV, IMISC_FULLREJUV)
	    || item.isScrollOf(SpellID::Healing);
}

int CountBeltHeals()
{
	int heals = 0;
	for (int i = 0; i < MaxBeltItems; i++) {
		if (IsHealItem(MyPlayer->SpdList[i]))
			heals++;
	}
	return heals;
}

int ActDrink()
{
	if (!CanAcceptPlayerAction("act_drink"))
		return 0;
	// 喝腰带上的第一瓶治疗类药水(与手柄快捷键 UseBeltItem 同路);
	// 无药时不发任何命令(空拍)。返回按键前的腰带治疗药数量
	const int heals = CountBeltHeals();
	if (heals > 0)
		UseBeltItem(BeltItemType::Healing);
	return heals;
}

int ActPickup()
{
	if (!CanAcceptPlayerAction("act_pickup"))
		return 0;
	// 走向并拾取最近的地面治疗药(与鼠标点击拾取同路 CMD_GOTOAGETITEM:
	// 引擎自动寻路、到位拾取、药水经 AutoPlaceItemInBelt 自动进腰带)。
	// 无目标时不发任何命令(空拍)。返回 0/1 = 是否发出了拾取命令
	bool beltHasRoom = false;
	for (int i = 0; i < MaxBeltItems; i++) {
		if (MyPlayer->SpdList[i].isEmpty()) {
			beltHasRoom = true;
			break;
		}
	}
	if (!beltHasRoom)
		return 0; // 腰带无空位:捡了会直落背包(喝药键与观测都看不见的价值黑洞),不发命令
	const Point me = MyPlayer->position.tile;
	int best = -1;
	int bestDist = 1 << 30;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int ii = ActiveItems[i];
		const Item &item = Items[ii];
		if (!IsHealItem(item))
			continue;
		const int dist = std::max(std::abs(item.position.x - me.x), std::abs(item.position.y - me.y));
		if (dist < bestDist) {
			bestDist = dist;
			best = ii;
		}
	}
	if (best < 0)
		return 0;
	NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, Items[best].position, static_cast<uint16_t>(best));
	return 1;
}

bool IsWantedGear(Item &item)
{
	// 值得捡的装备:对应身体槽位为空 + 属性需求达标。两个条件都在这里预判,
	// 因为引擎 AutoEquip 失败时会把装备落进背包——那是对观测与动作都不可见
	// 的价值黑洞(v13 审查的腰带满教训,装备版)。武器/盾牌不碰:战士出厂
	// 双手已满,AutoEquip 只填空槽(留给 v15 的"以旧换新"章)。
	if (item.isEmpty() || !item.isEquipment())
		return false;
	item.updateRequiredStatsCacheForPlayer(*MyPlayer); // 刷新 _iStatFlag(确定性纯函数)
	if (!item._iStatFlag)
		return false;
	const Player &p = *MyPlayer;
	switch (item._iLoc) {
	case ILOC_ARMOR:
		return p.InvBody[INVLOC_CHEST].isEmpty();
	case ILOC_HELM:
		return p.InvBody[INVLOC_HEAD].isEmpty();
	case ILOC_RING:
		return p.InvBody[INVLOC_RING_LEFT].isEmpty() || p.InvBody[INVLOC_RING_RIGHT].isEmpty();
	case ILOC_AMULET:
		return p.InvBody[INVLOC_AMULET].isEmpty();
	default:
		return false;
	}
}

int SweepBackpackGear()
{
	if (!CanAcceptPlayerAction("sweep_backpack_gear"))
		return 0;
	// PM_GOTHIT 时序窗(v14 审查确认):拾取请求与执行隔一个 tick,若中间挨了
	// 一记硬直(dam>>6 >= 等级),CanEquip 拒绝 _pmode>PM_WALK_SIDEWAYS,盔甲
	// 又进不了腰带(非 usable),于是静默沉入背包——对观测与动作双盲的价值
	// 黑洞。这里把背包里"本该穿上"的装备捞出来穿好(空槽+属性达标才动手,
	// 引擎自会重算 AC 与贴图,均有无头守卫)。返回本次上身件数
	Player &player = *MyPlayer;
	if (player._pmode > PM_WALK_SIDEWAYS)
		return 0;
	int equipped = 0;
	for (int iv = player._pNumInv - 1; iv >= 0; iv--) {
		if (!IsWantedGear(player.InvList[iv]))
			continue;
		const Item copy = player.InvList[iv];
		if (AutoEquip(player, copy, true, true)) {
			player.RemoveInvItem(iv);
			equipped++;
		}
	}
	return equipped;
}

int ActPickupGear()
{
	if (!CanAcceptPlayerAction("act_pickup_gear"))
		return 0;
	// 走向并拾取最近的"值得穿"的地面装备(与捡药同路 CMD_GOTOAGETITEM;
	// 引擎 AutoEquip 自动上身——EngineInit 已开启盔甲/头盔/首饰自动装备)。
	// 无目标时不发任何命令(空拍)。返回 0/1
	const Point me = MyPlayer->position.tile;
	int best = -1;
	int bestDist = 1 << 30;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int ii = ActiveItems[i];
		if (!IsWantedGear(Items[ii]))
			continue;
		const int dist = std::max(std::abs(Items[ii].position.x - me.x), std::abs(Items[ii].position.y - me.y));
		if (dist < bestDist) {
			bestDist = dist;
			best = ii;
		}
	}
	if (best < 0)
		return 0;
	NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, Items[best].position, static_cast<uint16_t>(best));
	return 1;
}

int ActPickupProgression(int x, int y)
{
	if (!CanAcceptPlayerAction("act_pickup_progression"))
		return 0;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
		return 0;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int itemId = ActiveItems[i];
		const Item &item = Items[itemId];
		if (item.IDidx != IDI_LAZSTAFF || item.position != Point { x, y })
			continue;
		NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, item.position,
		    static_cast<uint16_t>(itemId));
		return 1;
	}
	return 0;
}

} // namespace

PYBIND11_MODULE(_diablogym, m)
{
	m.doc() = "DiabloGym v0 —— DevilutionX 无头 RL 桥";
	m.def("engine_config", []() -> py::object {
		if (!gEngineInited)
			return py::none();
		EnsureEngineProcess("engine_config");
		return py::make_tuple(gAssetsDir, gSaveDir, gDataDir, gHeroClass);
	}, "读取已提交的原生单例配置；未初始化返回 None，用于 Python 异步异常恢复");
	m.def("init", &EngineInit, py::arg("assets_dir"), py::arg("save_dir"), py::arg("data_dir"),
	    py::arg("hero_class") = 0, py::arg("verbose") = false,
	    "一次性引擎初始化。data_dir 为 diabdat.mpq 所在目录；当前仅支持 hero_class=0(战士)");
	m.def("reset", &Reset, py::arg("seed"), "新开一局(全新 1 级英雄,确定性地牢种子),返回观测");
	m.def("step", &Step, py::arg("ticks") = 1, "推进游戏逻辑 N 个 tick(20 tick = 游戏内 1 秒),返回观测");
	m.def("observe", &Observe, "只读当前观测");
	m.def("act_walk", &ActWalk, py::arg("x"), py::arg("y"), "寻路走向目标格(网络命令层注入)");
	m.def("act_attack_monster", &ActAttackMonster, py::arg("monster_id"), "追击并近战指定怪物");
	m.def("act_attack_tile", &ActAttackTile, py::arg("x"), py::arg("y"), "原地朝目标格挥击");
	m.def("act_operate", &ActOperate, py::arg("x"), py::arg("y"), "操作目标格物体(开门等;引擎自动走近)");
	m.def("act_drink", &ActDrink, "喝腰带上的第一瓶治疗药(无药=无操作);返回按键前腰带治疗药数");
	m.def("act_pickup", &ActPickup, "走向并拾取最近的地面治疗药(无目标=无操作);返回 0/1");
	m.def("act_pickup_gear", &ActPickupGear, "走向并拾取最近的可穿戴装备(空槽+属性达标;无目标=无操作);返回 0/1");
	m.def("act_pickup_progression", &ActPickupProgression, py::arg("x"), py::arg("y"),
	    "仅拾取指定格的 Staff of Lazarus；供 action 10/11 必需剧情白名单使用");
	m.def("sweep_backpack_gear", &SweepBackpackGear, "把因硬直时序窗沉入背包的该穿装备捞出穿上;返回上身件数");
	m.def("end_game", &EndGame, "结束当前局(reset 会自动调用)");
	m.def("episode_generation", []() {
		EnsureEngineProcess("episode_generation");
		return gEpisodeGeneration;
	},
	    "当前原生游戏状态世代号(reset/end_game 时改变,用于缓存安全检查)");

	// ---- 探针专用接口(只用于发车前探针/验尸,训练与评估不得调用)----
	m.def("probe_add_experience", [](uint32_t xp) {
		EnsureInGame("probe_add_experience");
		// 直接注入经验(等级差按 0 计),触发引擎原生升级链
		// (NextPlrLevel → _pStatPts 累积 → Step 尾部 AutoSpendStatPoints)
		MyPlayer->addExperience(xp);
	}, py::arg("xp"), "探针:注入经验值,走原生升级路径");
	m.def("probe_modify_vit", [](int d) {
		EnsureInGame("probe_modify_vit");
		ModifyPlrVit(*MyPlayer, d);
	},
	    py::arg("d"), "探针:直接调体力(带封顶,同步 HP)");
	m.def("probe_bonus_ac", [](int d) {
		EnsureInGame("probe_bonus_ac");
		MyPlayer->_pIBonusAC += d;
	},
	    py::arg("d"), "探针:临时附加 AC(CalcPlrInv 会重算,战斗中不换装则稳定)");
	m.def("probe_invincible", [](bool enabled) {
		EnsureInGame("probe_invincible");
		MyPlayer->_pInvincible = enabled;
	}, py::arg("enabled"), "探针:切换无敌，仅供真实资源剧情/寻路验收");
	m.def("probe_stat_pts", []() {
		EnsureInGame("probe_stat_pts");
		return (int)MyPlayer->_pStatPts;
	},
	    "探针:读未花属性点(自动花点后应恒为 0)");
	m.def("probe_stats", []() {
		EnsureInGame("probe_stats");
		py::dict d;
		d["vit"] = (int)MyPlayer->_pVitality;
		d["str"] = (int)MyPlayer->_pStrength;
		d["max_hp"] = (int)(MyPlayer->_pMaxHP >> 6);
		return d;
	}, "探针:读属性明细");

	m.def("local_map", [](int radius) {
		EnsureInGame("local_map");
		const int maxRadius = std::max(static_cast<int>(MAXDUNX), static_cast<int>(MAXDUNY));
		if (radius < 0 || radius > maxRadius)
			throw std::invalid_argument("radius 必须在 [0, max(MAXDUNX, MAXDUNY)] 内");
		// 以玩家为中心的 (2r+1)² 局部地图:可走性 + 怪物占位 + 关闭的门
		// (C++ 端单次调用,避免逐格 probe 的开销)。
		// 注意:观测向量只消费 walkable/monster 两通道;door 通道仅供宏内部导航,
		// 不改变 286 维观测 —— 旧模型与排行榜完全兼容
		const Player &p = *MyPlayer;
		const int cx = p.position.tile.x, cy = p.position.tile.y;
		py::list walkable, monster, door;
		for (int dy = -radius; dy <= radius; dy++) {
			for (int dx = -radius; dx <= radius; dx++) {
				const int x = cx + dx, y = cy + dy;
				const bool inBounds = x >= 0 && x < MAXDUNX && y >= 0 && y < MAXDUNY;
				walkable.append(inBounds && IsTileWalkable({ x, y }, false) ? 1 : 0);
				monster.append(inBounds && dMonster[x][y] != 0 ? 1 : 0);
				// 关着的门:门对象在场且该格当前不可走。注意不能用 _oSolidFlag——
				// 引擎里门的封堵是靠门格地块换成实心(nSolidTable),门对象本身不置 solid。
				// 挡路的桶:实心但可破坏(operate 一击即碎,格子变可走)——seed 9005 的
				// 楼梯就被"门后一只桶"封死过,可通行规划必须认识这两种"软墙"
				bool closedDoor = false;
				bool barrel = false;
				if (inBounds) {
					Object *object = FindObjectAtPosition({ x, y });
					if (object != nullptr) {
						closedDoor = object->isDoor() && !IsTileWalkable({ x, y }, false);
						barrel = object->IsBreakable() && object->_oSolidFlag;
					}
				}
				door.append((closedDoor || barrel) ? 1 : 0);
			}
		}
		py::dict d;
		d["walkable"] = walkable;
		d["monster"] = monster;
		d["door"] = door;
		return d;
	}, py::arg("radius") = 5, "以玩家为中心的局部地图通道");

	m.def("probe_asset", [](const std::string &path) {
		EnsureEngineProcess("probe_asset");
		size_t size = 0;
		AssetHandle handle = OpenAsset(std::string_view(path), size);
		py::dict d;
		d["ok"] = handle.ok();
		d["size"] = static_cast<uint64_t>(size);
		return d;
	}, py::arg("path"), "调试:检查资产能否打开及其大小");

	m.def("probe_tile", [](int x, int y) {
		EnsureInGame("probe_tile");
		if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
			throw std::out_of_range("probe_tile 坐标越界");
		py::dict d;
		d["piece"] = static_cast<int>(dPiece[x][y]);
		d["monster"] = static_cast<int>(dMonster[x][y]);
		d["player"] = static_cast<int>(dPlayer[x][y]);
		d["object"] = static_cast<int>(dObject[x][y]);
		d["solid"] = IsTileSolid({ x, y });
		d["walkable"] = IsTileWalkable({ x, y }, false);
		Object *object = FindObjectAtPosition({ x, y });
		d["object_type"] = object != nullptr ? static_cast<int>(object->_otype) : -1;
		d["object_is_door"] = object != nullptr && object->isDoor();
		d["object_solid"] = object != nullptr && object->_oSolidFlag;
		d["object_selectable"] = object != nullptr && object->canInteractWith();
		return d;
	}, py::arg("x"), py::arg("y"), "调试:读取单格的占位/碰撞/物体状态");

	m.def("probe_is_spawn", []() {
		EnsureEngineProcess("probe_is_spawn");
		return gbIsSpawn;
	},
	    "探针:当前是否使用 shareware spawn.mpq");
	m.def("probe_warp_main_level", [](int level) {
		EnsureInGame("probe_warp_main_level");
		if (setlevel || level < 1 || level > 16)
			throw std::invalid_argument("探针主层须在 [1,16] 且当前不在 set-level");
		StartNewLvl(*MyPlayer, WM_DIABNEXTLVL, level);
	}, py::arg("level"), "探针:排队切换到指定主地牢层");
	m.def("probe_enter_set_level", [](int level) {
		EnsureInGame("probe_enter_set_level");
		if (setlevel)
			throw std::runtime_error("探针进入 set-level 前必须位于主地牢");
		const auto setLevel = static_cast<_setlevels>(level);
		switch (setLevel) {
		case SL_SKELKING:
			setlvltype = Quests[Q_SKELKING]._qlvltype;
			break;
		case SL_BONECHAMB:
			setlvltype = Quests[Q_SCHAMB]._qlvltype;
			break;
		case SL_POISONWATER:
			setlvltype = Quests[Q_PWATER]._qlvltype;
			break;
		case SL_VILEBETRAYER:
			setlvltype = Quests[Q_BETRAYER]._qlvltype;
			break;
		default:
			throw std::invalid_argument("探针只支持四个正式任务 set-level");
		}
		StartNewLvl(*MyPlayer, WM_DIABSETLVL, level);
	}, py::arg("level"), "探针:排队进入正式任务 set-level(1/2/4/5)");
	m.def("probe_return_set_level", []() {
		EnsureInGame("probe_return_set_level");
		if (!setlevel)
			throw std::runtime_error("探针返回前当前必须位于 set-level");
		StartNewLvl(*MyPlayer, WM_DIABRTNLVL, GetMapReturnLevel());
	}, "探针:排队从任务 set-level 返回对应主地牢层");

	// 触发点消息类型常量(观测 triggers[].msg 的取值)
	m.attr("WM_DIABNEXTLVL") = static_cast<int>(WM_DIABNEXTLVL);
	m.attr("WM_DIABPREVLVL") = static_cast<int>(WM_DIABPREVLVL);
	m.attr("WM_DIABSETLVL") = static_cast<int>(WM_DIABSETLVL);
	m.attr("WM_DIABRTNLVL") = static_cast<int>(WM_DIABRTNLVL);
	m.attr("WM_DIABTOWNWARP") = static_cast<int>(WM_DIABTOWNWARP);
	m.attr("WM_DIABTWARPUP") = static_cast<int>(WM_DIABTWARPUP);
	m.attr("PM_NEWLVL") = static_cast<int>(PM_NEWLVL);
}
