/**
 * @file diablogym.cpp
 *
 * DiabloGym v0 —— DevilutionX 无头嵌入桥(pybind11)。
 *
 * 嵌入方式与上游 test/timedemo_test.cpp 同源:HeadlessMode + loopback 单机,
 * 由 Python 侧逐 tick 驱动主循环(复刻 RunGameLoop 循环体,去掉墙钟限速与绘制),
 * 动作走网络命令层(NetSendCmd*)—— 与多人协议同一条路,天然支持日后联机部署。
 */

#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_SDL3
#include <SDL3/SDL.h>
#else
#include <SDL.h>
#endif

#include "DiabloUI/diabloui.h" // _uiheroinfo
#include "controls/control_mode.hpp"
#include "cursor.h"
#include "diablo.h"
#include "engine/render/scrollrt.h" // CalcViewportGeometry
#include "gmenu.h"
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
#include "monster.h"
#include "msg.h"
#include "multi.h"
#include "nthread.h"
#include "options.h"
#include "pfile.h"
#include "player.h"
#include "portal.h"
#include "quests.h"
#include "tables/monstdat.h"
#include "tables/playerdat.hpp"
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

bool DummyGetHeroInfo(_uiheroinfo * /*info*/)
{
	return true;
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
			SyncLoad(GetCustomEvent(event));
			nthread_ignore_mutex(false);
			continue;
		}
		// 无头环境不产生键鼠事件;动作全部经由网络命令层注入
	}
}

py::dict Observe()
{
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
	obs["dungeon_level"] = static_cast<int>(currlevel);
	obs["level_type"] = static_cast<int>(leveltype);
	obs["player_mode"] = static_cast<int>(player._pmode);
	obs["walkpath0"] = static_cast<int>(player.walkpath[0]);
	obs["future_x"] = static_cast<int>(player.position.future.x);
	obs["future_y"] = static_cast<int>(player.position.future.y);
	obs["dest_action"] = static_cast<int>(player.destAction);
	obs["dead"] = player._pmode == PM_DEATH || (player._pHitPoints >> 6) <= 0;
	obs["game_over"] = !gbRunGame;
	obs["victory"] = !IsDiabloAlive(false);

	py::list monsters;
	for (size_t i = 0; i < ActiveMonsterCount; i++) {
		const unsigned monsterId = ActiveMonsters[i];
		const Monster &monster = Monsters[monsterId];
		if (monster.hitPoints <= 0)
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
		const Item &item = Items[ActiveItems[i]];
		py::dict it;
		it["x"] = static_cast<int>(item.position.x);
		it["y"] = static_cast<int>(item.position.y);
		items.append(it);
	}
	obs["floor_items"] = items;

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
	if (gEngineInited)
		return;
	gHeroClass = heroClass;

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

	InitKeymapActions();
	LoadOptions();
	LuaInitialize();

	gbIsHellfire = false;
	gbMusicOn = false;
	gbSoundOn = false;

	// 无头下永远没有鼠标事件来把 ControlMode 设成键鼠模式;若停留在 None,
	// plrctrls 的 WalkInDir 会把"摇杆无输入"理解为松开手柄,每 tick 给寻路发刹车
	// (plrctrls.cpp:1744),导致走路命令只能执行一步。
	ControlMode = ControlTypes::KeyboardAndMouse;
	ControlDevice = ControlTypes::KeyboardAndMouse;

	LoadSpellData();
	LoadPlayerDataFiles();
	LoadMissileData();
	LoadMonsterData();
	LoadItemData();
	LoadObjectData();
	pfile_ui_set_hero_infos(DummyGetHeroInfo);
	AdjustToScreenGeometry(forceResolution);

	gEngineInited = true;
}

void EndGame()
{
	if (!gInGame)
		return;
	gbRunGame = false;
	// 上游 RunGameLoop 尾声还会调 FreeGame()(UI 贴图清理),但它在匿名命名空间里
	// 且无头模式下重开局时 Init* 会重建这些资源,故略过
	NetClose(); // 外层 StartGame 尾声(会清空 Players)
	gInGame = false;
}

py::dict Reset(uint32_t seed)
{
	if (!gEngineInited)
		throw std::runtime_error("先调用 init()");
	EndGame();

	CreateFreshHeroSave();
	gbLoadGame = false;

	if (!NetInit(/*bSinglePlayer=*/true))
		throw std::runtime_error("NetInit failed");

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
	nthread_ignore_mutex(false);
	lua::GameStart();
	gStartupTick = true;

	gInGame = true;
	return Observe();
}

py::dict Step(int ticks)
{
	if (!gInGame)
		throw std::runtime_error("先调用 reset()");
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
	return Observe();
}

void ActWalk(int x, int y)
{
	NetSendCmdLoc(MyPlayerId, true, CMD_WALKXY, { x, y });
}

void ActAttackMonster(uint16_t monsterId)
{
	NetSendCmdParam1(true, CMD_ATTACKID, monsterId);
}

void ActAttackTile(int x, int y)
{
	NetSendCmdLoc(MyPlayerId, true, CMD_SATTACKXY, { x, y });
}

} // namespace

PYBIND11_MODULE(_diablogym, m)
{
	m.doc() = "DiabloGym v0 —— DevilutionX 无头 RL 桥";
	m.def("init", &EngineInit, py::arg("assets_dir"), py::arg("save_dir"), py::arg("data_dir"),
	    py::arg("hero_class") = 0, py::arg("verbose") = false,
	    "一次性引擎初始化。data_dir 为 diabdat.mpq 所在目录。hero_class: 0=战士 1=游侠 2=法师");
	m.def("reset", &Reset, py::arg("seed"), "新开一局(全新 1 级英雄,确定性地牢种子),返回观测");
	m.def("step", &Step, py::arg("ticks") = 1, "推进游戏逻辑 N 个 tick(20 tick = 游戏内 1 秒),返回观测");
	m.def("observe", &Observe, "只读当前观测");
	m.def("act_walk", &ActWalk, py::arg("x"), py::arg("y"), "寻路走向目标格(网络命令层注入)");
	m.def("act_attack_monster", &ActAttackMonster, py::arg("monster_id"), "追击并近战指定怪物");
	m.def("act_attack_tile", &ActAttackTile, py::arg("x"), py::arg("y"), "原地朝目标格挥击");
	m.def("end_game", &EndGame, "结束当前局(reset 会自动调用)");

	m.def("local_map", [](int radius) {
		// 以玩家为中心的 (2r+1)² 局部地图:可走性 + 怪物占位(C++ 端单次调用,避免逐格 probe 的开销)
		const Player &p = *MyPlayer;
		const int cx = p.position.tile.x, cy = p.position.tile.y;
		py::list walkable, monster;
		for (int dy = -radius; dy <= radius; dy++) {
			for (int dx = -radius; dx <= radius; dx++) {
				const int x = cx + dx, y = cy + dy;
				const bool inBounds = x >= 0 && x < MAXDUNX && y >= 0 && y < MAXDUNY;
				walkable.append(inBounds && IsTileWalkable({ x, y }, false) ? 1 : 0);
				monster.append(inBounds && dMonster[x][y] != 0 ? 1 : 0);
			}
		}
		py::dict d;
		d["walkable"] = walkable;
		d["monster"] = monster;
		return d;
	}, py::arg("radius") = 5, "以玩家为中心的局部地图通道");

	m.def("probe_asset", [](const std::string &path) {
		size_t size = 0;
		AssetHandle handle = OpenAsset(std::string_view(path), size);
		py::dict d;
		d["ok"] = handle.ok();
		d["size"] = static_cast<uint64_t>(size);
		return d;
	}, py::arg("path"), "调试:检查资产能否打开及其大小");

	m.def("probe_tile", [](int x, int y) {
		py::dict d;
		d["piece"] = static_cast<int>(dPiece[x][y]);
		d["monster"] = static_cast<int>(dMonster[x][y]);
		d["player"] = static_cast<int>(dPlayer[x][y]);
		d["object"] = static_cast<int>(dObject[x][y]);
		d["solid"] = IsTileSolid({ x, y });
		d["walkable"] = IsTileWalkable({ x, y }, false);
		return d;
	}, py::arg("x"), py::arg("y"), "调试:读取单格的占位/碰撞状态");

	// 触发点消息类型常量(观测 triggers[].msg 的取值)
	m.attr("WM_DIABNEXTLVL") = static_cast<int>(WM_DIABNEXTLVL);
	m.attr("WM_DIABPREVLVL") = static_cast<int>(WM_DIABPREVLVL);
	m.attr("WM_DIABTOWNWARP") = static_cast<int>(WM_DIABTOWNWARP);
	m.attr("WM_DIABTWARPUP") = static_cast<int>(WM_DIABTWARPUP);
}
