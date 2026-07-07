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
#include "monster.h"
#include "msg.h"
#include "multi.h"
#include "nthread.h"
#include "objects.h" // FindObjectAtPosition / isDoor(下楼宏的门感知)
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

int CountBeltHeals();            // 定义在动作区(v12);Observe 的 raw 字段也要用
bool IsHealItem(const Item &);   // 定义在动作区(v13);floor_items 的 heal 标志也要用
bool IsWantedGear(Item &);       // 定义在动作区(v14);floor_items 的 gear 标志也要用

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
	obs["belt_heals"] = CountBeltHeals(); // v12 起入 raw;v13 起由 env 写进观测向量(瓶盲修复)
	obs["armor_class"] = player.GetArmor(); // v14:护甲值(_pIBonusAC + _pIAC + 敏捷/5)

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
		Item &item = Items[ActiveItems[i]];
		py::dict it;
		it["x"] = static_cast<int>(item.position.x);
		it["y"] = static_cast<int>(item.position.y);
		it["heal"] = IsHealItem(item);   // v13:捡药宏的目标标志
		it["gear"] = IsWantedGear(item); // v14:捡装备宏的目标标志(空槽+属性达标)
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

void ActOperate(int x, int y)
{
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
	// 喝腰带上的第一瓶治疗类药水(与手柄快捷键 UseBeltItem 同路);
	// 无药时不发任何命令(空拍)。返回按键前的腰带治疗药数量
	const int heals = CountBeltHeals();
	if (heals > 0)
		UseBeltItem(BeltItemType::Healing);
	return heals;
}

int ActPickup()
{
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
	m.def("act_operate", &ActOperate, py::arg("x"), py::arg("y"), "操作目标格物体(开门等;引擎自动走近)");
	m.def("act_drink", &ActDrink, "喝腰带上的第一瓶治疗药(无药=无操作);返回按键前腰带治疗药数");
	m.def("act_pickup", &ActPickup, "走向并拾取最近的地面治疗药(无目标=无操作);返回 0/1");
	m.def("act_pickup_gear", &ActPickupGear, "走向并拾取最近的可穿戴装备(空槽+属性达标;无目标=无操作);返回 0/1");
	m.def("sweep_backpack_gear", &SweepBackpackGear, "把因硬直时序窗沉入背包的该穿装备捞出穿上;返回上身件数");
	m.def("end_game", &EndGame, "结束当前局(reset 会自动调用)");

	m.def("local_map", [](int radius) {
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
		Object *object = FindObjectAtPosition({ x, y });
		d["object_type"] = object != nullptr ? static_cast<int>(object->_otype) : -1;
		d["object_is_door"] = object != nullptr && object->isDoor();
		d["object_solid"] = object != nullptr && object->_oSolidFlag;
		d["object_selectable"] = object != nullptr && object->canInteractWith();
		return d;
	}, py::arg("x"), py::arg("y"), "调试:读取单格的占位/碰撞/物体状态");

	// 触发点消息类型常量(观测 triggers[].msg 的取值)
	m.attr("WM_DIABNEXTLVL") = static_cast<int>(WM_DIABNEXTLVL);
	m.attr("WM_DIABPREVLVL") = static_cast<int>(WM_DIABPREVLVL);
	m.attr("WM_DIABTOWNWARP") = static_cast<int>(WM_DIABTOWNWARP);
	m.attr("WM_DIABTWARPUP") = static_cast<int>(WM_DIABTWARPUP);
}
