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
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "controls/plrctrls.h"
#include "cursor.h"
#include "diablo.h"
#include "engine/path.h"
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
#include "missiles.h"
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
int CountLegacyBeltHeals();      // 冻结 v3 观测含 Healing 卷轴；只读兼容账
int CountBeltFreeSlots();        // v4:捡药有效性不能用 heal 数反推，腰带还可能装别的物品
int InstantHealKind(const Item &); // 0=非即时治疗；1..4 对应四种药水，保留腰带槽位顺序
bool IsHealItem(const Item &);   // 定义在动作区(v13);floor_items 的 heal 标志也要用
bool IsLegacyHealItem(const Item &); // 仅重建旧 policy view，绝不驱动 v4 动作
bool IsWantedGear(const Item &); // 定义在动作区(v14);floor_items 的 gear 标志也要用
int ActPickupGearAt(
    int activeItemId, int x, int y, uint16_t seedHigh, uint16_t seedLow,
    uint16_t expectedCreateInfo, int expectedBaseId);

uint16_t LowWord(uint32_t value)
{
	return static_cast<uint16_t>(value & 0xFFFFU);
}

uint16_t HighWord(uint32_t value)
{
	return static_cast<uint16_t>(value >> 16);
}

bool MatchesItemIdentity(
    const Item &item, uint16_t seedHigh, uint16_t seedLow,
    uint16_t createInfo, int baseId)
{
	return HighWord(item._iSeed) == seedHigh
	    && LowWord(item._iSeed) == seedLow
	    && item._iCreateInfo == createInfo
	    && static_cast<int>(item.IDidx) == baseId;
}

uint64_t MonsterKillTotal()
{
	uint64_t total = 0;
	for (int monsterType = 0; monsterType < NUM_MAX_MTYPES; monsterType++) {
		const int count = MonsterKillCounts[monsterType];
		if (count < 0)
			throw std::runtime_error(
			    "MonsterKillCounts 含负数，拒绝发布损坏的累计击杀事实");
		const uint64_t unsignedCount = static_cast<uint64_t>(count);
		if (unsignedCount > std::numeric_limits<uint64_t>::max() - total)
			throw std::overflow_error("monster_kill_total uint64 汇总溢出");
		total += unsignedCount;
	}
	return total;
}

bool ItemEffectsActive(const Item &item)
{
	return !item.isEmpty()
	    && (item._iMagical == ITEM_QUALITY_NORMAL || item._iIdentified);
}

unsigned CountSetBits(uint32_t value)
{
	unsigned count = 0;
	while (value != 0) {
		count += value & 1U;
		value >>= 1;
	}
	return count;
}

constexpr uint32_t EffectBits(ItemSpecialEffect effect)
{
	return static_cast<uint32_t>(effect);
}

constexpr int LifeStealBasisPoints(uint32_t flags)
{
	// DevilutionX applies the fixed tiers as 3%/5% of the physical damage of
	// a successful strike; if both tier bits are present, the later 5% branch
	// overwrites the 3% result.  RandomStealLife is a separate heal and therefore
	// stacks with the effective fixed tier.  GenerateRnd(dam / 8) has an
	// asymptotic expectation of 1/16 damage, i.e. 6.25%.
	int basisPoints = 0;
	if ((flags & EffectBits(ItemSpecialEffect::StealLife5)) != 0)
		basisPoints = 500;
	else if ((flags & EffectBits(ItemSpecialEffect::StealLife3)) != 0)
		basisPoints = 300;
	if ((flags & EffectBits(ItemSpecialEffect::RandomStealLife)) != 0)
		basisPoints += 625;
	return basisPoints;
}

constexpr int InitialMeleeAttackSkippedFrames(uint32_t flags)
{
	// DiabloGym sends a fresh CMD_ATTACKID only after the preceding strike has
	// settled.  ProcessPlayers therefore reaches StartAttack with
	// includesFirstFrame=false: Quick skips 0, Fast skips 1, and both Faster
	// and Fastest skip 2.
	if ((flags & (static_cast<uint32_t>(ItemSpecialEffect::FastestAttack)
	                 | static_cast<uint32_t>(
	                     ItemSpecialEffect::FasterAttack)))
	    != 0)
		return 2;
	if ((flags & static_cast<uint32_t>(
	                 ItemSpecialEffect::FastAttack))
	    != 0)
		return 1;
	return 0;
}

/**
 * Deterministic item-local descriptor for the policy's floor/equipped rows.
 * It is deliberately NOT a replacement gate: summing it cannot reproduce
 * whole-loadout OR flags, resistance caps or attack-speed precedence.  The
 * gate/observation/reward authority is GearCombatProfile below.  A normal or
 * identified item exposes its affixes; unidentified equipped items only
 * receive their base value.
 */
uint32_t ItemCombatUtility(const Item &item)
{
	if (item.isEmpty() || !item.isEquipment() || !item._iStatFlag
	    || item._iClass == ICLASS_QUEST || item.IDidx == IDI_LAZSTAFF
	    || (item._iMaxDur > 0 && item._iMaxDur != DUR_INDESTRUCTIBLE
	        && item._iDurability <= 0))
		return 0;

	int64_t score = 0;
	if (item.isWeapon())
		score += static_cast<int64_t>(item._iMinDam + item._iMaxDam) * 512;
	score += static_cast<int64_t>(std::max<int>(0, item._iAC)) * 128;
	if (item._itype == ItemType::Shield)
		score += 1024; // a shield also enables the class-specific block roll

	if (ItemEffectsActive(item)) {
		score += static_cast<int64_t>(item._iPLDam) * 64;
		score += static_cast<int64_t>(item._iPLToHit) * 32;
		score += static_cast<int64_t>(item._iAC) * item._iPLAC * 128 / 100;
		score += static_cast<int64_t>(
		             item._iPLStr + item._iPLDex + item._iPLVit)
		    * 32;
		score += static_cast<int64_t>(
		             item._iPLFR + item._iPLLR + item._iPLMR)
		    * 32;
		// Life is stored in 26.6 fixed point.  Magic, mana and spell level
		// intentionally have no local combat value: none of the installed
		// 15 actions can cast a spell or spend mana.
		score += static_cast<int64_t>(item._iPLHP) * 4;
		score += static_cast<int64_t>(item._iPLDamMod) * 128;
		// Positive _iPLGetHit adds incoming damage; negative values reduce it.
		score -= static_cast<int64_t>(item._iPLGetHit) * 128;
		score += static_cast<int64_t>(item._iPLLight) * 512;
		score += static_cast<int64_t>(item._iPLEnAc) * 32;
		score += static_cast<int64_t>(
		             item._iFMinDam + item._iFMaxDam
		             + item._iLMinDam + item._iLMaxDam)
		    * 128;

		const uint32_t flags = static_cast<uint32_t>(item._iFlags);
		auto has = [flags](ItemSpecialEffect effect) {
			return (flags & static_cast<uint32_t>(effect)) != 0;
		};
		// Score only effects reachable through the installed melee controller.
		// Tiered OR flags use their effective highest tier, so redundant lower
		// bits cannot manufacture a larger policy hint.
		const int attackSkippedFrames
		    = InitialMeleeAttackSkippedFrames(flags);
		int hitRecoveryTier = 0;
		if (has(ItemSpecialEffect::FastestHitRecovery))
			hitRecoveryTier = 3;
		else if (has(ItemSpecialEffect::FasterHitRecovery))
			hitRecoveryTier = 2;
		else if (has(ItemSpecialEffect::FastHitRecovery))
			hitRecoveryTier = 1;
		score += static_cast<int64_t>(
		             attackSkippedFrames + hitRecoveryTier)
		    * 192;
		// This item-local number cannot know the equipped weapon's physical
		// throughput (life steal can live on a helm), so retain only a compact
		// monotonic hint in nominal percentage order.  Whole-loadout replacement
		// decisions use ScoreGearCombatProfile below, never this descriptor.
		score += static_cast<int64_t>(LifeStealBasisPoints(flags))
		    * 64 / 100;
		constexpr uint32_t UntieredMeleeEffectMask
		    = static_cast<uint32_t>(ItemSpecialEffect::Thorns)
		    | static_cast<uint32_t>(ItemSpecialEffect::HalfTrapDamage)
		    | static_cast<uint32_t>(ItemSpecialEffect::TripleDemonDamage);
		score += static_cast<int64_t>(
		             CountSetBits(flags & UntieredMeleeEffectMask))
		    * 192;
		// FastBlock has no effect without a loadout capable of blocking.
		if (item._itype == ItemType::Shield
		    && has(ItemSpecialEffect::FastBlock))
			score += 192;
		if (HasAnyOf(item._iFlags, ItemSpecialEffect::DrainLife))
			score -= 1536;
		if (HasAnyOf(item._iFlags, ItemSpecialEffect::ZeroResistance))
			score -= 4096;

		const uint8_t damAcFlags = static_cast<uint8_t>(item._iDamAcFlags);
		constexpr uint8_t BeneficialDamAcMask
		    = static_cast<uint8_t>(ItemSpecialEffectHf::Devastation)
		    | static_cast<uint8_t>(ItemSpecialEffectHf::Jesters)
		    | static_cast<uint8_t>(ItemSpecialEffectHf::ACAgainstDemons)
		    | static_cast<uint8_t>(ItemSpecialEffectHf::ACAgainstUndead);
		score += static_cast<int64_t>(
		             CountSetBits(damAcFlags & BeneficialDamAcMask))
		    * 256;
		if (HasAnyOf(item._iDamAcFlags, ItemSpecialEffectHf::Decay))
			score -= 512;
		if (HasAnyOf(item._iDamAcFlags, ItemSpecialEffectHf::Peril))
			score -= 512;
		if (HasAnyOf(
		        item._iDamAcFlags, ItemSpecialEffectHf::Doppelganger))
			score -= 512;
	}

	if (score <= 0)
		return 0;
	// Discount by absolute hits remaining, not only the percentage repaired:
	// 1/1 must not look as durable as 200/200.  The separate raw durability
	// fields remain available to the policy; this scalar is only a compact hint.
	if (item._iMaxDur > 0 && item._iMaxDur != DUR_INDESTRUCTIBLE) {
		const int durability = std::clamp(item._iDurability, 0, item._iMaxDur);
		score = score * durability / (durability + 16LL);
	}
	return static_cast<uint32_t>(std::min<int64_t>(
	    score, std::numeric_limits<uint32_t>::max()));
}

struct GearCombatProfile {
	uint32_t utility = 0;
	uint32_t effectFlags = 0;
	uint8_t damAcFlags = 0;
	int attackSpeedTier = 0;
	int hitRecoveryTier = 0;
	int lifeStealTier = 0;
	int manaStealTier = 0;
	int attackCycleFrames = 1;
	int attackImpactFrames = 1;
	int physicalMin = 0;
	int physicalMax = 0;
	int animalMin = 0;
	int animalMax = 0;
	int undeadMin = 0;
	int undeadMax = 0;
	int demonMin = 0;
	int demonMax = 0;
	int meleeToHit = 0;
	int meleePiercingToHit = 0;
	int magicToHit = 0;
	int armor = 0;
	bool blockEnabled = false;
	int blockChance = 0;
	int magicResistance = 0;
	int fireResistance = 0;
	int lightningResistance = 0;
	int lightRadius = 0;
	int currentHitPoints = 0;
	int maxHitPoints = 0;
	int maxMana = 0;
	int magic = 0;
	int getHit = 0;
	int enemyArmorReduction = 0;
	int fireMin = 0;
	int fireMax = 0;
	int lightningMin = 0;
	int lightningMax = 0;
	int spellLevelBonus = 0;
};

constexpr uint8_t DamAcEffectBits(ItemSpecialEffectHf effect)
{
	return static_cast<uint8_t>(effect);
}

constexpr int FourLevelEffectTier(
    uint32_t flags, ItemSpecialEffect first, ItemSpecialEffect second,
    ItemSpecialEffect third, ItemSpecialEffect fourth)
{
	if ((flags & EffectBits(fourth)) != 0)
		return 4;
	if ((flags & EffectBits(third)) != 0)
		return 3;
	if ((flags & EffectBits(second)) != 0)
		return 2;
	if ((flags & EffectBits(first)) != 0)
		return 1;
	return 0;
}

constexpr int TwoLevelEffectTier(
    uint32_t flags, ItemSpecialEffect first, ItemSpecialEffect second)
{
	if ((flags & EffectBits(second)) != 0)
		return 2;
	if ((flags & EffectBits(first)) != 0)
		return 1;
	return 0;
}

constexpr int ThreeLevelEffectTier(
    uint32_t flags, ItemSpecialEffect first, ItemSpecialEffect second,
    ItemSpecialEffect third)
{
	if ((flags & EffectBits(third)) != 0)
		return 3;
	if ((flags & EffectBits(second)) != 0)
		return 2;
	if ((flags & EffectBits(first)) != 0)
		return 1;
	return 0;
}

static_assert(FourLevelEffectTier(
                  EffectBits(ItemSpecialEffect::QuickAttack)
                      | EffectBits(ItemSpecialEffect::FastestAttack),
                  ItemSpecialEffect::QuickAttack,
                  ItemSpecialEffect::FastAttack,
                  ItemSpecialEffect::FasterAttack,
                  ItemSpecialEffect::FastestAttack)
    == 4);
static_assert(InitialMeleeAttackSkippedFrames(
                  EffectBits(ItemSpecialEffect::QuickAttack))
    == 0);
static_assert(InitialMeleeAttackSkippedFrames(
                  EffectBits(ItemSpecialEffect::FasterAttack))
    == 2);
static_assert(InitialMeleeAttackSkippedFrames(
                  EffectBits(ItemSpecialEffect::FastestAttack))
    == 2);
static_assert(LifeStealBasisPoints(
                  EffectBits(ItemSpecialEffect::StealLife3))
    == 300);
static_assert(LifeStealBasisPoints(
                  EffectBits(ItemSpecialEffect::StealLife5))
    == 500);
static_assert(LifeStealBasisPoints(
                  EffectBits(ItemSpecialEffect::StealLife3)
                      | EffectBits(ItemSpecialEffect::StealLife5))
    == 500);
static_assert(LifeStealBasisPoints(
                  EffectBits(ItemSpecialEffect::RandomStealLife))
    == 625);
static_assert(LifeStealBasisPoints(
                  EffectBits(ItemSpecialEffect::StealLife5)
                      | EffectBits(ItemSpecialEffect::RandomStealLife))
    == 1125);

uint32_t GearDurabilityReserve(const Player &player)
{
	uint64_t reserve = 0;
	for (const Item &item : player.InvBody) {
		if (item.isEmpty() || !item.isEquipment() || !item._iStatFlag)
			continue;
		if (item._iMaxDur == DUR_INDESTRUCTIBLE) {
			reserve += 4096;
			continue;
		}
		// Jewelry and other genuinely non-durable items use maxDur==0.  They
		// have no durability resource and must not receive a synthetic +4096
		// merely for occupying an empty slot.
		if (item._iMaxDur <= 0)
			continue;
		const int durability = std::clamp(item._iDurability, 0, item._iMaxDur);
		// Each point is one additional native durability-loss event survived.
		// The network/save item format preserves finite durability through 254
		// and reserves 255 for indestructible.  Capping this term at 32 made an
		// otherwise identical 200/200 weapon look equal to 32/32, permanently
		// hiding a real 168-hit replacement from action 14.  Sixteen points per
		// hit keeps the old per-slot maximum (4064 versus indestructible 4096)
		// while making the complete persisted finite domain strictly ordered.
		reserve += 16ULL * std::min(
		    durability, DUR_INDESTRUCTIBLE - 1);
	}
	return static_cast<uint32_t>(std::min<uint64_t>(
	    reserve, std::numeric_limits<uint32_t>::max()));
}

uint32_t ScoreGearCombatProfile(
    const GearCombatProfile &profile, uint32_t durabilityReserve)
{
	auto nonnegative = [](int value) {
		return static_cast<int64_t>(std::max(value, 0));
	};
	int64_t score = 0;
	// These are post-CalcPlrItemVals quantities, so global OR flags, the
	// resistance cap and stat dependency cascades are each represented once.
	// Damage, hit chance and animation speed are one throughput quantity, not
	// three additive bonuses.  The old formula could accept a slightly harder
	// but slower weapon whose damage/frame fell, or trade a large hit-probability
	// loss for paper damage that almost never lands.  Frames originate in int8
	// animation metadata; clamp defensively so corrupt/edge values cannot divide
	// by zero or amplify a score without bound.  Sixteen frames preserves the
	// historical scale around the Warrior's one-handed baseline.
	constexpr int ReferenceAttackCycleFrames = 16;
	const int boundedAttackCycleFrames
	    = std::clamp(profile.attackCycleFrames, 1, 127);
	const int64_t neutralPhysicalRange
	    = nonnegative(profile.physicalMin)
	    + nonnegative(profile.physicalMax);
	// Four neutral shares plus one share for each actual Diablo monster class
	// keep sword/mace/triple-demon trade-offs visible without a hard Pareto
	// lock.  A class-neutral weapon therefore contributes seven equal shares.
	const int64_t physicalRangePortfolio
	    = 4 * neutralPhysicalRange
	    + nonnegative(profile.animalMin)
	    + nonnegative(profile.animalMax)
	    + nonnegative(profile.undeadMin)
	    + nonnegative(profile.undeadMax)
	    + nonnegative(profile.demonMin)
	    + nonnegative(profile.demonMax);
	const int meleeHitBasisPoints
	    = std::clamp(profile.meleePiercingToHit, 5, 95) * 100;

	auto repeatedHitBasisPoints = [](int rawHit, int attempts) {
		const int hit = std::clamp(rawHit, 5, 95);
		int missBasisPoints = 10000;
		for (int attempt = 0; attempt < attempts; attempt++)
			missBasisPoints = missBasisPoints * (100 - hit) / 100;
		return 10000 - missBasisPoints;
	};
	// WeaponExplosion retries on the occupied tile until its animation ends:
	// fire has nine collision attempts and lightning seven.  Treat every
	// elemental target as resistant (quarter damage) for a conservative,
	// target-independent lower bound; immunity remains an unavoidable
	// target-specific exception.  Keeping fire/lightning separate preserves
	// their different retry counts.
	const int fireHitBasisPoints
	    = repeatedHitBasisPoints(profile.magicToHit, 9);
	const int lightningHitBasisPoints
	    = repeatedHitBasisPoints(profile.magicToHit, 7);
	const int64_t fireRangePortfolio
	    = 7 * (nonnegative(profile.fireMin)
	        + nonnegative(profile.fireMax));
	const int64_t lightningRangePortfolio
	    = 7 * (nonnegative(profile.lightningMin)
	        + nonnegative(profile.lightningMax));
	// Four times the physical term and one elemental term encode the 1/4
	// resistance factor without early integer truncation.  2048 is the damage
	// scale; 40000 closes the fourfold and basis-point denominators.
	const int64_t physicalDamageMass
	    = 4 * physicalRangePortfolio * meleeHitBasisPoints;
	const int64_t expectedDamageMass
	    = physicalDamageMass
	    + fireRangePortfolio * fireHitBasisPoints
	    + lightningRangePortfolio * lightningHitBasisPoints;
	score += expectedDamageMass * 2048 * ReferenceAttackCycleFrames
	    / (40000 * boundedAttackCycleFrames);
	// Life steal is produced only by a successful physical melee strike.  Use
	// the same post-hit, post-speed physical throughput as the damage score, not
	// a loadout-independent flag bounty; elemental explosions do not heal.  Do
	// not cap this potential by current missing HP, or a full-health pickup would
	// permanently discard future sustain.  Random and the effective fixed tier
	// are separate upstream branches and therefore stack.
	const int64_t expectedPhysicalDamageUtility
	    = physicalDamageMass * 2048 * ReferenceAttackCycleFrames
	    / (40000 * boundedAttackCycleFrames);
	score += expectedPhysicalDamageUtility
	    * LifeStealBasisPoints(profile.effectFlags) / 10000;
	score += nonnegative(profile.armor) * 1024;
	score += nonnegative(profile.magicResistance) * 512;
	score += nonnegative(profile.fireResistance) * 512;
	score += nonnegative(profile.lightningResistance) * 512;
	// Visibility is part of the action graph: controller targets must be lit.
	// Score the post-clamp radius, not raw item affixes, so stacked radiance or
	// darkness is represented exactly once.
	score += nonnegative(profile.lightRadius) * 4096;
	score += nonnegative(profile.maxHitPoints >> 6) * 256;
	// Positive _pIGetHit is extra physical damage received per hit; negative
	// values reduce it.  Keep this signed so both directions are symmetric.
	score -= static_cast<int64_t>(profile.getHit) * 512;
	if (profile.blockEnabled)
		score += 16384
		    + std::clamp(profile.blockChance, 0, 100) * 256;
	score += static_cast<int64_t>(profile.hitRecoveryTier) * 8192;
	// Impact latency matters for time-to-first-hit, but steady-state speed is
	// already represented exactly by the throughput term above.  Keep only a
	// small bounded latency preference to avoid double-counting attack speed.
	score += static_cast<int64_t>(
	             std::max(0, 64 - profile.attackImpactFrames))
	    * 512;

	constexpr uint32_t CurseMask
	    = EffectBits(ItemSpecialEffect::DrainLife)
	    | EffectBits(ItemSpecialEffect::ZeroResistance);
	// Only effects reachable through the installed 15-action melee controller
	// receive a generic value.  Arrow modifiers, mana steal, NoMana and other
	// spell-only bits are deliberately absent: there is no ranged/spell/mana
	// action, so rewarding them would create paper upgrades with no executable
	// combat gain.  Damage/speed/life-steal flags are already represented by
	// their post-Calc fields/tiers and must not be counted twice.
	constexpr uint32_t UntieredMeleeEffectMask
	    = EffectBits(ItemSpecialEffect::Thorns)
	    | EffectBits(ItemSpecialEffect::HalfTrapDamage);
	score += static_cast<int64_t>(
	             CountSetBits(
	                 profile.effectFlags & UntieredMeleeEffectMask))
	    * 2048;
	if (profile.blockEnabled
	    && (profile.effectFlags & EffectBits(ItemSpecialEffect::FastBlock))
	        != 0)
		score += 2048;
	// Removing a curse must be expressible as a strict scalar improvement.
	score += static_cast<int64_t>(
	             2 - CountSetBits(profile.effectFlags & CurseMask))
	    * 16384;
	constexpr uint8_t DamAcCurseMask
	    = DamAcEffectBits(ItemSpecialEffectHf::Decay)
	    | DamAcEffectBits(ItemSpecialEffectHf::Peril)
	    | DamAcEffectBits(ItemSpecialEffectHf::Doppelganger);
	constexpr uint8_t BeneficialDamAcMask
	    = DamAcEffectBits(ItemSpecialEffectHf::Devastation)
	    | DamAcEffectBits(ItemSpecialEffectHf::Jesters)
	    | DamAcEffectBits(ItemSpecialEffectHf::ACAgainstDemons)
	    | DamAcEffectBits(ItemSpecialEffectHf::ACAgainstUndead);
	score += static_cast<int64_t>(
	             CountSetBits(profile.damAcFlags & BeneficialDamAcMask))
	    * 2048;
	score += static_cast<int64_t>(
	             3 - CountSetBits(profile.damAcFlags & DamAcCurseMask))
	    * 16384;
	score += durabilityReserve;
	return static_cast<uint32_t>(std::clamp<int64_t>(
	    score, 0, std::numeric_limits<uint32_t>::max()));
}

GearCombatProfile GearCombatProfileFromPlayer(const Player &player)
{
	GearCombatProfile profile;
	profile.effectFlags = static_cast<uint32_t>(player._pIFlags);
	profile.damAcFlags = static_cast<uint8_t>(player.pDamAcFlags);
	profile.attackSpeedTier = FourLevelEffectTier(
	    profile.effectFlags, ItemSpecialEffect::QuickAttack,
	    ItemSpecialEffect::FastAttack, ItemSpecialEffect::FasterAttack,
	    ItemSpecialEffect::FastestAttack);
	profile.hitRecoveryTier = ThreeLevelEffectTier(
	    profile.effectFlags, ItemSpecialEffect::FastHitRecovery,
	    ItemSpecialEffect::FasterHitRecovery,
	    ItemSpecialEffect::FastestHitRecovery);
	profile.lifeStealTier = TwoLevelEffectTier(
	    profile.effectFlags, ItemSpecialEffect::StealLife3,
	    ItemSpecialEffect::StealLife5);
	profile.manaStealTier = TwoLevelEffectTier(
	    profile.effectFlags, ItemSpecialEffect::StealMana3,
	    ItemSpecialEffect::StealMana5);
	const int skippedFrames
	    = InitialMeleeAttackSkippedFrames(profile.effectFlags);
	profile.attackCycleFrames
	    = std::max(1, static_cast<int>(player._pAFrames) - skippedFrames);
	profile.attackImpactFrames
	    = std::max(1, static_cast<int>(player._pAFNum) - skippedFrames);
	const int damageModifier
	    = player._pIBonusDamMod + player._pDamageMod;
	profile.physicalMin = player._pIMinDam
	    + player._pIBonusDam * player._pIMinDam / 100
	    + damageModifier;
	profile.physicalMax = player._pIMaxDam
	    + player._pIBonusDam * player._pIMaxDam / 100
	    + damageModifier;
	ItemType weaponClass = ItemType::None;
	for (const inv_body_loc hand :
	    { INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT }) {
		if (player.InvBody[hand]._itype == ItemType::Sword)
			weaponClass = ItemType::Sword;
		if (player.InvBody[hand]._itype == ItemType::Mace)
			weaponClass = ItemType::Mace; // matches PlrHitMonst precedence
	}
	auto classDamage = [weaponClass](
	                       int damage, MonsterClass monsterClass) {
		if ((weaponClass == ItemType::Sword
		        && monsterClass == MonsterClass::Undead)
		    || (weaponClass == ItemType::Mace
		        && monsterClass == MonsterClass::Animal))
			return damage - damage / 2;
		if ((weaponClass == ItemType::Sword
		        && monsterClass == MonsterClass::Animal)
		    || (weaponClass == ItemType::Mace
		        && monsterClass == MonsterClass::Undead))
			return damage + damage / 2;
		return damage;
	};
	profile.animalMin
	    = classDamage(profile.physicalMin, MonsterClass::Animal);
	profile.animalMax
	    = classDamage(profile.physicalMax, MonsterClass::Animal);
	profile.undeadMin
	    = classDamage(profile.physicalMin, MonsterClass::Undead);
	profile.undeadMax
	    = classDamage(profile.physicalMax, MonsterClass::Undead);
	const int demonMultiplier = (profile.effectFlags
	                                & EffectBits(
	                                    ItemSpecialEffect::TripleDemonDamage))
	        != 0
	    ? 3
	    : 1;
	profile.demonMin = profile.physicalMin * demonMultiplier;
	profile.demonMax = profile.physicalMax * demonMultiplier;
	profile.meleeToHit = player.GetMeleeToHit();
	profile.meleePiercingToHit = player.GetMeleePiercingToHit();
	profile.magicToHit = player.GetMagicToHit();
	profile.armor = player.GetArmor();
	profile.blockEnabled = player._pBlockFlag;
	profile.blockChance = player.GetBlockChance();
	profile.magicResistance = player._pMagResist;
	profile.fireResistance = player._pFireResist;
	profile.lightningResistance = player._pLghtResist;
	profile.lightRadius = player._pLightRad;
	profile.currentHitPoints = player._pHitPoints;
	profile.maxHitPoints = player._pMaxHP;
	profile.maxMana = player._pMaxMana;
	profile.magic = player._pMagic;
	profile.getHit = player._pIGetHit;
	profile.enemyArmorReduction = player._pIEnAc;
	profile.fireMin = player._pIFMinDam;
	profile.fireMax = player._pIFMaxDam;
	profile.lightningMin = player._pILMinDam;
	profile.lightningMax = player._pILMaxDam;
	profile.spellLevelBonus = player._pISplLvlAdd;
	profile.utility = ScoreGearCombatProfile(
	    profile, GearDurabilityReserve(player));
	return profile;
}

GearCombatProfile SimulateGearCombatProfile(
    const Player &source, std::array<Item, NUM_INVLOC> &body)
{
	// Player owns animation resources and is intentionally move-only.  Build a
	// stat-only inactive-level instance instead of touching the live player;
	// CalcPlrItemVals then remains the single authority for OR/cap/class rules.
	Player simulated = {};
	simulated._pClass = source._pClass;
	simulated.setCharacterLevel(source.getCharacterLevel());
	simulated._pBaseStr = source._pBaseStr;
	simulated._pBaseMag = source._pBaseMag;
	simulated._pBaseDex = source._pBaseDex;
	simulated._pBaseVit = source._pBaseVit;
	simulated._pHPBase = source._pHPBase;
	simulated._pMaxHPBase = source._pMaxHPBase;
	simulated._pManaBase = source._pManaBase;
	simulated._pMaxManaBase = source._pMaxManaBase;
	simulated._pSpellFlags = source._pSpellFlags;
	simulated._pRSpell = source._pRSpell;
	simulated._pRSplType = source._pRSplType;
	simulated._pSBkSpell = source._pSBkSpell;
	simulated._pMemSpells = source._pMemSpells;
	simulated._pAblSpells = source._pAblSpells;
	std::copy_n(std::begin(source._pSplLvl), std::size(source._pSplLvl),
	    std::begin(simulated._pSplLvl));
	// Force isOnActiveLevel() false, preventing the stat-only light-radius
	// calculation from mutating the live dungeon light/vision tables.
	simulated.plrIsOnSetLevel = !setlevel;
	for (int slot = 0; slot < NUM_INVLOC; slot++)
		simulated.InvBody[slot] = body[slot];
	CalcPlrInv(simulated, false);
	SetPlrAnims(simulated);
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		body[slot]._iStatFlag = simulated.InvBody[slot]._iStatFlag;
	}
	return GearCombatProfileFromPlayer(simulated);
}

void MakeGearUtilityRelativeToEmpty(
    GearCombatProfile &profile, const GearCombatProfile &emptyBaseline)
{
	// Keep the public value unsigned while preserving both cursed (negative
	// marginal value) and beneficial loadouts.  Subtracting a same-player empty
	// baseline prevents level-up/base HP/base mana from masquerading as a gear
	// pickup reward.
	constexpr int64_t NeutralLoadoutUtility = 1LL << 30;
	const int64_t relative = NeutralLoadoutUtility
	    + static_cast<int64_t>(profile.utility)
	    - static_cast<int64_t>(emptyBaseline.utility);
	profile.utility = static_cast<uint32_t>(std::clamp<int64_t>(
	    relative, 0, std::numeric_limits<uint32_t>::max()));
}

GearCombatProfile EmptyGearCombatBaseline(const Player &source)
{
	std::array<Item, NUM_INVLOC> emptyBody {};
	return SimulateGearCombatProfile(source, emptyBody);
}

GearCombatProfile LoadoutGearCombatProfile(const Player &player)
{
	const GearCombatProfile emptyBaseline
	    = EmptyGearCombatBaseline(player);
	GearCombatProfile profile = GearCombatProfileFromPlayer(player);
	MakeGearUtilityRelativeToEmpty(profile, emptyBaseline);
	return profile;
}

bool IsConservativeGearUpgrade(
    const GearCombatProfile &previous, const GearCombatProfile &next)
{
	// CalcPlrLifeMana can reduce current life when an equipped +HP/+VIT item
	// is replaced.  The stat-only simulation deliberately preserves HPBase,
	// so this is the exact life the live atomic commit would produce.  Diablo
	// considers fixed-point values 1..63 dead as well (`hasNoLife` shifts by
	// six); never expose or commit a replacement that crosses that boundary.
	if ((next.currentHitPoints >> 6) <= 0)
		return false;

	// Removing +HP/+VIT subtracts an absolute amount from both current and max
	// life.  Merely checking "still alive" allowed 101 HP on a +100 HP item to
	// become 1 HP after a high-damage replacement.  Require the projected
	// health fraction to stay at least as safe.  At full health both fractions
	// are 1, so healing later unlocks the same trade instead of permanently
	// freezing the slot.  Fixed-point values are bounded by 2000<<6; int64
	// cross-products are exact and avoid division/rounding.
	const int64_t nextHealthFraction
	    = static_cast<int64_t>(next.currentHitPoints)
	    * previous.maxHitPoints;
	const int64_t previousHealthFraction
	    = static_cast<int64_t>(previous.currentHitPoints)
	    * next.maxHitPoints;
	if (nextHealthFraction < previousHealthFraction)
		return false;

	// The simulation is deterministic integer arithmetic: there is no rounding
	// noise to suppress.  Requiring an arbitrary 1024 margin silently discarded
	// real +HP/+resistance/+elemental upgrades and reduced a14 opportunities.
	if (next.utility <= previous.utility)
		return false;

	constexpr uint32_t CurseMask
	    = EffectBits(ItemSpecialEffect::DrainLife)
	    | EffectBits(ItemSpecialEffect::ZeroResistance);
	if ((next.effectFlags & ~previous.effectFlags & CurseMask) != 0)
		return false;
	constexpr uint8_t DamAcCurseMask
	    = DamAcEffectBits(ItemSpecialEffectHf::Decay)
	    | DamAcEffectBits(ItemSpecialEffectHf::Peril)
	    | DamAcEffectBits(ItemSpecialEffectHf::Doppelganger);
	if ((next.damAcFlags & static_cast<uint8_t>(~previous.damAcFlags)
	        & DamAcCurseMask)
	    != 0)
		return false;

	// Everything else is already priced in the one whole-loadout scalar.
	// Re-imposing per-field monotonicity here locks the source slot forever as
	// soon as it receives a rare affix (the original growth plateau).
	return true;
}

Item PrepareGearCandidate(const Item &item, const Player &player)
{
	Item candidate = item;
	// The one-way training task has no Cain/identify action.  The gear macro's
	// semantics therefore include identification, and both targeting and the
	// eventual atomic replacement must score exactly that same post-identify
	// item.
	candidate._iIdentified = true;
	candidate.updateRequiredStatsCacheForPlayer(player);
	return candidate;
}

void AppendItemCombatState(
    py::dict &entry, const Item &item, int activeItemId,
    bool scoreAsPickupCandidate = false)
{
	const bool effectsActive = ItemEffectsActive(item);
	const Item scored = scoreAsPickupCandidate && MyPlayer != nullptr
	    ? PrepareGearCandidate(item, *MyPlayer)
	    : item;
	const uint32_t combatUtility = ItemCombatUtility(scored);
	entry["active_id"] = activeItemId;
	entry["base_id"] = static_cast<int>(item.IDidx);
	entry["item_type"] = static_cast<int>(item._itype);
	entry["equip_loc"] = static_cast<int>(item._iLoc);
	entry["base_ac"] = static_cast<int>(item._iAC);
	entry["identified"] = item._iIdentified;
	entry["quality"] = static_cast<int>(item._iMagical);
	entry["durability"] = item._iDurability;
	entry["max_durability"] = item._iMaxDur;
	entry["effects_active"] = effectsActive;
	entry["min_damage"] = item._iMinDam;
	entry["max_damage"] = item._iMaxDam;
	// Keep raw affixes observable even while inactive.  effects_active tells
	// the policy whether they currently apply; floor gear is auto-identified
	// by a14, so hiding these values would make its utility irreconstructible.
	entry["effect_damage"] = item._iPLDam;
	entry["effect_to_hit"] = item._iPLToHit;
	entry["effect_ac_percent"] = item._iPLAC;
	entry["effect_strength"] = item._iPLStr;
	entry["effect_magic"] = item._iPLMag;
	entry["effect_dexterity"] = item._iPLDex;
	entry["effect_vitality"] = item._iPLVit;
	entry["effect_fire_resist"] = item._iPLFR;
	entry["effect_lightning_resist"] = item._iPLLR;
	entry["effect_magic_resist"] = item._iPLMR;
	entry["effect_mana"] = item._iPLMana;
	entry["effect_hp"] = item._iPLHP;
	entry["effect_damage_mod"] = item._iPLDamMod;
	entry["effect_get_hit"] = item._iPLGetHit;
	entry["effect_light"] = item._iPLLight;
	entry["effect_spell_level"] = item._iSplLvlAdd;
	entry["effect_enemy_ac"] = item._iPLEnAc;
	entry["effect_fire_min"] = item._iFMinDam;
	entry["effect_fire_max"] = item._iFMaxDam;
	entry["effect_lightning_min"] = item._iLMinDam;
	entry["effect_lightning_max"] = item._iLMaxDam;
	entry["seed_hi"] = HighWord(item._iSeed);
	entry["seed_lo"] = LowWord(item._iSeed);
	entry["create_info"] = item._iCreateInfo;
	entry["item_class"] = static_cast<int>(item._iClass);
	entry["misc_id"] = static_cast<int>(item._iMiscId);
	entry["spell_id"] = static_cast<int>(item._iSpell);
	entry["charges"] = item._iCharges;
	entry["max_charges"] = item._iMaxCharges;
	entry["min_strength"] = item._iMinStr;
	entry["min_magic"] = item._iMinMag;
	entry["min_dexterity"] = item._iMinDex;
	entry["stat_usable"] = item._iStatFlag;
	entry["effect_flags"] = static_cast<uint32_t>(item._iFlags);
	entry["effect_dam_ac_flags"] = static_cast<uint8_t>(item._iDamAcFlags);
	entry["combat_utility"] = combatUtility;
	entry["combat_utility_hi"] = HighWord(combatUtility);
	entry["combat_utility_lo"] = LowWord(combatUtility);
}

bool CanReachTarget(Point target)
{
	const Player &player = *MyPlayer;
	if (player.position.future == target)
		return true;
	int8_t path[MaxPathLengthPlayer];
	return FindPath(CanStep,
	           [&player](Point position) { return PosOkPlayer(player, position); },
	           player.position.future, target, path, MaxPathLengthPlayer)
	    > 0;
}

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

void ClearEpisodePersistentGameplayState()
{
	// FreeGame/InitLevelMonsters follow the upstream lifetime model: most
	// Monster fields are overwritten for the next level, but the complete
	// objects are not value-reset.  That is safe for one ordinary game and
	// unsafe for thousands of independent Gym episodes in one process:
	// position.last and other AI fields survived reset and could consume a
	// different number of global-RNG draws before otherwise identical drops.
	for (Monster &monster : Monsters)
		monster = {};

	// Bestiary kill counts are hero state, not process state.  The normal UI
	// load path restores them from the selected hero; our fresh-save bridge
	// must explicitly establish the new-hero zero value before every episode.
	std::fill(
	    std::begin(MonsterKillCounts), std::end(MonsterKillCounts), 0);

	// Upstream clears this only inside
	// LoadGameLevelFirstFlagEntry()'s !HeadlessMode branch.  DiabloGym is
	// permanently headless, so without this call a unique generated in one
	// episode can never be generated by the same seed again.  Seed 7001
	// exposed the leak as CF_UNIQUE (0x200) disappearing on the second replay.
	ClearUniqueItemFlags();
}

void NormalizeStarterItemSeeds(Player &player, uint32_t episodeSeed)
{
	// CreatePlayer seeds its starter items from SDL_GetTicks before Reset can
	// take ownership of the global RNG.  Normalize identity-only _iSeed values
	// after NetInit has reloaded the fresh hero.  Use a separate fixed-domain
	// mt19937 so this consumes neither DungeonSeeds nor the gameplay LCG.
	constexpr uint32_t StarterItemSeedDomain = 0xA17D1AB1U;
	std::mt19937 itemSeeds(episodeSeed ^ StarterItemSeedDomain);
	auto normalize = [&itemSeeds](Item &item) {
		if (item.isEmpty())
			return;
		uint32_t deterministicSeed;
		do {
			deterministicSeed = static_cast<uint32_t>(itemSeeds());
		} while (deterministicSeed == 0);
		item._iSeed = deterministicSeed;
	};
	for (Item &item : player.InvBody)
		normalize(item);
	for (Item &item : player.InvList)
		normalize(item);
	for (Item &item : player.SpdList)
		normalize(item);
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
	// 新 dual wire 需要保留 26.6 定点低位；旧整数 HP/Mana 字段继续冻结。
	// 两个 uint16 word 各自可被 float32 精确表示，避免除以尺度后吞低位。
	obs["hp_fixed_hi"] = HighWord(static_cast<uint32_t>(player._pHitPoints));
	obs["hp_fixed_lo"] = LowWord(static_cast<uint32_t>(player._pHitPoints));
	obs["max_hp_fixed_hi"] = HighWord(static_cast<uint32_t>(player._pMaxHP));
	obs["max_hp_fixed_lo"] = LowWord(static_cast<uint32_t>(player._pMaxHP));
	obs["mana_fixed_hi"] = HighWord(static_cast<uint32_t>(player._pMana));
	obs["mana_fixed_lo"] = LowWord(static_cast<uint32_t>(player._pMana));
	obs["max_mana_fixed_hi"] = HighWord(static_cast<uint32_t>(player._pMaxMana));
	obs["max_mana_fixed_lo"] = LowWord(static_cast<uint32_t>(player._pMaxMana));
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
	// ActiveMonsters 快照无法看见同一宏动作内“生成后又死亡”的怪物；
	// 引擎的按类型永久击杀账才是 spawn→die 闭环事实源。用 uint64 安全
	// 汇总并直接交给 Python int，避免长期训练中窄整数回绕。
	obs["monster_kill_total"] = MonsterKillTotal();
	obs["belt_heals"] = CountBeltHeals(); // v12 起入 raw;v13 起由 env 写进观测向量(瓶盲修复)
	// v3 曾把 Healing 卷轴误归为一按即用的治疗药。v4 动作必须继续排除
	// 它，冻结网络的兼容视图却要逐字看到旧计数；两个事实源显式并存，
	// 不能让 Python 从已修正的 bool 猜回已经丢失的旧分类。
	obs["legacy_belt_heals"] = CountLegacyBeltHeals();
	obs["belt_free_slots"] = CountBeltFreeSlots(); // v4 raw-only:精确屏蔽“腰带满”的捡药空按
	// action12 消耗“从左到右第一瓶即时治疗药”。只给总数会把
	// [小红,大紫] 与 [大紫,小红] 伪装成同一状态，尽管下一次按键的治疗量
	// 不同。逐槽公开 0..4 原始类别；Python 的新 Worker 线协议将其展开为
	// 8×4 one-hot，旧 295/298 维兼容视图完全不变。
	py::list beltHealKinds;
	py::list beltSlotKinds;
	for (int i = 0; i < MaxBeltItems; i++)
	{
		const Item &beltItem = player.SpdList[i];
		const int healKind = InstantHealKind(beltItem);
		beltHealKinds.append(healKind);
		// 六个互斥类:0 empty,1 other,2..5 对应即时治疗 1..4。
		// 旧 heal_kinds 的 0 同时代表 empty/other，只给 free 总数仍无法
		// 确定 a13 自动落入哪一槽以及随后 a12 会先喝哪瓶。
		beltSlotKinds.append(
		    beltItem.isEmpty() ? 0 : (healKind == 0 ? 1 : healKind + 1));
	}
	obs["belt_heal_kinds"] = beltHealKinds;
	obs["belt_slot_kinds"] = beltSlotKinds;
	obs["armor_class"] = player.GetArmor(); // v14:护甲值(_pIBonusAC + _pIAC + 敏捷/5)
	// a9 的真实转移由这些“已汇总、当前生效”的战斗量决定。只给 HP/AC
	// 会把不同武器伤害、命中、抗性、格挡、攻速/吸血 flags 的角色压成
	// 同一状态；dual Worker 以固定尺度编码标量并把两个 flags 展成 bit。
	obs["hero_class"] = static_cast<int>(player._pClass);
	obs["strength"] = player._pStrength;
	obs["magic"] = player._pMagic;
	obs["dexterity"] = player._pDexterity;
	obs["vitality"] = player._pVitality;
	obs["melee_to_hit"] = player.GetMeleeToHit();
	obs["melee_piercing_to_hit"] = player.GetMeleePiercingToHit();
	obs["block_chance"] = player.GetBlockChance();
	obs["item_min_damage"] = player._pIMinDam;
	obs["item_max_damage"] = player._pIMaxDam;
	obs["damage_mod"] = player._pDamageMod;
	obs["item_bonus_damage"] = player._pIBonusDam;
	obs["item_bonus_to_hit"] = player._pIBonusToHit;
	obs["item_bonus_damage_mod"] = player._pIBonusDamMod;
	obs["item_get_hit"] = player._pIGetHit;
	obs["item_enemy_ac"] = player._pIEnAc;
	obs["magic_resist"] = player._pMagResist;
	obs["fire_resist"] = player._pFireResist;
	obs["lightning_resist"] = player._pLghtResist;
	obs["item_fire_min"] = player._pIFMinDam;
	obs["item_fire_max"] = player._pIFMaxDam;
	obs["item_lightning_min"] = player._pILMinDam;
	obs["item_lightning_max"] = player._pILMaxDam;
	obs["block_enabled"] = player._pBlockFlag;
	obs["item_effect_flags"] = static_cast<uint32_t>(player._pIFlags);
	obs["item_dam_ac_flags"] = static_cast<uint8_t>(player.pDamAcFlags);
	const uint32_t boundedGearUtility
	    = LoadoutGearCombatProfile(player).utility;
	obs["gear_combat_utility"] = boundedGearUtility;
	obs["gear_combat_utility_hi"] = HighWord(boundedGearUtility);
	obs["gear_combat_utility_lo"] = LowWord(boundedGearUtility);
	py::list equippedItems;
	for (int bodyLocation = 0; bodyLocation < NUM_INVLOC; bodyLocation++) {
		const Item &equipped = player.InvBody[bodyLocation];
		py::dict entry;
		entry["present"] = !equipped.isEmpty();
		// active_id 仅属于地面数组；已穿槽以 0 作稳定占位，其余字段与
		// floor gear 共用完整 CalcPlrInv/耐久转移协议。
		AppendItemCombatState(entry, equipped, 0);
		equippedItems.append(entry);
	}
	obs["equipped_items"] = equippedItems;
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
		// CMD_ATTACKID/MakePlrPath 与 reachable 都以 future（动画提交后的
		// 占位格）为准；tile 只是当前渲染格。两者在走路动画期间会连续
		// 多个 tick 不同，宏的止损几何必须与引擎采用同一坐标。
		m["future_x"] = static_cast<int>(monster.position.future.x);
		m["future_y"] = static_cast<int>(monster.position.future.y);
		m["hp"] = monster.hitPoints >> 6;
		m["max_hp"] = monster.maxHitPoints >> 6;
		m["hp_fixed_hi"] = HighWord(static_cast<uint32_t>(monster.hitPoints));
		m["hp_fixed_lo"] = LowWord(static_cast<uint32_t>(monster.hitPoints));
		m["max_hp_fixed_hi"] = HighWord(static_cast<uint32_t>(monster.maxHitPoints));
		m["max_hp_fixed_lo"] = LowWord(static_cast<uint32_t>(monster.maxHitPoints));
		// 同 type/HP/坐标的怪物若处于攻击伤害帧前与 idle，下一拍伤亡
		// 分布完全不同。公开驱动 MonsterMode/AI 与当前动画推进的有限
		// 状态，以及已缩放过的即时攻防量；dual wire 对 flags 逐 bit 编码。
		m["mode"] = static_cast<int>(monster.mode);
		m["direction"] = static_cast<int>(monster.direction);
		m["anim_frame"] = static_cast<int>(monster.animInfo.currentFrame);
		m["anim_tick"] = static_cast<int>(
		    monster.animInfo.tickCounterOfCurrentFrame);
		m["anim_ticks_per_frame"] = static_cast<int>(
		    monster.animInfo.ticksPerFrame);
		m["anim_num_frames"] = static_cast<int>(
		    monster.animInfo.numberOfFrames);
		m["anim_progress"] = monster.animInfo.ticksPerFrame > 0
		    ? static_cast<int>(monster.animInfo.getAnimationProgress())
		    : 0;
		m["anim_petrified"] = monster.animInfo.isPetrified;
		m["goal"] = static_cast<int>(monster.goal);
		m["goal_var1"] = monster.goalVar1;
		m["goal_var2"] = monster.goalVar2;
		m["goal_var3"] = monster.goalVar3;
		m["var1"] = monster.var1;
		m["var2"] = monster.var2;
		m["var3"] = monster.var3;
		m["enemy_dx"] = static_cast<int>(monster.enemyPosition.x)
		    - static_cast<int>(player.position.tile.x);
		m["enemy_dy"] = static_cast<int>(monster.enemyPosition.y)
		    - static_cast<int>(player.position.tile.y);
		m["old_dx"] = static_cast<int>(monster.position.old.x)
		    - static_cast<int>(player.position.tile.x);
		m["old_dy"] = static_cast<int>(monster.position.old.y)
		    - static_cast<int>(player.position.tile.y);
		m["temp_dx"] = static_cast<int>(monster.position.temp.x)
		    - static_cast<int>(player.position.tile.x);
		m["temp_dy"] = static_cast<int>(monster.position.temp.y)
		    - static_cast<int>(player.position.tile.y);
		m["last_dx"] = static_cast<int>(monster.position.last.x)
		    - static_cast<int>(player.position.tile.x);
		m["last_dy"] = static_cast<int>(monster.position.last.y)
		    - static_cast<int>(player.position.tile.y);
		m["active_for_ticks"] = monster.activeForTicks;
		m["path_count"] = monster.pathCount;
		m["enemy_id"] = monster.enemy;
		m["is_invalid"] = monster.isInvalid;
		m["monster_level_type"] = monster.levelType;
		m["ai"] = static_cast<int>(monster.ai);
		m["intelligence"] = monster.intelligence;
		m["min_damage"] = monster.minDamage;
		m["max_damage"] = monster.maxDamage;
		m["min_damage_special"] = monster.minDamageSpecial;
		m["max_damage_special"] = monster.maxDamageSpecial;
		m["armor_class"] = monster.armorClass;
		m["resistance"] = monster.resistance;
		m["unique_type"] = static_cast<int>(monster.uniqueType);
		m["reduce_strength"] = monster.reducePlayerStrength;
		m["reduce_magic"] = monster.reducePlayerMagic;
		m["reduce_dexterity"] = monster.reducePlayerDexterity;
		m["reduce_vitality"] = monster.reducePlayerVitality;
		m["reduce_max_hp"] = monster.reducePlayerMaxHP;
		m["reduce_max_mana"] = monster.reducePlayerMaxMana;
		m["leader"] = monster.leader;
		m["leader_relation"] = static_cast<int>(monster.leaderRelation);
		m["pack_size"] = monster.packSize;
		m["who_hit"] = monster.whoHit;
		m["unique_trans"] = monster.uniqTrans;
		m["corpse_id"] = monster.corpseId;
		m["light_id"] = monster.lightId;
		m["golem_to_hit"] = monster.golemToHit;
		m["talk_msg"] = static_cast<int>(monster.talkMsg);
		m["monster_level"] = monster.level(sgGameInitInfo.nDifficulty);
		m["to_hit"] = monster.toHit(sgGameInitInfo.nDifficulty);
		m["to_hit_special"] = monster.toHitSpecial(sgGameInitInfo.nDifficulty);
		const uint32_t experience = monster.exp(sgGameInitInfo.nDifficulty);
		m["experience_hi"] = HighWord(experience);
		m["experience_lo"] = LowWord(experience);
		m["rnd_item_seed_hi"] = HighWord(monster.rndItemSeed);
		m["rnd_item_seed_lo"] = LowWord(monster.rndItemSeed);
		m["ai_seed_hi"] = HighWord(monster.aiSeed);
		m["ai_seed_lo"] = LowWord(monster.aiSeed);
		m["combat_flags"] = static_cast<uint16_t>(monster.flags);
		// 全层列表仍是奖励/击杀的事实源；策略只能消费当前人类玩家也能
		// 看见并经原生寻路到达的子集，禁止用未探索房间里的全知坐标作战。
		const bool visible = (monster.flags & MFLAG_HIDDEN) == 0
		    && !monster.isPlayerMinion()
		    && IsTileLit(monster.position.tile);
		m["visible"] = visible;
		m["reachable"] = visible && CanReachTarget(monster.position.future);
		monsters.append(m);
	}
	obs["monsters"] = monsters;

	py::list items;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int activeItemId = ActiveItems[i];
		const Item &item = Items[activeItemId];
		py::dict it;
		it["active_id"] = activeItemId;
		it["x"] = static_cast<int>(item.position.x);
		it["y"] = static_cast<int>(item.position.y);
		it["heal"] = IsHealItem(item);   // v13:捡药宏的目标标志
		it["heal_kind"] = InstantHealKind(item);
		it["legacy_heal"] = IsLegacyHealItem(item); // 冻结 v3 只读观测
		it["gear"] = IsWantedGear(item); // v14:捡装备宏的目标标志(空槽+属性达标)
		// a14 的坐标与 bool 不足以决定转移：同一格可以承载不同装备，
		// 其槽位、基础 AC 及已生效词缀会改变即时奖励和后续战斗。把
		// CalcPlrItemVals 会消费的物品事实完整留在 raw；Python 只为
		// snapshot 选中的那一件编码，不扩大旧 295/298 兼容观测。
		AppendItemCombatState(it, item, activeItemId, true);
		const bool visible = IsTileLit(item.position);
		it["visible"] = visible;
		it["reachable"] = visible && CanReachTarget(item.position);
		items.append(it);
	}
	obs["floor_items"] = items;

	// 投射物会在一次 option 的数个 engine tick 内独立移动/命中。只公开
	// 怪物与格子会把“火球已贴脸”和“尚未发射”压成同一 Worker 状态。
	// 这里保留 SaveMissile 的全部标量状态并补上重复碰撞 hash；Python
	// 仅选择 radius-12 内固定槽，并用 uint16 words 无损编码 int32。
	py::list missiles;
	for (Missile &missile : Missiles) {
		py::dict entry;
		const MissileSource sourceType = missile.sourceType();
		const bool visible = IsTileLit(missile.position.tile);
		const bool startVisible = IsTileLit(missile.position.start);
		bool sourceVisible = false;
		if (sourceType == MissileSource::Player) {
			const bool sourceInBounds = missile._misource >= 0
			    && static_cast<size_t>(missile._misource) < Players.size();
			if (sourceInBounds) {
				const Player &source = Players[missile._misource];
				// A bounded player slot is not necessarily a live entity:
				// inactive/other-level/transitioning slots retain coordinates
				// that must not make source identity policy-observable.
				sourceVisible = source.plractive
				    && source.isOnActiveLevel()
				    && !source._pLvlChanging
				    && IsTileLit(source.position.tile);
			}
		} else if (sourceType == MissileSource::Monster) {
			const bool sourceInBounds = missile._misource >= 0
			    && static_cast<size_t>(missile._misource) < MaxMonsters;
			bool sourceActive = false;
			if (sourceInBounds) {
				for (size_t i = 0; i < ActiveMonsterCount; ++i) {
					if (ActiveMonsters[i]
					    == static_cast<unsigned>(missile._misource)) {
						sourceActive = true;
						break;
					}
				}
			}
			if (sourceActive) {
				const Monster &source = Monsters[missile._misource];
				sourceVisible = !source.isInvalid
				    && !source.hasNoLife()
				    && (source.flags & MFLAG_HIDDEN) == 0
				    && !source.isPlayerMinion()
				    && IsTileLit(source.position.tile);
			}
		}
		entry["type"] = static_cast<int>(missile._mitype);
		entry["visible"] = visible;
		entry["source_visible"] = sourceVisible;
		entry["start_visible"] = startVisible;
		entry["tile_dx"] = static_cast<int>(missile.position.tile.x)
		    - static_cast<int>(player.position.tile.x);
		entry["tile_dy"] = static_cast<int>(missile.position.tile.y)
		    - static_cast<int>(player.position.tile.y);
		entry["offset_x"] = missile.position.offset.deltaX;
		entry["offset_y"] = missile.position.offset.deltaY;
		entry["velocity_x"] = missile.position.velocity.deltaX;
		entry["velocity_y"] = missile.position.velocity.deltaY;
		entry["start_dx"] = startVisible
		    ? static_cast<int>(missile.position.start.x)
		        - static_cast<int>(player.position.tile.x)
		    : 0;
		entry["start_dy"] = startVisible
		    ? static_cast<int>(missile.position.start.y)
		        - static_cast<int>(player.position.tile.y)
		    : 0;
		entry["traveled_x"] = missile.position.traveled.deltaX;
		entry["traveled_y"] = missile.position.traveled.deltaY;
		entry["direction"] = missile.getFrameGroupRaw();
		entry["spell_level"] = missile._mispllvl;
		entry["deleted"] = missile._miDelFlag;
		entry["anim_type"] = static_cast<int>(missile._miAnimType);
		entry["anim_flags"] = static_cast<int>(missile._miAnimFlags);
		entry["anim_delay"] = missile._miAnimDelay;
		entry["anim_len"] = missile._miAnimLen;
		entry["anim_width"] = missile._miAnimWidth;
		entry["anim_width2"] = missile._miAnimWidth2;
		entry["anim_count"] = missile._miAnimCnt;
		entry["anim_add"] = missile._miAnimAdd;
		entry["anim_frame"] = missile._miAnimFrame;
		entry["draw"] = missile._miDrawFlag;
		entry["light"] = missile._miLightFlag;
		entry["pre"] = missile._miPreFlag;
		entry["uniq_trans"] = missile._miUniqTrans;
		entry["duration"] = missile.duration;
		entry["source_id"] = sourceVisible ? missile._misource : -1;
		entry["caster"] = static_cast<int>(missile._micaster);
		entry["damage"] = missile._midam;
		entry["hit"] = missile._miHitFlag;
		entry["distance"] = missile._midist;
		entry["light_id"] = missile._mlid;
		entry["random"] = missile._mirnd;
		entry["var1"] = missile.var1;
		entry["var2"] = missile.var2;
		entry["var3"] = missile.var3;
		entry["var4"] = missile.var4;
		entry["var5"] = missile.var5;
		entry["var6"] = missile.var6;
		entry["var7"] = missile.var7;
		entry["limit_reached"] = missile.limitReached;
		entry["last_collision_target_hash"] = missile.lastCollisionTargetHash;
		entry["source_type"] = static_cast<int>(sourceType);
		// TARGET_BOTH also enters PlayerMHit in CheckMissileCol.  "hostile"
		// means capable of colliding with the controlled player, not merely
		// "cast by a monster"; damage/type let the policy distinguish harmless
		// portals and other TARGET_BOTH effects.
		entry["hostile"] = missile._micaster != TARGET_MONSTERS;
		missiles.append(entry);
	}
	obs["missiles"] = missiles;

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
	const bool archiveIsSpawn = gbIsSpawn;
	std::string expectedArchiveName =
	    std::filesystem::path(expectedMainArchive).filename().string();
	std::transform(
	    expectedArchiveName.begin(), expectedArchiveName.end(),
	    expectedArchiveName.begin(),
	    [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	const bool expectedIsSpawn = expectedArchiveName == "spawn.mpq";
	if (archiveIsSpawn != expectedIsSpawn)
		throw std::runtime_error(
		    "主档案模式与 gbIsSpawn 不一致: archive="
		    + expectedArchiveName + ",gbIsSpawn="
		    + std::to_string(archiveIsSpawn ? 1 : 0));
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
	// 训练转移不能继承 save_dir/diablo.ini 的个人 QoL、速度或任务配置。
	// 尤其 autoRefillBelt 会让 action12 优先消耗不可见背包药，自动拾取
	// 会在无策略动作时改 inventory，autoEquipWeapons 还可能把 Lazarus
	// 法杖穿到手上而逃过单调任务适配器。所有会改变世界/动作结果的选项
	// 在 Lua 和新局初始化前钉成协议常量。
	Options &options = GetOptions();
	const auto activeMods = options.Mods.GetActiveModList();
	if (!activeMods.empty()) {
		std::string names;
		for (std::string_view name : activeMods) {
			if (!names.empty())
				names += ",";
			names += name;
		}
		throw std::runtime_error(
		    "DiabloGym 禁止启用 Lua mods；save_dir 中检测到 active mods: "
		    + names);
	}
	options.GameMode.gameMode.SetValue(StartUpGameMode::Diablo);
	// shareware 的回调会直接重写 gbIsSpawn；恢复为实际加载主档案决定的
	// 模式，既阻断 ini 把完整 DIABDAT 降成试玩版，也保留 spawn.mpq 支持。
	options.GameMode.shareware.SetValue(archiveIsSpawn);
	if (gbIsSpawn != archiveIsSpawn)
		throw std::runtime_error(
		    "冻结 GameMode.shareware 后 gbIsSpawn 与主档案不一致");
	options.Gameplay.tickRate.SetValue(20);
	options.Gameplay.runInTown.SetValue(false);
	options.Gameplay.randomizeQuests.SetValue(true);
	options.Gameplay.theoQuest.SetValue(false);
	options.Gameplay.cowQuest.SetValue(false);
	options.Gameplay.testBard.SetValue(false);
	options.Gameplay.testBarbarian.SetValue(false);
	options.Gameplay.friendlyFire.SetValue(true);
	options.Gameplay.multiplayerFullQuests.SetValue(false);
	options.Gameplay.autoGoldPickup.SetValue(false);
	options.Gameplay.autoElixirPickup.SetValue(false);
	options.Gameplay.autoOilPickup.SetValue(false);
	options.Gameplay.autoPickupInTown.SetValue(false);
	options.Gameplay.numHealPotionPickup.SetValue(0);
	options.Gameplay.numFullHealPotionPickup.SetValue(0);
	options.Gameplay.numManaPotionPickup.SetValue(0);
	options.Gameplay.numFullManaPotionPickup.SetValue(0);
	options.Gameplay.numRejuPotionPickup.SetValue(0);
	options.Gameplay.numFullRejuPotionPickup.SetValue(0);
	options.Gameplay.autoRefillBelt.SetValue(false);
	options.Gameplay.autoEquipWeapons.SetValue(false);
	options.Gameplay.autoEquipShields.SetValue(false);
	options.Gameplay.autoEquipArmor.SetValue(true);
	options.Gameplay.autoEquipHelms.SetValue(true);
	options.Gameplay.autoEquipJewelry.SetValue(true);
	options.Gameplay.quickCast.SetValue(false);
	options.Gameplay.disableCripplingShrines.SetValue(false);
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
	ClearEpisodePersistentGameplayState();
	gStallPrints = 0;
	gMonotonicQuestTurnInUsed = false;

	CreateFreshHeroSave();
	gbLoadGame = false;

	if (!NetInit(/*bSinglePlayer=*/true))
		throw std::runtime_error("NetInit failed");
	NormalizeStarterItemSeeds(*MyPlayer, seed);
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

int ActWait()
{
	if (!CanAcceptPlayerAction("act_wait"))
		return 0;
	Player &player = *MyPlayer;
	if (!gbRunGame || player._pmode == PM_DEATH || player._pmode == PM_QUIT)
		return 0;

	// v4 明确的 wait/cancel 语义：
	// 1. 立即清掉已经由上一拍 OnWalk/OnAttack 写入的长路径与延迟动作；
	// 2. 再把“走到当前 future 格”排到 loopback 网络队列末尾，作为 FIFO
	//    栅栏。只做第 1 步仍有竞态：调用方可能先排 act_attack_monster，
	//    再在同一 bridge.step 前排 wait，下一拍旧攻击包会重新写回 destAction。
	// 已经进入攻击/施法动画时必须立刻回站立：若只清 destAction，本拍尚未
	// 到伤害帧的一刀会在下一个 Gym 动作中命中，把击杀错误记到 action0/
	// 喝药。走路动画不硬切，允许当前已提交的一格自然收尾，随后停止。
	if (IsAnyOf(player._pmode, PM_ATTACK, PM_RATTACK, PM_SPELL))
		StartStand(player, player._pdir);
	player.Stop();
	LastPlayerAction = PlayerActionType::None;
	NetSendCmdLoc(MyPlayerId, true, CMD_WALKXY, player.position.future);
	return 1;
}

void ActWalk(int x, int y)
{
	if (!CanAcceptPlayerAction("act_walk"))
		return;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
		return; // 地图边缘的越界走格按 Gym 无效动作处理
	NetSendCmdLoc(MyPlayerId, true, CMD_WALKXY, { x, y });
}

struct DynamicTileDanger {
	Object *object = nullptr;
	bool damagingHazard = false;
	bool explosiveBreakable = false;
};

DynamicTileDanger InspectDynamicTileDanger(Point position)
{
	DynamicTileDanger danger;
	if (position.x < 0 || position.x >= MAXDUNX
	    || position.y < 0 || position.y >= MAXDUNY)
		return danger;
	danger.object = FindObjectAtPosition(position);
	if (danger.object != nullptr) {
		danger.damagingHazard
		    = danger.object->_otype == OBJ_FLAMEHOLE
		    && danger.object->_oVar2 == 0
		    && danger.object->_oVar4 != 0;
		danger.explosiveBreakable
		    = danger.object->IsBreakable()
		    && danger.object->isExplosive();
	}
	// Burning crosses damage the tile immediately north of their object.
	Object *southObject = position.y + 1 < MAXDUNY
	    ? FindObjectAtPosition({ position.x, position.y + 1 })
	    : nullptr;
	danger.damagingHazard = danger.damagingHazard
	    || (southObject != nullptr
	        && IsAnyOf(
	            southObject->_otype, OBJ_BCROSS, OBJ_TBCROSS));
	return danger;
}

constexpr int AbsoluteDelta(int value)
{
	return value < 0 ? -value : value;
}

constexpr bool IsAdjacentControllerDelta(int dx, int dy)
{
	return (AbsoluteDelta(dx) > AbsoluteDelta(dy)
	               ? AbsoluteDelta(dx)
	               : AbsoluteDelta(dy))
	    == 1;
}

constexpr bool IsExactControllerEdge(int dx, int dy, int pathSteps)
{
	return IsAdjacentControllerDelta(dx, dy) && pathSteps == 1;
}

static_assert(IsExactControllerEdge(1, 1, 1),
    "斜向 Worker 原子键必须能表达为一步");
static_assert(!IsExactControllerEdge(1, 1, 2),
    "相邻目标若只能绕路两步必须 fail-close");
static_assert(!IsExactControllerEdge(2, 0, 1),
    "远 waypoint 不得伪装成控制器原子边");

int ActExploreWalk(
    int x,
    int y,
    const std::vector<std::pair<int, int>> &protectedTiles,
    int centerX,
    int centerY,
    int radius)
{
	if (!CanAcceptPlayerAction("act_explore_walk"))
		return 0;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY
	    || centerX < 0 || centerX >= MAXDUNX
	    || centerY < 0 || centerY >= MAXDUNY
	    || radius < 0
	    || std::abs(x - centerX) > radius
	    || std::abs(y - centerY) > radius)
		return 0;

	// ActWait 在上一宏末尾放入一个 loopback FIFO 栅栏；安全路径必须在
	// 它之后写入，否则下一 Step 处理旧 wait 包时会把新 walkpath 擦掉。
	ProcessGameMessagePackets();
	if (!CanAcceptPlayerAction("act_explore_walk"))
		return 0;

	auto inWindow = [centerX, centerY, radius](Point position) {
		return std::abs(position.x - centerX) <= radius
		    && std::abs(position.y - centerY) <= radius;
	};
	auto encode = [](int tx, int ty) { return tx * MAXDUNY + ty; };
	std::unordered_set<int> forbidden;
	for (const auto &[tx, ty] : protectedTiles) {
		if (tx < 0 || tx >= MAXDUNX || ty < 0 || ty >= MAXDUNY)
			throw std::out_of_range("act_explore_walk protected 坐标越界");
		if (std::abs(tx - centerX) > radius
		    || std::abs(ty - centerY) > radius)
			throw std::out_of_range(
			    "act_explore_walk protected 坐标超出固定快照");
		forbidden.insert(encode(tx, ty));
	}
	if (forbidden.count(encode(x, y)) != 0)
		return 0;

	Player &player = *MyPlayer;
	ClrPlrPath(player);
	player.destAction = ACTION_NONE;
	LastPlayerAction = PlayerActionType::Walk;
	if (player.position.future == Point { x, y })
		return 1;
	const Point start = player.position.future;
	// Worker 原子方向动作含四个斜向键；只拒绝非相邻 waypoint。堵角等
	// 斜向非法边仍由 CanStep + 最终 steps==1 逐边 fail-close，不能绕路。
	if (!IsAdjacentControllerDelta(x - start.x, y - start.y)
	    || !inWindow(start))
		return 0;
	const Point target { x, y };
	const DynamicTileDanger danger = InspectDynamicTileDanger(target);
	// Python 的冻结快照之后，火焰机关可能刚进入伤害相位；爆炸桶也可能
	// 在宏执行过程中出现在下一格。提交边的最后一刻重读当前已照亮事实，
	// 防止安全路径在 TOCTOU 窗口里变成踩火/撞桶。
	if ((danger.damagingHazard && IsTileLit(target))
	    || danger.explosiveBreakable)
		return 0;

	const int steps = FindPath(CanStep,
	    [&player, &forbidden, &encode, &inWindow](Point position) {
		    return inWindow(position)
		        && forbidden.count(encode(position.x, position.y)) == 0
		        && PosOkPlayer(player, position);
	    },
	    start, target, player.walkpath, MaxPathLengthPlayer);
	// The requested target is adjacent, so any answer other than exactly one
	// step means the engine found a detour through state outside the fixed
	// controller edge.  Fail closed instead of installing that hidden route.
	if (!IsExactControllerEdge(x - start.x, y - start.y, steps)) {
		ClrPlrPath(player);
		return 0;
	}
	return 1;
}

void ActAttackMonster(uint16_t monsterId)
{
	if (!CanAcceptPlayerAction("act_attack_monster"))
		return;
	if (monsterId >= MaxMonsters)
		throw std::out_of_range("monster_id 越界");
	NetSendCmdParam1(true, CMD_ATTACKID, monsterId);
}

int ActControllerAttackMonster(
    uint16_t monsterId, int centerX, int centerY, int radius)
{
	if (!CanAcceptPlayerAction("act_controller_attack_monster"))
		return 0;
	if (monsterId >= MaxMonsters)
		throw std::out_of_range("monster_id 越界");
	if (centerX < 0 || centerX >= MAXDUNX
	    || centerY < 0 || centerY >= MAXDUNY || radius < 0)
		return 0;
	bool active = false;
	for (size_t i = 0; i < ActiveMonsterCount; i++)
		active = active || ActiveMonsters[i] == monsterId;
	if (!active || Monsters[monsterId].hasNoLife())
		return 0;
	const Point playerPosition = MyPlayer->position.future;
	const Point monsterPosition = Monsters[monsterId].position.future;
	auto inWindow = [centerX, centerY, radius](Point position) {
		return std::abs(position.x - centerX) <= radius
		    && std::abs(position.y - centerY) <= radius;
	};
	// 只允许快照窗内的邻接挥刀。远目标必须由 Python 固定快照路径逐
	// 相邻步接近，严禁 CMD_ATTACKID 在这里偷偷安装一条全图追击路径。
	if (!inWindow(playerPosition) || !inWindow(monsterPosition)
	    || std::max(
	           std::abs(playerPosition.x - monsterPosition.x),
	           std::abs(playerPosition.y - monsterPosition.y))
	        > 1)
		return 0;
	NetSendCmdParam1(true, CMD_ATTACKID, monsterId);
	return 1;
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

int ActControllerOperate(int x, int y, int centerX, int centerY, int radius)
{
	if (!CanAcceptPlayerAction("act_controller_operate"))
		return 0;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY
	    || centerX < 0 || centerX >= MAXDUNX
	    || centerY < 0 || centerY >= MAXDUNY || radius < 0)
		return 0;
	const Point playerPosition = MyPlayer->position.future;
	auto inWindow = [centerX, centerY, radius](Point position) {
		return std::abs(position.x - centerX) <= radius
		    && std::abs(position.y - centerY) <= radius;
	};
	const Point target { x, y };
	const DynamicTileDanger danger = InspectDynamicTileDanger(target);
	// CMD_OPOBJXY normally installs a path automatically. Controller actions
	// may issue it only while still adjacent to the observed door; a knockback
	// before a stall reissue therefore fails closed instead of pathing unseen.
	if (!inWindow(playerPosition) || !inWindow(target)
	    || danger.explosiveBreakable
	    || std::max(
	           std::abs(playerPosition.x - target.x),
	           std::abs(playerPosition.y - target.y))
	        > 1)
		return 0;
	NetSendCmdLoc(MyPlayerId, true, CMD_OPOBJXY, target);
	return 1;
}

bool IsHealItem(const Item &item)
{
	// Healing 卷轴需要随后指定施法目标，不是“按下即回血”的腰带药。
	// 把它算进治疗药会让 act_drink 选中卷轴、留下十字光标，并把后续
	// 所有按键都短路成假成功。这里只承认同步生效且会被消耗的药水。
	return InstantHealKind(item) != 0;
}

int InstantHealKind(const Item &item)
{
	if (item.isEmpty())
		return 0;
	if (item._iMiscId == IMISC_HEAL)
		return 1;
	if (item._iMiscId == IMISC_FULLHEAL)
		return 2;
	if (item._iMiscId == IMISC_REJUV)
		return 3;
	if (item._iMiscId == IMISC_FULLREJUV)
		return 4;
	return 0;
}

bool IsLegacyHealItem(const Item &item)
{
	if (item.isEmpty())
		return false;
	// Exact protocol-v3 observation predicate.  This deliberately preserves
	// the historical Healing-scroll misclassification only for frozen-network
	// input reconstruction; v4 pickup/drink legality continues to use
	// IsHealItem above.
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

int CountLegacyBeltHeals()
{
	int heals = 0;
	for (int i = 0; i < MaxBeltItems; i++) {
		if (IsLegacyHealItem(MyPlayer->SpdList[i]))
			heals++;
	}
	return heals;
}

int CountBeltFreeSlots()
{
	int freeSlots = 0;
	for (int i = 0; i < MaxBeltItems; i++) {
		if (MyPlayer->SpdList[i].isEmpty())
			freeSlots++;
	}
	return freeSlots;
}

int ActDrink()
{
	if (!CanAcceptPlayerAction("act_drink"))
		return 0;
	// 只喝“按下即回血”的药水。上游 UseBeltItem(BeltItemType::Healing)
	// 也会匹配 Healing 卷轴；卷轴只会进入选目标状态，却曾被本桥误报为
	// 已喝药，随后 pcurs != CURSOR_HAND 又令所有动作静默短路。
	//
	// UseInvItem 的 bool 也只是“按键已处理”，在商店/聊天/非手形光标等
	// 情况仍可返回 true。因此回执必须以后置事实认证：HP 确实增加，或
	// 腰带即时治疗药数确实减少（auto-refill 可能改喝背包同类药，此时
	// 腰带数不变但 HP 会增加）。只有认证成功才返回请求前的药数。
	const int heals = CountBeltHeals();
	if (heals == 0 || (MyPlayer->_pHitPoints >> 6) >= (MyPlayer->_pMaxHP >> 6))
		return 0;
	int beltSlot = -1;
	for (int i = 0; i < MaxBeltItems; i++) {
		if (IsHealItem(MyPlayer->SpdList[i])) {
			beltSlot = i;
			break;
		}
	}
	if (beltSlot < 0)
		return 0; // CountBeltHeals 与选择谓词必须同源；纵深防御。

	const int hitPointsBefore = MyPlayer->_pHitPoints;
	UseInvItem(INVITEM_BELT_FIRST + beltSlot);
	const int hitPointsAfter = MyPlayer->_pHitPoints;
	const int healsAfter = CountBeltHeals();
	if (hitPointsAfter <= hitPointsBefore && healsAfter >= heals)
		return 0;
	return heals;
}

int ActPickup()
{
	if (!CanAcceptPlayerAction("act_pickup"))
		return 0;
	// Public convenience entry may consume only a heal under the exact future
	// tile.  Navigation belongs to the frozen-snapshot controller; allowing
	// this API to select a distant nearest item would bypass every path/hazard
	// proof even though ActPickupAt itself is fail-closed.
	if (CountBeltFreeSlots() == 0)
		return 0; // 腰带无空位:捡了会直落背包(喝药键与观测都看不见的价值黑洞),不发命令
	const Point me = MyPlayer->position.future;
	int best = -1;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int ii = ActiveItems[i];
		const Item &item = Items[ii];
		if (!IsHealItem(item) || !IsTileLit(item.position)
		    || item.position != me)
			continue;
		best = ii;
		break;
	}
	if (best < 0)
		return 0;
	NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, Items[best].position, static_cast<uint16_t>(best));
	return 1;
}

int ActPickupAt(
    int activeItemId, int x, int y, uint16_t seedHigh, uint16_t seedLow,
    uint16_t createInfo, int baseId)
{
	if (!CanAcceptPlayerAction("act_pickup_at"))
		return 0;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY
	    || CountBeltFreeSlots() == 0
	    || activeItemId < 0 || activeItemId >= MAXITEMS)
		return 0;
	bool active = false;
	for (int i = 0; i < ActiveItemCount; i++)
		active = active || ActiveItems[i] == activeItemId;
	const Item &item = Items[activeItemId];
	if (!active || item.position != Point { x, y }
	    || !MatchesItemIdentity(
	        item, seedHigh, seedLow, createInfo, baseId)
	    || !IsHealItem(item))
		return 0;
	const Point start = MyPlayer->position.future;
	// Snapshot controllers prove every preceding edge themselves.  The native
	// commit may consume only an item literally under the player's future
	// tile; CMD_GOTOAGETITEM must not get a final chance to re-path 1-2 tiles
	// through state that changed after the frozen snapshot.
	if (start != item.position)
		return 0;
	NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, item.position,
	    static_cast<uint16_t>(activeItemId));
	return 1;
}

struct GearUpgradePlan {
	bool valid = false;
	inv_body_loc target = INVLOC_HEAD;
	std::array<bool, NUM_INVLOC> clearSlots {};
	Item candidate {};
	uint32_t nextUtility = 0;
	int nextCurrentHitPoints = 0;
	int nextMaxHitPoints = 0;
};

struct PlayerResourceState {
	int hitPoints;
	int maxHitPoints;
	int hitPointsBase;
	int maxHitPointsBase;
	int mana;
	int maxMana;
	int manaBase;
	int maxManaBase;
};

PlayerResourceState CapturePlayerResourceState(const Player &player)
{
	return {
		player._pHitPoints,
		player._pMaxHP,
		player._pHPBase,
		player._pMaxHPBase,
		player._pMana,
		player._pMaxMana,
		player._pManaBase,
		player._pMaxManaBase,
	};
}

void RestoreGearUpgradeTransaction(
    Player &player, const std::array<Item, NUM_INVLOC> &body,
    const PlayerResourceState &resources)
{
	for (int slot = 0; slot < NUM_INVLOC; slot++)
		player.InvBody[slot] = body[slot];
	// A fatal candidate makes CalcPlrLifeMana call SetPlayerHitPoints(0),
	// which also rewrites HPBase.  Restoring only InvBody would therefore
	// leave a rejected transaction at a different life total.
	player._pHitPoints = resources.hitPoints;
	player._pMaxHP = resources.maxHitPoints;
	player._pHPBase = resources.hitPointsBase;
	player._pMaxHPBase = resources.maxHitPointsBase;
	player._pMana = resources.mana;
	player._pMaxMana = resources.maxMana;
	player._pManaBase = resources.manaBase;
	player._pMaxManaBase = resources.maxManaBase;
	CalcPlrInv(player, true);
	if (player._pHitPoints != resources.hitPoints
	    || player._pMaxHP != resources.maxHitPoints
	    || player._pHPBase != resources.hitPointsBase
	    || player._pMaxHPBase != resources.maxHitPointsBase
	    || player._pMana != resources.mana
	    || player._pMaxMana != resources.maxMana
	    || player._pManaBase != resources.manaBase
	    || player._pMaxManaBase != resources.maxManaBase)
		throw std::runtime_error(
		    "action14 事务回滚未逐位恢复生命/法力状态");
}

bool CanPairOneHanded(
    const Player &player, const Item &candidate, const Item &other)
{
	if (other.isEmpty())
		return true;
	if (player.GetItemLocation(candidate) != ILOC_ONEHAND
	    || player.GetItemLocation(other) != ILOC_ONEHAND)
		return false;
	if (candidate._iClass != other._iClass)
		return true; // the normal weapon + shield pairing
	const ClassAttributes &attributes = GetClassAttributes(player._pClass);
	return HasAnyOf(attributes.classFlags, PlayerClassFlag::DualWield)
	    && candidate.isWeapon() && other.isWeapon()
	    && IsAnyOf(candidate._itype, ItemType::Sword, ItemType::Mace)
	    && IsAnyOf(other._itype, ItemType::Sword, ItemType::Mace);
}

void ConsiderGearUpgradePlan(
    const Player &player, const Item &candidate,
    inv_body_loc target, std::array<bool, NUM_INVLOC> clearSlots,
    const GearCombatProfile &emptyBaseline,
    const GearCombatProfile &previousProfile, GearUpgradePlan &best)
{
	std::array<Item, NUM_INVLOC> body;
	for (int slot = 0; slot < NUM_INVLOC; slot++)
		body[slot] = player.InvBody[slot];
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		if (clearSlots[slot])
			body[slot].clear();
	}
	body[target] = candidate;
	GearCombatProfile nextProfile
	    = SimulateGearCombatProfile(player, body);
	MakeGearUtilityRelativeToEmpty(nextProfile, emptyBaseline);
	if (!body[target]._iStatFlag
	    || !IsConservativeGearUpgrade(previousProfile, nextProfile))
		return;
	// Highest actual aggregate utility wins.  Equal candidates retain the
	// first enumerated slot, giving rings/hands a stable deterministic tie.
	if (best.valid && nextProfile.utility <= best.nextUtility)
		return;
	best.valid = true;
	best.target = target;
	best.clearSlots = clearSlots;
	best.candidate = candidate;
	best.nextUtility = nextProfile.utility;
	best.nextCurrentHitPoints = nextProfile.currentHitPoints;
	best.nextMaxHitPoints = nextProfile.maxHitPoints;
}

GearUpgradePlan PlanGearUpgrade(const Player &player, const Item &item)
{
	GearUpgradePlan best;
	if (item.isEmpty() || !item.isEquipment()
	    || item._iClass == ICLASS_QUEST || item.IDidx == IDI_LAZSTAFF)
		return best;
	Item candidate = PrepareGearCandidate(item, player);
	if (!candidate._iStatFlag)
		return best;
	const GearCombatProfile emptyBaseline
	    = EmptyGearCombatBaseline(player);
	GearCombatProfile previousProfile
	    = GearCombatProfileFromPlayer(player);
	MakeGearUtilityRelativeToEmpty(previousProfile, emptyBaseline);

	auto singleSlot = [&](inv_body_loc slot) {
		std::array<bool, NUM_INVLOC> clearSlots {};
		clearSlots[slot] = true;
		ConsiderGearUpgradePlan(
		    player, candidate, slot, clearSlots, emptyBaseline,
		    previousProfile, best);
	};
	switch (candidate._iLoc) {
	case ILOC_ARMOR:
		singleSlot(INVLOC_CHEST);
		break;
	case ILOC_HELM:
		singleSlot(INVLOC_HEAD);
		break;
	case ILOC_RING:
		singleSlot(INVLOC_RING_LEFT);
		singleSlot(INVLOC_RING_RIGHT);
		break;
	case ILOC_AMULET:
		singleSlot(INVLOC_AMULET);
		break;
	case ILOC_ONEHAND:
	case ILOC_TWOHAND: {
		const item_equip_type location = player.GetItemLocation(candidate);
		if (location == ILOC_TWOHAND) {
			std::array<bool, NUM_INVLOC> clearSlots {};
			clearSlots[INVLOC_HAND_LEFT] = true;
			clearSlots[INVLOC_HAND_RIGHT] = true;
			ConsiderGearUpgradePlan(
			    player, candidate, INVLOC_HAND_LEFT, clearSlots,
			    emptyBaseline, previousProfile, best);
			break;
		}
		if (location != ILOC_ONEHAND
		    || (!candidate.isWeapon()
		        && candidate._itype != ItemType::Shield))
			break;

		const bool leftTwoHanded
		    = !player.InvBody[INVLOC_HAND_LEFT].isEmpty()
		    && player.GetItemLocation(
		           player.InvBody[INVLOC_HAND_LEFT])
		        == ILOC_TWOHAND;
		const bool rightTwoHanded
		    = !player.InvBody[INVLOC_HAND_RIGHT].isEmpty()
		    && player.GetItemLocation(
		           player.InvBody[INVLOC_HAND_RIGHT])
		        == ILOC_TWOHAND;
		if (leftTwoHanded || rightTwoHanded) {
			std::array<bool, NUM_INVLOC> clearSlots {};
			clearSlots[INVLOC_HAND_LEFT] = true;
			clearSlots[INVLOC_HAND_RIGHT] = true;
			const inv_body_loc target = rightTwoHanded
			    ? INVLOC_HAND_RIGHT
			    : INVLOC_HAND_LEFT;
			ConsiderGearUpgradePlan(
			    player, candidate, target, clearSlots,
			    emptyBaseline, previousProfile, best);
			break;
		}

		const std::array<inv_body_loc, 2> order = candidate.isWeapon()
		    ? std::array<inv_body_loc, 2> {
		        INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT }
		    : std::array<inv_body_loc, 2> {
		        INVLOC_HAND_RIGHT, INVLOC_HAND_LEFT };
		for (const inv_body_loc target : order) {
			const inv_body_loc other = target == INVLOC_HAND_LEFT
			    ? INVLOC_HAND_RIGHT
			    : INVLOC_HAND_LEFT;
			if (!CanPairOneHanded(
			        player, candidate, player.InvBody[other]))
				continue;
			singleSlot(target);
		}
		break;
	}
	default:
		break;
	}
	return best;
}

bool IsWantedGear(const Item &item)
{
	// Pure read: plan and all dependency/cascade checks operate on copies.
	// A target is exposed only when the post-identification, post-replacement
	// aggregate utility is strictly greater than the currently equipped set.
	return MyPlayer != nullptr && PlanGearUpgrade(*MyPlayer, item).valid;
}

int ActPickupGear()
{
	if (!CanAcceptPlayerAction("act_pickup_gear"))
		return 0;
	// Public convenience entry is intentionally exact-tile too; navigation is
	// the Python controller's frozen-snapshot responsibility.  This prevents a
	// caller from bypassing ActPickupGearAt by selecting a nearby floor item.
	const Point me = MyPlayer->position.future;
	int best = -1;
	for (int i = 0; i < ActiveItemCount; i++) {
		const int ii = ActiveItems[i];
		if (!IsWantedGear(Items[ii])
		    || !IsTileLit(Items[ii].position)
		    || Items[ii].position != me)
			continue;
		best = ii;
		break;
	}
	if (best < 0)
		return 0;
	const Item &item = Items[best];
	return ActPickupGearAt(
	    best, item.position.x, item.position.y, HighWord(item._iSeed),
	    LowWord(item._iSeed), item._iCreateInfo, static_cast<int>(item.IDidx));
}

int ActPickupGearAt(
    int activeItemId, int x, int y, uint16_t seedHigh, uint16_t seedLow,
    uint16_t expectedCreateInfo, int expectedBaseId)
{
	if (!CanAcceptPlayerAction("act_pickup_gear_at"))
		return 0;
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY
	    || activeItemId < 0 || activeItemId >= MAXITEMS)
		return 0;
	bool active = false;
	for (int i = 0; i < ActiveItemCount; i++)
		active = active || ActiveItems[i] == activeItemId;
	Item &item = Items[activeItemId];
	if (!active || item.position != Point { x, y }
	    || !MatchesItemIdentity(
	        item, seedHigh, seedLow, expectedCreateInfo, expectedBaseId))
		return 0;
	Player &player = *MyPlayer;
	const Point start = player.position.future;
	// The Python controller has already proven and executed every path edge.
	// Immediate equipment must consume only the item under the exact future
	// tile, never use adjacency as a final unobserved teleport/re-path.
	if (start != item.position)
		return 0;
	const GearUpgradePlan plan = PlanGearUpgrade(player, item);
	if (!plan.valid)
		return 0;

	// 不走 AutoGetItem 的 “AutoEquip 失败→腰带→隐藏背包”回退链。
	// 复制全套身体槽后原子替换；旧装备按训练协议明确销毁，不进背包、
	// 不落到可能被下一次宏误捞的地面。CalcPlrInv 是最终权威校验；
	// 任一属性依赖级联使 aggregate utility 不再严格增长就完整回滚。
	const Point position = item.position;
	const uint32_t seed = item._iSeed;
	const _item_indexes baseId = item.IDidx;
	const uint16_t createInfo = item._iCreateInfo;
	std::array<Item, NUM_INVLOC> previousBody;
	for (int slot = 0; slot < NUM_INVLOC; slot++)
		previousBody[slot] = player.InvBody[slot];
	const PlayerResourceState previousResources
	    = CapturePlayerResourceState(player);
	const GearCombatProfile emptyBaseline
	    = EmptyGearCombatBaseline(player);
	GearCombatProfile previousProfile
	    = GearCombatProfileFromPlayer(player);
	MakeGearUtilityRelativeToEmpty(previousProfile, emptyBaseline);
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		if (plan.clearSlots[slot])
			player.InvBody[slot].clear();
	}
	Item equipped = plan.candidate;
	equipped._iCreateInfo &= ~CF_PREGEN;
	player.InvBody[plan.target] = equipped;
	CalcPlrInv(player, true);
	GearCombatProfile nextProfile
	    = GearCombatProfileFromPlayer(player);
	MakeGearUtilityRelativeToEmpty(nextProfile, emptyBaseline);
	if (nextProfile.utility != plan.nextUtility
	    || nextProfile.currentHitPoints != plan.nextCurrentHitPoints
	    || nextProfile.maxHitPoints != plan.nextMaxHitPoints) {
		RestoreGearUpgradeTransaction(
		    player, previousBody, previousResources);
		throw std::runtime_error(
		    "action14 模拟规划与原子提交的效用/当前及最大生命不一致");
	}
	if (player.hasNoLife()
	    || (nextProfile.currentHitPoints >> 6) <= 0) {
		RestoreGearUpgradeTransaction(
		    player, previousBody, previousResources);
		throw std::runtime_error(
		    "action14 规划安全门失效：原子提交会使玩家死亡");
	}
	if (!player.InvBody[plan.target]._iStatFlag
	    || !IsConservativeGearUpgrade(previousProfile, nextProfile)) {
		RestoreGearUpgradeTransaction(
		    player, previousBody, previousResources);
		return 0;
	}

	// Publish the floor-item removal before the equipment delta, matching the
	// engine's normal OnRequestGetItem ordering.  The looped-back CMD_GETITEM
	// cannot equip the item a second time: bMaster is the local player, so
	// OnGetItem only updates the level delta and returns.  We remove the exact
	// local item ourselves after the commit has passed its final validation.
	NetSendCmdGItem(
	    false, CMD_GETITEM, player, static_cast<uint8_t>(activeItemId));
	SyncGetItem(position, seed, baseId, createInfo);
	SetItemRecord(seed, createInfo, static_cast<int>(baseId));

	// Keep remote inventory mirrors coherent.  CHANGE carries the identified
	// replacement; explicit DEL messages cover any additional hand cleared by
	// a two-hand transition.  The local player has already committed atomically.
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		if (slot != plan.target
		    && plan.clearSlots[slot]
		    && !previousBody[slot].isEmpty())
			NetSendCmdDelItem(false, static_cast<uint8_t>(slot));
	}
	NetSendCmdChItem(false, static_cast<uint8_t>(plan.target), true);
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
		if (MyPlayer->position.future != item.position)
			return 0;
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
	m.def("act_wait", &ActWait,
	    "取消遗留寻路/目标动作并原地等待；攻击立即中止，已提交的单格移动可自然收尾，返回 0/1");
	m.def("act_walk", &ActWalk, py::arg("x"), py::arg("y"), "寻路走向目标格(网络命令层注入)");
	m.def("act_explore_walk", &ActExploreWalk,
	    py::arg("x"), py::arg("y"), py::arg("protected_tiles"),
	    py::arg("center_x"), py::arg("center_y"), py::arg("radius"),
	    "控制器专用相邻步：只在固定快照窗内安装不穿受保护格的一步路径，返回 0/1");
	m.def("act_attack_monster", &ActAttackMonster, py::arg("monster_id"), "追击并近战指定怪物");
	m.def("act_controller_attack_monster", &ActControllerAttackMonster,
	    py::arg("monster_id"), py::arg("center_x"),
	    py::arg("center_y"), py::arg("radius"),
	    "固定控制快照专用：只对窗内相邻怪物挥刀，拒绝原生远程追击");
	m.def("act_attack_tile", &ActAttackTile, py::arg("x"), py::arg("y"), "原地朝目标格挥击");
	m.def("act_operate", &ActOperate, py::arg("x"), py::arg("y"), "操作目标格物体(开门等;引擎自动走近)");
	m.def("act_controller_operate", &ActControllerOperate,
	    py::arg("x"), py::arg("y"), py::arg("center_x"),
	    py::arg("center_y"), py::arg("radius"),
	    "固定控制快照专用：只操作窗内相邻物体，拒绝原生远程寻路");
	m.def("act_drink", &ActDrink, "喝腰带上的第一瓶即时治疗药;仅在确认回血/消耗后返回按键前药数，否则返回 0");
	m.def("act_pickup", &ActPickup,
	    "只拾取玩家 future 同格的首瓶治疗药；不负责寻路，返回 0/1");
	m.def("act_pickup_at", &ActPickupAt,
	    py::arg("active_item_id"), py::arg("x"), py::arg("y"),
	    py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"),
	    py::arg("base_id"),
	    "只在玩家 future 精确同格时拾取固定快照逐字匹配身份的即时治疗药，返回 0/1");
	m.def("act_pickup_gear", &ActPickupGear,
	    "只原子装备玩家 future 同格的首件安全升级；不负责寻路，返回 0/1");
	m.def("act_pickup_gear_at", &ActPickupGearAt,
	    py::arg("active_item_id"), py::arg("x"), py::arg("y"),
	    py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"),
	    py::arg("base_id"),
	    "只在玩家 future 精确同格时原子装备固定快照逐字匹配身份的物品，返回 0/1");
	m.def("act_pickup_progression", &ActPickupProgression, py::arg("x"), py::arg("y"),
	    "仅在玩家 future 精确同格时拾取指定 Staff of Lazarus；供 action 10/11 使用");
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
	m.def("probe_inventory_item_count", []() {
		EnsureInGame("probe_inventory_item_count");
		return static_cast<int>(MyPlayer->_pNumInv);
	}, "探针:读背包物品数，验证原子换装不产生隐藏背包回退");
	m.def("probe_kill_monster", [](unsigned monsterId) {
		EnsureInGame("probe_kill_monster");
		if (monsterId >= MaxMonsters)
			throw std::out_of_range("probe_kill_monster 怪物 id 越界");
		bool active = false;
		for (size_t i = 0; i < ActiveMonsterCount; i++) {
			if (ActiveMonsters[i] == monsterId) {
				active = true;
				break;
			}
		}
		Monster &monster = Monsters[monsterId];
		if (!active || monster.hasNoLife())
			throw std::invalid_argument(
			    "probe_kill_monster 只接受当前活动且存活的怪物");
		const uint64_t before = MonsterKillTotal();
		const int monsterType = static_cast<int>(monster.type().type);
		M_StartKill(monster, *MyPlayer);
		const uint64_t after = MonsterKillTotal();
		if (before == std::numeric_limits<uint64_t>::max()
		    || after != before + 1)
			throw std::logic_error(
			    "原生 MonsterDeath 未精确递增 monster_kill_total");
		py::dict result;
		result["monster_type"] = monsterType;
		result["before"] = before;
		result["after"] = after;
		return result;
	}, py::arg("monster_id"),
	    "探针:经原生 M_StartKill 杀死活动怪物并核验累计击杀事实恰好 +1");
	m.def("probe_gear_combat_profile", []() {
		EnsureInGame("probe_gear_combat_profile");
		const GearCombatProfile profile
		    = LoadoutGearCombatProfile(*MyPlayer);
		py::dict result;
		result["utility"] = profile.utility;
		result["effect_flags"] = profile.effectFlags;
		result["dam_ac_flags"] = profile.damAcFlags;
		result["attack_speed_tier"] = profile.attackSpeedTier;
		result["attack_cycle_frames"] = profile.attackCycleFrames;
		result["attack_impact_frames"] = profile.attackImpactFrames;
		result["physical_min"] = profile.physicalMin;
		result["physical_max"] = profile.physicalMax;
		result["animal_min"] = profile.animalMin;
		result["animal_max"] = profile.animalMax;
		result["undead_min"] = profile.undeadMin;
		result["undead_max"] = profile.undeadMax;
		result["demon_min"] = profile.demonMin;
		result["demon_max"] = profile.demonMax;
		result["melee_to_hit"] = profile.meleeToHit;
		result["melee_piercing_to_hit"] = profile.meleePiercingToHit;
		result["magic_to_hit"] = profile.magicToHit;
		result["fire_min"] = profile.fireMin;
		result["fire_max"] = profile.fireMax;
		result["lightning_min"] = profile.lightningMin;
		result["lightning_max"] = profile.lightningMax;
		result["armor"] = profile.armor;
		result["block_enabled"] = profile.blockEnabled;
		result["block_chance"] = profile.blockChance;
		result["magic_resist"] = profile.magicResistance;
		result["fire_resist"] = profile.fireResistance;
		result["lightning_resist"] = profile.lightningResistance;
		result["light_radius"] = profile.lightRadius;
		result["current_hp_fixed"] = profile.currentHitPoints;
		result["max_hp_fixed"] = profile.maxHitPoints;
		result["max_mana_fixed"] = profile.maxMana;
		result["magic"] = profile.magic;
		result["get_hit"] = profile.getHit;
		result["hit_recovery_tier"] = profile.hitRecoveryTier;
		result["life_steal_tier"] = profile.lifeStealTier;
		result["mana_steal_tier"] = profile.manaStealTier;
		result["spell_level_bonus"] = profile.spellLevelBonus;
		return result;
	}, "探针:读取整套装备经 CalcPlrItemVals 后的保守战斗评分/硬门事实");
	m.def("probe_set_current_hit_points", [](int hitPoints) {
		EnsureInGame("probe_set_current_hit_points");
		const int maxHitPoints = MyPlayer->_pMaxHP >> 6;
		if (hitPoints <= 0 || hitPoints > maxHitPoints)
			throw std::invalid_argument(
			    "probe_set_current_hit_points 必须在 [1,max_hp] 内");
		SetPlayerHitPoints(*MyPlayer, hitPoints << 6);
		return MyPlayer->_pHitPoints;
	}, py::arg("hit_points"),
	    "探针:设置玩家当前整点生命，供换装 projected-HP 安全门回归");
	m.def("probe_spawn_test_gear", [](int baseId, int minDamage,
	                                      int maxDamage, int armorClass,
	                                      int magicDamageBonus,
	                                      uint32_t effectFlags,
	                                      int fireResistance,
	                                      int lightningResistance,
	                                      int magicResistance,
	                                      int toHitBonus,
	                                      int getHitPenalty,
	                                      int lifeBonusPoints,
	                                      int vitalityBonus,
	                                      int magicBonus,
	                                      int manaBonusPoints,
	                                      int spellLevelBonus,
	                                      int lightBonus,
	                                      int damAcFlags,
	                                      int durability,
	                                      int maxDurability,
	                                      int fireMinDamage,
	                                      int fireMaxDamage,
	                                      int lightningMinDamage,
	                                      int lightningMaxDamage) {
		EnsureInGame("probe_spawn_test_gear");
		if (!IsItemAvailable(baseId))
			throw std::invalid_argument("probe_spawn_test_gear base_id 不可用");
		if (minDamage < 0 || minDamage > 255
		    || maxDamage < minDamage || maxDamage > 255
		    || armorClass < 0 || armorClass > 32767
		    || magicDamageBonus < -32768 || magicDamageBonus > 32767
		    || fireResistance < -32768 || fireResistance > 32767
			    || lightningResistance < -32768
			    || lightningResistance > 32767
			    || magicResistance < -32768 || magicResistance > 32767
			    || toHitBonus < -32768 || toHitBonus > 32767
			    || getHitPenalty < -32768 || getHitPenalty > 32767
			    || lifeBonusPoints < -511 || lifeBonusPoints > 511
			    || vitalityBonus < -32768 || vitalityBonus > 32767
			    || magicBonus < -32768 || magicBonus > 32767
			    || manaBonusPoints < -511 || manaBonusPoints > 511
			    || spellLevelBonus < -128 || spellLevelBonus > 127
			    || lightBonus < -32768 || lightBonus > 32767
			    || damAcFlags < 0 || damAcFlags > 255
			    || durability < -1 || durability > DUR_INDESTRUCTIBLE
			    || maxDurability < -1
			    || maxDurability > DUR_INDESTRUCTIBLE
			    || fireMinDamage < 0 || fireMinDamage > 32767
			    || fireMaxDamage < fireMinDamage
			    || fireMaxDamage > 32767
			    || lightningMinDamage < 0
			    || lightningMinDamage > 32767
			    || lightningMaxDamage < lightningMinDamage
			    || lightningMaxDamage > 32767
			    || ((durability == -1) != (maxDurability == -1))
			    || (durability != -1
			        && (durability > maxDurability
			            || ((durability == DUR_INDESTRUCTIBLE)
			                != (maxDurability == DUR_INDESTRUCTIBLE)))))
				throw std::invalid_argument("probe_spawn_test_gear 数值越界");
		Item item;
		InitializeItem(item, static_cast<_item_indexes>(baseId));
		if (!item.isEquipment() && baseId != IDI_LAZSTAFF)
			throw std::invalid_argument("probe_spawn_test_gear 只接受装备");
		GenerateNewSeed(item);
		item._iMinDam = static_cast<uint8_t>(minDamage);
		item._iMaxDam = static_cast<uint8_t>(maxDamage);
		item._iAC = static_cast<int16_t>(armorClass);
		if (durability != -1) {
			item._iDurability = durability;
			item._iMaxDur = maxDurability;
		}
		item._iFMinDam = static_cast<int16_t>(fireMinDamage);
		item._iFMaxDam = static_cast<int16_t>(fireMaxDamage);
		item._iLMinDam = static_cast<int16_t>(lightningMinDamage);
		item._iLMaxDam = static_cast<int16_t>(lightningMaxDamage);
		if (magicDamageBonus != 0 || effectFlags != 0
			    || fireResistance != 0 || lightningResistance != 0
			    || magicResistance != 0 || toHitBonus != 0
			    || getHitPenalty != 0 || lifeBonusPoints != 0
			    || vitalityBonus != 0 || magicBonus != 0
			    || manaBonusPoints != 0 || spellLevelBonus != 0
			    || lightBonus != 0 || damAcFlags != 0
			    || fireMaxDamage != 0 || lightningMaxDamage != 0) {
			item._iMagical = ITEM_QUALITY_MAGIC;
			item._iIdentified = false;
			item._iPLDam = static_cast<int16_t>(magicDamageBonus);
			item._iFlags = static_cast<ItemSpecialEffect>(effectFlags);
			if (fireMaxDamage != 0) {
				item._iFlags = static_cast<ItemSpecialEffect>(
				    static_cast<uint32_t>(item._iFlags)
				    | EffectBits(ItemSpecialEffect::FireDamage));
			}
			if (lightningMaxDamage != 0) {
				item._iFlags = static_cast<ItemSpecialEffect>(
				    static_cast<uint32_t>(item._iFlags)
				    | EffectBits(ItemSpecialEffect::LightningDamage));
			}
			item._iPLFR = static_cast<int16_t>(fireResistance);
			item._iPLLR = static_cast<int16_t>(lightningResistance);
				item._iPLMR = static_cast<int16_t>(magicResistance);
				item._iPLToHit = static_cast<int16_t>(toHitBonus);
			item._iPLGetHit = static_cast<int16_t>(getHitPenalty);
			item._iPLHP = static_cast<int16_t>(
			    lifeBonusPoints * 64);
			item._iPLVit = static_cast<int16_t>(vitalityBonus);
			item._iPLMag = static_cast<int16_t>(magicBonus);
			item._iPLMana = static_cast<int16_t>(
			    manaBonusPoints * 64);
			item._iSplLvlAdd = static_cast<int8_t>(spellLevelBonus);
			item._iPLLight = static_cast<int16_t>(lightBonus);
			item._iDamAcFlags
			    = static_cast<ItemSpecialEffectHf>(damAcFlags);
		}
		item.updateRequiredStatsCacheForPlayer(*MyPlayer);
		const std::optional<Point> position = FindClosestValidPosition(
		    ItemSpaceOk, MyPlayer->position.tile, 1, 2);
		if (!position.has_value())
			throw std::runtime_error("probe_spawn_test_gear 玩家两格内无地面空位");
			const int activeItemId = PlaceItemInWorld(
			    std::move(item), *position);
			const Item &placed = Items[activeItemId];
			py::dict result;
			result["active_id"] = activeItemId;
			result["x"] = static_cast<int>(position->x);
			result["y"] = static_cast<int>(position->y);
			result["seed_hi"] = HighWord(placed._iSeed);
			result["seed_lo"] = LowWord(placed._iSeed);
			result["create_info"] = placed._iCreateInfo;
			result["base_id"] = static_cast<int>(placed.IDidx);
			return result;
	}, py::arg("base_id"), py::arg("min_damage"),
	    py::arg("max_damage"), py::arg("armor_class") = 0,
	    py::arg("magic_damage_bonus") = 0,
	    py::arg("effect_flags") = 0,
	    py::arg("fire_resistance") = 0,
		    py::arg("lightning_resistance") = 0,
		    py::arg("magic_resistance") = 0,
		    py::arg("to_hit_bonus") = 0,
	    py::arg("get_hit_penalty") = 0,
	    py::arg("life_bonus_points") = 0,
	    py::arg("vitality_bonus") = 0,
	    py::arg("magic_bonus") = 0,
	    py::arg("mana_bonus_points") = 0,
	    py::arg("spell_level_bonus") = 0,
	    py::arg("light_bonus") = 0,
	    py::arg("dam_ac_flags") = 0,
	    py::arg("durability") = -1,
	    py::arg("max_durability") = -1,
	    py::arg("fire_min_damage") = 0,
	    py::arg("fire_max_damage") = 0,
	    py::arg("lightning_min_damage") = 0,
	    py::arg("lightning_max_damage") = 0,
	    "探针:在玩家两格内生成确定装备，供整套 OR/cap/攻速/诅咒/耐久回归");

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
		py::list walkable, monster, door, closedDoorOnly;
		py::list hazard, explosiveSoftwall;
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
					DynamicTileDanger danger;
					if (inBounds) {
						danger = InspectDynamicTileDanger({ x, y });
						if (danger.object != nullptr) {
							closedDoor = danger.object->isDoor()
							    && !IsTileWalkable({ x, y }, false);
							barrel = danger.object->IsBreakable()
							    && danger.object->_oSolidFlag;
						}
					}
					door.append((closedDoor || barrel) ? 1 : 0);
				// action10 可在仍有普通边疆时顺路优先开门，但不应把房内
				// 每只桶都当高优先级出口；保留 door=门或桶的旧规划口径，
					// 另给探索一个只含可交互闭门的精确通道。
					closedDoorOnly.append(
					    (closedDoor && danger.object != nullptr
					        && danger.object->canInteractWith())
					        ? 1
					        : 0);
					// 只公开玩家已照亮的活动伤害相位；未发现机关仍明确属于
					// 部分可观测环境。控制器将该格当硬墙，避免自动踩火。
					hazard.append(
					    (inBounds && danger.damagingHazard
					        && IsTileLit({ x, y }))
					        ? 1
					        : 0);
					explosiveSoftwall.append(
					    (barrel && danger.explosiveBreakable) ? 1 : 0);
			}
		}
		py::dict d;
		d["walkable"] = walkable;
		d["monster"] = monster;
			d["door"] = door;
			d["closed_door"] = closedDoorOnly;
			d["hazard"] = hazard;
			d["explosive_softwall"] = explosiveSoftwall;
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
		const DynamicTileDanger danger
		    = InspectDynamicTileDanger({ x, y });
		d["object_type"] = object != nullptr ? static_cast<int>(object->_otype) : -1;
		d["object_is_door"] = object != nullptr && object->isDoor();
		d["object_solid"] = object != nullptr && object->_oSolidFlag;
		d["object_selectable"] = object != nullptr && object->canInteractWith();
		d["lit"] = IsTileLit({ x, y });
		d["damaging_hazard"] = danger.damagingHazard;
		d["explosive_breakable"] = danger.explosiveBreakable;
		return d;
	}, py::arg("x"), py::arg("y"),
	    "调试:读取单格占位/碰撞/物体及动态伤害/爆炸事实");

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
	m.attr("PM_STAND") = static_cast<int>(PM_STAND);
	m.attr("PM_NEWLVL") = static_cast<int>(PM_NEWLVL);
	m.attr("PM_WALK_NORTHWARDS") = static_cast<int>(PM_WALK_NORTHWARDS);
	m.attr("PM_WALK_SOUTHWARDS") = static_cast<int>(PM_WALK_SOUTHWARDS);
	m.attr("PM_WALK_SIDEWAYS") = static_cast<int>(PM_WALK_SIDEWAYS);
	m.attr("PM_ATTACK") = static_cast<int>(PM_ATTACK);
	m.attr("ACTION_NONE") = static_cast<int>(ACTION_NONE);
	m.attr("ACTION_ATTACKMON") = static_cast<int>(ACTION_ATTACKMON);
	m.attr("WALK_NONE") = static_cast<int>(WALK_NONE);
	m.attr("IDI_CLEAVER") = static_cast<int>(IDI_CLEAVER);
	m.attr("IDI_LAZSTAFF") = static_cast<int>(IDI_LAZSTAFF);
	m.attr("IDI_WARRSHLD") = static_cast<int>(IDI_WARRSHLD);
	m.attr("IDI_WARRCLUB") = static_cast<int>(IDI_WARRCLUB);
	m.attr("IDI_INFRARING") = static_cast<int>(IDI_INFRARING);
	m.attr("DUR_INDESTRUCTIBLE") = static_cast<int>(DUR_INDESTRUCTIBLE);
	m.attr("ITEM_EFFECT_DRAIN_LIFE")
	    = EffectBits(ItemSpecialEffect::DrainLife);
	m.attr("ITEM_EFFECT_RANDOM_ARROW_VELOCITY")
	    = EffectBits(ItemSpecialEffect::RandomArrowVelocity);
	m.attr("ITEM_EFFECT_FIRE_ARROWS")
	    = EffectBits(ItemSpecialEffect::FireArrows);
	m.attr("ITEM_EFFECT_MULTIPLE_ARROWS")
	    = EffectBits(ItemSpecialEffect::MultipleArrows);
	m.attr("ITEM_EFFECT_KNOCKBACK")
	    = EffectBits(ItemSpecialEffect::Knockback);
	m.attr("ITEM_EFFECT_STEAL_MANA5")
	    = EffectBits(ItemSpecialEffect::StealMana5);
	m.attr("ITEM_EFFECT_LIGHTNING_ARROWS")
	    = EffectBits(ItemSpecialEffect::LightningArrows);
	m.attr("ITEM_EFFECT_NO_MANA")
	    = EffectBits(ItemSpecialEffect::NoMana);
	m.attr("ITEM_EFFECT_ZERO_RESISTANCE")
	    = EffectBits(ItemSpecialEffect::ZeroResistance);
	m.attr("ITEM_EFFECT_QUICK_ATTACK")
	    = EffectBits(ItemSpecialEffect::QuickAttack);
	m.attr("ITEM_EFFECT_FASTEST_HIT_RECOVERY")
	    = EffectBits(ItemSpecialEffect::FastestHitRecovery);
	m.attr("ITEM_EFFECT_STEAL_LIFE3")
	    = EffectBits(ItemSpecialEffect::StealLife3);
	m.attr("ITEM_EFFECT_STEAL_LIFE5")
	    = EffectBits(ItemSpecialEffect::StealLife5);
	m.attr("ITEM_EFFECT_RANDOM_STEAL_LIFE")
	    = EffectBits(ItemSpecialEffect::RandomStealLife);
	m.attr("ITEM_EFFECT_FASTEST_ATTACK")
	    = EffectBits(ItemSpecialEffect::FastestAttack);
	m.attr("ITEM_DAM_AC_DECAY")
	    = DamAcEffectBits(ItemSpecialEffectHf::Decay);
	m.attr("ITEM_DAM_AC_PERIL")
	    = DamAcEffectBits(ItemSpecialEffectHf::Peril);
	m.attr("ITEM_DAM_AC_DOPPELGANGER")
	    = DamAcEffectBits(ItemSpecialEffectHf::Doppelganger);
}
