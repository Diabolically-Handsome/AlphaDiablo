/**
 * @file diablogym.cpp
 *
 * DiabloGym v0: headless DevilutionX embedding bridge (pybind11).
 *
 * Embedded the same way as upstream test/timedemo_test.cpp: HeadlessMode + loopback single player,
 * with Python driving the main loop tick by tick (a copy of the RunGameLoop body without wall-clock throttling or drawing);
 * actions go through the network command layer (NetSendCmd*), the same path as the multiplayer protocol, so online deployment is naturally possible later.
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
#include "preparation_goal.hpp"
#include "quest_entrance_visibility.hpp"

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
#include "inv.h"     // v14: AutoEquip (backpack salvage; fixes the PM_GOTHIT timing window)
#include "options.h" // v14: auto-equip options (armor/helmet/jewelry off by default)
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
#include "objects.h" // FindObjectAtPosition / isDoor (door awareness for the descend macro)
#include "options.h"
#include "pfile.h"
#include "player.h"
#include "panels/info_box.hpp"
#include "portal.h"
#include "qol/chatlog.h"
#include "qol/stash.h"
#include "quests.h"
#include "spells.h" // R18-F portal-v1: IsValidSpellFrom for act_cast_town_portal
#include "tables/monstdat.h"
#include "tables/playerdat.hpp"
#include "stores.h"
#include "towners.h"
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
bool gManualControl = false;
uint64_t gManualTicks = 0;
bool ManualTransitionGuard(const Player &, interface_mode, int);
// Explicit, between-episode experiment switch. The legacy protocol stays off.
bool gGearWearScoring = false;
bool gGearHitScoring = false;
bool gPreparationEquipment = false;
bool gQuestEntranceObservation = false;
bool gStartupTick = true; // corresponds to gbGameLoopStartup in RunGameLoop
int gHeroClass = 0;       // HeroClass::Warrior
int gStallPrints = 0;     // print budget for logic-stall diagnostics
std::string gAssetsDir;
std::string gSaveDir;
std::string gDataDir;
uint64_t gEpisodeGeneration = 0;
bool gLuaInitialized = false;
bool gExitCleanupRegistered = false;
int64_t gEnginePid = -1;
bool gMonotonicQuestTurnInUsed = false;
bool gResourceProtocol = false;
bool gResourceLootEconomy = false;
Player ResourceUnequipSimulation(const Player &source);
void RecordLootGearRetention(const Player &, const std::array<Item, NUM_INVLOC> &,
    const std::array<bool, NUM_INVLOC> &, int);
bool EquipmentReadyForPreservation(const Player &player);
py::dict ObserveResourceState();
void ResetResourceEpisode(uint32_t episodeSeed);
void SaveLevelForTransition();
void ResourceAfterLoad();

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
		    + ": reusing a DevilutionX instance initialized by the parent process in a fork child is forbidden; use spawn");
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
		throw std::runtime_error(std::string(operation) + ": call reset() first");
}

bool CanAcceptPlayerAction(const char *operation)
{
	EnsureInGame(operation);
	// Step normally settles an end-of-tick level change before returning to Python; this is still defense in depth, covering
	// the PM_NEWLVL window left by a probe queuing directly, an exception, or a future new entry point.
	// If an action were pushed onto the network command queue now, SyncLoad would switch to the new map first, and then
	// ProcessGameMessagePackets would apply the old scene's command to the new scene.
	return MyPlayer->_pmode != PM_NEWLVL && !MyPlayer->_pLvlChanging;
}

int ConceptualDungeonDepth()
{
	if (!setlevel)
		return static_cast<int>(currlevel);
	const int returnLevel = GetMapReturnLevel();
	if (returnLevel <= 0)
		throw std::runtime_error(
		    "DiabloGym hit a set-level that cannot be mapped to a main dungeon depth: "
		    + std::to_string(static_cast<int>(setlvlnum)));
	return returnLevel;
}

void DiscardPendingEvents()
{
	// Level-change events enter the SDL queue at the end of a game_loop tick. If that tick happens to hit
	// episode truncation, the next episode's reset comes before the next PumpSdlEvents; without clearing the queue
	// the previous episode's WM_DIABNEXTLVL would be applied to the new hero, leaking state across episodes.
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
	}
}

bool DummyGetHeroInfo(_uiheroinfo * /*info*/)
{
	return true;
}

int CountBeltHeals();            // defined in the action section (v12); Observe's raw fields also use it
int CountLegacyBeltHeals();      // the frozen v3 observation counts Healing scrolls; read-only compatibility count
int CountBeltFreeSlots();        // v4: potion-pickup validity cannot be inferred from the heal count; the belt may hold other items
int InstantHealKind(const Item &); // 0 = not an instant heal; 1..4 = the four potions, keeping belt slot order
bool IsHealItem(const Item &);   // defined in the action section (v13); the floor_items heal flag also uses it
bool IsLegacyHealItem(const Item &); // only rebuilds the old policy view, never drives v4 actions
bool IsWantedGear(const Item &); // defined in the action section (v14); the floor_items gear flag also uses it
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
			    "MonsterKillCounts contains a negative value; refusing to publish a corrupt cumulative kill fact");
		const uint64_t unsignedCount = static_cast<uint64_t>(count);
		if (unsignedCount > std::numeric_limits<uint64_t>::max() - total)
			throw std::overflow_error("monster_kill_total uint64 sum overflow");
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
	bool preparationSword = false;
	bool preparationShield = false;
	uint32_t utility = 0;
	// Internal protocol guard only; never part of the actor input or utility.
	bool nativeEquipmentReady = false;
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

// Private Diablo/Butcher task recipe: a fixed public AC50 reference, not an
// omniscient read of the current monster. Apply the native hit floor/cap AFTER
// armor subtraction; capping raw accuracy first discards useful equipment.
constexpr int GearReferenceArmor = 50;
int GearHitChance(int rawPiercingToHit, int enemyArmor)
{
	return static_cast<int>(std::clamp<int64_t>(
	    static_cast<int64_t>(rawPiercingToHit) - enemyArmor, 5, 95));
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
	    = (gGearHitScoring
	            ? GearHitChance(profile.meleePiercingToHit, GearReferenceArmor)
	            : std::clamp(profile.meleePiercingToHit, 5, 95)) * 100;

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

uint32_t ScoreGearWearProfile(const Player &player, uint32_t presentUtility,
    bool dungeonCombat);

GearCombatProfile GearCombatProfileFromPlayer(const Player &player,
    bool evaluateWear = true, bool dungeonCombat = false)
{
	GearCombatProfile profile;
	for (const inv_body_loc hand : { INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT }) {
		const Item &item = player.InvBody[hand];
		if (item.isEmpty() || !item._iStatFlag || item._iDurability == 0
		    || (item._iMagical != ITEM_QUALITY_NORMAL && !item._iIdentified)) continue;
		profile.preparationSword |= item._itype == ItemType::Sword && player.GetItemLocation(item) == ILOC_ONEHAND;
		profile.preparationShield |= item._itype == ItemType::Shield;
	}
	profile.nativeEquipmentReady = EquipmentReadyForPreservation(player);
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
	if (gGearWearScoring && evaluateWear)
		profile.utility = ScoreGearWearProfile(player, profile.utility,
		    dungeonCombat || leveltype != DTYPE_TOWN);
	return profile;
}

GearCombatProfile SimulateGearCombatProfile(
    const Player &source, std::array<Item, NUM_INVLOC> &body,
    bool dungeonCombat = false, bool evaluateWear = true)
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
	SetPlrAnims(simulated, dungeonCombat);
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		body[slot]._iStatFlag = simulated.InvBody[slot]._iStatFlag;
	}
	return GearCombatProfileFromPlayer(simulated, evaluateWear, dungeonCombat);
}

uint32_t ScoreGearWearProfile(const Player &player, uint32_t presentUtility,
    bool dungeonCombat)
{
	// Six deterministic stress snapshots, NOT a forecast of actual RNG or time.
	// Each unit advances three independent hypothetical event counters: being
	// staggered, blocking, and landing a melee hit. The arithmetic never calls
	// Step, RNG, real repair/equip, or mutates live inventory/HP.
	std::array<Item, NUM_INVLOC> body;
	std::copy(std::begin(player.InvBody), std::end(player.InvBody), body.begin());
	std::array<int, NUM_INVLOC> accumulated {};
	uint64_t sum = presentUtility;
	auto wear = [&](inv_body_loc slot, int numerator) {
		Item &item = body[slot];
		if (item.isEmpty() || item._iMaxDur <= 0 || item._iMaxDur == DUR_INDESTRUCTIBLE)
			return;
		accumulated[slot] += numerator;
		if (accumulated[slot] >= 120) {
			accumulated[slot] -= 120;
			if (--item._iDurability <= 0)
				item.clear();
		}
	};
	for (int pressure = 1; pressure <= 30; ++pressure) {
		const bool chest = !body[INVLOC_CHEST].isEmpty();
		const bool head = !body[INVLOC_HEAD].isEmpty();
		if (chest) wear(INVLOC_CHEST, head ? 60 : 90);
		if (head) wear(INVLOC_HEAD, chest ? 30 : 90);
		for (const inv_body_loc hand : { INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT }) {
			if (body[hand]._itype == ItemType::Shield)
				wear(hand, 12);
			else if (body[hand].isWeapon())
				wear(hand, 4);
		}
		if (pressure % 6 == 0)
			sum += SimulateGearCombatProfile(player, body, dungeonCombat, false).utility;
	}
	return static_cast<uint32_t>(sum / 6);
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

GearCombatProfile EmptyGearCombatBaseline(const Player &source, bool dungeonCombat = false)
{
	std::array<Item, NUM_INVLOC> emptyBody {};
	return SimulateGearCombatProfile(source, emptyBody, dungeonCombat);
}

GearCombatProfile LoadoutGearCombatProfile(const Player &player)
{
	const GearCombatProfile emptyBaseline
	    = EmptyGearCombatBaseline(player);
	GearCombatProfile profile = GearCombatProfileFromPlayer(player);
	MakeGearUtilityRelativeToEmpty(profile, emptyBaseline);
	return profile;
}

PreparationProgress PreparationGoal(const GearCombatProfile &p)
{
	return ::PreparationGoal(p.armor, p.demonMin, p.demonMax, p.preparationSword, p.preparationShield);
}

py::dict PreparationEquipmentState(const Player &player)
{
	// Detached previews lack attack animations. Simulate both compared sets
	// with the same dungeon animation metadata, without touching live state.
	std::array<Item, NUM_INVLOC> body;
	std::copy(std::begin(player.InvBody), std::end(player.InvBody), body.begin());
	const auto p = SimulateGearCombatProfile(player, body, true, false);
	py::dict d;
	d["protocol"] = "butcher-equipment-progress/2";
	d["armor"] = p.armor; d["minimum"] = p.demonMin; d["maximum"] = p.demonMax;
	d["sword"] = p.preparationSword; d["shield"] = p.preparationShield;
	d["to_hit"] = p.meleePiercingToHit; d["utility"] = p.utility;
	d["attack_cycle_frames"] = p.attackCycleFrames;
	d["hp_fixed"] = p.currentHitPoints; d["max_hp_fixed"] = p.maxHitPoints;
	return d;
}

bool IsConservativeGearUpgrade(
    const GearCombatProfile &previous, const GearCombatProfile &next)
{
	// The opt-in native predicate is computed from the full recalculated player
	// for both candidate exposure and live commit. Missing health, potions or
	// level must not waive equipment readiness that this loadout already has.
	if (!gPreparationEquipment && previous.nativeEquipmentReady && !next.nativeEquipmentReady)
		return false;
	const auto beforeGoal = PreparationGoal(previous), afterGoal = PreparationGoal(next);
	if (gPreparationEquipment && !PreparationPreserves(beforeGoal, afterGoal)) return false;

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
	if (next.utility <= previous.utility
	    && !(gPreparationEquipment && beforeGoal != afterGoal))
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
	throw std::runtime_error("data_dir is missing DIABDAT.MPQ/diabdat.mpq/spawn.mpq: " + dataDir);
}

// Empty event handler: demo::FetchMessage refuses to emit events while CurrentEventHandler==DisableInputEventHandler
// (demomode.cpp:727); an "in game" handler must be installed to unlock the event stream.
// Events are actually dispatched in PumpSdlEvents; nothing to handle here.
void GymEventHandler(const SDL_Event & /*event*/, uint16_t /*modState*/)
{
}

void CreateFreshHeroSave()
{
	// Rebuild save slot 0 every episode -> every episode is a brand-new level-1 hero, reproducibly
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

// Synchronous level loading: a copy of the branches of interfac.cpp DoLoad.
// Headless mode needs no progress animation, so this bypasses upstream's threaded ShowProgress and loads on one thread.
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
		SaveLevelForTransition();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_MAIN);
		break;
	case WM_DIABPREVLVL:
		SaveLevelForTransition();
		FreeGameMem();
		currlevel--;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_PREV);
		break;
	case WM_DIABSETLVL:
		SaveLevelForTransition();
		setlevel = true;
		leveltype = setlvltype;
		currlevel = static_cast<uint8_t>(setlvlnum);
		FreeGameMem();
		loadResult = LoadGameLevel(false, ENTRY_SETLVL);
		break;
	case WM_DIABRTNLVL:
		SaveLevelForTransition();
		setlevel = false;
		FreeGameMem();
		currlevel = GetMapReturnLevel();
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_RTNLVL);
		break;
	case WM_DIABWARPLVL:
		SaveLevelForTransition();
		FreeGameMem();
		GetPortalLevel();
		loadResult = LoadGameLevel(false, ENTRY_WARPLVL);
		break;
	case WM_DIABTOWNWARP:
		SaveLevelForTransition();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_TWARPDN);
		break;
	case WM_DIABTWARPUP:
		SaveLevelForTransition();
		FreeGameMem();
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_TWARPUP);
		break;
	case WM_DIABRETOWN:
		SaveLevelForTransition();
		FreeGameMem();
		setlevel = false;
		currlevel = myPlayer.plrlevel;
		leveltype = GetLevelType(currlevel);
		loadResult = LoadGameLevel(false, ENTRY_MAIN);
		break;
	default:
		throw std::runtime_error("SyncLoad: unsupported interface_mode " + std::to_string(static_cast<int>(uMsg)));
	}

	if (!loadResult.has_value())
		throw std::runtime_error("Level load failed: " + loadResult.error());

	// The headless-required part of the ProgressEventHandler WM_DONE branch: announce joining the level
	NetSendCmdLocParam2(true, CMD_PLAYER_JOINLEVEL, myPlayer.position.tile, myPlayer.plrlevel,
	    myPlayer.plrIsOnSetLevel ? 1 : 0);
	// This bridge is always loopback single player; for an already active local player the only
	// synchronizing effect of OnPlayerJoinLevel is clearing _pLvlChanging. After the end-of-tick synchronous load we
	// Observe immediately and cannot wait until after the next Python action to process the join packet, or the action
	// guard would swallow one more legal action in the new scene. Clearing the queued join packet again next tick
	// is idempotent and still keeps the same message path as upstream.
	if (!gbIsMultiplayer)
		myPlayer._pLvlChanging = false;
	// Replicates upstream's NewCursor(CURSOR_HAND) in the WM_DONE branch (interfac.cpp; skipped under headless by
	// skipRendering): the pickup arrival check requires pcurs==CURSOR_HAND (player.cpp);
	// reasserting it on every level change pins this implicit invariant (found in the v13 review)
	NewCursor(CURSOR_HAND);
	gStartupTick = true;
	ResourceAfterLoad();
}

// Replicates the custom-event branch of GameEventHandler (level changes etc.), using synchronous loading instead
void PumpSdlEvents()
{
	SDL_Event event;
	uint16_t modState;
	// Note this must be devilution::FetchMessage (the real event pump in events.hpp);
	// demo::FetchMessage swallows every event except QUIT outside demo mode
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
		// The headless environment produces no keyboard/mouse events; all actions are injected through the network command layer
	}
}

// R18-H (2026-09-07) sweep-v1: chest/barrel observation channel (default off).
#include "resource_sweep.hpp"

py::dict ObserveQuestEntrances()
{
	py::dict result;
	result["protocol"] = "visible-quest-entrances/1";
	result["dungeon_level"] = ConceptualDungeonDepth();
	result["engine_level"] = static_cast<int>(currlevel);
	result["is_set_level"] = setlevel;
	result["set_level_id"] = setlevel ? static_cast<int>(setlvlnum) : 0;
	py::list entries;
	for (const Quest &quest : Quests) {
		const QuestEntranceFacts facts {
			gbIsSpawn, UseMultiplayerQuests(), setlevel,
			quest._qactive != QUEST_NOTAVAIL, quest._qidx == Q_BETRAYER,
			IsTileVisible(quest.position), IsTileLit(quest.position),
			InDungeonBounds(quest.position), static_cast<int>(currlevel),
			static_cast<int>(quest._qlevel), static_cast<int>(quest._qslvl),
			static_cast<int>(quest._qvar1)
		};
		if (!ExportQuestEntrance(facts))
			continue;
		py::dict entry;
		entry["kind"] = "quest_set_level";
		entry["x"] = static_cast<int>(quest.position.x);
		entry["y"] = static_cast<int>(quest.position.y);
		entry["destination_set_level_id"] = static_cast<int>(quest._qslvl);
		entry["enter_action"] = "walk";
		entry["requires_stand"] = true;
		entry["visible"] = true;
		entries.append(entry);
	}
	result["entries"] = entries;
	return result;
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
	// The new dual wire needs to keep the low bits of the 26.6 fixed point; the old integer HP/Mana fields stay frozen.
	// Each of the two uint16 words is exactly representable in float32, avoiding low-bit loss from dividing by the scale.
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
	// In quest set-levels `currlevel` is reused as a `_setlevels` enum value (e.g.
	// Vile Betrayer=5) and is never the main-line depth. Training reward, death pricing and leaderboard
	// depth must use the main level the set-level returns to; the scene identity is kept as well, so that entering/leaving
	// a quest set-level does not count the disappearance of the whole old map's monsters as kills.
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
	// An ActiveMonsters snapshot cannot see monsters that "spawn and then die" within the same macro action;
	// the engine's permanent per-type kill ledger is the source of truth for the spawn->die loop. Sum it safely as uint64
	// and hand it straight to a Python int, avoiding narrow-integer wraparound in long training runs.
	obs["monster_kill_total"] = MonsterKillTotal();
	obs["belt_heals"] = CountBeltHeals(); // in raw since v12; since v13 env writes it into the observation vector (bottle-blindness fix)
	// v3 wrongly classed Healing scrolls as instant heal potions. v4 actions must keep excluding
	// them, while the frozen network's compatibility view must still see the old count verbatim; both sources coexist explicitly,
	// so Python never has to guess the lost old classification back from the corrected bool.
	obs["legacy_belt_heals"] = CountLegacyBeltHeals();
	obs["belt_free_slots"] = CountBeltFreeSlots(); // v4 raw-only: exactly masks empty potion-pickup presses when the belt is full
	// action12 consumes "the first instant heal potion from left to right". Publishing only the total would make
	// [small red, large purple] and [large purple, small red] look like the same state, although the next press heals
	// a different amount. Publish the raw class 0..4 per slot; the new Python Worker wire protocol expands it to
	// 8x4 one-hot, and the old 295/298-dim compatibility views stay unchanged.
	py::list beltHealKinds;
	py::list beltSlotKinds;
	for (int i = 0; i < MaxBeltItems; i++)
	{
		const Item &beltItem = player.SpdList[i];
		const int healKind = InstantHealKind(beltItem);
		beltHealKinds.append(healKind);
		// Six mutually exclusive classes: 0 empty, 1 other, 2..5 = instant heal 1..4.
		// In the old heal_kinds, 0 meant both empty and other, and the free total alone still cannot
		// determine which slot a13 auto-fills and which potion a12 then drinks first.
		beltSlotKinds.append(
		    beltItem.isEmpty() ? 0 : (healKind == 0 ? 1 : healKind + 1));
	}
	obs["belt_heal_kinds"] = beltHealKinds;
	obs["belt_slot_kinds"] = beltSlotKinds;
	obs["armor_class"] = player.GetArmor(); // v14: armor class (_pIBonusAC + _pIAC + dexterity/5)
	// a9's real transition is determined by these "aggregated, currently effective" combat values. Giving only HP/AC
	// would collapse characters with different weapon damage, to-hit, resistances, blocking and attack-speed/life-steal flags into
	// the same state; the dual Worker encodes the scalars at fixed scales and expands the two flag words into bits.
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
		// active_id belongs only to the floor array; equipped slots use 0 as a stable placeholder, and the other fields
		// share the full CalcPlrInv/durability-transfer protocol with floor gear.
		AppendItemCombatState(entry, equipped, 0);
		equippedItems.append(entry);
	}
	obs["equipped_items"] = equippedItems;
	// A one-way training task cannot return to town to see Cain, but the original single-player Lazarus main quest strictly requires
	// "pick up the staff -> return to town and give it to Cain -> go down to L15". After the staff is picked up the bridge automatically performs only this one
	// equivalent hand-over (see Step); the quest state and whether the adapter was used stay in raw for probes and
	// trajectory audits. They do not enter the policy vector and do not pretend to be the original natural flow.
	const Quest &betrayerQuest = Quests[Q_BETRAYER];
	obs["betrayer_quest_active"] = static_cast<int>(betrayerQuest._qactive);
	obs["betrayer_quest_stage"] = static_cast<int>(betrayerQuest._qvar1);
	obs["betrayer_portal_stage"] = static_cast<int>(betrayerQuest._qvar2);
	obs["monotonic_quest_turn_in_used"] = gMonotonicQuestTurnInUsed;

	py::list monsters;
	for (size_t i = 0; i < ActiveMonsterCount; i++) {
		const unsigned monsterId = ActiveMonsters[i];
		const Monster &monster = Monsters[monsterId];
		// The engine judges death by the integer part of the fixed-point HP (hasNoLife). Comparing only raw
		// hitPoints<=0 would expose a dead monster with 0<HP<1 as hp=0 for one extra tick, making the Python
		// side lose the kill reward when it disappears on the next tick.
		if (monster.hasNoLife())
			continue;
		py::dict m;
		m["id"] = monsterId;
		m["type"] = static_cast<int>(monster.type().type);
		m["x"] = static_cast<int>(monster.position.tile.x);
		m["y"] = static_cast<int>(monster.position.tile.y);
		// CMD_ATTACKID/MakePlrPath and reachable both use future (the occupied tile after the animation
		// commits); tile is only the current render tile. The two differ for several consecutive ticks during a walk
		// animation, so the macro's stop-loss geometry must use the same coordinate as the engine.
		m["future_x"] = static_cast<int>(monster.position.future.x);
		m["future_y"] = static_cast<int>(monster.position.future.y);
		m["hp"] = monster.hitPoints >> 6;
		m["max_hp"] = monster.maxHitPoints >> 6;
		m["hp_fixed_hi"] = HighWord(static_cast<uint32_t>(monster.hitPoints));
		m["hp_fixed_lo"] = LowWord(static_cast<uint32_t>(monster.hitPoints));
		m["max_hp_fixed_hi"] = HighWord(static_cast<uint32_t>(monster.maxHitPoints));
		m["max_hp_fixed_lo"] = LowWord(static_cast<uint32_t>(monster.maxHitPoints));
		// Monsters with the same type/HP/position have completely different next-tick casualty distributions when one is
		// before an attack's damage frame and the other is idle. Publish the finite state that drives MonsterMode/AI and the current animation progress,
		// plus the already scaled instant attack/defense values; the dual wire encodes the flags bit by bit.
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
		// The full-level list remains the source of truth for rewards/kills; the policy may only consume the subset a human player
		// could also see and reach by native pathfinding, and must never fight with omniscient coordinates in unexplored rooms.
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
		it["heal"] = IsHealItem(item);   // v13: target flag for the potion-pickup macro
		it["heal_kind"] = InstantHealKind(item);
		it["legacy_heal"] = IsLegacyHealItem(item); // frozen v3 read-only observation
		it["gear"] = IsWantedGear(item); // v14: target flag for the gear-pickup macro (empty slot + stat requirements met)
		// a14's coordinates and bool are not enough to determine the transition: the same tile can hold different gear,
		// whose slot, base AC and active affixes change the immediate reward and later combat. Keep every
		// item fact that CalcPlrItemVals consumes in raw; Python encodes only the one item selected by the
		// snapshot, without enlarging the old 295/298 compatibility observation.
		AppendItemCombatState(it, item, activeItemId, true);
		const bool visible = IsTileLit(item.position);
		it["visible"] = visible;
		it["reachable"] = visible && CanReachTarget(item.position);
		items.append(it);
	}
	obs["floor_items"] = items;

	// Missiles move/hit independently across the several engine ticks of one option. Publishing only
	// monsters and tiles would collapse "the fireball is already in your face" and "not fired yet" into the same Worker state.
	// This keeps all scalar state of SaveMissile and adds a duplicate-collision hash; Python
	// selects only fixed slots within radius 12 and losslessly encodes int32 as uint16 words.
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

	// The "next required story target" shared by actions 10/11. The allowlist contains only interactions without which
	// Diablo cannot be reached, and never turns ordinary chests/shrines/side-quest objects into omniscient
	// automation. goal is the tile to actually reach; the two Vile books in particular require the player to stand exactly
	// on the pentagram south-west of the book, and operating them from any adjacent tile is silently refused upstream.
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

	// Level entrances/exits (stairs/portals): the agent's navigation targets
	py::list triggers;
	for (int i = 0; i < numtrigs; i++) {
		py::dict t;
		t["x"] = static_cast<int>(trigs[i].position.x);
		t["y"] = static_cast<int>(trigs[i].position.y);
		t["msg"] = static_cast<int>(trigs[i]._tmsg);
		triggers.append(t);
	}
	obs["triggers"] = triggers;
	// Independent opt-in channel: never insert set-level entrances into the
	// frozen stair list or change any existing neural observation field.
	if (gQuestEntranceObservation)
		obs["quest_entrances"] = ObserveQuestEntrances();
	// R18-H (2026-09-07) sweep-v1: the chest/barrel channel. Absent unless the
	// new default-false flag is on, so the frozen observation dict is byte-identical.
	if (gObjectObservation)
		obs["objects"] = ObserveSweepObjects();
	if (gResourceProtocol)
		obs["resource_state"] = ObserveResourceState();

	return obs;
}

void EngineInit(const std::string &assetsDir, const std::string &saveDir, const std::string &dataDir, int heroClass, bool verbose)
{
	if (heroClass != static_cast<int>(HeroClass::Warrior))
		throw std::invalid_argument("the current action/auto stat-allocation contract only supports hero_class=0 (Warrior)");
	if (gEngineInited) {
		EnsureEngineProcess("init");
		if (assetsDir != gAssetsDir || saveDir != gSaveDir || dataDir != gDataDir || heroClass != gHeroClass)
			throw std::runtime_error("DevilutionX is an in-process singleton; init() cannot be repeated with a different configuration");
		return;
	}
	gHeroClass = heroClass;
	const std::string expectedMainArchive = ExpectedMainArchivePath(dataDir);

	// Set headless first so that no later error path can pop up a GUI dialog (matches test/main.cpp:84)
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

	// Upstream registers custom SDL events only when creating the window (display.cpp); a headless embedding must register them itself,
	// or level-change events (WM_DIABNEXTLVL etc.) are not recognized after being pushed and the player gets stuck in PM_NEWLVL
	RegisterCustomEvents();

	// MPQ search order: BasePath -> PrefPath -> ConfigPath (assets.cpp GetMPQSearchPaths).
	// BasePath points at the game data directory; Pref/Config point at scratch, isolating saves from the user's real game
	paths::SetBasePath(dataDir + "/");
	paths::SetAssetsPath(assetsDir + "/");
	paths::SetPrefPath(saveDir + "/");
	paths::SetConfigPath(saveDir + "/");

	LoadCoreArchives();
	LoadGameArchives(); // if diabdat.mpq is not found, falls back to spawn.mpq automatically and sets gbIsSpawn
	if (!HaveMainData())
		throw std::runtime_error("neither diabdat.mpq nor spawn.mpq was found (the default search includes "
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
		    "main archive mode disagrees with gbIsSpawn: archive="
		    + expectedArchiveName + ",gbIsSpawn="
		    + std::to_string(archiveIsSpawn ? 1 : 0));
#ifndef UNPACKED_MPQS
	// LoadGameArchives also searches scratch, system directories and the current working directory. If a
	// higher-priority/same-name MPQ happens to be there, hashing data_dir alone would bind the evaluation identity to the wrong
	// file. Embedded mode must accept only the main archive chosen by engine priority from the caller's explicit data_dir,
	// so that the content SHA of training/evaluation corresponds one-to-one to the bytes actually loaded.
	const auto mainArchive = MpqArchives.find(MainMpqPriority);
	if (mainArchive == MpqArchives.end()
	    || mainArchive->second.path() != expectedMainArchive) {
		const std::string actual = mainArchive == MpqArchives.end()
		    ? "<missing>"
		    : mainArchive->second.path();
		MpqArchives.clear();
		if (SDL_WasInit((~0U) & ~SDL_INIT_HAPTIC) != 0)
			SDL_Quit();
		throw std::runtime_error("actual main MPQ did not come from data_dir: actual=" + actual
		                         + ", expected=" + expectedMainArchive);
	}
#endif

	InitKeymapActions();
	LoadOptions();
	// Training transitions must not inherit personal QoL, speed or quest settings from save_dir/diablo.ini.
	// In particular autoRefillBelt would make action12 consume invisible backpack potions first, auto-pickup
	// would change the inventory without any policy action, and autoEquipWeapons could put the Lazarus
	// staff in hand and escape the monotone quest adapter. Every option that changes world/action outcomes
	// is pinned to a protocol constant before Lua and new-game initialization.
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
		    "DiabloGym forbids enabling Lua mods; active mods detected in save_dir: "
		    + names);
	}
	options.GameMode.gameMode.SetValue(StartUpGameMode::Diablo);
	// The shareware callback rewrites gbIsSpawn directly; restore the mode determined by the main archive actually loaded,
	// which both stops the ini from downgrading a full DIABDAT to the demo and keeps spawn.mpq support.
	options.GameMode.shareware.SetValue(archiveIsSpawn);
	if (gbIsSpawn != archiveIsSpawn)
		throw std::runtime_error(
		    "gbIsSpawn disagrees with the main archive after freezing GameMode.shareware");
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
	options.Gameplay.autoEquipArmor.SetValue(!gManualControl);
	options.Gameplay.autoEquipHelms.SetValue(!gManualControl);
	options.Gameplay.autoEquipJewelry.SetValue(!gManualControl);
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
		throw std::runtime_error("cannot register DevilutionX process-exit cleanup");
	}
	gExitCleanupRegistered = true;

	gbIsHellfire = false;
	gbMusicOn = false;
	gbSoundOn = false;

	// Under headless there are never mouse events to set ControlMode to keyboard/mouse; if it stays None,
	// plrctrls' WalkInDir reads "no stick input" as a released gamepad and brakes the pathing every tick
	// (plrctrls.cpp:1744), so walk commands only execute one step.
	ControlMode = ControlTypes::KeyboardAndMouse;
	ControlDevice = ControlTypes::KeyboardAndMouse;
	// DiabloGym's task space only allows progressing downward. Otherwise ordinary FARM/worker movement
	// could accidentally step on an up-stairs trigger, return to town and waste the remaining 3000 steps.
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
	// Replicates FreeGame() at the end of upstream RunGameLoop. It lives in an anonymous namespace in diablo.cpp
	// and cannot be called directly, but it cannot be omitted: InitCursor explicitly requires
	// the previous game to have run FreeCursor, and quest text/panel/player graphics caches must not cross episodes.
	CleanupGameResources();
	NetClose(); // end of the outer StartGame (clears Players)
	gInGame = false;
	gEpisodeGeneration++; // makes a direct end_game() immediately invalidate the Python wrapper's raw cache
	DiscardPendingEvents();
}

void EngineShutdownAtExit() noexcept
{
	// After fork only the calling thread survives; inherited SDL/network/Lua locks and thread state must not
	// be destroyed in the child. A plain return would still run upstream C++ global static destructors such as
	// CurrentLuaState, re-exposing the cross-translation-unit destruction-order UAF. The only legal end points of a fork child
	// are exec/os._exit; if it mistakenly takes a normal exit/SystemExit, fail closed here
	// by terminating immediately with failure, skipping the remaining atexit handlers and all static destructors.
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
		throw std::runtime_error("call init() first");
	EndGame();
	ClearEpisodePersistentGameplayState();
	gStallPrints = 0;
	gMonotonicQuestTurnInUsed = false;
	ResetResourceEpisode(seed);
	gManualTicks = 0;

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
	if (gManualControl && (sgGameInitInfo.nDifficulty != DIFF_NORMAL || MyPlayer->_pClass != HeroClass::Warrior))
		throw std::runtime_error("manual experiment requires native normal difficulty and Warrior");

	// Determinism: overwrite all dungeon seeds with the user seed (the engine just filled them from an entropy source in NetInit)
	std::mt19937 rng(seed);
	for (int i = 0; i < NUMLEVELS; i++) {
		DungeonSeeds[i] = static_cast<uint32_t>(rng());
		LevelSeeds[i] = std::nullopt;
	}
	// Defensively take over the global RNG: CreatePlayer (via pfile_ui_save_create) just called SetRndSeed with wall-clock milliseconds
	// (player.cpp). In the pinned engine quest selection is unaffected (InitQuests goes through
	// InitialiseQuestPools(DungeonSeeds[15]), a local RNG whose seed was taken over in the loop above),
	// and level loading also reseeds per level; this overwrite pins "the global RNG belongs to the episode seed" as
	// an invariant that cannot break as upstream evolves. Measured: 32-seed evaluation fingerprints are bit-identical before and after the fix.
	SetRndSeed(static_cast<uint32_t>(rng()));

	// New-game initialization of the outer StartGame(bNewGame=true)
	InitLevels();
	InitQuests();
	InitPortals();
	InitDungMsgs(*MyPlayer);
	DeltaSyncJunk();
	giNumberOfLevels = gbIsHellfire ? 25 : 17;

	// The prologue of RunGameLoop before its while loop (headless version, skipping drawing/fades/discord).
	// The inner StartGame(uMsg) is in an anonymous namespace; below is a copy using its public API
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

// v20: automatic stat allocation, fixing the "stat point black hole". The engine grants 5 stat points per level
// (NextPlrLevel only accumulates _pStatPts; spending has always relied on a human clicking the UI), and nineteen generations of this bridge never
// called the spending path, so the level -> survivability chain was broken and "farm first, dive later" did not hold mechanically
// (a naked clvl3 fighting a median L3 monster pack has negative expectation; the only comeback line needs +10 vitality + armor).
// Warrior definition: each batch of points is split 3 vitality : 2 strength (vitality dominates low-level survival, strength feeds damage and
// carrying capacity). ModifyPlr* applies the stat caps and recomputes derived values (HP/to-hit/capacity); points overflowing
// a cap are discarded in place, as for a human player. Hooked at the end of Step: level-ups happen inside ticks, so
// the points land at most one env step (4 ticks) later, which is immediate for the policy.
static void AutoSpendStatPoints()
{
	Player &p = *MyPlayer;
	const int pts = p._pStatPts;
	if (pts <= 0)
		return;
	const int vit = (pts * 3 + 4) / 5; // 3/5 rounded up goes to vitality
	ModifyPlrVit(p, vit);
	ModifyPlrStr(p, pts - vit);
	p._pStatPts = 0;
}

// The original single-player main quest requires bringing the Staff of Lazarus back to town and handing it to Cain. DiabloGym's
// training task dives strictly one way from L1, with neither a town-return action nor level regression; if
// that UI precondition stayed, a full clear would be unreachable in the action graph. This only replicates
// the one-time hand-over state transition for that item in TalkToStoryteller, without skipping the staff stand, the pickup,
// the L15 entrance, the Vile mechanism or combat, and permanently marks in raw that this game used the adapter.
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
		throw std::invalid_argument("ticks must be a positive integer");
	for (int i = 0; i < ticks && gbRunGame; i++) {
		PumpSdlEvents();
		if (!gbRunGame)
			break;
		ProcessGameMessagePackets();
		const bool advanced = game_loop(gStartupTick);
		if (gManualControl && advanced) ++gManualTicks;
		if (!advanced && gStallPrints < 8) {
			std::fprintf(stderr, "[diablogym] game_loop stalled (multi_handle_delta got no turn), destroyed=%d\n",
			    gbGameDestroyed ? 1 : 0);
			gStallPrints++;
		}
		gStartupTick = false;
	}
	// StartNewLvl can put the custom event into the SDL queue only at the end of the last game_loop
	// tick. Pump once more before returning to Python so that the level-change scene and reward
	// belong to the env step that actually triggered the stairs, not to the next policy action
	// forced to spend an empty tick. This only loads; it consumes no extra game-logic ticks.
	if (gbRunGame)
		PumpSdlEvents();
	if (!gManualControl) {
		AutoSpendStatPoints();
		AutoTurnInBetrayerStaffForMonotonicTask();
	}
	return Observe();
}

int ActWait()
{
	if (!CanAcceptPlayerAction("act_wait"))
		return 0;
	Player &player = *MyPlayer;
	if (!gbRunGame || player._pmode == PM_DEATH || player._pmode == PM_QUIT)
		return 0;

	// Explicit v4 wait/cancel semantics:
	// 1. Immediately clear the long path and delayed action already written by the previous tick's OnWalk/OnAttack;
	// 2. Then queue "walk to the current future tile" at the end of the loopback network queue as a FIFO
	//    fence. Doing only step 1 still races: the caller may queue act_attack_monster first,
	//    then queue wait before the same bridge.step, and the old attack packet would write destAction back next tick.
	// An attack/spell animation already in progress must return to standing immediately: if only destAction were cleared, a swing
	// that has not reached its damage frame this tick would land during the next Gym action, crediting the kill to action0/
	// drinking by mistake. Walk animations are not cut hard; the committed tile may finish naturally, then the player stops.
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
		return; // an out-of-bounds step at the map edge is treated as an invalid Gym action
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
    "a diagonal Worker atomic key must be expressible as one step");
static_assert(!IsExactControllerEdge(1, 1, 2),
    "an adjacent target reachable only by a two-step detour must fail closed");
static_assert(!IsExactControllerEdge(2, 0, 1),
    "a far waypoint must not masquerade as a controller atomic edge");

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

	// ActWait put a loopback FIFO fence at the end of the previous macro; the safe path must be written
	// after it, or the next Step would erase the new walkpath when processing the old wait packet.
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
			throw std::out_of_range("act_explore_walk protected coordinate out of range");
		if (std::abs(tx - centerX) > radius
		    || std::abs(ty - centerY) > radius)
			throw std::out_of_range(
			    "act_explore_walk protected coordinate outside the fixed snapshot");
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
	// Worker atomic direction actions include four diagonal keys; only non-adjacent waypoints are refused. Illegal
	// diagonal edges such as blocked corners still fail closed edge by edge via CanStep + final steps==1, with no detours.
	if (!IsAdjacentControllerDelta(x - start.x, y - start.y)
	    || !inWindow(start))
		return 0;
	const Point target { x, y };
	const DynamicTileDanger danger = InspectDynamicTileDanger(target);
	// After Python's frozen snapshot, a fire trap may just have entered its damage phase; an explosive barrel may also
	// appear on the next tile while the macro runs. Re-read the currently lit facts at the last moment before committing an edge,
	// so that the safe path cannot turn into stepping into fire/bumping a barrel inside the TOCTOU window.
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
		throw std::out_of_range("monster_id out of range");
	NetSendCmdParam1(true, CMD_ATTACKID, monsterId);
}

int ActControllerAttackMonster(
    uint16_t monsterId, int centerX, int centerY, int radius)
{
	if (!CanAcceptPlayerAction("act_controller_attack_monster"))
		return 0;
	if (monsterId >= MaxMonsters)
		throw std::out_of_range("monster_id out of range");
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
	// Only adjacent swings inside the snapshot window are allowed. Far targets must be approached along Python's fixed snapshot path
	// one adjacent step at a time; CMD_ATTACKID must never quietly install a whole-map chase path here.
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
	// Operate the object on the target tile (door/chest/lever): the engine walks there and operates it, the same path as a mouse click
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
	// A Healing scroll needs a spell target afterwards; it is not a belt potion that "heals on press".
	// Counting it as a heal potion would make act_drink choose the scroll, leave a cross-hair cursor, and short-circuit
	// every later key press into a fake success. Only potions that take effect synchronously and are consumed count here.
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
	// Drink only potions that "heal on press". Upstream UseBeltItem(BeltItemType::Healing)
	// also matches Healing scrolls; a scroll only enters target selection, but this bridge once misreported it as
	// a drink, after which pcurs != CURSOR_HAND silently short-circuited every action.
	//
	// UseInvItem's bool also only means "the key press was handled", and can still return true with a store/chat/non-hand cursor
	// and similar cases. The receipt must therefore be certified by post-facts: HP actually rose, or
	// the belt's instant heal count actually fell (auto-refill may drink a same-kind backpack potion instead; then
	// the belt count stays the same but HP rises). Only on successful certification is the pre-request potion count returned.
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
		return 0; // CountBeltHeals and the selection predicate must share one source; defense in depth.

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
		return 0; // no free belt slot: a pickup would drop straight into the backpack (a value black hole invisible to the drink key and the observation), so no command is sent
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
	int nextArmorClass = 0;
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
		    "action14 transaction rollback did not restore life/mana state bit for bit");
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

// Loot mode stages every displaced object using the engine's real packing rule.
// No live inventory, scroll state, RNG or network queue is touched by this copy.
bool StageLootReplacedInventory(const Player &player,
    const std::array<bool, NUM_INVLOC> &clearSlots, Player &staged)
{
	for (int slot = 0; slot < NUM_INVLOC; ++slot) {
		if (clearSlots[slot] && !player.InvBody[slot].isEmpty()
		    && !AutoPlaceItemInInventory(staged, player.InvBody[slot], false))
			return false;
	}
	return true;
}

void ConsiderGearUpgradePlan(
    const Player &player, const Item &candidate,
    inv_body_loc target, std::array<bool, NUM_INVLOC> clearSlots,
    const GearCombatProfile &emptyBaseline,
    const GearCombatProfile &previousProfile, GearUpgradePlan &best,
    bool requireRetentionCapacity, bool dungeonCombat)
{
	if (gResourceLootEconomy && requireRetentionCapacity) {
		Player staged = ResourceUnequipSimulation(player);
		if (!StageLootReplacedInventory(player, clearSlots, staged)) return;
	}
	std::array<Item, NUM_INVLOC> body;
	for (int slot = 0; slot < NUM_INVLOC; slot++)
		body[slot] = player.InvBody[slot];
	for (int slot = 0; slot < NUM_INVLOC; slot++) {
		if (clearSlots[slot])
			body[slot].clear();
	}
	body[target] = candidate;
	GearCombatProfile nextProfile
	    = SimulateGearCombatProfile(player, body, dungeonCombat);
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
	best.nextArmorClass = nextProfile.armor;
}

GearUpgradePlan PlanGearUpgrade(const Player &player, const Item &item,
    bool requireRetentionCapacity = true, bool dungeonCombat = false)
{
	GearUpgradePlan best;
	if (item.isEmpty() || !item.isEquipment()
	    || item._iClass == ICLASS_QUEST || item.IDidx == IDI_LAZSTAFF)
		return best;
	Item candidate = PrepareGearCandidate(item, player);
	if (!candidate._iStatFlag)
		return best;
	const GearCombatProfile emptyBaseline
	    = EmptyGearCombatBaseline(player, dungeonCombat);
	GearCombatProfile previousProfile;
	if (dungeonCombat) {
		// A town player retains cached attack frames from the previous dungeon,
		// while a fresh town copy has none. Normalize all three loadouts through
		// the same native dungeon-animation table without touching the live player.
		std::array<Item, NUM_INVLOC> currentBody;
		std::copy(std::begin(player.InvBody), std::end(player.InvBody), currentBody.begin());
		previousProfile = SimulateGearCombatProfile(player, currentBody, true);
	} else {
		previousProfile = GearCombatProfileFromPlayer(player);
	}
	MakeGearUtilityRelativeToEmpty(previousProfile, emptyBaseline);

	auto singleSlot = [&](inv_body_loc slot) {
		std::array<bool, NUM_INVLOC> clearSlots {};
		clearSlots[slot] = true;
		ConsiderGearUpgradePlan(
		    player, candidate, slot, clearSlots, emptyBaseline,
		    previousProfile, best, requireRetentionCapacity, dungeonCombat);
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
			    emptyBaseline, previousProfile, best, requireRetentionCapacity, dungeonCombat);
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
			    emptyBaseline, previousProfile, best, requireRetentionCapacity, dungeonCombat);
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
	if (gResourceLootEconomy && (!IsTileLit(item.position) || player._pmode != PM_STAND || player.hasNoLife() || !player.HoldItem.isEmpty())) return 0;
	const GearUpgradePlan plan = PlanGearUpgrade(player, item);
	if (!plan.valid)
		return 0;

	std::optional<Player> retainedInventory;
	const int goldBefore = player._pGold;
	if (gResourceLootEconomy) {
		retainedInventory.emplace(ResourceUnequipSimulation(player));
		if (!StageLootReplacedInventory(player, plan.clearSlots, *retainedInventory)) return 0;
	}

	// Do not use AutoGetItem's "AutoEquip fails -> belt -> hidden backpack" fallback chain.
	// Copy the whole set of body slots, then replace atomically; legacy old gear is explicitly destroyed, loot mode has pre-checked retention.
	// legacy old items do not enter the backpack
	// or drop to the floor where the next macro might pick them up by mistake. CalcPlrInv is the final authoritative check;
	// if any stat-dependent cascade stops the aggregate utility from strictly increasing, roll back completely.
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
		    "action14 simulated plan and atomic commit disagree on utility/current and max life");
	}
	if (player.hasNoLife()
	    || (nextProfile.currentHitPoints >> 6) <= 0) {
		RestoreGearUpgradeTransaction(
		    player, previousBody, previousResources);
		throw std::runtime_error(
		    "action14 planning safety gate failed: the atomic commit would kill the player");
	}
	if (!player.InvBody[plan.target]._iStatFlag
	    || !IsConservativeGearUpgrade(previousProfile, nextProfile)) {
		RestoreGearUpgradeTransaction(
		    player, previousBody, previousResources);
		return 0;
	}

	const int previousInventoryCount = player._pNumInv;
	if (retainedInventory) {
		player._pNumInv = retainedInventory->_pNumInv;
		std::copy(std::begin(retainedInventory->InvList), std::end(retainedInventory->InvList), std::begin(player.InvList));
		std::copy(std::begin(retainedInventory->InvGrid), std::end(retainedInventory->InvGrid), std::begin(player.InvGrid));
		player.CalcScrolls();
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
	if (retainedInventory) {
		for (int cell = 0; cell < InventoryGridCells; ++cell)
			if (player.InvGrid[cell] > previousInventoryCount) NetSendCmdChInvItem(false, cell);
		RecordLootGearRetention(player, previousBody, plan.clearSlots, goldBefore);
	}
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

#include "resource_protocol.hpp"
#include "resource_combinations.hpp"
#include "manual_control.hpp"
#include "unbelt_exact.hpp"
#include "witch_trade.hpp"
#include "combat_telemetry.hpp"
#include "butcher_event.hpp"
#include "field_equipment.hpp"

} // namespace

PYBIND11_MODULE(_diablogym, m)
{
	m.def("manual_configure", []() {
		if (gEngineInited) throw std::runtime_error("manual mode must precede init");
		gManualControl = true;
		gQuestEntranceObservation = true;
	});
	m.def("manual_seed_has_king", &ManualSeedHasKing);
	m.def("manual_observe", &ManualObserveR4);
	m.def("manual_action", &ManualActionR4);
	m.def("manual_checkpoint", &ManualCheckpointR4);
	m.def("butcher_events", &ButcherEvents);
	m.def("manual_can_unbelt", &ManualCanUnbelt);
	m.def("manual_can_standing_attack", &ManualCanStandingAttack);
	m.def("manual_combat_telemetry", &ManualCombatTelemetry);
	m.doc() = "DiabloGym v0: headless DevilutionX RL bridge";
	m.def("configure_quest_entrance_observation", [](bool enabled) {
		EnsureEngineProcess("configure_quest_entrance_observation");
		if (gInGame && enabled != gQuestEntranceObservation)
			throw std::runtime_error("quest entrance observation may change only between episodes");
		gQuestEntranceObservation = enabled;
	});
	m.def("quest_entrance_protocol", []() {
		return gQuestEntranceObservation ? "visible-quest-entrances/1" : "off";
	});
	m.def("configure_field_equipment", [](bool enabled) {
		EnsureEngineProcess("configure_field_equipment");
		if (gInGame && enabled != gFieldEquipment)
			throw std::runtime_error("field equipment may change only between episodes");
		if (enabled && (gGearWearScoring || gGearHitScoring))
			throw std::invalid_argument("field-v1 requires legacy global gear scoring");
		gFieldEquipment = enabled;
	});
	m.def("field_equipment_protocol", []() {
		return gFieldEquipment ? "field-inventory-wear30-v1" : "off";
	});
	m.def("preview_field_equipment", &PreviewFieldEquipment);
	m.def("act_equip_field_inventory_item", &ActEquipFieldInventoryItem,
	    py::arg("index"), py::arg("seed_hi"), py::arg("seed_lo"),
	    py::arg("create_info"), py::arg("base_id"));
#include "gear_wear_fixture.hpp"
	m.def("configure_gear_hit_experiment", [](bool enabled) {
		EnsureEngineProcess("configure_gear_hit_experiment");
		if (gInGame && enabled != gGearHitScoring)
			throw std::runtime_error("hit scoring may change only between episodes");
		if (enabled && (gGearWearScoring || gFieldEquipment))
			throw std::invalid_argument("hit-ac50-v1 cannot combine wear or field candidates");
		gGearHitScoring = enabled;
	});
	m.def("configure_preparation_equipment", [](bool enabled) {
		EnsureEngineProcess("configure_preparation_equipment");
		if (gInGame) throw std::runtime_error("preparation equipment must be configured before reset");
		gPreparationEquipment = enabled;
	});
	m.def("configure_resource_full_smith_catalog", &ConfigureResourceFullSmithCatalog);
	m.def("resource_smith_catalog_protocol", []() {
		return gResourceFullSmithCatalog ? "smith-normal-and-premium-identified/1" : "smith-ordinary/1";
	});
	m.def("preparation_equipment_protocol", []() { return gPreparationEquipment ? "butcher-equipment-progress/2" : "off"; });
	m.def("probe_gear_hit_probability", &GearHitChance,
	    py::arg("raw_piercing_to_hit"), py::arg("enemy_armor"));
	m.def("configure_gear_wear_experiment", [](bool enabled) {
		EnsureEngineProcess("configure_gear_wear_experiment");
		if (gInGame && enabled != gGearWearScoring)
			throw std::runtime_error("gear scoring may change only between episodes");
		if (enabled && gGearHitScoring)
			throw std::invalid_argument("wear candidate cannot combine hit-ac50-v1");
		gGearWearScoring = enabled;
	});
	m.def("gear_scoring_protocol", []() {
		if (gGearHitScoring) return "hit-ac50-v1";
		return gGearWearScoring ? "wear-pressure30-v2" : "legacy";
	});
	m.def("engine_config", []() -> py::object {
		if (!gEngineInited)
			return py::none();
		EnsureEngineProcess("engine_config");
		return py::make_tuple(gAssetsDir, gSaveDir, gDataDir, gHeroClass);
	}, "Read the committed native singleton configuration; returns None before init; used by Python's async exception recovery");
	m.def("init", &EngineInit, py::arg("assets_dir"), py::arg("save_dir"), py::arg("data_dir"),
	    py::arg("hero_class") = 0, py::arg("verbose") = false,
	    "One-time engine initialization. data_dir is the directory holding diabdat.mpq; currently only hero_class=0 (Warrior) is supported");
	m.def("reset", &Reset, py::arg("seed"), "Start a new game (brand-new level-1 hero, deterministic dungeon seeds) and return the observation");
	m.def("step", &Step, py::arg("ticks") = 1, "Advance game logic by N ticks (20 ticks = 1 in-game second) and return the observation");
	m.def("observe", &Observe, "Read the current observation only");
	m.def("configure_resource_protocol", &ConfigureResourceProtocol, py::arg("enabled"), py::arg("ordinary_armor_scope") = false,
	    py::arg("preserve_equipment_readiness") = false, py::arg("loot_economy") = false,
	    py::arg("readiness_advisory") = false, py::arg("retreat") = false, py::arg("portal") = false,
	    py::arg("identify") = false);
	m.def("configure_town_service", &ConfigureTownService, py::arg("authorized"));
	m.def("configure_resource_trip_limits", &ConfigureResourceTripLimits, py::arg("enabled"),
	    "Independent growth candidate: toggle count quotas only, between episodes");
	m.def("configure_retreat", &ConfigureRetreat, py::arg("authorized"),
	    "R18-A retreat-v1: authorize (or revoke) one one-floor ascent from main L2+");
	m.def("configure_resource_portal", &ConfigureResourcePortal, py::arg("authorized"),
	    "R18-F portal-v1: authorize (or revoke) one town-portal transit (main L2+ -> town, or town -> the open portal's floor)");
	m.def("configure_object_observation", &ConfigureObjectObservation, py::arg("enabled"),
	    "R18-H sweep-v1: enable (or disable) the raw \"objects\" chest/barrel channel; "
	    "off by default and off = byte-identical observation dict");
	m.def("configure_resource_weapon_purchase", &ConfigureResourceWeaponPurchase, py::arg("enabled"),
	    "R18-K2b weapon-purchase-v1: let Griswold sell, and the town equip path wear, an ordinary ONE-HANDED weapon (default off; between episodes only)");
	m.def("project_seen_resource_smith_items", &ProjectSeenResourceSmithItems, py::arg("identities"));
	m.def("preview_resource_equipment_combinations", &PreviewResourceEquipmentCombinations, py::arg("sequences"), py::arg("max_gold_cost"));
	m.def("act_pickup_loot_at", &ActPickupLootAt, py::arg("item_id"), py::arg("x"), py::arg("y"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"));
	m.def("act_sell_inventory_item", &ActSellInventoryItem, py::arg("vendor"), py::arg("index"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"), py::arg("expected_price"));
	m.def("act_pickup_gold_at", &ActPickupGoldAt, py::arg("item_id"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"));
	m.def("act_talk_towner", &ActTalkTowner, py::arg("towner_id"));
	m.def("act_dismiss_dialog", &ActDismissDialog);
	m.def("act_buy_store_item", &ActBuyStoreItem, py::arg("vendor"), py::arg("index"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"));
	m.def("act_cast_town_portal", &ActCastTownPortal, py::arg("spell_from"),
	    "R18-F portal-v1: read a carried Scroll of Town Portal via CMD_SPELLXY (never UseInvItem, which would strand the cursor in CURSOR_TELEPORT)");
	m.def("act_identify", &ActIdentifyItem, py::arg("equipped"), py::arg("index"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"), py::arg("expected_price"),
	    "R18-H identify-v1: pay Cain his fixed fee to identify one carried or worn magic item (town, inside an authorized resource town trip, standing next to the storyteller)");
	m.def("act_repair_equipped_item", &ActRepairEquippedItem, py::arg("slot"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"), py::arg("expected_durability"), py::arg("expected_price"));
	m.def("act_equip_inventory_item", &ActEquipInventoryItem, py::arg("index"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"));
	m.def("act_unequip_equipped_item", &ActUnequipEquippedItem, py::arg("slot"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("base_id"));
	m.def("probe_resource_recreate_equipment", [](int slot, int baseId, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int durability, int bonusStrength, int bonusHp) {
		EnsureInGame("probe_resource_recreate_equipment");
		if (slot < 0 || slot >= NUM_INVLOC || baseId < 0 || static_cast<size_t>(baseId) >= AllItemsList.size())
			throw std::invalid_argument("invalid fixture item or slot");
		Item item = {};
		RecreateItem(*MyPlayer, item, static_cast<_item_indexes>(baseId), createInfo,
		    (uint32_t { seedHigh } << 16) | seedLow, 0, 0);
		if (!item.isEquipment() || durability < 0 || durability > item._iMaxDur)
			throw std::invalid_argument("invalid fixture equipment durability");
		item._iIdentified = true;
		item._iDurability = static_cast<uint8_t>(durability);
		item._iPLStr += bonusStrength;
		item._iPLHP += bonusHp;
		MyPlayer->InvBody[slot] = item;
		CalcPlrInv(*MyPlayer, true);
	}, py::arg("slot"), py::arg("base_id"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("durability"), py::arg("bonus_strength") = 0, py::arg("bonus_hp_fixed") = 0,
	    "Engineering fixture only: recreate one real item identity, optionally add a stat-boundary affix; never use for effect evaluation");
	m.def("probe_resource_armor_plan", [](int inventoryIndex) {
		EnsureInGame("probe_resource_armor_plan");
		const Player &player = *MyPlayer;
		if (inventoryIndex < 0 || inventoryIndex >= player._pNumInv)
			throw std::invalid_argument("invalid engineering inventory index");
		auto serialize = [](const GearCombatProfile &profile) {
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
		};
		const auto empty = EmptyGearCombatBaseline(player);
		auto live = GearCombatProfileFromPlayer(player);
		MakeGearUtilityRelativeToEmpty(live, empty);
		std::array<Item, NUM_INVLOC> currentBody;
		std::copy(std::begin(player.InvBody), std::end(player.InvBody), currentBody.begin());
		auto simulatedCurrent = SimulateGearCombatProfile(player, currentBody);
		MakeGearUtilityRelativeToEmpty(simulatedCurrent, empty);
		const auto upgrade = PlanGearUpgrade(player, player.InvList[inventoryIndex]);
		const auto legal = PlanResourceArmor(player, player.InvList[inventoryIndex], inventoryIndex);
		py::dict result;
		result["live_profile"] = serialize(live);
		result["simulated_current_profile"] = serialize(simulatedCurrent);
		result["old_upgrade_valid"] = upgrade.valid;
		result["old_upgrade_target"] = upgrade.valid ? static_cast<int>(upgrade.target) : -1;
		result["old_upgrade_next_utility"] = upgrade.nextUtility;
		py::list cleared;
		for (int slot = 0; slot < NUM_INVLOC; ++slot)
			if (upgrade.clearSlots[slot]) cleared.append(slot);
		result["old_upgrade_clear_slots"] = cleared;
		if (legal.gear.valid) {
			for (int slot = 0; slot < NUM_INVLOC; ++slot)
				if (legal.gear.clearSlots[slot]) currentBody[slot].clear();
			currentBody[legal.gear.target] = legal.gear.candidate;
			auto next = SimulateGearCombatProfile(player, currentBody);
			MakeGearUtilityRelativeToEmpty(next, empty);
			result["simulated_candidate_profile"] = serialize(next);
		}
		return result;
	}, py::arg("inventory_index"), "Read-only engineering audit: old utility plan versus live and detached profiles; never an action");
	m.def("probe_resource_armor_item", [](const std::string &destination, int baseId, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int durability, int quality, int armorClass) {
		EnsureInGame("probe_resource_armor_item");
		if (baseId < 0 || static_cast<size_t>(baseId) >= AllItemsList.size()
		    || (destination != "smith" && destination != "inventory")
		    || quality < ITEM_QUALITY_NORMAL || quality > ITEM_QUALITY_UNIQUE)
			throw std::invalid_argument("invalid engineering item fixture");
		Item item = {};
		RecreateItem(*MyPlayer, item, static_cast<_item_indexes>(baseId), createInfo,
		    (uint32_t { seedHigh } << 16) | seedLow, 0, 0);
		if (!item.isEquipment() || durability < 0 || durability > item._iMaxDur)
			throw std::invalid_argument("invalid fixture durability or equipment");
		item._iDurability = static_cast<uint8_t>(durability);
		item._iMagical = static_cast<item_quality>(quality);
		if (armorClass != -1) {
			if (armorClass < AllItemsList[baseId].iMinAC || armorClass > AllItemsList[baseId].iMaxAC)
				throw std::invalid_argument("fixture AC outside real base-item range");
			item._iAC = armorClass;
		}
		item._iIdentified = true;
		item._iStatFlag = MyPlayer->CanUseItem(item);
		if (destination == "smith") {
			const int index = static_cast<int>(SmithItems.size());
			SmithItems.push_back(item);
			return index;
		}
		const int index = MyPlayer->_pNumInv;
		if (!AutoPlaceItemInInventory(*MyPlayer, item, false))
			throw std::runtime_error("engineering inventory fixture has no room");
		return index;
	}, py::arg("destination"), py::arg("base_id"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"), py::arg("durability"), py::arg("quality") = 0, py::arg("armor_class") = -1,
	    "Engineering fixture only: recreate a real item (optional AC within its native range) into Smith stock or inventory; never policy-effect evidence");
	m.def("probe_resource_identify_fixture", [](const std::string &destination, int baseId, uint16_t seedHigh, uint16_t seedLow, int level, int slot) {
		EnsureInGame("probe_resource_identify_fixture");
		if (baseId < 0 || static_cast<size_t>(baseId) >= AllItemsList.size()
		    || (destination != "inventory" && destination != "body")
		    || level < 1 || level > 30)
			throw std::invalid_argument("invalid identify fixture");
		// R18-H identify-v1 fixture: build the item through the ENGINE's own drop
		// path (SetupAllItems with onlygood/uper15), so the affixes, values and the
		// unidentified flag are the real ones a floor drop would carry. It is
		// refused unless the seed actually produced an unidentified magic item.
		Item item = {};
		SetupAllItems(*MyPlayer, item, static_cast<_item_indexes>(baseId),
		    (uint32_t { seedHigh } << 16) | seedLow, level, 15, true, false);
		if (item.isEmpty() || item._iMagical == ITEM_QUALITY_NORMAL || item._iIdentified)
			throw std::runtime_error("identify fixture seed produced no unidentified magic item");
		item._iStatFlag = MyPlayer->CanUseItem(item);
		if (destination == "body") {
			if (slot < 0 || slot >= NUM_INVLOC) throw std::invalid_argument("invalid fixture body slot");
			MyPlayer->InvBody[slot] = item;
			CalcPlrInv(*MyPlayer, true);
			return slot;
		}
		const int index = MyPlayer->_pNumInv;
		if (!AutoPlaceItemInInventory(*MyPlayer, item, false))
			throw std::runtime_error("identify fixture has no inventory room");
		return index;
	}, py::arg("destination"), py::arg("base_id"), py::arg("seed_hi"), py::arg("seed_lo"), py::arg("level") = 10, py::arg("slot") = -1,
	    "Engineering fixture only: generate a REAL unidentified magic item with the engine drop path into inventory or a body slot; never policy-effect evidence");
	m.def("probe_loot_carry_floor_item", [](int activeId) {
		EnsureInGame("probe_loot_carry_floor_item");
		bool active = false;
		for (int i = 0; i < ActiveItemCount; ++i) active = active || ActiveItems[i] == activeId;
		if (!gResourceLootEconomy || !active || activeId < 0 || activeId >= MAXITEMS || !Items[activeId].isEquipment())
			throw std::invalid_argument("invalid loot floor fixture");
		const Item item = Items[activeId];
		const int index = MyPlayer->_pNumInv;
		if (!AutoPlaceItemInInventory(*MyPlayer, item, false)) throw std::runtime_error("fixture inventory full");
		SyncGetItem(item.position, item._iSeed, item.IDidx, item._iCreateInfo);
		return index;
	}, py::arg("active_id"), "Engineering fixture only: carry an engineered floor item to test inventory upgrade protection; never a policy action");
	m.def("probe_loot_inventory_value", [](int index, int value) {
		EnsureInGame("probe_loot_inventory_value");
		if (!gResourceLootEconomy || index < 0 || index >= MyPlayer->_pNumInv || value < 1 || value > 1000000)
			throw std::invalid_argument("invalid loot value fixture");
		Item &item = MyPlayer->InvList[index];
		if (item._itype == ItemType::Gold && value > MaxGold)
			throw std::invalid_argument("fixture gold stack exceeds native maximum");
		item._ivalue = value;
		item._iIvalue = value;
		if (item._itype == ItemType::Gold) {
			SetPlrHandGoldCurs(item);
			MyPlayer->_pGold = CalculateGold(*MyPlayer);
		}
	}, py::arg("index"), py::arg("value"),
	    "Engineering fixture only: set one real inventory item's value for sale-capacity or low-cash boundaries; never used by resource actions");
	m.def("probe_resource_fill_inventory", [](int leaveFree) {
		EnsureInGame("probe_resource_fill_inventory");
		// R18-K2b review round (2026-09-07): leave_free lets a test build a
		// TIGHT pack (the regime where the counter and the equip path can
		// disagree about capacity) instead of only a full one. An ordinary
		// potion occupies exactly one cell, so the count is exact, and
		// leave_free = 0 -- the only value every existing caller passes -- is
		// the frozen fixture: fill every remaining cell.
		if (leaveFree < 0) throw std::invalid_argument("leave_free must not be negative");
		int freeCells = 0;
		for (const auto cell : MyPlayer->InvGrid) freeCells += cell == 0;
		for (int index = 0; index < freeCells - leaveFree; ++index) {
			Item item = {};
			GetItemAttrs(item, IDI_HEAL, 1);
			item._iSeed = 0xF1110000U + index;
			if (!AutoPlaceItemInInventory(*MyPlayer, item, false)) break;
		}
	}, py::arg("leave_free") = 0, "Engineering fixture only: fill remaining inventory slots with ordinary potions, keeping leave_free cells empty");
	m.def("probe_resource_inventory_snapshot", []() {
		EnsureInGame("probe_resource_inventory_snapshot");
		py::dict snapshot;
		py::list items, grid;
		for (int index = 0; index < MyPlayer->_pNumInv; ++index) {
			py::dict item;
			AppendItemCombatState(item, MyPlayer->InvList[index], 0);
			item["value"] = MyPlayer->InvList[index]._ivalue;
			item["identified_value"] = MyPlayer->InvList[index]._iIvalue;
			items.append(item);
		}
		for (const auto value : MyPlayer->InvGrid) grid.append(value);
		snapshot["items"] = items;
		snapshot["grid"] = grid;
		snapshot["gold"] = MyPlayer->_pGold;
		snapshot["held_empty"] = MyPlayer->HoldItem.isEmpty();
		return snapshot;
	}, "Engineering observation only: actual inventory placement and item fields for atomicity checks");
	m.def("probe_resource_set_belt_heals", [](int count) {
		EnsureInGame("probe_resource_set_belt_heals");
		if (count < 0 || count > MaxBeltItems) throw std::invalid_argument("count outside belt capacity");
		for (int index = 0; index < MaxBeltItems; ++index) {
			MyPlayer->SpdList[index].clear();
			if (index < count) {
				GetItemAttrs(MyPlayer->SpdList[index], IDI_HEAL, 1);
				MyPlayer->SpdList[index]._iSeed = 0xB3170000U + index;
			}
		}
		MyPlayer->CalcScrolls();
	}, py::arg("count"), "Test fixture only: replace belt with a specified number of ordinary healing potions");
	m.def("probe_resource_set_durability", [](int durability) {
		EnsureInGame("probe_resource_set_durability");
		if (durability < 0 || durability > 255) throw std::invalid_argument("durability outside uint8");
		for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
			Item &item = MyPlayer->InvBody[slot];
			if (!item.isEmpty()) item._iDurability = static_cast<uint8_t>(durability);
		}
	}, py::arg("durability"), "Test fixture only: set equipped durability for readiness boundary checks");
	m.def("probe_resource_set_hp_fixed", [](int hitPoints) {
		EnsureInGame("probe_resource_set_hp_fixed");
		if (hitPoints <= 0 || hitPoints > MyPlayer->_pMaxHP) throw std::invalid_argument("HP outside alive/max range");
		SetPlayerHitPoints(*MyPlayer, hitPoints);
	}, py::arg("hit_points"), "Test fixture only: set exact fixed-point HP for the 80 percent boundary");
	m.def("probe_resource_add_gold", [](int amount) {
		EnsureInGame("probe_resource_add_gold");
		if (amount <= 0 || amount > MaxGold) throw std::invalid_argument("one gold stack must be in [1,MaxGold]");
		Item gold = {};
		MakeGoldStack(gold, amount);
		if (!AutoPlaceItemInInventory(*MyPlayer, gold, false)) throw std::runtime_error("fixture gold does not fit");
		MyPlayer->_pGold = CalculateGold(*MyPlayer);
	}, py::arg("amount"), "Test fixture only: add a real inventory gold stack, never for policy evaluation");
	m.def("probe_resource_transition", [](int mode, int target) {
		EnsureInGame("probe_resource_transition");
		const auto request = static_cast<interface_mode>(mode);
		if (request == WM_DIABWARPLVL) StartWarpLvl(*MyPlayer, 0);
		else if (request == WM_DIABRETOWN) RestartTownLvl(*MyPlayer);
		else if (IsAnyOf(request, WM_DIABNEXTLVL, WM_DIABPREVLVL, WM_DIABTOWNWARP, WM_DIABTWARPUP, WM_DIABRTNLVL)) StartNewLvl(*MyPlayer, request, target);
		else throw std::invalid_argument("unsupported transition probe message");
	}, py::arg("mode"), py::arg("target"), "Test fixture only: request an engine transition without advancing a tick");
	m.def("probe_resource_town_portal_fixture", []() {
		EnsureInGame("probe_resource_town_portal_fixture");
		if (setlevel || currlevel != 0 || MyPlayer->_pmode != PM_STAND)
			throw std::runtime_error("portal fixture requires standing in town");
		Missile *portal = AddMissile(MyPlayer->position.tile, MyPlayer->position.tile,
		    Direction::South, MissileID::TownPortal, TARGET_MONSTERS, *MyPlayer, 0, 0);
		if (portal == nullptr) throw std::runtime_error("portal fixture could not allocate a missile");
		MyPlayer->walkpath[0] = 1; // Non-empty queued path makes ClrPlrPath observable.
	}, "Test fixture only: create an actual portal under the town player with a pending path");
	m.def("probe_resource_process_town_portal", []() {
		EnsureInGame("probe_resource_process_town_portal");
		for (Missile &missile : Missiles) {
			if (missile._mitype == MissileID::TownPortal && missile.position.tile == MyPlayer->position.tile) {
				ProcessTownPortal(missile);
				return;
			}
		}
		throw std::runtime_error("portal fixture is not under player");
	}, "Test fixture only: execute the actual portal collision handler without a game tick");
	m.def("probe_resource_add_portal_scroll", [](bool belt) {
		EnsureInGame("probe_resource_add_portal_scroll");
		// GetItemAttrs at item level 1 is exactly how SpawnWitch builds the
		// pinned scroll (items.cpp), and it draws no RNG, so a fixture scroll
		// does not shift the gameplay stream the way a purchase would.
		Item scroll = {};
		GetItemAttrs(scroll, IDI_PORTAL, 1);
		scroll._iCreateInfo = 1;
		scroll._iIdentified = true;
		scroll._iStatFlag = MyPlayer->CanUseItem(scroll);
		const bool placed = belt ? AutoPlaceItemInBelt(*MyPlayer, scroll, true, false)
		                         : AutoPlaceItemInInventory(*MyPlayer, scroll, false);
		if (!placed) throw std::runtime_error("fixture portal scroll does not fit");
		MyPlayer->CalcScrolls();
	}, py::arg("belt") = false,
	    "Test fixture only: place one real Scroll of Town Portal without a purchase");
	m.def("probe_resource_portal_fixture", [](int level, int x, int y) {
		EnsureInGame("probe_resource_portal_fixture");
		if (level < 0 || level > 16) throw std::invalid_argument("fixture portal level out of range");
		SetPortalStats(MyPlayerId, level > 0, Point { x, y }, level, DTYPE_CATHEDRAL, false);
	}, py::arg("level"), py::arg("x") = 0, py::arg("y") = 0,
	    "Test fixture only: set this player's Portals[] record (level 0 closes it) to exercise the WM_DIABWARPLVL guard branch without a real cast");
	m.def("probe_resource_set_death_ui_flag", [](bool dead) {
		EnsureInGame("probe_resource_set_death_ui_flag");
		MyPlayerIsDead = dead;
	}, py::arg("dead"), "Test fixture only: set the death UI flag to check restart rejection atomicity");
	m.def("probe_resource_queue_restart_town", []() {
		EnsureInGame("probe_resource_queue_restart_town");
		NetSendCmd(true, CMD_RETOWN);
	}, "Test fixture only: queue the normal restart-town network message");
	m.def("probe_resource_process_game_packets", []() {
		EnsureInGame("probe_resource_process_game_packets");
		ProcessGameMessagePackets();
	}, "Test fixture only: drain normal game packets to expose a forbidden queued warp");
	m.def("probe_resource_transition_snapshot", []() {
		EnsureInGame("probe_resource_transition_snapshot");
		py::dict result;
		const Player &player = *MyPlayer;
		result["player_level"] = player.plrlevel;
		result["level"] = currlevel;
		result["set_level"] = setlevel;
		result["set_level_type"] = static_cast<int>(setlvltype);
		result["mode"] = static_cast<int>(player._pmode);
		result["changing"] = player._pLvlChanging;
		result["invincible"] = player._pInvincible;
		result["mana_shield"] = player.pManaShield;
		result["my_player_dead"] = MyPlayerIsDead;
		result["hp_fixed"] = player._pHitPoints;
		result["mana_fixed"] = player._pMana;
		result["rng"] = GetLCGEngineState();
		result["town_warp_from"] = TWarpFrom;
		result["missiles"] = Missiles.size();
		py::list path, occupancy;
		for (const auto value : player.walkpath) path.append(value);
		for (const auto &row : dPlayer) for (const auto value : row) occupancy.append(value);
		result["walkpath"] = path;
		result["player_occupancy"] = occupancy;
		return result;
	}, "Test fixture only: gameplay state touched by InitLevelChange");
	m.def("act_wait", &ActWait,
	    "Cancel leftover pathing/target actions and wait in place; attacks stop immediately, a committed single-tile move may finish naturally; returns 0/1");
	m.def("act_walk", &ActWalk, py::arg("x"), py::arg("y"), "Path toward the target tile (injected through the network command layer)");
	m.def("act_explore_walk", &ActExploreWalk,
	    py::arg("x"), py::arg("y"), py::arg("protected_tiles"),
	    py::arg("center_x"), py::arg("center_y"), py::arg("radius"),
	    "Controller-only adjacent step: installs a one-step path inside the fixed snapshot window that avoids protected tiles; returns 0/1");
	m.def("act_attack_monster", &ActAttackMonster, py::arg("monster_id"), "Chase and melee the given monster");
	m.def("act_controller_attack_monster", &ActControllerAttackMonster,
	    py::arg("monster_id"), py::arg("center_x"),
	    py::arg("center_y"), py::arg("radius"),
	    "Fixed control snapshot only: swings only at adjacent monsters inside the window; refuses native long-range chasing");
	m.def("act_attack_tile", &ActAttackTile, py::arg("x"), py::arg("y"), "Swing at the target tile in place");
	m.def("act_operate", &ActOperate, py::arg("x"), py::arg("y"), "Operate the object on the target tile (open doors etc.; the engine walks there automatically)");
	m.def("act_controller_operate", &ActControllerOperate,
	    py::arg("x"), py::arg("y"), py::arg("center_x"),
	    py::arg("center_y"), py::arg("radius"),
	    "Fixed control snapshot only: operates only adjacent objects inside the window; refuses native long-range pathing");
	m.def("act_drink", &ActDrink, "Drink the first instant heal potion in the belt; returns the pre-press potion count only after the heal/consumption is confirmed, otherwise 0");
	m.def("act_pickup", &ActPickup,
	    "Pick up only the first heal potion on the player's future tile; does no pathing; returns 0/1");
	m.def("act_pickup_at", &ActPickupAt,
	    py::arg("active_item_id"), py::arg("x"), py::arg("y"),
	    py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"),
	    py::arg("base_id"),
	    "Pick up the instant heal potion whose identity matches the fixed snapshot exactly, only when the player's future tile is exactly that tile; returns 0/1");
	m.def("act_pickup_gear", &ActPickupGear,
	    "Atomically equip only the first safe upgrade on the player's future tile; does no pathing; returns 0/1");
	m.def("act_pickup_gear_at", &ActPickupGearAt,
	    py::arg("active_item_id"), py::arg("x"), py::arg("y"),
	    py::arg("seed_hi"), py::arg("seed_lo"), py::arg("create_info"),
	    py::arg("base_id"),
	    "Atomically equip the item whose identity matches the fixed snapshot exactly, only when the player's future tile is exactly that tile; returns 0/1");
	m.def("act_pickup_progression", &ActPickupProgression, py::arg("x"), py::arg("y"),
	    "Pick up the given Staff of Lazarus only when the player's future tile is exactly that tile; used by actions 10/11");
	m.def("end_game", &EndGame, "End the current game (reset calls it automatically)");
	m.def("episode_generation", []() {
		EnsureEngineProcess("episode_generation");
		return gEpisodeGeneration;
	},
	    "Generation number of the current native game state (changes on reset/end_game; used for cache safety checks)");

	// ---- Probe-only interface (only for pre-launch probes/post-mortems; training and evaluation must not call it) ----
	m.def("probe_add_experience", [](uint32_t xp) {
		EnsureInGame("probe_add_experience");
		// Inject experience directly (level difference counted as 0), triggering the engine's native level-up chain
		// (NextPlrLevel -> _pStatPts accumulates -> AutoSpendStatPoints at the end of Step)
		MyPlayer->addExperience(xp);
	}, py::arg("xp"), "Probe: inject experience, following the native level-up path");
	m.def("probe_modify_vit", [](int d) {
		EnsureInGame("probe_modify_vit");
		ModifyPlrVit(*MyPlayer, d);
	},
	    py::arg("d"), "Probe: adjust vitality directly (capped, HP synced)");
	m.def("probe_bonus_ac", [](int d) {
		EnsureInGame("probe_bonus_ac");
		MyPlayer->_pIBonusAC += d;
	},
	    py::arg("d"), "Probe: add temporary AC (CalcPlrInv recomputes it; stable if gear is not changed in combat)");
	m.def("probe_invincible", [](bool enabled) {
		EnsureInGame("probe_invincible");
		MyPlayer->_pInvincible = enabled;
	}, py::arg("enabled"), "Probe: toggle invincibility, only for real resource story/pathing acceptance checks");
	m.def("probe_stat_pts", []() {
		EnsureInGame("probe_stat_pts");
		return (int)MyPlayer->_pStatPts;
	},
	    "Probe: read unspent stat points (always 0 after auto-spending)");
	m.def("probe_stats", []() {
		EnsureInGame("probe_stats");
		py::dict d;
		d["vit"] = (int)MyPlayer->_pVitality;
		d["str"] = (int)MyPlayer->_pStrength;
		d["max_hp"] = (int)(MyPlayer->_pMaxHP >> 6);
		return d;
	}, "Probe: read stat details");
	m.def("probe_inventory_item_count", []() {
		EnsureInGame("probe_inventory_item_count");
		return static_cast<int>(MyPlayer->_pNumInv);
	}, "Probe: read the backpack item count, verifying that the atomic gear swap produces no hidden backpack fallback");
	m.def("probe_kill_monster", [](unsigned monsterId) {
		EnsureInGame("probe_kill_monster");
		if (monsterId >= MaxMonsters)
			throw std::out_of_range("probe_kill_monster monster id out of range");
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
			    "probe_kill_monster only accepts a currently active, living monster");
		const uint64_t before = MonsterKillTotal();
		const int monsterType = static_cast<int>(monster.type().type);
		M_StartKill(monster, *MyPlayer);
		const uint64_t after = MonsterKillTotal();
		if (before == std::numeric_limits<uint64_t>::max()
		    || after != before + 1)
			throw std::logic_error(
			    "native MonsterDeath did not increment monster_kill_total by exactly one");
		py::dict result;
		result["monster_type"] = monsterType;
		result["before"] = before;
		result["after"] = after;
		return result;
	}, py::arg("monster_id"),
	    "Probe: kill an active monster through native M_StartKill and verify that the cumulative kill fact rises by exactly +1");
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
	}, "Probe: read the conservative combat score/hard-gate facts of the whole gear set after CalcPlrItemVals");
	m.def("probe_set_current_hit_points", [](int hitPoints) {
		EnsureInGame("probe_set_current_hit_points");
		const int maxHitPoints = MyPlayer->_pMaxHP >> 6;
		if (hitPoints <= 0 || hitPoints > maxHitPoints)
			throw std::invalid_argument(
			    "probe_set_current_hit_points must be within [1,max_hp]");
		SetPlayerHitPoints(*MyPlayer, hitPoints << 6);
		return MyPlayer->_pHitPoints;
	}, py::arg("hit_points"),
	    "Probe: set the player's current whole-point life, for projected-HP safety-gate regression tests of gear swaps");
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
			throw std::invalid_argument("probe_spawn_test_gear base_id unavailable");
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
				throw std::invalid_argument("probe_spawn_test_gear value out of range");
		Item item;
		InitializeItem(item, static_cast<_item_indexes>(baseId));
		if (!item.isEquipment() && baseId != IDI_LAZSTAFF)
			throw std::invalid_argument("probe_spawn_test_gear accepts only equipment");
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
			throw std::runtime_error("probe_spawn_test_gear: no free floor tile within two tiles of the player");
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
	    "Probe: spawn deterministic gear within two tiles of the player, for whole-set OR/cap/attack-speed/curse/durability regression tests");

	m.def("local_map", [](int radius) {
		EnsureInGame("local_map");
		const int maxRadius = std::max(static_cast<int>(MAXDUNX), static_cast<int>(MAXDUNY));
		if (radius < 0 || radius > maxRadius)
			throw std::invalid_argument("radius must be within [0, max(MAXDUNX, MAXDUNY)]");
		// Local (2r+1)^2 map centred on the player: walkability + monster occupancy + closed doors
		// (a single C++ call, avoiding the overhead of per-tile probes).
		// Note: the observation vector consumes only the walkable/monster channels; the door channel is only for in-macro navigation
		// and does not change the 286-dim observation, so old models and the leaderboard stay fully compatible
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
				// Closed door: a door object is present and the tile is currently not walkable. Note that _oSolidFlag cannot be used:
				// in the engine a door blocks by swapping the door tile's piece for a solid one (nSolidTable); the door object itself is not solid.
				// Blocking barrel: solid but breakable (operate smashes it in one hit and the tile becomes walkable). On seed 9005 the
				// stairs were once sealed by "a barrel behind a door"; passability planning must recognize both kinds of "soft wall"
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
				// action10 may prefer opening doors on the way while ordinary frontier remains, but must not treat every barrel in the room
				// as a high-priority exit; keep the old planning definition door = door or barrel,
					// and give exploration a separate exact channel containing only interactable closed doors.
					closedDoorOnly.append(
					    (closedDoor && danger.object != nullptr
					        && danger.object->canInteractWith())
					        ? 1
					        : 0);
					// Only publish active damage phases the player has already lit; undiscovered traps still clearly belong to a
					// partially observable environment. The controller treats the tile as a hard wall to avoid stepping into fire automatically.
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
	}, py::arg("radius") = 5, "Local map channels centred on the player");

	m.def("probe_asset", [](const std::string &path) {
		EnsureEngineProcess("probe_asset");
		size_t size = 0;
		AssetHandle handle = OpenAsset(std::string_view(path), size);
		py::dict d;
		d["ok"] = handle.ok();
		d["size"] = static_cast<uint64_t>(size);
		return d;
	}, py::arg("path"), "Debug: check whether an asset can be opened, and its size");

	m.def("probe_tile", [](int x, int y) {
		EnsureInGame("probe_tile");
		if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY)
			throw std::out_of_range("probe_tile coordinate out of range");
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
	    "Debug: read one tile's occupancy/collision/object and dynamic damage/explosion facts");

	m.def("probe_is_spawn", []() {
		EnsureEngineProcess("probe_is_spawn");
		return gbIsSpawn;
	},
	    "Probe: whether the shareware spawn.mpq is in use");
	m.def("probe_warp_main_level", [](int level) {
		EnsureInGame("probe_warp_main_level");
		if (setlevel || level < 1 || level > 16)
			throw std::invalid_argument("probe main level must be within [1,16] and not currently in a set-level");
		StartNewLvl(*MyPlayer, WM_DIABNEXTLVL, level);
	}, py::arg("level"), "Probe: queue a switch to the given main dungeon level");
	m.def("probe_enter_set_level", [](int level) {
		EnsureInGame("probe_enter_set_level");
		if (setlevel)
			throw std::runtime_error("probe must be in the main dungeon before entering a set-level");
		const auto setLevel = static_cast<_setlevels>(level);
		dungeon_type requestedType;
		switch (setLevel) {
		case SL_SKELKING:
			requestedType = Quests[Q_SKELKING]._qlvltype;
			break;
		case SL_BONECHAMB:
			requestedType = Quests[Q_SCHAMB]._qlvltype;
			break;
		case SL_POISONWATER:
			requestedType = Quests[Q_PWATER]._qlvltype;
			break;
		case SL_VILEBETRAYER:
			requestedType = Quests[Q_BETRAYER]._qlvltype;
			break;
		default:
			throw std::invalid_argument("probe supports only the four official quest set-levels");
		}
		if (!IsPlayerLevelTransitionAllowed(*MyPlayer, WM_DIABSETLVL, level))
			return;
		setlvltype = requestedType;
		StartNewLvl(*MyPlayer, WM_DIABSETLVL, level);
	}, py::arg("level"), "Probe: queue entry into an official quest set-level (1/2/4/5)");
	m.def("probe_return_set_level", []() {
		EnsureInGame("probe_return_set_level");
		if (!setlevel)
			throw std::runtime_error("probe must currently be in a set-level before returning");
		StartNewLvl(*MyPlayer, WM_DIABRTNLVL, GetMapReturnLevel());
	}, "Probe: queue a return from a quest set-level to its main dungeon level");

	// Trigger message-type constants (the values of the observation's triggers[].msg)
	m.attr("WM_DIABNEXTLVL") = static_cast<int>(WM_DIABNEXTLVL);
	m.attr("WM_DIABWARPLVL") = static_cast<int>(WM_DIABWARPLVL);
	m.attr("WM_DIABRETOWN") = static_cast<int>(WM_DIABRETOWN);
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
