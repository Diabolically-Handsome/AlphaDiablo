// Included inside the bridge's private namespace after the gear helpers.
// Opt-in L2 curriculum: the native player transition hook is the final gate.

// Ordinary-armor recipes opt in; old one-argument configuration stays chest-only.
bool gResourceOrdinaryArmorScope = false;
bool gResourcePreserveEquipmentReadiness = false;
// R17.1 chairman ruling 3 (2026-09-06): "coach-v03" law. When set, the L1->L2
// transition guard only RECORDS the readiness verdict (advisory) instead of
// vetoing, the deeper-floor wall is lifted (Python per-floor tables govern L3+),
// and the receipt carries the six-condition law verdict (health excluded).
bool gResourceReadinessAdvisory = false;
using ResourceItemIdentity = std::array<int, 4>;
std::vector<ResourceItemIdentity> gResourceSeenSmithItems;
uint64_t gResourceSeenSmithGeneration = 0;
uint32_t gResourceSeenSmithTownSeed = 0;

ResourceItemIdentity ResourceIdentity(const Item &item)
{
	return { HighWord(item._iSeed), LowWord(item._iSeed), item._iCreateInfo, static_cast<int>(item.IDidx) };
}

bool gTownServiceAuthorized = false;
bool gTownServiceTrip = false;
int gResourceServiceTripsStarted = 0;
constexpr int MaxLootServiceTrips = 2;
// R18-A retreat-v1 (2026-09-06): a Python-authorized one-floor ascent from
// main L2+ (the return-to-town interface). Each completed retreat earns one
// extra loot-economy town trip. Off = every frozen verdict byte for byte.
bool gResourceRetreatEnabled = false;
bool gRetreatAuthorized = false;
int gResourceRetreatsStarted = 0;
constexpr int MaxResourceRetreats = 3;
std::vector<ResourceItemIdentity> gResourceRetainedGear;
std::vector<ResourceItemIdentity> gResourceLastRetainedGear;
int gResourceLastRetentionGold = 0;
bool gResourceVisitedL1 = false;
int gResourceMaxDepth = 0;
// A separate stream retains normal town restocking without consuming gameplay RNG.
constexpr int RequiredResourceBeltHeals = 4;
constexpr uint32_t TownRestockSeedDomain = 0x74574E31U; // "tWN1"
std::mt19937 gTownRestockRng;
uint64_t gTownRestockSequence = 0;
struct ResourceTransitionReceipt {
	uint64_t sequence = 0;
	bool accepted = false;
	bool ready = false;
	bool readyLaw = false; // six-condition law (health excluded), R17.1 ruling 3
	bool sourceIsSet = false;
	bool targetIsSet = false;
	int source = 0;
	int target = 0;
	int message = 0;
	std::string reason = "none";
};
ResourceTransitionReceipt gResourceTransition;

struct ResourceReadiness {
	int level = 0;
	int armor = 0;
	int damage = 0;
	int hp = 0;
	int maxHp = 0;
	int heals = 0;
	int minimumDurability = DUR_INDESTRUCTIBLE;
	bool weapon = false;
	std::vector<std::string> failures;
	bool ready() const { return failures.empty(); }
	// Six-condition readiness law (R17.1 ruling 3): clvl/AC/dmg/belt/durability/weapon,
	// HP deliberately excluded (HP belongs to the drink reflex and the town-trip trigger).
	bool readyExcludingHealth() const
	{
		for (const std::string &failure : failures)
			if (failure != "health") return false;
		return true;
	}
};

ResourceReadiness EvaluateResourceReadiness(const Player &player)
{
	ResourceReadiness result;
	result.level = player.getCharacterLevel();
	result.armor = player.GetArmor();
	result.damage = player._pIMaxDam + player._pDamageMod;
	result.hp = player._pHitPoints;
	result.maxHp = player._pMaxHP;
	for (const Item &item : player.SpdList)
		result.heals += InstantHealKind(item) != 0;
	for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
		const Item &item = player.InvBody[slot];
		if (item.isEmpty())
			continue;
		if (item.isWeapon() && item._iStatFlag && item._iDurability != 0)
			result.weapon = true;
		if (item._iMaxDur != DUR_INDESTRUCTIBLE)
			result.minimumDurability = std::min(result.minimumDurability, static_cast<int>(item._iDurability));
	}
	if (result.level < 2) result.failures.emplace_back("level");
	if (result.armor < 9) result.failures.emplace_back("armor");
	if (result.damage < 6) result.failures.emplace_back("damage");
	if (result.maxHp <= 0 || int64_t { 5 } * result.hp < int64_t { 4 } * result.maxHp) result.failures.emplace_back("health");
	if (result.heals < RequiredResourceBeltHeals) result.failures.emplace_back("potions");
	if (result.minimumDurability < 15) result.failures.emplace_back("durability");
	if (!result.weapon) result.failures.emplace_back("weapon");
	return result;
}

bool EquipmentReadyForPreservation(const Player &player)
{
	// Preserve the old path without evaluating or exposing any new state.
	if (!gResourceProtocol || !gResourcePreserveEquipmentReadiness)
		return false;
	const ResourceReadiness readiness = EvaluateResourceReadiness(player);
	for (const std::string &failure : readiness.failures) {
		if (failure == "armor" || failure == "damage"
		    || failure == "weapon" || failure == "durability")
			return false;
	}
	return true;
}

bool ResourceTransitionGuard(const Player &player, interface_mode mode, int target)
{
	if (!gResourceProtocol || &player != MyPlayer)
		return true;
	const int source = ConceptualDungeonDepth();
	const ResourceReadiness readiness = EvaluateResourceReadiness(player);
	bool accepted = false;
	bool targetIsSet = false;
	std::string reason = "unauthorized_transition";
	if (player.hasNoLife() || player._pmode == PM_DEATH) {
		reason = "dead";
	} else if (player._pLvlChanging || player._pmode == PM_NEWLVL) {
		reason = "transition_pending";
	} else if (mode == WM_DIABWARPLVL || mode == WM_DIABRETOWN || mode == WM_DIABTWARPUP || mode == WM_DIABTOWNWARP) {
		reason = "unsupported_portal_or_warp";
	} else if (mode == WM_DIABSETLVL) {
		// Quest entry is not descent; it must not be subject to L2 readiness.
		targetIsSet = true;
		target = source;
		accepted = source > 0 && source <= 2;
		reason = accepted ? "quest_entry" : "curriculum_boundary";
	} else if (mode == WM_DIABRTNLVL && setlevel) {
		accepted = target == GetMapReturnLevel() && target > 0 && target <= 2;
		reason = accepted ? "quest_return" : "curriculum_boundary";
	} else if (!setlevel && mode == WM_DIABPREVLVL && source >= 2 && target == source - 1) {
		// R18-A retreat-v1: a Python-authorized one-floor ascent. With the
		// feature off this stays the old default verdict (unauthorized_transition).
		accepted = gResourceRetreatEnabled && gRetreatAuthorized;
		reason = accepted ? "retreat_ascent"
		                  : (gResourceRetreatEnabled ? "retreat_not_authorized" : "unauthorized_transition");
	} else if (target > 2) {
		// coach-v03: deeper floors are governed by the Python per-floor tables;
		// the engine only records the transition.
		accepted = gResourceReadinessAdvisory;
		reason = accepted ? "deeper_advisory" : "curriculum_boundary";
	} else if (!setlevel && mode == WM_DIABPREVLVL && source == 1 && target == 0) {
		accepted = gTownServiceAuthorized && (!gResourceLootEconomy || gResourceServiceTripsStarted < MaxLootServiceTrips + gResourceRetreatsStarted);
		reason = accepted ? "town_service_departure" : (gResourceLootEconomy && gResourceServiceTripsStarted >= MaxLootServiceTrips + gResourceRetreatsStarted ? "town_service_trip_limit" : "town_service_not_authorized");
	} else if (!setlevel && mode == WM_DIABNEXTLVL && source == 0 && target == 1) {
		accepted = !gResourceVisitedL1 || gTownServiceTrip;
		reason = accepted ? (gResourceVisitedL1 ? "town_service_return" : "initial_dungeon_entry") : "town_service_not_authorized";
	} else if (!setlevel && mode == WM_DIABNEXTLVL && source == 1 && target == 2) {
		if (gResourceReadinessAdvisory) {
			// Advisory: never veto; the receipt records whether the six-condition
			// law held (forced-unready descents are paid no escrow on the Python side).
			accepted = true;
			reason = readiness.readyExcludingHealth() ? "ready" : "forced_unready";
		} else {
			accepted = readiness.ready();
			reason = accepted ? "ready" : "not_ready";
		}
	}
	++gResourceTransition.sequence;
	gResourceTransition.accepted = accepted;
	gResourceTransition.ready = readiness.ready();
	gResourceTransition.readyLaw = readiness.readyExcludingHealth();
	gResourceTransition.sourceIsSet = setlevel;
	gResourceTransition.targetIsSet = targetIsSet;
	gResourceTransition.source = source;
	gResourceTransition.target = target;
	gResourceTransition.message = static_cast<int>(mode);
	gResourceTransition.reason = reason;
	return accepted;
}

void ConfigureResourceProtocol(bool enabled, bool ordinaryArmorScope = false,
    bool preserveEquipmentReadiness = false, bool lootEconomy = false,
    bool readinessAdvisory = false, bool retreat = false)
{
	EnsureEngineProcess("configure_resource_protocol");
	if (ordinaryArmorScope && !enabled)
		throw std::invalid_argument("ordinary armor scope requires l2-town-v1");
	if (preserveEquipmentReadiness && !enabled)
		throw std::invalid_argument("equipment readiness preservation requires l2-town-v1");
	if (lootEconomy && (!enabled || !ordinaryArmorScope))
		throw std::invalid_argument("loot economy requires resource protocol and ordinary armor scope");
	if (readinessAdvisory && !enabled)
		throw std::invalid_argument("readiness advisory (coach-v03) requires l2-town-v1");
	if (retreat && (!enabled || !readinessAdvisory))
		throw std::invalid_argument("retreat-v1 requires l2-town-v1 under coach-v03");
	if (gInGame && (enabled != gResourceProtocol || ordinaryArmorScope != gResourceOrdinaryArmorScope
	                  || preserveEquipmentReadiness != gResourcePreserveEquipmentReadiness
	                  || lootEconomy != gResourceLootEconomy
	                  || readinessAdvisory != gResourceReadinessAdvisory
	                  || retreat != gResourceRetreatEnabled))
		throw std::runtime_error("resource protocol may change only between episodes");
	if (gResourceOrdinaryArmorScope && !ordinaryArmorScope) gResourceSeenSmithItems.clear();
	gResourceProtocol = enabled;
	gResourceOrdinaryArmorScope = ordinaryArmorScope;
	gResourcePreserveEquipmentReadiness = preserveEquipmentReadiness;
	gResourceLootEconomy = lootEconomy;
	gResourceReadinessAdvisory = readinessAdvisory;
	gResourceRetreatEnabled = retreat;
	LevelTransitionGuard = enabled ? ResourceTransitionGuard : nullptr;
	DisableLevelBacktracking = !enabled;
}

void ResetResourceEpisode(uint32_t episodeSeed)
{
	if (gResourceOrdinaryArmorScope) gResourceSeenSmithItems.clear();
	gTownRestockRng.seed(episodeSeed ^ TownRestockSeedDomain);
	gTownRestockSequence = 0;
	gTownServiceAuthorized = false;
	gTownServiceTrip = false;
	gResourceServiceTripsStarted = 0;
	gRetreatAuthorized = false;
	gResourceRetreatsStarted = 0;
	gResourceRetainedGear.clear();
	gResourceLastRetainedGear.clear();
	gResourceLastRetentionGold = 0;
	gResourceVisitedL1 = false;
	gResourceMaxDepth = 0;
	gResourceTransition = {};
	LevelTransitionGuard = gResourceProtocol ? ResourceTransitionGuard : nullptr;
	DisableLevelBacktracking = !gResourceProtocol;
}

void SaveLevelForTransition()
{
	const bool restockTown = gResourceProtocol && !setlevel && leveltype == DTYPE_TOWN;
	// SaveLevel normally refreshes DungeonSeeds[0] from the wall-clock-seeded
	// Xoshiro generator. Reset only owns the gameplay LCG, so the first
	// town -> L1 save otherwise makes a later town visit irreproducible.
	// Keep the real save and normal per-departure refresh. Override only the
	// new protocol's town seed after a successful save, leaving legacy mode
	// and the dungeon/combat RNG untouched. Failed saves consume no sequence.
	pfile_save_level();
	if (restockTown) {
		if (gResourceOrdinaryArmorScope) gResourceSeenSmithItems.clear();
		DungeonSeeds[0] = static_cast<uint32_t>(gTownRestockRng());
		++gTownRestockSequence;
	}
}

void ResourceAfterLoad()
{
	if (!gResourceProtocol)
		return;
	// Gym creates a fresh hero each episode. Shared stash wealth is not part
	// of this protocol and may never subsidize a purchase across episodes.
	if (!gInGame)
		Stash = {};
	if (Stash.gold != 0)
		throw std::runtime_error("l2-town-v1 forbids shared-stash gold");
	if (!setlevel) {
		const int depth = static_cast<int>(currlevel);
		gResourceMaxDepth = std::max(gResourceMaxDepth, depth);
		if (gRetreatAuthorized && gResourceTransition.accepted
		    && gResourceTransition.reason == "retreat_ascent" && depth == gResourceTransition.target) {
			gRetreatAuthorized = false;
			++gResourceRetreatsStarted;
		}
		if (depth == 0 && !gTownServiceTrip && gResourceTransition.accepted && gResourceTransition.reason == "town_service_departure") {
			gTownServiceTrip = true;
			if (gResourceLootEconomy) ++gResourceServiceTripsStarted;
		}
		if (depth == 1) {
			gResourceVisitedL1 = true;
			if (gTownServiceTrip) {
				gTownServiceTrip = false;
				gTownServiceAuthorized = false;
			}
		}
	}
}

void ConfigureTownService(bool authorized)
{
	EnsureInGame("configure_town_service");
	if (!gResourceProtocol)
		throw std::runtime_error("town service requires l2-town-v1");
	if (authorized && !gTownServiceTrip && (setlevel || currlevel != 1))
		throw std::runtime_error("town service can depart only from main L1");
	if (gResourceLootEconomy && authorized && !gTownServiceTrip && gResourceServiceTripsStarted >= MaxLootServiceTrips + gResourceRetreatsStarted)
		throw std::runtime_error("loot economy allows at most two town service trips per episode");
	if (gResourceOrdinaryArmorScope && authorized && !gTownServiceTrip) gResourceSeenSmithItems.clear();
	gTownServiceAuthorized = authorized;
}

void ConfigureRetreat(bool authorized)
{
	EnsureInGame("configure_retreat");
	if (!gResourceProtocol || !gResourceRetreatEnabled)
		throw std::runtime_error("retreat requires l2-town-v1 with retreat-v1 enabled");
	if (authorized && (setlevel || currlevel < 2))
		throw std::runtime_error("retreat can depart only from main L2 or deeper");
	if (authorized && gResourceRetreatsStarted >= MaxResourceRetreats)
		throw std::runtime_error("retreat-v1 allows at most three retreats per episode");
	gRetreatAuthorized = authorized;
}

const char *ResourceVendorName(TalkID store)
{
	switch (store) {
	case TalkID::Smith: case TalkID::SmithBuy: return "smith";
	case TalkID::Healer: case TalkID::HealerBuy: return "healer";
	default: return "none";
	}
}

bool ResourceTownActionAllowed(const char *operation)
{
	return CanAcceptPlayerAction(operation) && gResourceProtocol
	    && !setlevel && currlevel == 0 && gTownServiceTrip
	    && !MyPlayer->hasNoLife() && MyPlayer->HoldItem.isEmpty();
}

bool IsOrdinaryResourceArmor(const Item &item)
{
	return !item.isEmpty() && item._iClass != ICLASS_QUEST
	    && item._iMagical == ITEM_QUALITY_NORMAL
	    && (item.isArmor() || item.isHelm() || item.isShield());
}

void AppendOrdinaryResourceArmorState(py::dict &entry, const Item &item, int inventoryIndex);

py::dict ResourceItemState(const Item &source, int inventoryIndex = -1)
{
	Item item = source;
	item._iStatFlag = MyPlayer->CanUseItem(item);
	const GearUpgradePlan plan = PlanGearUpgrade(*MyPlayer, item);
	py::dict entry;
	entry["seed_hi"] = HighWord(item._iSeed);
	entry["seed_lo"] = LowWord(item._iSeed);
	entry["create_info"] = item._iCreateInfo;
	entry["base_id"] = static_cast<int>(item.IDidx);
	entry["price"] = item._iIvalue;
	entry["heal_kind"] = InstantHealKind(item);
	entry["armor_class"] = item._iAC;
	entry["can_use"] = MyPlayer->CanUseItem(item);
	entry["can_fit"] = StoreAutoPlace(item, false);
	entry["is_armor"] = item.isArmor() && item._iMagical == ITEM_QUALITY_NORMAL;
	entry["durable"] = item._iMaxDur == DUR_INDESTRUCTIBLE || item._iDurability >= 15;
	entry["projected_armor_class"] = plan.valid ? plan.nextArmorClass : MyPlayer->GetArmor();
	entry["upgrade"] = plan.valid;
	entry["meets_armor_gate"] = item.isArmor() && item._iMagical == ITEM_QUALITY_NORMAL
	    && MyPlayer->CanUseItem(item) && (item._iMaxDur == DUR_INDESTRUCTIBLE || item._iDurability >= 15)
	    && plan.valid && plan.target == INVLOC_CHEST && plan.nextArmorClass >= 9;
	if (gResourceOrdinaryArmorScope)
		AppendOrdinaryResourceArmorState(entry, item, inventoryIndex);
	return entry;
}

py::dict ResourceReadinessState(const Player &player)
{
	const ResourceReadiness state = EvaluateResourceReadiness(player);
	py::dict ready;
	bool repairNeededForGate = false;
	for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
		const Item &item = player.InvBody[slot];
		repairNeededForGate = repairNeededForGate || (!item.isEmpty()
		    && item._iMaxDur != DUR_INDESTRUCTIBLE && item._iDurability < 15 && item._iMaxDur >= 15);
	}
	ready["repair_service_needed"] = repairNeededForGate;
	ready["required_belt_heals"] = RequiredResourceBeltHeals;
	ready["ready"] = state.ready();
	ready["ready_excluding_health"] = state.readyExcludingHealth();
	ready["target_main_depth"] = 2;
	ready["failures"] = state.failures;
	ready["clvl"] = state.level;
	ready["armor_class"] = state.armor;
	ready["damage"] = state.damage;
	ready["hp"] = state.hp >> 6;
	ready["max_hp"] = state.maxHp >> 6;
	ready["hp_fixed"] = state.hp;
	ready["max_hp_fixed"] = state.maxHp;
	ready["belt_heals"] = state.heals;
	ready["belt_free_slots"] = std::count_if(std::begin(player.SpdList), std::end(player.SpdList),
	    [](const Item &item) { return item.isEmpty(); });
	ready["belt_capacity"] = MaxBeltItems;
	const Item &chest = player.InvBody[INVLOC_CHEST];
	ready["armor_service_needed"] = state.armor < 9
	    || (!chest.isEmpty() && chest._iMaxDur != DUR_INDESTRUCTIBLE && chest._iDurability < 15);
	ready["min_finite_durability"] = state.minimumDurability;
	ready["weapon_equipped"] = state.weapon;
	return ready;
}

// The same normal shift-click rule as CheckInvCut: place the intact item into
// the inventory, remove its body slot, and recompute with CalcPlrInv. Preview
// uses an inactive detached player so no world, RNG or network state is touched.
Player ResourceUnequipSimulation(const Player &source)
{
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
	std::copy(std::begin(source._pSplLvl), std::end(source._pSplLvl), std::begin(simulated._pSplLvl));
	simulated.plrIsOnSetLevel = !setlevel;
	simulated._pNumInv = source._pNumInv;
	std::copy(std::begin(source.InvBody), std::end(source.InvBody), std::begin(simulated.InvBody));
	std::copy(std::begin(source.InvList), std::end(source.InvList), std::begin(simulated.InvList));
	std::copy(std::begin(source.InvGrid), std::end(source.InvGrid), std::begin(simulated.InvGrid));
	std::copy(std::begin(source.SpdList), std::end(source.SpdList), std::begin(simulated.SpdList));
	return simulated;
}

// Ordinary armor previews and commits use the same native legal plan. Every
// displaced item must survive in the real inventory; a hand plan can clear
// more than its target slot. Detached staging cannot publish network commands.
bool StageResourceArmorInventory(const Player &player, int inventoryIndex,
    const GearUpgradePlan &gear, Player &staged)
{
	if (inventoryIndex >= 0)
		staged.RemoveInvItem(inventoryIndex, false, false);
	for (int slot = 0; slot < NUM_INVLOC; ++slot) {
		if (gear.clearSlots[slot] && !player.InvBody[slot].isEmpty()
		    && !AutoPlaceItemInInventory(staged, player.InvBody[slot], false))
			return false;
	}
	return true;
}

void ApplyResourceArmorPlan(Player &player, const GearUpgradePlan &gear)
{
	for (int slot = 0; slot < NUM_INVLOC; ++slot)
		if (gear.clearSlots[slot]) player.InvBody[slot].clear();
	player.InvBody[gear.target] = gear.candidate;
}

struct ResourceArmorPlan {
	GearUpgradePlan gear;
	bool canFit = false;
	bool safe = false;
	const char *reason = "unsupported_item";
	py::dict readiness;
};

ResourceArmorPlan PlanResourceArmor(const Player &player, const Item &item, int inventoryIndex)
{
	ResourceArmorPlan result;
	if (!IsOrdinaryResourceArmor(item)) return result;
	Item candidate = item;
	candidate._iStatFlag = player.CanUseItem(candidate);
	if (!candidate._iStatFlag) {
		result.reason = "cannot_use";
		return result;
	}
	// This is normal inventory equipment, not action14's strict utility
	// upgrade policy. An AC decrease can still fix a readiness deficit.
	// Keep PlanGearUpgrade and every legacy caller completely unchanged.
	Player simulated = ResourceUnequipSimulation(player);
	auto clearSlot = [&](inv_body_loc slot) {
		result.gear.clearSlots[slot] = true;
		simulated.InvBody[slot].clear();
	};
	if (item.isHelm()) {
		clearSlot(INVLOC_HEAD);
		result.gear.target = INVLOC_HEAD;
	} else if (item.isArmor()) {
		clearSlot(INVLOC_CHEST);
		result.gear.target = INVLOC_CHEST;
	} else {
		// Match ordinary hand compatibility and the shift-click preference:
		// replace a shield, retain a compatible weapon, or clear a two-hand
		// weapon. AutoEquip below validates the real engine pairing rules.
		const Item &left = player.InvBody[INVLOC_HAND_LEFT];
		const Item &right = player.InvBody[INVLOC_HAND_RIGHT];
		const bool twoHands = (!left.isEmpty() && player.GetItemLocation(left) == ILOC_TWOHAND)
		    || (!right.isEmpty() && player.GetItemLocation(right) == ILOC_TWOHAND);
		if (twoHands) {
			clearSlot(INVLOC_HAND_LEFT);
			clearSlot(INVLOC_HAND_RIGHT);
		} else if (!left.isEmpty() && left._iClass == candidate._iClass) {
			clearSlot(INVLOC_HAND_LEFT);
		} else if (!right.isEmpty() && right._iClass == candidate._iClass) {
			clearSlot(INVLOC_HAND_RIGHT);
		}
		result.gear.target = simulated.InvBody[INVLOC_HAND_LEFT].isEmpty()
		    ? INVLOC_HAND_LEFT : INVLOC_HAND_RIGHT;
	}
	// persist=false is the engine's pure legality probe: no network, RNG,
	// light/stat mutation, or item moved into the live player's hand.
	if (!AutoEquip(simulated, candidate, false, false)) {
		result.reason = "incompatible_slot";
		return result;
	}
	result.gear.valid = true;
	result.gear.candidate = candidate;
	result.canFit = StageResourceArmorInventory(player, inventoryIndex, result.gear, simulated);
	ApplyResourceArmorPlan(simulated, result.gear);
	CalcPlrInv(simulated, false);
	result.gear.nextArmorClass = simulated.GetArmor();
	result.gear.nextCurrentHitPoints = simulated._pHitPoints;
	result.gear.nextMaxHitPoints = simulated._pMaxHP;
	// Full readiness (including health fraction and the remaining weapon)
	// is observable separately. Normal legal equipment need only leave the
	// player alive and the newly equipped item usable after stat cascades.
	const bool usable = simulated.InvBody[result.gear.target]._iStatFlag;
	const bool alive = (simulated._pHitPoints >> 6) > 0;
	result.safe = alive && usable;
	result.readiness = ResourceReadinessState(simulated);
	result.readiness["block_enabled"] = simulated._pBlockFlag;
	result.readiness["block_chance"] = simulated.GetBlockChance();
	result.reason = !alive ? "unsafe_life" : (!usable ? "cannot_use_after_swap"
	    : (!result.canFit ? "no_room_for_replaced_items" : "ready"));
	return result;
}

void AppendOrdinaryResourceArmorState(py::dict &entry, const Item &item, int inventoryIndex)
{
	AppendItemCombatState(entry, item, 0);
	const ResourceArmorPlan plan = PlanResourceArmor(*MyPlayer, item, inventoryIndex);
	py::list replaced;
	if (plan.gear.valid)
		for (int slot = 0; slot < NUM_INVLOC; ++slot)
			if (plan.gear.clearSlots[slot] && !MyPlayer->InvBody[slot].isEmpty()) replaced.append(slot);
	entry["is_ordinary_armor"] = IsOrdinaryResourceArmor(item);
	entry["target_slot"] = plan.gear.valid ? static_cast<int>(plan.gear.target) : -1;
	entry["replaced_slots"] = replaced;
	entry["can_equip"] = plan.gear.valid && plan.safe && plan.canFit;
	entry["equip_reason"] = plan.reason;
	entry["projected_readiness"] = plan.readiness;
	if (plan.gear.valid) entry["projected_armor_class"] = plan.gear.nextArmorClass;
	const bool durable = item._iMaxDur == DUR_INDESTRUCTIBLE || item._iDurability >= 15;
	bool projectedCombatGate = false;
	if (plan.gear.valid && plan.safe) {
		const auto failures = py::cast<std::vector<std::string>>(plan.readiness["failures"]);
		projectedCombatGate = std::none_of(failures.begin(), failures.end(), [](const std::string &failure) {
			return failure == "armor" || failure == "damage" || failure == "weapon";
		});
	}
	entry["meets_armor_gate"] = IsOrdinaryResourceArmor(item) && durable
	    && plan.gear.valid && plan.safe && plan.canFit && projectedCombatGate;
}

bool IsResourceUnequipSlot(const Player &player, int slot)
{
	if (slot < 0 || slot >= NUM_INVLOC) return false;
	const Item &item = player.InvBody[slot];
	if (item.isEmpty() || item._iClass == ICLASS_QUEST) return false;
	return (slot == INVLOC_HEAD && item.isHelm())
	    || (slot == INVLOC_CHEST && item.isArmor())
	    || ((slot == INVLOC_HAND_LEFT || slot == INVLOC_HAND_RIGHT) && item.isShield());
}

bool ResourceUnequipActionAllowed()
{
	return ResourceTownActionAllowed("act_unequip_equipped_item") && !qtextflag
	    && MyPlayer->_pmode <= PM_WALK_SIDEWAYS;
}

struct ResourceUnequipPlan {
	bool canFit = false;
	bool safe = false;
	py::dict readiness;
};

ResourceUnequipPlan PlanResourceUnequip(const Player &player, int slot)
{
	ResourceUnequipPlan plan;
	if (!IsResourceUnequipSlot(player, slot)) return plan;
	Player simulated = ResourceUnequipSimulation(player);
	const Item item = simulated.InvBody[slot];
	plan.canFit = AutoPlaceItemInInventory(simulated, item, false);
	// Even when full, expose the prospective native stats separately from
	// executability; a missing inventory slot must never become a free delete.
	RemoveEquipment(simulated, static_cast<inv_body_loc>(slot), false, false);
	CalcPlrInv(simulated, false);
	// hasNoLife() deliberately ignores life in town; reject an unequip
	// that would leave the player dead when normal dungeon rules resume.
	plan.safe = (simulated._pHitPoints >> 6) > 0;
	plan.readiness = ResourceReadinessState(simulated);
	if (gResourceOrdinaryArmorScope) {
		plan.readiness["block_enabled"] = simulated._pBlockFlag;
		plan.readiness["block_chance"] = simulated.GetBlockChance();
	}
	return plan;
}

py::list ObserveUnequipCandidates()
{
	py::list candidates;
	// New projections belong only to an executable town inventory action;
	// dungeon/default-off observations must not simulate equipment changes.
	if (!gResourceProtocol || setlevel || currlevel != 0 || !gTownServiceTrip
	    || !ResourceUnequipActionAllowed()) return candidates;
	const bool available = true;
	const ResourceReadiness current = EvaluateResourceReadiness(*MyPlayer);
	for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
		if (!IsResourceUnequipSlot(*MyPlayer, slot)) continue;
		const Item &item = MyPlayer->InvBody[slot];
		const ResourceUnequipPlan plan = PlanResourceUnequip(*MyPlayer, slot);
		py::dict entry;
		entry["slot"] = static_cast<int>(slot);
		entry["seed_hi"] = HighWord(item._iSeed);
		entry["seed_lo"] = LowWord(item._iSeed);
		entry["create_info"] = item._iCreateInfo;
		entry["base_id"] = static_cast<int>(item.IDidx);
		entry["durability"] = item._iDurability;
		entry["max_durability"] = item._iMaxDur;
		entry["can_fit"] = plan.canFit;
		entry["can_unequip"] = available && plan.canFit && plan.safe;
		entry["reason"] = !available ? "unavailable" : !plan.canFit ? "no_room" : !plan.safe ? "unsafe_life" : "available";
		entry["projected_readiness"] = plan.readiness;
		entry["removes_durability_failure"] = current.minimumDurability < 15
		    && py::cast<int>(plan.readiness["min_finite_durability"]) >= 15;
		candidates.append(entry);
	}
	return candidates;
}

// New loot APIs remain absent from legacy raw observations.
#include "resource_loot.hpp"

py::dict ObserveResourceState()
{
	py::dict ready = ResourceReadinessState(*MyPlayer);
	py::dict transition;
	transition["sequence"] = gResourceTransition.sequence;
	transition["accepted"] = gResourceTransition.accepted;
	transition["reason"] = gResourceTransition.reason;
	transition["source_depth"] = gResourceTransition.source;
	transition["target_depth"] = gResourceTransition.target;
	transition["message"] = gResourceTransition.message;
	transition["pretransition_ready"] = gResourceTransition.ready;
	transition["pretransition_ready_law"] = gResourceTransition.readyLaw;
	transition["source_is_set"] = gResourceTransition.sourceIsSet;
	transition["target_is_set"] = gResourceTransition.targetIsSet;
	py::dict town;
	town["active_vendor"] = ResourceVendorName(ActiveStore);
	town["dialog_active"] = qtextflag;
	py::list npcs, stock, gold, inventory, repairs, inventoryEquipment;
	if (!setlevel && currlevel == 0) {
		for (size_t index = 0; index < Towners.size(); ++index) {
			const Towner &npc = Towners[index];
			if (npc._ttype != TOWN_SMITH && npc._ttype != TOWN_HEALER) continue;
			py::dict entry;
			entry["id"] = index;
			entry["type"] = npc._ttype == TOWN_SMITH ? "smith" : "healer";
			entry["x"] = npc.position.x;
			entry["y"] = npc.position.y;
			npcs.append(entry);
		}
		auto appendStock = [&stock](const auto &items, const char *vendor) {
			for (size_t index = 0; index < items.size(); ++index) {
				py::dict entry = ResourceItemState(items[index]);
				entry["vendor"] = vendor;
				entry["index"] = index;
				stock.append(entry);
				// Remember only identities actually exported at the visited shop.
				if (gResourceOrdinaryArmorScope && gTownServiceTrip && std::string(vendor) == "smith") {
					if (gResourceSeenSmithGeneration != gTownRestockSequence || gResourceSeenSmithTownSeed != DungeonSeeds[0])
						gResourceSeenSmithItems.clear();
					gResourceSeenSmithGeneration = gTownRestockSequence;
					gResourceSeenSmithTownSeed = DungeonSeeds[0];
					const auto identity = ResourceIdentity(items[index]);
					if (std::find(gResourceSeenSmithItems.begin(), gResourceSeenSmithItems.end(), identity) == gResourceSeenSmithItems.end())
						gResourceSeenSmithItems.push_back(identity);
				}
			}
		};
		// Stock becomes policy-visible only at the corresponding vendor.
		if (std::string(ResourceVendorName(ActiveStore)) == "smith") {
			appendStock(SmithItems, "smith");
			if (!qtextflag) {
				for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
					const Item &item = MyPlayer->InvBody[slot];
					const std::optional<int> price = GetStoreRepairPrice(item);
					if (!price.has_value()) continue;
					py::dict entry;
					entry["slot"] = static_cast<int>(slot);
					entry["seed_hi"] = HighWord(item._iSeed);
					entry["seed_lo"] = LowWord(item._iSeed);
					entry["create_info"] = item._iCreateInfo;
					entry["base_id"] = static_cast<int>(item.IDidx);
					entry["price"] = *price;
					entry["durability"] = item._iDurability;
					entry["max_durability"] = item._iMaxDur;
					entry["repair_needed_for_gate"] = item._iDurability < 15 && item._iMaxDur >= 15;
					repairs.append(entry);
				}
			}
		}
		if (std::string(ResourceVendorName(ActiveStore)) == "healer") appendStock(HealerItems, "healer");
	}
	for (int index = 0; index < MyPlayer->_pNumInv; ++index) {
		if (MyPlayer->InvList[index].isEquipment()) {
			py::dict entry;
			AppendItemCombatState(entry, MyPlayer->InvList[index], 0);
			entry["index"] = index;
			inventoryEquipment.append(entry);
		}
		if (gResourceOrdinaryArmorScope ? !IsOrdinaryResourceArmor(MyPlayer->InvList[index]) : !MyPlayer->InvList[index].isArmor()) continue;
		py::dict entry = ResourceItemState(MyPlayer->InvList[index], index);
		entry["index"] = index;
		inventory.append(entry);
	}
	for (int i = 0; i < ActiveItemCount; ++i) {
		const int id = ActiveItems[i];
		const Item &item = Items[id];
		if (item._itype != ItemType::Gold || !IsTileLit(item.position)) continue;
		py::dict entry;
		entry["active_id"] = id;
		entry["x"] = item.position.x;
		entry["y"] = item.position.y;
		entry["seed_hi"] = HighWord(item._iSeed);
		entry["seed_lo"] = LowWord(item._iSeed);
		entry["create_info"] = item._iCreateInfo;
		entry["base_id"] = static_cast<int>(item.IDidx);
		entry["value"] = item._ivalue;
		gold.append(entry);
	}
	town["npcs"] = npcs;
	town["stock"] = stock;
	town["repair_quotes"] = repairs;
	py::dict result;
	result["protocol"] = "l2-town-v1";
	result["enabled"] = true;
	if (gResourceOrdinaryArmorScope) result["ordinary_armor_scope"] = true;
	if (gResourcePreserveEquipmentReadiness) result["preserve_equipment_readiness"] = true;
	if (gResourceReadinessAdvisory) result["readiness_advisory"] = true;
	result["service_authorized"] = gTownServiceAuthorized;
	result["service_trip"] = gTownServiceTrip;
	result["retreat_enabled"] = gResourceRetreatEnabled;
	if (gResourceRetreatEnabled) {
		result["retreat_authorized"] = gRetreatAuthorized;
		result["retreats_started"] = gResourceRetreatsStarted;
		result["max_retreats"] = MaxResourceRetreats;
	}
	result["max_main_depth_reached"] = gResourceMaxDepth;
	// coach-v03 lifts the native curriculum wall (Python per-floor tables govern L3+).
	result["curriculum_max_depth"] = gResourceReadinessAdvisory ? 16 : 2;
	result["town_restock_sequence"] = gTownRestockSequence;
	result["town_seed"] = DungeonSeeds[0];
	result["readiness"] = ready;
	result["transition"] = transition;
	result["town"] = town;
	result["gold_items"] = gold;
	result["inventory_items"] = inventory;
	result["unequip_candidates"] = ObserveUnequipCandidates();
	result["inventory_equipment"] = inventoryEquipment;
	if (gResourceLootEconomy) AppendResourceLootState(result, town);
	return result;
}

py::list ProjectSeenResourceSmithItems(const std::vector<ResourceItemIdentity> &identities)
{
	EnsureInGame("project_seen_resource_smith_items");
	if (!gResourceOrdinaryArmorScope || !ResourceTownActionAllowed("project_seen_resource_smith_items")
	    || MyPlayer->_pmode != PM_STAND || qtextflag)
		throw std::runtime_error("known Smith projection requires idle ordinary-armor town service");
	if (gResourceSeenSmithGeneration != gTownRestockSequence || gResourceSeenSmithTownSeed != DungeonSeeds[0])
		throw std::invalid_argument("Smith projection belongs to another town generation");
	// Check the whole request against observation memory before looking at
	// live stock. Unknown identities cannot be used as an inventory oracle.
	std::vector<ResourceItemIdentity> requested;
	for (const auto &identity : identities) {
		if (std::find(requested.begin(), requested.end(), identity) != requested.end())
			throw std::invalid_argument("duplicate known Smith identity");
		if (std::find(gResourceSeenSmithItems.begin(), gResourceSeenSmithItems.end(), identity) == gResourceSeenSmithItems.end())
			throw std::invalid_argument("unobserved Smith identity");
		requested.push_back(identity);
	}
	py::list result;
	for (const auto &identity : requested) {
		py::dict entry;
		bool found = false;
		for (size_t index = 0; index < SmithItems.size(); ++index) {
			if (ResourceIdentity(SmithItems[index]) != identity) continue;
			entry = ResourceItemState(SmithItems[index]);
			entry["vendor"] = "smith";
			entry["index"] = index;
			entry["status"] = "projected";
			found = true;
			break;
		}
		if (!found) {
			entry["seed_hi"] = identity[0];
			entry["seed_lo"] = identity[1];
			entry["create_info"] = identity[2];
			entry["base_id"] = identity[3];
			entry["status"] = "stale_item"; // no projection or old-cache fallback
		}
		entry["projection_origin"] = "known-smith-live-player";
		entry["town_restock_sequence"] = gTownRestockSequence;
		entry["town_seed"] = DungeonSeeds[0];
		result.append(entry);
	}
	return result;
}

int ActPickupGoldAt(int activeItemId, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId)
{
	if (!CanAcceptPlayerAction("act_pickup_gold_at") || !gResourceProtocol || MyPlayer->hasNoLife()
	    || activeItemId < 0 || activeItemId >= MAXITEMS)
		return 0;
	bool active = false;
	for (int i = 0; i < ActiveItemCount; ++i) active = active || ActiveItems[i] == activeItemId;
	const Item &item = Items[activeItemId];
	if (!active || item._itype != ItemType::Gold || !IsTileLit(item.position)
	    || MyPlayer->position.future != item.position
	    || !MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId)) return 0;
	NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, item.position, static_cast<uint16_t>(activeItemId));
	return 1;
}

int ActTalkTowner(int index)
{
	if (!ResourceTownActionAllowed("act_talk_towner") || qtextflag || index < 0 || static_cast<size_t>(index) >= Towners.size()) return 0;
	const Towner &npc = Towners[index];
	if ((npc._ttype != TOWN_HEALER && npc._ttype != TOWN_SMITH)
	    || MyPlayer->position.tile.WalkingDistance(npc.position) >= 2) return 0;
	NetSendCmdLocParam1(true, CMD_TALKXY, npc.position, static_cast<uint16_t>(index));
	return 1;
}

int ActDismissDialog()
{
	if (!ResourceTownActionAllowed("act_dismiss_dialog")) return 0;
	if (qtextflag) {
		qtextflag = false;
		stream_stop();
		return 1;
	}
	if (!IsPlayerInStore()) return 0;
	StoreESC();
	return 1;
}

py::dict ResourceActionResult(bool accepted, const char *reason, int paid = 0)
{
	py::dict result;
	result["accepted"] = accepted;
	result["reason"] = reason;
	result["price"] = paid;
	return result;
}

py::dict ActBuyStoreItem(const std::string &vendor, int index, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId)
{
	if (!ResourceTownActionAllowed("act_buy_store_item") || qtextflag) return ResourceActionResult(false, "unavailable");
	const bool healer = vendor == "healer";
	if ((!healer && vendor != "smith") || vendor != ResourceVendorName(ActiveStore)) return ResourceActionResult(false, "wrong_vendor");
	const Towner *npc = GetTowner(healer ? TOWN_HEALER : TOWN_SMITH);
	if (npc == nullptr || MyPlayer->position.tile.WalkingDistance(npc->position) >= 2) return ResourceActionResult(false, "not_adjacent");
	const std::span<const Item> items = healer ? std::span<const Item>(HealerItems) : std::span<const Item>(SmithItems);
	if (index < 0 || static_cast<size_t>(index) >= items.size()) return ResourceActionResult(false, "invalid_item");
	const Item &item = items[index];
	if (!MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId)) return ResourceActionResult(false, "stale_item");
	if (!MyPlayer->CanUseItem(item)) return ResourceActionResult(false, "cannot_use");
	if (healer ? InstantHealKind(item) == 0 : (gResourceOrdinaryArmorScope ? !IsOrdinaryResourceArmor(item) : (!item.isArmor() || item._iMagical != ITEM_QUALITY_NORMAL))) return ResourceActionResult(false, "unsupported_item");
	if (Stash.gold != 0) throw std::runtime_error("l2-town-v1 forbids shared-stash gold");
	const auto purchase = TryBuyStoreItem(healer ? StoreVendor::Healer : StoreVendor::Smith, static_cast<size_t>(index), (uint32_t { seedHigh } << 16) | seedLow, createInfo, baseId);
	switch (purchase.status) {
	case StoreBuyStatus::Success: return ResourceActionResult(true, "purchased", purchase.paid);
	case StoreBuyStatus::InvalidItem: return ResourceActionResult(false, "invalid_item");
	case StoreBuyStatus::StaleItem: return ResourceActionResult(false, "stale_item");
	case StoreBuyStatus::NoMoney: return ResourceActionResult(false, "no_money");
	case StoreBuyStatus::NoRoom: return ResourceActionResult(false, "no_room");
	}
	throw std::runtime_error("unknown store transaction status");
}

py::dict ActRepairEquippedItem(int slot, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId, int expectedDurability, int expectedPrice)
{
	if (!ResourceTownActionAllowed("act_repair_equipped_item") || qtextflag) return ResourceActionResult(false, "unavailable");
	if (std::string(ResourceVendorName(ActiveStore)) != "smith") return ResourceActionResult(false, "wrong_vendor");
	const Towner *npc = GetTowner(TOWN_SMITH);
	if (npc == nullptr || MyPlayer->position.tile.WalkingDistance(npc->position) >= 2) return ResourceActionResult(false, "not_adjacent");
	if (slot != INVLOC_HEAD && slot != INVLOC_HAND_LEFT && slot != INVLOC_HAND_RIGHT && slot != INVLOC_CHEST) return ResourceActionResult(false, "invalid_item");
	if (Stash.gold != 0) throw std::runtime_error("l2-town-v1 forbids shared-stash gold");
	const auto repair = TryRepairStoreItem(true, static_cast<size_t>(slot), (uint32_t { seedHigh } << 16) | seedLow,
	    createInfo, baseId, expectedDurability, expectedPrice);
	switch (repair.status) {
	case StoreBuyStatus::Success: return ResourceActionResult(true, "repaired", repair.paid);
	case StoreBuyStatus::InvalidItem: return ResourceActionResult(false, "invalid_item");
	case StoreBuyStatus::StaleItem: return ResourceActionResult(false, "stale_item");
	case StoreBuyStatus::NoMoney: return ResourceActionResult(false, "no_money");
	case StoreBuyStatus::NoRoom: return ResourceActionResult(false, "no_room");
	}
	throw std::runtime_error("unknown store repair status");
}

py::dict ActEquipOrdinaryInventoryArmor(int index, const Item &candidate)
{
	Player &player = *MyPlayer;
	if (player._pmode > PM_WALK_SIDEWAYS)
		return ResourceActionResult(false, "unavailable");
	const ResourceArmorPlan plan = PlanResourceArmor(player, candidate, index);
	if (!plan.gear.valid || !plan.safe || !plan.canFit)
		return ResourceActionResult(false, plan.reason);
	Player staged = ResourceUnequipSimulation(player);
	if (!StageResourceArmorInventory(player, index, plan.gear, staged))
		return ResourceActionResult(false, "no_room_for_replaced_items");
	ApplyResourceArmorPlan(staged, plan.gear);
	CalcPlrInv(staged, false);
	const auto resources = CapturePlayerResourceState(player);
	std::array<Item, NUM_INVLOC> previousBody;
	std::copy(std::begin(player.InvBody), std::end(player.InvBody), previousBody.begin());
	ApplyResourceArmorPlan(player, plan.gear);
	CalcPlrInv(player, true);
	if ((player._pHitPoints >> 6) <= 0 || !player.InvBody[plan.gear.target]._iStatFlag
	    || player.GetArmor() != plan.gear.nextArmorClass
	    || player._pHitPoints != plan.gear.nextCurrentHitPoints
	    || player._pMaxHP != plan.gear.nextMaxHitPoints) {
		RestoreGearUpgradeTransaction(player, previousBody, resources);
		return ResourceActionResult(false, "equipment_validation_failed");
	}
	// Publish only after both the capacity and post-stat transaction succeed.
	// Inventory removal compacts indices, so send all old/new origins exactly
	// as the shared GUI purchase kernel does instead of guessing old indices.
	std::array<int8_t, InventoryGridCells> previousGrid;
	std::copy(std::begin(player.InvGrid), std::end(player.InvGrid), previousGrid.begin());
	player._pNumInv = staged._pNumInv;
	std::copy(std::begin(staged.InvList), std::end(staged.InvList), std::begin(player.InvList));
	std::copy(std::begin(staged.InvGrid), std::end(staged.InvGrid), std::begin(player.InvGrid));
	// CalcPlrInv refreshes carried requirements only for MyPlayer. Detached
	// inventory staging therefore cannot supply the final post-swap caches.
	for (int itemIndex = 0; itemIndex < player._pNumInv; ++itemIndex)
		player.InvList[itemIndex].updateRequiredStatsCacheForPlayer(player);
	for (Item &item : player.SpdList)
		item.updateRequiredStatsCacheForPlayer(player);
	player.CalcScrolls();
	std::array<bool, InventoryGridCells> sent {};
	for (int cell = 0; cell < InventoryGridCells; ++cell) {
		const int oldIndex = std::abs(previousGrid[cell]) - 1;
		if (oldIndex >= 0 && !sent[oldIndex]) {
			NetSendCmdParam1(false, CMD_DELINVITEMS, static_cast<uint16_t>(cell));
			sent[oldIndex] = true;
		}
	}
	sent.fill(false);
	for (int cell = 0; cell < InventoryGridCells; ++cell) {
		const int newIndex = std::abs(player.InvGrid[cell]) - 1;
		if (newIndex >= 0 && !sent[newIndex]) {
			NetSendCmdChInvItem(false, cell);
			sent[newIndex] = true;
		}
	}
	py::list replaced;
	for (int slot = 0; slot < NUM_INVLOC; ++slot) {
		if (!plan.gear.clearSlots[slot] || previousBody[slot].isEmpty()) continue;
		replaced.append(slot);
		if (slot != plan.gear.target) NetSendCmdDelItem(false, static_cast<uint8_t>(slot));
	}
	NetSendCmdChItem(false, static_cast<uint8_t>(plan.gear.target), true);
	py::dict result = ResourceActionResult(true, "equipped");
	result["target_slot"] = static_cast<int>(plan.gear.target);
	result["replaced_slots"] = replaced;
	py::dict equipped;
	AppendItemCombatState(equipped, player.InvBody[plan.gear.target], 0);
	result["equipped_item"] = equipped;
	result["readiness_after"] = ResourceReadinessState(player);
	return result;
}

py::dict ActEquipInventoryItem(int index, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId)
{
	if (!ResourceTownActionAllowed("act_equip_inventory_item") || qtextflag) return ResourceActionResult(false, "unavailable");
	Player &player = *MyPlayer;
	if (index < 0 || index >= player._pNumInv) return ResourceActionResult(false, "invalid_item");
	const Item candidate = player.InvList[index];
	if (!MatchesItemIdentity(candidate, seedHigh, seedLow, createInfo, baseId)) return ResourceActionResult(false, "stale_item");
	if (gResourceOrdinaryArmorScope) return ActEquipOrdinaryInventoryArmor(index, candidate);
	if (!candidate.isArmor() || candidate._iMagical != ITEM_QUALITY_NORMAL || !player.CanUseItem(candidate)) return ResourceActionResult(false, "cannot_use");
	const GearUpgradePlan plan = PlanGearUpgrade(player, candidate);
	if (!plan.valid || plan.target != INVLOC_CHEST) return ResourceActionResult(false, "not_upgrade");
	// Stage the normal inventory removal/placement rules on a detached player.
	// No live world mutation or network command occurs before capacity passes.
	Player staged = {};
	staged._pNumInv = player._pNumInv;
	std::copy(std::begin(player.InvList), std::end(player.InvList), std::begin(staged.InvList));
	std::copy(std::begin(player.InvGrid), std::end(player.InvGrid), std::begin(staged.InvGrid));
	int removedGrid = -1;
	for (int i = 0; i < InventoryGridCells; ++i)
		if (std::abs(player.InvGrid[i]) == index + 1 && removedGrid < 0) removedGrid = i;
	staged.RemoveInvItem(index, false, false);
	const Item oldArmor = player.InvBody[INVLOC_CHEST];
	if (!oldArmor.isEmpty() && !AutoPlaceItemInInventory(staged, oldArmor, false)) return ResourceActionResult(false, "no_room_for_old_armor");
	const auto resources = CapturePlayerResourceState(player);
	std::array<Item, NUM_INVLOC> previousBody;
	std::copy(std::begin(player.InvBody), std::end(player.InvBody), previousBody.begin());
	player.InvBody[INVLOC_CHEST] = candidate;
	CalcPlrInv(player, true);
	if (player.hasNoLife() || !player.InvBody[INVLOC_CHEST]._iStatFlag || player.GetArmor() != plan.nextArmorClass) {
		RestoreGearUpgradeTransaction(player, previousBody, resources);
		return ResourceActionResult(false, "equipment_validation_failed");
	}
	player._pNumInv = staged._pNumInv;
	std::copy(std::begin(staged.InvList), std::end(staged.InvList), std::begin(player.InvList));
	std::copy(std::begin(staged.InvGrid), std::end(staged.InvGrid), std::begin(player.InvGrid));
	player.CalcScrolls();
	if (removedGrid >= 0) NetSendCmdParam1(false, CMD_DELINVITEMS, static_cast<uint16_t>(removedGrid));
	if (!oldArmor.isEmpty()) {
		for (int grid = 0; grid < InventoryGridCells; ++grid) {
			if (player.InvGrid[grid] == player._pNumInv) {
				NetSendCmdChInvItem(false, grid);
				break;
			}
		}
	}
	NetSendCmdChItem(false, INVLOC_CHEST, true);
	return ResourceActionResult(true, "equipped");
}


py::dict ActUnequipEquippedItem(int slot, uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId)
{
	if (!ResourceUnequipActionAllowed()) return ResourceActionResult(false, "unavailable");
	Player &player = *MyPlayer;
	if (!IsResourceUnequipSlot(player, slot)) return ResourceActionResult(false, "invalid_item");
	const Item item = player.InvBody[slot];
	if (!MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId)) return ResourceActionResult(false, "stale_item");
	const ResourceUnequipPlan plan = PlanResourceUnequip(player, slot);
	if (!plan.canFit) return ResourceActionResult(false, "no_room");
	if (!plan.safe) return ResourceActionResult(false, "unsafe_life");
	// The preflight and commit execute synchronously on the same unchanged
	// inventory. AutoPlace performs no mutation if its capacity check fails.
	const int inventoryIndex = player._pNumInv;
	if (!AutoPlaceItemInInventory(player, item, false)) return ResourceActionResult(false, "no_room");
	RemoveEquipment(player, static_cast<inv_body_loc>(slot), false, false);
	CalcPlrInv(player, true);
	// Publish only the completed ordinary transaction. No failed request can
	// leave an inventory addition or equipment deletion in the network queue.
	for (int grid = 0; grid < InventoryGridCells; ++grid) {
		if (player.InvGrid[grid] == inventoryIndex + 1) {
			NetSendCmdChInvItem(false, grid);
			break;
		}
	}
	NetSendCmdDelItem(false, static_cast<inv_body_loc>(slot));
	py::dict result = ResourceActionResult(true, "unequipped");
	result["slot"] = slot;
	result["inventory_index"] = inventoryIndex;
	result["readiness_after"] = ResourceReadinessState(player);
	py::dict carried;
	AppendItemCombatState(carried, player.InvList[inventoryIndex], 0);
	carried["index"] = inventoryIndex;
	result["inventory_item"] = carried;
	return result;
}
