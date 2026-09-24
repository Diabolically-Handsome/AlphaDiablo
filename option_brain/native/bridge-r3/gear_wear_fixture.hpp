// Engineering-only detached loadout comparison. No real item creation or RNG.
m.def("probe_gear_score_fixture", [](const std::vector<py::dict> &descriptors,
    const py::dict &candidateDescriptor, int targetSlot) {
	EnsureInGame("probe_gear_score_fixture");
	if (descriptors.size() != NUM_INVLOC
	    || targetSlot < 0 || targetSlot >= NUM_INVLOC)
		throw std::invalid_argument("fixture requires seven body slots");
	auto decode = [](const py::dict &d) {
		Item item;
		if (!py::cast<bool>(d["present"])) return item;
		auto integer = [&](const char *key) { return py::cast<int>(d[key]); };
#define GEAR_FIELD(field, key) item.field = static_cast<decltype(item.field)>(integer(key))
		GEAR_FIELD(_itype, "item_type"); GEAR_FIELD(IDidx, "base_id");
		GEAR_FIELD(_iLoc, "equip_loc"); GEAR_FIELD(_iClass, "item_class");
		GEAR_FIELD(_iMagical, "quality"); GEAR_FIELD(_iDurability, "durability");
		GEAR_FIELD(_iMaxDur, "max_durability"); GEAR_FIELD(_iAC, "base_ac");
		GEAR_FIELD(_iMinDam, "min_damage"); GEAR_FIELD(_iMaxDam, "max_damage");
		GEAR_FIELD(_iPLDam, "effect_damage"); GEAR_FIELD(_iPLToHit, "effect_to_hit");
		GEAR_FIELD(_iPLAC, "effect_ac_percent"); GEAR_FIELD(_iPLStr, "effect_strength");
		GEAR_FIELD(_iPLMag, "effect_magic"); GEAR_FIELD(_iPLDex, "effect_dexterity");
		GEAR_FIELD(_iPLVit, "effect_vitality"); GEAR_FIELD(_iPLFR, "effect_fire_resist");
		GEAR_FIELD(_iPLLR, "effect_lightning_resist"); GEAR_FIELD(_iPLMR, "effect_magic_resist");
		GEAR_FIELD(_iPLHP, "effect_hp"); GEAR_FIELD(_iPLMana, "effect_mana");
		GEAR_FIELD(_iPLDamMod, "effect_damage_mod"); GEAR_FIELD(_iPLGetHit, "effect_get_hit");
		GEAR_FIELD(_iPLLight, "effect_light"); GEAR_FIELD(_iSplLvlAdd, "effect_spell_level");
		GEAR_FIELD(_iPLEnAc, "effect_enemy_ac"); GEAR_FIELD(_iFMinDam, "effect_fire_min");
		GEAR_FIELD(_iFMaxDam, "effect_fire_max"); GEAR_FIELD(_iLMinDam, "effect_lightning_min");
		GEAR_FIELD(_iLMaxDam, "effect_lightning_max"); GEAR_FIELD(_iFlags, "effect_flags");
		GEAR_FIELD(_iDamAcFlags, "effect_dam_ac_flags"); GEAR_FIELD(_iMinStr, "min_strength");
		GEAR_FIELD(_iMinMag, "min_magic"); GEAR_FIELD(_iMinDex, "min_dexterity");
		GEAR_FIELD(_iCreateInfo, "create_info");
#undef GEAR_FIELD
		item._iSeed = (static_cast<uint32_t>(integer("seed_hi")) << 16) | integer("seed_lo");
		item._iIdentified = py::cast<bool>(d["identified"]);
		item._iStatFlag = true;
		return item;
	};
	Player fixture = ResourceUnequipSimulation(*MyPlayer);
	fixture.setCharacterLevel(6);
	fixture._pBaseStr = 40; fixture._pBaseDex = 10;
	fixture._pBaseMag = 10; fixture._pBaseVit = 40;
	fixture._pHPBase = fixture._pMaxHPBase = 110 << 6;
	for (int slot = 0; slot < NUM_INVLOC; ++slot) fixture.InvBody[slot] = decode(descriptors[slot]);
	CalcPlrInv(fixture, false); SetPlrAnims(fixture, true);
	std::array<Item, NUM_INVLOC> current;
	std::copy(std::begin(fixture.InvBody), std::end(fixture.InvBody), current.begin());
	auto next = current;
	const Item candidate = decode(candidateDescriptor);
	next[targetSlot] = candidate;
	const auto oldNow = SimulateGearCombatProfile(fixture, current, true, false);
	const auto nextNow = SimulateGearCombatProfile(fixture, next, true, false);
	const auto oldWear = SimulateGearCombatProfile(fixture, current, true);
	const auto nextWear = SimulateGearCombatProfile(fixture, next, true);
	const auto plan = PlanGearUpgrade(fixture, candidate, false, true);
	py::dict result;
	result["legacy_delta"] = static_cast<int64_t>(nextNow.utility) - oldNow.utility;
	result["wear_delta"] = static_cast<int64_t>(nextWear.utility) - oldWear.utility;
	result["wear_current"] = oldWear.utility;
	result["wear_candidate"] = nextWear.utility;
	result["planned_valid"] = plan.valid;
	result["reserved_upgrade"] = ResourceLootUpgrade(fixture, candidate);
	result["next_armor"] = nextWear.armor;
	result["next_block"] = nextWear.blockEnabled;
	result["next_current_hp"] = nextWear.currentHitPoints;
	result["hit_scoring_enabled"] = gGearHitScoring;
	result["reference_armor"] = GearReferenceArmor;
	result["reference_hit_before"] = GearHitChance(oldNow.meleePiercingToHit, GearReferenceArmor);
	result["reference_hit_after"] = GearHitChance(nextNow.meleePiercingToHit, GearReferenceArmor);
	return result;
});
