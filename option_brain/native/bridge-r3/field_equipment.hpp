// Separate, explicitly enabled inventory equipment in the main dungeon.
// No town transaction gate is weakened. No item or durability is created.
bool gFieldEquipment = false;

bool FieldEquipmentAllowed(const char *operation)
{
	return gFieldEquipment && !gGearWearScoring && gResourceProtocol
	    && gResourceOrdinaryArmorScope && CanAcceptPlayerAction(operation)
	    && !setlevel && currlevel > 0 && !MyPlayer->hasNoLife()
	    && MyPlayer->_pmode == PM_STAND && MyPlayer->HoldItem.isEmpty()
	    && pcurs == CURSOR_HAND && !qtextflag && !IsPlayerInStore();
}

uint32_t FieldWearUtility(const Player &player)
{
	// This does not switch a14, sale or shop scoring. The earlier fixed wear
	// model is only a read-only comparison for the new inventory service.
	return ScoreGearWearProfile(player,
	    GearCombatProfileFromPlayer(player, false).utility, true);
}

uint32_t FieldProjectedUtility(const Player &player, const ResourceArmorPlan &plan)
{
	Player simulated = ResourceUnequipSimulation(player);
	ApplyResourceArmorPlan(simulated, plan.gear);
	CalcPlrInv(simulated, false);
	SetPlrAnims(simulated, true);
	return FieldWearUtility(simulated);
}

py::list PreviewFieldEquipment()
{
	EnsureInGame("preview_field_equipment");
	py::list result;
	if (!FieldEquipmentAllowed("preview_field_equipment")) return result;
	const Player &player = *MyPlayer;
	const uint32_t before = FieldWearUtility(player);
	for (int index = 0; index < player._pNumInv; ++index) {
		const Item &item = player.InvList[index];
		if (!IsOrdinaryResourceArmor(item)) continue;
		const auto plan = PlanResourceArmor(player, item, index);
		if (!plan.gear.valid || !plan.safe || !plan.canFit
		    || plan.gear.nextCurrentHitPoints > player._pHitPoints) continue;
		const uint32_t after = FieldProjectedUtility(player, plan);
		if (after <= before) continue;
		py::dict row;
		row["index"] = index;
		row["seed_hi"] = HighWord(item._iSeed);
		row["seed_lo"] = LowWord(item._iSeed);
		row["create_info"] = item._iCreateInfo;
		row["base_id"] = static_cast<int>(item.IDidx);
		row["target_slot"] = static_cast<int>(plan.gear.target);
		row["utility_before"] = before;
		row["utility_after"] = after;
		row["utility_delta"] = after - before;
		result.append(row);
	}
	return result;
}

py::dict ActEquipFieldInventoryItem(int index, uint16_t seedHigh, uint16_t seedLow,
    uint16_t createInfo, int baseId)
{
	EnsureInGame("act_equip_field_inventory_item");
	if (!FieldEquipmentAllowed("act_equip_field_inventory_item"))
		return ResourceActionResult(false, "unavailable");
	Player &player = *MyPlayer;
	if (index < 0 || index >= player._pNumInv)
		return ResourceActionResult(false, "invalid_item");
	const Item item = player.InvList[index];
	if (!MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId))
		return ResourceActionResult(false, "stale_item");
	if (!IsOrdinaryResourceArmor(item))
		return ResourceActionResult(false, "unsupported_item");
	const auto plan = PlanResourceArmor(player, item, index);
	if (!plan.gear.valid || !plan.safe || !plan.canFit)
		return ResourceActionResult(false, plan.reason);
	if (plan.gear.nextCurrentHitPoints > player._pHitPoints)
		return ResourceActionResult(false, "would_heal");
	const uint32_t before = FieldWearUtility(player);
	const uint32_t projected = FieldProjectedUtility(player, plan);
	if (projected <= before)
		return ResourceActionResult(false, "not_field_upgrade");
	const int goldBefore = player._pGold;
	const int hpBefore = player._pHitPoints;
	// Reuses the existing atomic capacity, equip and displaced-item kernel.
	py::dict result = ActEquipOrdinaryInventoryArmor(index, item);
	if (py::cast<bool>(result["accepted"])) {
		result["utility_before"] = before;
		result["utility_projected"] = projected;
		result["utility_after"] = FieldWearUtility(player);
		result["gold_before"] = goldBefore;
		result["gold_after"] = player._pGold;
		result["hp_fixed_before"] = hpBefore;
		result["hp_fixed_after"] = player._pHitPoints;
		result["inventory_source_index"] = index;
	}
	return result;
}
