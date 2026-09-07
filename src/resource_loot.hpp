// Included inside resource_protocol.hpp after real inventory/armor helpers.
// Loot mode changes capabilities, never actor dimensions or the readiness gate.

bool ResourceLootQuestItem(const Item &item)
{
	return item._iClass == ICLASS_QUEST || item.IDidx == IDI_LAZSTAFF;
}

bool ResourceLootUpgrade(const Player &player, const Item &item)
{
	// A useful upgrade does not become sale stock merely because its replaced
	// equipment cannot currently fit. This check deliberately ignores capacity.
	// Compare empty/current/candidate in real dungeon combat context even at Smith.
	return PlanGearUpgrade(player, item, false, true).valid;
}

std::optional<int> ResourceLootSellPrice(const Item &item)
{
	if (item.isEmpty() || !item.isEquipment() || ResourceLootQuestItem(item)) return std::nullopt;
	return GetStoreSellPrice(StoreVendor::Smith, item);
}

py::dict ResourceLootItemState(const Item &item, int index)
{
	py::dict entry;
	entry["empty"] = item.isEmpty();
	entry["index"] = index;
	if (item.isEmpty()) return entry;
	AppendItemCombatState(entry, item, 0);
	entry["identity"] = ResourceIdentity(item);
	entry["value"] = item._ivalue;
	entry["identified_value"] = item._iIvalue;
	entry["is_quest"] = ResourceLootQuestItem(item);
	entry["is_equipment"] = item.isEquipment();
	const auto price = ResourceLootSellPrice(item);
	const bool upgrade = price.has_value() && ResourceLootUpgrade(*MyPlayer, item);
	entry["upgrade"] = upgrade;
	entry["reserved_upgrade"] = upgrade;
	entry["sellable"] = price.has_value() && !upgrade;
	entry["price"] = price.has_value() ? py::cast(*price) : py::none();
	entry["sell_price"] = price.has_value() ? py::cast(*price) : py::none();
	entry["retained_from_a14"] = std::find(gResourceRetainedGear.begin(), gResourceRetainedGear.end(), ResourceIdentity(item)) != gResourceRetainedGear.end();
	return entry;
}

py::dict ResourceLootInventoryState()
{
	py::dict state;
	py::list items, grid;
	for (int i = 0; i < MyPlayer->_pNumInv; ++i) items.append(ResourceLootItemState(MyPlayer->InvList[i], i));
	int free = 0;
	for (const auto cell : MyPlayer->InvGrid) {
		grid.append(cell);
		if (cell == 0) ++free;
	}
	state["inventory_count"] = MyPlayer->_pNumInv;
	state["items"] = items;
	state["grid"] = grid;
	state["free_cells"] = free; // Cells are not a promise of rectangular fit.
	state["gold"] = MyPlayer->_pGold;
	state["held_empty"] = MyPlayer->HoldItem.isEmpty();
	return state;
}

void RecordLootGearRetention(const Player &player,
    const std::array<Item, NUM_INVLOC> &body,
    const std::array<bool, NUM_INVLOC> &clearSlots, int goldBefore)
{
	gResourceLastRetainedGear.clear();
	gResourceLastRetentionGold = goldBefore;
	for (int slot = 0; slot < NUM_INVLOC; ++slot) {
		if (!clearSlots[slot] || body[slot].isEmpty()) continue;
		const auto identity = ResourceIdentity(body[slot]);
		gResourceLastRetainedGear.push_back(identity);
		if (std::find(gResourceRetainedGear.begin(), gResourceRetainedGear.end(), identity) == gResourceRetainedGear.end())
			gResourceRetainedGear.push_back(identity);
	}
	if (player._pGold != goldBefore) throw std::runtime_error("gear retention changed gold");
}

bool ResourceLootAtSmith()
{
	if (!ResourceTownActionAllowed("loot_smith_service") || qtextflag
	    || MyPlayer->_pmode != PM_STAND || std::string(ResourceVendorName(ActiveStore)) != "smith") return false;
	for (const auto &npc : Towners)
		if (npc._ttype == TOWN_SMITH && MyPlayer->position.tile.WalkingDistance(npc.position) < 2) return true;
	return false;
}

void AppendResourceLootState(py::dict &result, py::dict &town)
{
	result["loot_economy"] = true;
	result["service_trips_started"] = gResourceServiceTripsStarted;
	result["max_service_trips"] = MaxLootServiceTrips;
	result["inventory_state"] = ResourceLootInventoryState();
	py::dict retention;
	retention["retained_identities"] = gResourceLastRetainedGear;
	retention["gold_before"] = gResourceLastRetentionGold;
	retention["gold_after"] = gResourceLastRetentionGold;
	result["last_a14_retention"] = retention;
	py::list loot, quotes;
	for (int i = 0; i < ActiveItemCount; ++i) {
		const int id = ActiveItems[i];
		const Item &item = Items[id];
		if (!IsTileLit(item.position) || !ResourceLootSellPrice(item).has_value()
		    || ResourceLootUpgrade(*MyPlayer, item)) continue;
		py::dict entry = ResourceLootItemState(item, -1);
		entry["active_id"] = id;
		entry["x"] = item.position.x;
		entry["y"] = item.position.y;
		entry["visible"] = true;
		entry["can_fit"] = CanFitItemInInventory(*MyPlayer, item);
		loot.append(entry);
	}
	// No vendor discovery: quotes are exported only during a real Smith visit.
	if (ResourceLootAtSmith()) {
		for (int i = 0; i < MyPlayer->_pNumInv; ++i) {
			const Item &item = MyPlayer->InvList[i];
			if (!ResourceLootSellPrice(item).has_value() || ResourceLootUpgrade(*MyPlayer, item)) continue;
			py::dict entry = ResourceLootItemState(item, i);
			entry["vendor"] = "smith";
			quotes.append(entry);
		}
	}
	result["loot_items"] = loot;
	town["sell_quotes"] = quotes;
}

py::dict ResourceLootReceipt(bool accepted, const char *reason, int index,
    uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId,
    int goldBefore, int received = 0)
{
	py::dict receipt;
	receipt["accepted"] = accepted;
	receipt["reason"] = reason;
	receipt["index"] = index;
	receipt["seed_hi"] = seedHigh;
	receipt["seed_lo"] = seedLow;
	receipt["create_info"] = createInfo;
	receipt["base_id"] = baseId;
	receipt["received"] = received;
	receipt["price"] = 0; // Paid expenditure; sale proceeds have their own field.
	receipt["gold_before"] = goldBefore;
	receipt["gold_after"] = MyPlayer->_pGold;
	return receipt;
}

py::dict ActPickupLootAt(int activeItemId, int x, int y,
    uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId)
{
	EnsureInGame("act_pickup_loot_at");
	const int goldBefore = MyPlayer->_pGold;
	auto receipt = [&](bool accepted, const char *reason) {
		return ResourceLootReceipt(accepted, reason, -1, seedHigh, seedLow, createInfo, baseId, goldBefore);
	};
	if (!gResourceProtocol || !gResourceLootEconomy || !CanAcceptPlayerAction("act_pickup_loot_at")
	    || MyPlayer->hasNoLife() || MyPlayer->_pmode != PM_STAND || !MyPlayer->HoldItem.isEmpty()) return receipt(false, "unavailable");
	if (x < 0 || x >= MAXDUNX || y < 0 || y >= MAXDUNY || activeItemId < 0 || activeItemId >= MAXITEMS)
		return receipt(false, "invalid_item");
	bool active = false;
	for (int i = 0; i < ActiveItemCount; ++i) active = active || ActiveItems[i] == activeItemId;
	const Item &floor = Items[activeItemId];
	if (!active || floor.position != Point { x, y }
	    || !MatchesItemIdentity(floor, seedHigh, seedLow, createInfo, baseId)) return receipt(false, "stale_item");
	if (!IsTileLit(floor.position) || MyPlayer->position.future != floor.position)
		return receipt(false, "not_at_visible_item");
	if (!ResourceLootSellPrice(floor).has_value()) return receipt(false, "not_sellable");
	if (ResourceLootUpgrade(*MyPlayer, floor)) return receipt(false, "reserved_upgrade");
	if (!CanFitItemInInventory(*MyPlayer, floor)) return receipt(false, "no_room");
	Item carried = floor;
	carried._iCreateInfo &= ~CF_PREGEN;
	carried.updateRequiredStatsCacheForPlayer(*MyPlayer);
	const int index = MyPlayer->_pNumInv;
	// Normal engine packing only; do not route through AutoGetItem's auto-equip.
	if (!AutoPlaceItemInInventory(*MyPlayer, carried, false)) return receipt(false, "no_room");
	NetSendCmdGItem(false, CMD_GETITEM, *MyPlayer, static_cast<uint8_t>(activeItemId));
	SyncGetItem(Point { x, y }, (uint32_t { seedHigh } << 16) | seedLow,
	    static_cast<_item_indexes>(baseId), createInfo);
	SetItemRecord((uint32_t { seedHigh } << 16) | seedLow, createInfo, baseId);
	for (int cell = 0; cell < InventoryGridCells; ++cell)
		if (MyPlayer->InvGrid[cell] == index + 1) NetSendCmdChInvItem(false, cell);
	py::dict result = receipt(true, "picked_up");
	result["index"] = index;
	result["active_id"] = activeItemId;
	result["inventory_item"] = ResourceLootItemState(MyPlayer->InvList[index], index);
	return result;
}

py::dict ActSellInventoryItem(const std::string &vendor, int index,
    uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId, int expectedPrice)
{
	EnsureInGame("act_sell_inventory_item");
	const int goldBefore = MyPlayer->_pGold;
	auto receipt = [&](bool accepted, const char *reason, int received = 0) {
		py::dict result = ResourceLootReceipt(accepted, reason, index, seedHigh, seedLow, createInfo, baseId, goldBefore, received);
		result["vendor"] = vendor;
		result["quoted_price"] = expectedPrice;
		return result;
	};
	if (!gResourceProtocol || !gResourceLootEconomy || !ResourceTownActionAllowed("act_sell_inventory_item") || qtextflag)
		return receipt(false, "unavailable");
	if (vendor != "smith" || vendor != ResourceVendorName(ActiveStore)) return receipt(false, "wrong_vendor");
	if (!ResourceLootAtSmith()) return receipt(false, "unavailable");
	if (index < 0 || index >= MyPlayer->_pNumInv) return receipt(false, "stale_item");
	const Item &item = MyPlayer->InvList[index];
	if (!MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId)) return receipt(false, "stale_item");
	if (!ResourceLootSellPrice(item).has_value()) return receipt(false, "not_sellable");
	if (ResourceLootUpgrade(*MyPlayer, item)) return receipt(false, "reserved_upgrade");
	const StoreSellResult sale = TrySellStoreItem(StoreVendor::Smith, index,
	    (uint32_t { seedHigh } << 16) | seedLow, createInfo, baseId, expectedPrice);
	switch (sale.status) {
	case StoreSellStatus::Success: {
		if (MyPlayer->_pGold - goldBefore != sale.received) throw std::runtime_error("sale gold receipt mismatch");
		py::dict result = receipt(true, "sold", sale.received);
		result["inventory_after"] = ResourceLootInventoryState();
		return result;
	}
	case StoreSellStatus::InvalidItem: return receipt(false, "invalid_item");
	case StoreSellStatus::StaleItem: return receipt(false, "stale_item");
	case StoreSellStatus::NoRoom: return receipt(false, "no_room");
	}
	throw std::runtime_error("unknown store sell status");
}
