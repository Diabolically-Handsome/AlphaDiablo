// R18-H (2026-09-07) 凯恩鉴定 (Cain identify): the storyteller's fixed-fee
// identify service, exposed to the gym the way resource_loot.hpp exposes the
// Smith's sale counter.
//
// Included inside resource_protocol.hpp immediately after resource_loot.hpp, so
// ResourceTownActionAllowed / ResourceIdentity / AppendItemCombatState /
// GetStoreSellPrice are all already visible.
//
// 未鉴定的魔法物品按 _ivalue 出售、并且不带任何词缀生效；凯恩以固定
// StorytellerIdentifyPrice 金币翻开 _iIdentified 之后，同一件物品按 _iIvalue
// 出售（stores.cpp NormalStoreSellPrice）。本文件只做那一笔交易，不改任何阈值。
//
// UI 独立性:引擎侧 TryIdentifyItem 不读任何商店/滚动条状态,所以这里 **不**
// 需要先和凯恩开店(bridge 的 ActTalkTowner 也从不为 TOWN_STORY 开店)。可执行
// 前提只有"真的站在凯恩旁边、且没有别的店面开着",与真人操作一致。

// The seven body slots StartStorytellerIdentify lists, in its own order.
constexpr std::array<inv_body_loc, 7> ResourceIdentifyBodySlots = {
	INVLOC_HEAD, INVLOC_CHEST, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT,
	INVLOC_RING_LEFT, INVLOC_RING_RIGHT, INVLOC_AMULET
};

// Smith quote for this item as it would stand AFTER identification. The engine
// keeps _iIvalue populated on unidentified magic items, and the loot state
// already exports it, so this reveals nothing new; it just saves Python from
// re-deriving NormalStoreSellPrice.
std::optional<int> ResourceIdentifiedSalePrice(const Item &item)
{
	if (item.isEmpty()) return std::nullopt;
	Item identified = item;
	identified._iIdentified = true;
	return GetStoreSellPrice(StoreVendor::Smith, identified);
}

py::dict ResourceIdentifyQuote(const Item &item, bool equipped, int index, int slot)
{
	py::dict entry;
	entry["equipped"] = equipped;
	entry["index"] = index;
	entry["slot"] = slot;
	entry["seed_hi"] = HighWord(item._iSeed);
	entry["seed_lo"] = LowWord(item._iSeed);
	entry["create_info"] = item._iCreateInfo;
	entry["base_id"] = static_cast<int>(item.IDidx);
	entry["name"] = std::string(item.getName().str());
	entry["quality"] = static_cast<int>(item._iMagical);
	entry["identified"] = item._iIdentified;
	entry["is_equipment"] = item.isEquipment();
	entry["is_quest"] = ResourceLootQuestItem(item);
	entry["value"] = item._ivalue;
	entry["identified_value"] = item._iIvalue;
	const auto price = GetStoreIdentifyPrice(item);
	entry["identify_price"] = price.has_value() ? py::cast(*price) : py::none();
	const auto now = ResourceLootSellPrice(item);
	entry["sale_price"] = now.has_value() ? py::cast(*now) : py::none();
	const auto after = ResourceIdentifiedSalePrice(item);
	entry["sale_price_identified"] = after.has_value() ? py::cast(*after) : py::none();
	return entry;
}

// Every carried or worn item Cain would list, body slots first in his own
// order. Exported only while identify-v1 is on, so every frozen arm keeps its
// observation byte for byte.
py::list ObserveResourceIdentifyQuotes()
{
	py::list quotes;
	const Player &player = *MyPlayer;
	for (size_t i = 0; i < ResourceIdentifyBodySlots.size(); ++i) {
		const inv_body_loc slot = ResourceIdentifyBodySlots[i];
		const Item &item = player.InvBody[slot];
		if (!GetStoreIdentifyPrice(item).has_value()) continue;
		quotes.append(ResourceIdentifyQuote(item, true, static_cast<int>(slot), static_cast<int>(slot)));
	}
	for (int index = 0; index < player._pNumInv; ++index) {
		const Item &item = player.InvList[index];
		if (!GetStoreIdentifyPrice(item).has_value()) continue;
		quotes.append(ResourceIdentifyQuote(item, false, index, -1));
	}
	return quotes;
}

// R18-H review round (2026-09-07): the same gate, but it now says WHICH half
// refused, in the vocabulary ActBuyStoreItem already uses -- "unavailable" for
// the law (feature off, no authorized town-service trip, wrong level, dead,
// holding an item), "busy" for a transient UI/motion state, "not_adjacent" for
// the distance alone. The telemetry `failures` list and the economy audit could
// not tell a walk problem from a law violation while all three said
// "not_adjacent". Order of evaluation is unchanged; nullptr = the gate is open.
const char *ResourceIdentifyGateRefusal()
{
	if (!gResourceIdentifyEnabled || !ResourceTownActionAllowed("act_identify")) return "unavailable";
	if (qtextflag || MyPlayer->_pmode != PM_STAND || IsPlayerInStore()) return "busy";
	const Towner *npc = GetTowner(TOWN_STORY);
	if (npc == nullptr || MyPlayer->position.tile.WalkingDistance(npc->position) >= 2) return "not_adjacent";
	return nullptr;
}

// True only while the pair is really standing next to Cain with no store page
// open. Deliberately independent of ActiveStore/TalkID: the shared kernel needs
// no UI state, and the bridge never opens the storyteller's pages.
bool ResourceIdentifyAtStoryteller()
{
	return ResourceIdentifyGateRefusal() == nullptr;
}

py::dict ResourceIdentifyReceipt(bool accepted, const char *reason, bool equipped, int index,
    uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId,
    int quotedPrice, int goldBefore, int price = 0, int goldAfter = -1)
{
	py::dict receipt;
	receipt["accepted"] = accepted;
	receipt["reason"] = reason;
	receipt["equipped"] = equipped;
	receipt["index"] = index;
	receipt["seed_hi"] = seedHigh;
	receipt["seed_lo"] = seedLow;
	receipt["create_info"] = createInfo;
	receipt["base_id"] = baseId;
	receipt["quoted_price"] = quotedPrice;
	receipt["price"] = price;      // Paid expenditure, the buy/repair convention.
	receipt["received"] = 0;       // Identification is never income.
	receipt["gold_before"] = goldBefore;
	receipt["gold_after"] = goldAfter < 0 ? MyPlayer->_pGold : goldAfter;
	return receipt;
}

py::dict ActIdentifyItem(bool equipped, int index,
    uint16_t seedHigh, uint16_t seedLow, uint16_t createInfo, int baseId, int expectedPrice)
{
	EnsureInGame("act_identify");
	const int goldBefore = MyPlayer->_pGold;
	auto receipt = [&](bool accepted, const char *reason, int price = 0) {
		return ResourceIdentifyReceipt(accepted, reason, equipped, index, seedHigh, seedLow,
		    createInfo, baseId, expectedPrice, goldBefore, price);
	};
	if (!gResourceProtocol || !gResourceIdentifyEnabled || !gResourceLootEconomy)
		return receipt(false, "unavailable");
	if (const char *refusal = ResourceIdentifyGateRefusal(); refusal != nullptr)
		return receipt(false, refusal);
	if (index < 0 || (equipped ? index >= NUM_INVLOC : index >= MyPlayer->_pNumInv))
		return receipt(false, "invalid_item");
	const Item &item = equipped ? MyPlayer->InvBody[index] : MyPlayer->InvList[index];
	if (!MatchesItemIdentity(item, seedHigh, seedLow, createInfo, baseId))
		return receipt(false, "stale_item");
	if (!GetStoreIdentifyPrice(item).has_value()) return receipt(false, "not_identifiable");
	if (ResourceLootQuestItem(item)) return receipt(false, "quest_item");
	if (Stash.gold != 0) throw std::runtime_error("l2-town-v1 forbids shared-stash gold");
	const StoreIdentifyResult identified = TryIdentifyItem(equipped, static_cast<size_t>(index),
	    (uint32_t { seedHigh } << 16) | seedLow, createInfo, baseId, expectedPrice);
	switch (identified.status) {
	case StoreIdentifyStatus::Success: {
		if (identified.goldBefore != goldBefore
		    || MyPlayer->_pGold != identified.goldAfter
		    || goldBefore - MyPlayer->_pGold != identified.price)
			throw std::runtime_error("identify gold receipt mismatch");
		// Paying Cain can empty a gold pile and compact InvList (TakePlayerGold
		// spends the SMALL piles first), so the requested inventory index may no
		// longer address our item. Re-find it by its unchanged identity; body
		// slots never move. Never report the stale slot as the identified item.
		int indexAfter = index;
		const Item *after = nullptr;
		if (equipped) {
			after = &MyPlayer->InvBody[index];
		} else {
			for (int i = 0; i < MyPlayer->_pNumInv; ++i) {
				if (!MatchesItemIdentity(MyPlayer->InvList[i], seedHigh, seedLow, createInfo, baseId))
					continue;
				after = &MyPlayer->InvList[i];
				indexAfter = i;
				break;
			}
		}
		if (after == nullptr || !MatchesItemIdentity(*after, seedHigh, seedLow, createInfo, baseId))
			throw std::runtime_error("identified item is no longer carried");
		if (!after->_iIdentified) throw std::runtime_error("identify left the item unidentified");
		py::dict result = receipt(true, "identified", identified.price);
		result["index_after"] = indexAfter;
		result["item"] = ResourceIdentifyQuote(*after, equipped, indexAfter, equipped ? index : -1);
		result["inventory_after"] = ResourceLootInventoryState();
		return result;
	}
	case StoreIdentifyStatus::InvalidItem: return receipt(false, "invalid_item");
	case StoreIdentifyStatus::StaleItem: return receipt(false, "stale_item");
	case StoreIdentifyStatus::NoMoney: return receipt(false, "no_money");
	}
	throw std::runtime_error("unknown store identify status");
}
