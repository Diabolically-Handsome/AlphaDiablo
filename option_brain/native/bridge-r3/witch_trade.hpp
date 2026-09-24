// Bridge-only ordinary Witch UI sales. Frozen native engine is not patched.
// There is no resource insertion, price override, or automatic item selection.
py::dict ManualObserveR4()
{
	auto d = ManualObserve();
	if (!qtextflag && ManualNear("witch") && std::string(ManualVendor()) == "witch") {
		py::list inv = py::cast<py::list>(d["inventory"]);
		for (auto entry : inv) {
			auto item = py::cast<py::dict>(entry);
			const int index = py::cast<int>(item["index"]);
			const Item &native = MyPlayer->InvList[index];
			if (!WitchWillBuy(native)) continue;
			// Exact read-only NormalStoreSellPrice formula (stores.cpp:527).
			const int value = native._iMagical != ITEM_QUALITY_NORMAL && native._iIdentified
			    ? native._iIvalue : native._ivalue;
			item["sale_price"] = std::max(value / 4, 1);
		}
	}
	return d;
}

py::dict ManualCheckpointR4()
{
	auto d = ManualCheckpoint();
	d["public"] = ManualObserveR4();
	return d;
}

py::dict ManualActionR4(const std::string &kind, const py::dict &a)
{
	if (kind != "sell" || !a.contains("vendor") || py::cast<std::string>(a["vendor"]) != "witch")
		return ManualActionR3(kind, a);
	EnsureInGame("manual_witch_sale");
	if (!gManualControl) throw std::runtime_error("manual mode is off");
	Player &p = *MyPlayer;
	if (p.hasNoLife() || p._pmode != PM_STAND || p.position.tile != p.position.future || p.walkpath[0] != WALK_NONE)
		return ResourceActionResult(false, "native_busy");
	if (!ManualNear("witch") || std::string(ManualVendor()) != "witch" || qtextflag)
		return ResourceActionResult(false, "not_at_witch");
	const int index = py::cast<int>(a["index"]), price = py::cast<int>(a["price"]);
	const auto identity = py::cast<ResourceItemIdentity>(a["identity"]);
	if (py::cast<bool>(a["equipped"]) || index < 0 || index >= p._pNumInv)
		return ResourceActionResult(false, "invalid_inventory_item");
	const Item &item = p.InvList[index];
	if (item.isEmpty() || ResourceIdentity(item) != identity || !WitchWillBuy(item))
		return ResourceActionResult(false, "stale_or_unsellable_item");
	const int goldBefore = p._pGold, countBefore = p._pNumInv;
	// Use the normal menu's own quote and native confirmation transaction.
	StartStore(TalkID::WitchSell);
	int quoted = -1;
	for (int i = 0; i < CurrentItemIndex; ++i)
		if (ResourceIdentity(PlayerItems[i]) == identity) { quoted = i; break; }
	if (quoted < 0 || PlayerItems[quoted]._iIvalue != price) {
		StartStore(TalkID::Witch);
		return ResourceActionResult(false, "stale_native_quote");
	}
	for (int i = 0; i < quoted; ++i) StoreDown();
	StoreEnter();
	if (ActiveStore != TalkID::Confirm || ResourceIdentity(TempItem) != identity || TempItem._iIvalue != price) {
		StartStore(TalkID::Witch);
		return ResourceActionResult(false, "native_confirmation_unavailable");
	}
	CurrentTextLine = 18; // The ordinary Yes button, stores.cpp ConfirmEnter.
	StoreEnter();
	bool remains = false;
	for (int i = 0; i < p._pNumInv; ++i) remains |= ResourceIdentity(p.InvList[i]) == identity;
	const bool accepted = !remains && p._pNumInv == countBefore - 1 && p._pGold - goldBefore == price;
	StartStore(TalkID::Witch);
	return ResourceActionResult(accepted, "native_witch_ui_sell", goldBefore - p._pGold);
}
