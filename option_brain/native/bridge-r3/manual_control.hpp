// Independent manual interface. No old resource controller or neural worker.
// Native engine unchanged; commands mirror the normal GUI and transaction kernels.
bool ManualSeen(Point p)
{
	// Town has no dungeon lighting/vision flags. Use a conservative camera
	// diamond (inside the 640x480 viewport), not the whole fixed town map.
	if (!setlevel && currlevel == 0 && MyPlayer != nullptr) {
		const Point c = MyPlayer->position.tile;
		return InDungeonBounds(p) && std::abs(p.x-c.x)+std::abs(p.y-c.y) <= 8;
	}
	return InDungeonBounds(p) && IsTileVisible(p) && IsTileLit(p);
}

bool ManualTransitionGuard(const Player &p, interface_mode mode, int target)
{
	if (&p != MyPlayer) return true;
	if (p.hasNoLife() || mode == WM_DIABRETOWN) return false;
	if (mode == WM_DIABSETLVL) return target == SL_SKELKING;
	if (mode == WM_DIABRTNLVL) return setlevel && setlvlnum == SL_SKELKING;
	if (mode == WM_DIABWARPLVL) {
		const Portal &portal = Portals[MyPlayerId];
		return portal.open && !portal.setlvl && portal.level >= 1 && portal.level <= 3;
	}
	return target >= 0 && target <= 3;
}

bool ManualSeedHasKing(uint32_t seed)
{
	// Only a local copy of the quest lottery. No world/reset or global RNG call.
	std::mt19937 rng(seed);
	uint32_t questSeed = 0;
	for (int i = 0; i <= 15; ++i) questSeed = static_cast<uint32_t>(rng());
	Quest local[MAXQUESTS] {};
	for (auto &q : local) q._qactive = QUEST_INIT;
	InitialiseQuestPools(questSeed, local);
	return local[Q_SKELKING]._qactive != QUEST_NOTAVAIL;
}

py::dict ManualItem(const Item &it, int index, const char *place)
{
	py::dict d;
	d["index"] = index; d["place"] = place; d["empty"] = it.isEmpty();
	if (it.isEmpty()) return d;
	d["identity"] = ResourceIdentity(it);
	d["name"] = std::string(it.getName());
	d["identified"] = it._iIdentified || it._iMagical == ITEM_QUALITY_NORMAL;
	d["quality"] = static_cast<int>(it._iMagical);
	d["location"] = static_cast<int>(it._iLoc);
	d["durability"] = it._iDurability; d["max_durability"] = it._iMaxDur;
	d["min_strength"] = it._iMinStr; d["min_magic"] = it._iMinMag;
	d["min_dexterity"] = it._iMinDex; d["can_use"] = MyPlayer->CanUseItem(it);
	d["min_damage"] = it._iMinDam; d["max_damage"] = it._iMaxDam;
	d["armor"] = it._iAC; d["heal_kind"] = InstantHealKind(it);
	d["gold"] = it._itype == ItemType::Gold ? it._ivalue : 0;
	d["portal_scroll"] = it.isScrollOf(SpellID::TownPortal);
	// Do not expose unidentified affixes, identified value, RNG, or upgrade scores.
	if (it._iIdentified || it._iMagical == ITEM_QUALITY_NORMAL) {
		d["bonus_damage_percent"] = it._iPLDam; d["bonus_to_hit"] = it._iPLToHit;
		d["bonus_armor_percent"] = it._iPLAC;
		d["bonus_strength"] = it._iPLStr; d["bonus_dexterity"] = it._iPLDex;
		d["bonus_vitality"] = it._iPLVit; d["bonus_hp_fixed"] = it._iPLHP;
	}
	return d;
}

const char *ManualVendor()
{
	if (ActiveStore == TalkID::Storyteller || ActiveStore == TalkID::StorytellerIdentify) return "cain";
	return ResourceVendorName(ActiveStore);
}

bool ManualNear(const std::string &vendor)
{
	const Towner *npc = GetTowner(vendor == "smith" ? TOWN_SMITH : vendor == "healer" ? TOWN_HEALER : vendor == "witch" ? TOWN_WITCH : TOWN_STORY);
	return !setlevel && currlevel == 0 && npc != nullptr && MyPlayer->position.tile.WalkingDistance(npc->position) < 2;
}

bool ManualCanAttack(const Monster &monster)
{
	const Player &p = *MyPlayer;
	return CanAcceptPlayerAction("manual_attack_legality") && !p.hasNoLife()
	    && !IsPlayerInStore() && !qtextflag && !monster.hasNoLife()
	    && ManualSeen(monster.position.tile) && (monster.flags & MFLAG_HIDDEN) == 0
	    && p.position.future.WalkingDistance(monster.position.future) <= 1;
}

bool ManualCanStandingAttack(unsigned id)
{
	if (id >= MaxMonsters || MyPlayer == nullptr) return false;
	bool active = false;
	for (size_t i=0; i<ActiveMonsterCount; ++i) active |= ActiveMonsters[i] == id;
	const Player &p = *MyPlayer;
	return active && ManualCanAttack(Monsters[id])
	    && p.position.tile == p.position.future && p.walkpath[0] == WALK_NONE
	    && p.position.tile.WalkingDistance(Monsters[id].position.tile) <= 1;
}

py::dict ManualObserve()
{
	EnsureInGame("manual_observe");
	if (!gManualControl) throw std::runtime_error("manual mode is off");
	const Player &p = *MyPlayer;
	py::dict d, hero;
	d["tick"] = gManualTicks;
	d["normal_difficulty"] = sgGameInitInfo.nDifficulty == DIFF_NORMAL;
	d["scene"] = py::make_tuple(ConceptualDungeonDepth(), setlevel ? static_cast<int>(setlvlnum) : 0);
	hero["x"] = p.position.tile.x; hero["y"] = p.position.tile.y;
	hero["future_x"] = p.position.future.x; hero["future_y"] = p.position.future.y;
	hero["mode"] = static_cast<int>(p._pmode); hero["walkpath0"] = static_cast<int>(p.walkpath[0]);
	hero["hp_fixed"] = p._pHitPoints; hero["max_hp_fixed"] = p._pMaxHP;
	hero["mana_fixed"] = p._pMana; hero["max_mana_fixed"] = p._pMaxMana;
	hero["level"] = p.getCharacterLevel(); hero["xp"] = p._pExperience;
	hero["warrior"] = p._pClass == HeroClass::Warrior;
	hero["gold"] = p._pGold; hero["armor"] = p.GetArmor();
	hero["strength"] = p._pStrength; hero["magic"] = p._pMagic;
	hero["dexterity"] = p._pDexterity; hero["vitality"] = p._pVitality;
	hero["unspent_stats"] = p._pStatPts; hero["belt_heals"] = CountBeltHeals();
	hero["min_damage"] = p._pIMinDam; hero["max_damage"] = p._pIMaxDam;
	hero["damage_modifier"] = p._pDamageMod; hero["damage_bonus_percent"] = p._pIBonusDam;
	hero["dead"] = p.hasNoLife() || p._pmode == PM_DEATH;
	d["hero"] = hero; d["dialog"] = qtextflag; d["vendor"] = ManualVendor();
	d["kills"] = MonsterKillTotal();
	py::list tiles, enemies, floor, objects, npcs, exits, inventory, belt, equipment, stock;
	for (int y = 0; y < MAXDUNY; ++y) for (int x = 0; x < MAXDUNX; ++x) {
		const Point pt {x, y};
		if (!ManualSeen(pt)) continue;
		// Terrain only; not occupancy of unseen monsters.
		tiles.append(py::make_tuple(x, y, IsTileWalkable(pt, false)));
	}
	for (size_t i = 0; i < ActiveMonsterCount; ++i) {
		const unsigned id = ActiveMonsters[i]; const Monster &m = Monsters[id];
		if (m.hasNoLife() || m.isPlayerMinion() || (m.flags & MFLAG_HIDDEN) != 0 || !ManualSeen(m.position.tile)) continue;
		py::dict e; e["id"] = id; e["name"] = std::string(m.name());
		e["x"] = m.position.tile.x; e["y"] = m.position.tile.y;
		e["hp"] = m.hitPoints >> 6; e["max_hp"] = m.maxHitPoints >> 6;
		e["mode"] = static_cast<int>(m.mode); e["king"] = m.type().type == MT_SKING;
		// A legality bit, not the hidden destination/AI intent. Use the SAME
		// predicate as submission; tile proximity alone is insufficient mid-walk.
		e["attack_legal"] = ManualCanAttack(m);
		enemies.append(e);
	}
	for (int i = 0; i < ActiveItemCount; ++i) {
		const int id = ActiveItems[i]; const Item &it = Items[id];
		if (!ManualSeen(it.position)) continue;
		auto f = ManualItem(it, id, "floor"); f["x"] = it.position.x; f["y"] = it.position.y; floor.append(f);
	}
	for (int i = 0; i < ActiveObjectCount; ++i) {
		const int id = ActiveObjects[i]; const Object &o = Objects[id];
		if (!ManualSeen(o.position) || !o.canInteractWith()) continue;
		py::dict e; e["id"] = id; e["x"] = o.position.x; e["y"] = o.position.y;
		// No hidden trapped/explosive distinction.
		e["kind"] = o.isDoor() ? "door" : o.IsBreakable() ? "barrel" : "object";
		objects.append(e);
	}
	for (int i = 0; i < numtrigs; ++i) if (ManualSeen(trigs[i].position)) {
		py::dict e; e["x"] = trigs[i].position.x; e["y"] = trigs[i].position.y;
		e["message"] = static_cast<int>(trigs[i]._tmsg); exits.append(e);
	}
	if (!setlevel && currlevel == 0) for (size_t i = 0; i < Towners.size(); ++i) {
		const Towner &n = Towners[i]; if (!ManualSeen(n.position)) continue;
		py::dict e; e["id"] = i; e["x"] = n.position.x; e["y"] = n.position.y;
		e["kind"] = n._ttype == TOWN_SMITH ? "smith" : n._ttype == TOWN_HEALER ? "healer" : n._ttype == TOWN_WITCH ? "witch" : n._ttype == TOWN_STORY ? "cain" : "other";
		npcs.append(e);
	}
	for (int i = 0; i < p._pNumInv; ++i) {
		auto it = ManualItem(p.InvList[i], i, "inventory");
		if (ManualNear("smith") && std::string(ManualVendor()) == "smith") { auto price = GetStoreSellPrice(StoreVendor::Smith, p.InvList[i]); if (price) it["sale_price"] = *price; }
		if (ManualNear("cain")) { auto price = GetStoreIdentifyPrice(p.InvList[i]); if (price) it["identify_price"] = *price; }
		inventory.append(it);
	}
	for (int i = 0; i < MaxBeltItems; ++i) belt.append(ManualItem(p.SpdList[i], i, "belt"));
	for (int i = 0; i < NUM_INVLOC; ++i) {
		auto it = ManualItem(p.InvBody[i], i, "equipment");
		if (ManualNear("smith") && std::string(ManualVendor()) == "smith") { auto price = GetStoreRepairPrice(p.InvBody[i]); if (price) it["repair_price"] = *price; }
		if (ManualNear("cain")) { auto price = GetStoreIdentifyPrice(p.InvBody[i]); if (price) it["identify_price"] = *price; }
		equipment.append(it);
	}
	const std::string vendor = ManualVendor();
	auto appendStock = [&](const auto &items, int offset) {
		for (size_t i = 0; i < items.size(); ++i) if (!items[i].isEmpty()) {
			auto it = ManualItem(items[i], static_cast<int>(i) + offset, "stock");
			it["price"] = items[i]._iIvalue; it["vendor"] = vendor;
			Item copy = items[i]; it["fits"] = StoreAutoPlace(copy, false); stock.append(it);
		}
	};
	if (!qtextflag && ManualNear(vendor)) {
		if (vendor == "smith") { appendStock(SmithItems, 0); appendStock(PremiumItems, ResourcePremiumIndexBase); }
		if (vendor == "healer") appendStock(HealerItems, 0);
		if (vendor == "witch") appendStock(WitchItems, 0);
	}
	d["tiles"] = tiles; d["enemies"] = enemies; d["floor"] = floor; d["objects"] = objects;
	d["npcs"] = npcs; d["exits"] = exits; d["quest_entrances"] = ObserveQuestEntrances();
	d["inventory"] = inventory; d["belt"] = belt; d["equipment"] = equipment; d["stock"] = stock;
	d["king_kills"] = MonsterKillCounts[MT_SKING]; d["king_quest_done"] = Quests[Q_SKELKING]._qactive == QUEST_DONE;
	return d;
}

py::dict ManualCheckpoint()
{
	EnsureInGame("manual_checkpoint");
	py::dict d; d["raw"] = Observe(); d["public"] = ManualObserve();
	d["lcg"] = GetLCGEngineState(); d["town_sequence"] = gTownRestockSequence;
	py::list seeds; for (const auto seed : DungeonSeeds) seeds.append(seed); d["seeds"] = seeds;
	d["king_present"] = Quests[Q_SKELKING]._qactive != QUEST_NOTAVAIL;
	d["auto_quest_used"] = gMonotonicQuestTurnInUsed;
	return d; // Private integrity data: never supplied to either brain.
}

bool ManualCanUnbelt(int index, const ResourceItemIdentity &identity)
{
	if (!gManualControl || MyPlayer == nullptr || MyPlayer->hasNoLife()) return false;
	const Player &p = *MyPlayer;
	if (p._pmode != PM_STAND || p.position.tile != p.position.future || p.walkpath[0] != WALK_NONE) return false;
	if (qtextflag || IsPlayerInStore() || !p.HoldItem.isEmpty()) return false;
	if (index < 0 || index >= MaxBeltItems || p.SpdList[index].isEmpty() || ResourceIdentity(p.SpdList[index]) != identity) return false;
	return CanFitItemInInventory(p, p.SpdList[index]);
}

py::dict ManualAction(const std::string &kind, py::dict a)
{
	if (!gManualControl || !CanAcceptPlayerAction("manual_action") || MyPlayer->hasNoLife()) return ResourceActionResult(false, "unavailable");
	Player &p = *MyPlayer;
	auto integer = [&](const char *key) { return py::cast<int>(a[key]); };
	auto same = [&](const Item &it) { return ResourceIdentity(it) == py::cast<ResourceItemIdentity>(a["identity"]); };
	const bool idle = p._pmode == PM_STAND && p.position.tile == p.position.future && p.walkpath[0] == WALK_NONE;
	if (kind == "walk") {
		Point q { integer("x"), integer("y") };
		if (!ManualSeen(q) || p.position.future.WalkingDistance(q) != 1 || IsPlayerInStore() || qtextflag) return ResourceActionResult(false, "invalid_edge");
		if (!IsTileWalkable(q, false)) return ResourceActionResult(false, "blocked");
		// One observed edge. No implicit long path, stop, danger policy or auto attack.
		NetSendCmdLoc(MyPlayerId, true, CMD_WALKXY, q); return ResourceActionResult(true, "queued");
	}
	if (kind == "attack_stand") {
		const int id = integer("id");
		if (id < 0 || !ManualCanStandingAttack(static_cast<unsigned>(id)))
			return ResourceActionResult(false, "stationary_target_unavailable");
		// Exactly the human Shift+attack command: a tile swing, never chase.
		NetSendCmdLoc(MyPlayerId, true, CMD_SATTACKXY, Monsters[id].position.tile);
		return ResourceActionResult(true, "native_shift_attack_queued");
	}
	if (kind == "attack") {
		const unsigned id = integer("id"); bool active = false;
		for (size_t i=0; i<ActiveMonsterCount; ++i) active |= ActiveMonsters[i] == id;
		if (!active || id >= MaxMonsters) return ResourceActionResult(false, "target_gone");
		const auto &m = Monsters[id];
		if (!ManualCanAttack(m)) return ResourceActionResult(false, "target_unavailable");
		NetSendCmdParam1(true, CMD_ATTACKID, id); return ResourceActionResult(true, "queued");
	}
	if (kind == "drink") {
		const int slot = integer("index");
		if (slot < 0 || slot >= MaxBeltItems || !same(p.SpdList[slot]) || !IsHealItem(p.SpdList[slot])) return ResourceActionResult(false, "stale_potion");
		const int before = p._pHitPoints; const auto identity = ResourceIdentity(p.SpdList[slot]);
		UseInvItem(INVITEM_BELT_FIRST + slot);
		auto r = ResourceActionResult(p._pHitPoints > before || p.SpdList[slot].isEmpty() || ResourceIdentity(p.SpdList[slot]) != identity, "native_potion");
		r["hp_before"] = before; r["hp_after"] = p._pHitPoints; return r;
	}
	if (kind == "dismiss") {
		if (qtextflag) { qtextflag = false; stream_stop(); return ResourceActionResult(true, "dialog_closed"); }
		if (IsPlayerInStore()) { StoreESC(); return ResourceActionResult(true, "store_closed"); }
		return ResourceActionResult(false, "no_dialog");
	}
	if (!idle) return ResourceActionResult(false, "native_busy");
	if (kind == "pickup") {
		const int id = integer("index"); bool active = false;
		for (int i=0; i<ActiveItemCount; ++i) active |= ActiveItems[i] == id;
		if (!active || id < 0 || id >= MAXITEMS) return ResourceActionResult(false, "item_gone");
		const Item &it = Items[id];
		if (!same(it) || !ManualSeen(it.position) || p.position.tile != it.position) return ResourceActionResult(false, "stale_or_not_on_item");
		NetSendCmdLocParam1(true, CMD_GOTOAGETITEM, it.position, static_cast<uint16_t>(id));
		return ResourceActionResult(true, "queued");
	}
	if (kind == "operate") {
		const int id = integer("id"); bool active = false;
		for (int i=0; i<ActiveObjectCount; ++i) active |= ActiveObjects[i] == id;
		if (!active) return ResourceActionResult(false, "object_gone");
		const Object &o=Objects[id];
		if (!ManualSeen(o.position) || !o.canInteractWith() || p.position.tile.WalkingDistance(o.position) > 1) return ResourceActionResult(false, "object_unavailable");
		ActOperate(o.position.x, o.position.y); return ResourceActionResult(true, "queued");
	}
	if (kind == "talk") {
		int id=integer("id"); if (setlevel || currlevel != 0 || id < 0 || static_cast<size_t>(id) >= Towners.size()) return ResourceActionResult(false, "invalid_npc");
		const Towner &n=Towners[id]; if (!ManualSeen(n.position) || p.position.tile.WalkingDistance(n.position) >= 2) return ResourceActionResult(false, "not_adjacent");
		NetSendCmdLocParam1(true, CMD_TALKXY, n.position, static_cast<uint16_t>(id)); return ResourceActionResult(true, "queued");
	}
	if (kind == "stat") {
		const std::string attr=py::cast<std::string>(a["attribute"]); const auto &limits=GetClassAttributes(p._pClass);
		int current, maximum; _cmd_id cmd;
		if (attr=="strength") {current=p._pBaseStr; maximum=limits.maxStr; cmd=CMD_ADDSTR;}
		else if(attr=="dexterity") {current=p._pBaseDex; maximum=limits.maxDex; cmd=CMD_ADDDEX;}
		else if(attr=="vitality") {current=p._pBaseVit; maximum=limits.maxVit; cmd=CMD_ADDVIT;}
		else if(attr=="magic") {current=p._pBaseMag; maximum=limits.maxMag; cmd=CMD_ADDMAG;}
		else return ResourceActionResult(false,"invalid_stat");
		if(p._pStatPts<=0 || current>=maximum) return ResourceActionResult(false,"no_earned_point");
		NetSendCmdParam1(true,cmd,1); --p._pStatPts; return ResourceActionResult(true,"queued_earned_point");
	}
	if(kind=="unbelt") {
		const int index=integer("index");
		if (!ManualCanUnbelt(index, py::cast<ResourceItemIdentity>(a["identity"]))) return ResourceActionResult(false,"unbelt_unavailable");
		const Point previous=MousePosition; const bool oldFlag=invflag;
		const auto &box=InvRect[SLOTXY_BELT_FIRST+index];
		MousePosition=GetMainPanel().position + Displacement{box.position.x+2,box.position.y+2}; invflag=true;
		CheckInvItem(true,false); MousePosition=previous; invflag=oldFlag;
		return ResourceActionResult(true,"native_belt_shift_click_verify_after");
	}
	if(kind=="equip" || kind=="unequip" || kind=="belt") {
		if(qtextflag || IsPlayerInStore() || !p.HoldItem.isEmpty()) return ResourceActionResult(false,"ui_busy");
		int index=integer("index"), rect=-1;
		if(kind=="unequip") { if(index<0 || index>=NUM_INVLOC || !same(p.InvBody[index])) return ResourceActionResult(false,"stale_item"); rect=index; }
		else {
			if(index<0 || index>=p._pNumInv || !same(p.InvList[index])) return ResourceActionResult(false,"stale_item");
			if(kind=="equip" && !p.InvList[index].isEquipment()) return ResourceActionResult(false,"not_equipment");
			if(kind=="belt" && !CanBePlacedOnBelt(p,p.InvList[index])) return ResourceActionResult(false,"not_belt_item");
			for(int i=0;i<InventoryGridCells;++i) if(std::abs(p.InvGrid[i])==index+1) {rect=SLOTXY_INV_FIRST+i;break;}
		}
		if(rect<0) return ResourceActionResult(false,"no_inventory_cell");
		const Point previous=MousePosition; const bool oldFlag=invflag;
		const auto &box=InvRect[rect]; MousePosition=GetRightPanel().position + Displacement{box.position.x+2,box.position.y+2}; invflag=true;
		CheckInvItem(true,false); MousePosition=previous; invflag=oldFlag;
		return ResourceActionResult(true,"native_shift_click_verify_after");
	}
	if(kind=="buy") {
		const std::string vendor=py::cast<std::string>(a["vendor"]); int index=integer("index");
		if(qtextflag || vendor!=ManualVendor() || !ManualNear(vendor)) return ResourceActionResult(false,"not_at_vendor");
		const auto identity=py::cast<ResourceItemIdentity>(a["identity"]);
		if(vendor=="smith" && index>=ResourcePremiumIndexBase) {
			const int n=index-ResourcePremiumIndexBase;
			if(n<0 || static_cast<size_t>(n)>=PremiumItems.size() || !same(PremiumItems[n])) return ResourceActionResult(false,"stale_stock");
			const Item &it=PremiumItems[n]; Item check=it; const int price=it._iIvalue;
			if(!PlayerCanAfford(price)) return ResourceActionResult(false,"no_money");
			if(!StoreAutoPlace(check,false)) return ResourceActionResult(false,"no_room");
			int visible=0;for(int i=0;i<n;++i) visible+=!PremiumItems[i].isEmpty();
			int before=p._pGold; StartStore(TalkID::SmithPremiumBuy);for(int i=0;i<visible;++i) StoreDown();StoreEnter();
			if(ActiveStore!=TalkID::Confirm || !same(TempItem)) throw std::runtime_error("premium selection mismatch");
			CurrentTextLine=18;StoreEnter();StartStore(TalkID::Smith);
			if(before-p._pGold!=price) throw std::runtime_error("premium cash mismatch");
			return ResourceActionResult(true,"purchased",price);
		}
		if(vendor!="smith" && vendor!="healer" && vendor!="witch") return ResourceActionResult(false,"invalid_vendor");
		auto r=TryBuyStoreItem(vendor=="smith"?StoreVendor::Smith:vendor=="healer"?StoreVendor::Healer:StoreVendor::Witch,index,(uint32_t(identity[0])<<16)|identity[1],identity[2],identity[3]);
		return ResourceActionResult(r.status==StoreBuyStatus::Success,"native_buy",r.paid);
	}
	if(kind=="sell" || kind=="repair" || kind=="identify") {
		const auto identity=py::cast<ResourceItemIdentity>(a["identity"]); const int index=integer("index"),price=integer("price");
		const bool equipped=py::cast<bool>(a["equipped"]);
		if(kind=="identify") {
			if(!ManualNear("cain") || qtextflag) return ResourceActionResult(false,"not_at_cain");
			auto r=TryIdentifyItem(equipped,index,(uint32_t(identity[0])<<16)|identity[1],identity[2],identity[3],price);
			return ResourceActionResult(r.status==StoreIdentifyStatus::Success,"native_identify",r.price);
		}
		if(!ManualNear("smith") || std::string(ManualVendor())!="smith" || qtextflag) return ResourceActionResult(false,"not_at_smith");
		if(kind=="repair") {auto r=TryRepairStoreItem(equipped,index,(uint32_t(identity[0])<<16)|identity[1],identity[2],identity[3],integer("durability"),price);return ResourceActionResult(r.status==StoreBuyStatus::Success,"native_repair",r.paid);}
		auto r=TrySellStoreItem(StoreVendor::Smith,index,(uint32_t(identity[0])<<16)|identity[1],identity[2],identity[3],price);
		return ResourceActionResult(r.status==StoreSellStatus::Success,"native_sell",-r.received);
	}
	return ResourceActionResult(false,"unsupported");
}
