// Included inside the private bridge namespace after resource_protocol.hpp.
// Explicit, bounded read-only previews. No global player substitution, native
// transaction invocation, stock regeneration, RNG operation or world command.
constexpr size_t ResourceCombinationBatchLimit = 256;
constexpr size_t ResourceCombinationStepLimit = 8;
constexpr const char *ResourceCombinationVersion = "observed-equipment-combinations-v1";

struct ResourceCombinationIntent {
	std::string operation;
	ResourceItemIdentity identity {};
	int index = -1;
	int durability = -1;
	int price = -1;
};

int ResourceCombinationInteger(py::handle value)
{
	if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
		throw std::invalid_argument("combination integer field must be an integer");
	return py::cast<int>(value);
}

ResourceItemIdentity ResourceCombinationIdentity(const py::sequence &command, size_t offset)
{
	ResourceItemIdentity identity;
	for (size_t i = 0; i < identity.size(); ++i)
		identity[i] = ResourceCombinationInteger(command[offset + i]);
	for (size_t i = 0; i < 3; ++i)
		if (identity[i] < 0 || identity[i] > 65535)
			throw std::invalid_argument("combination identity word out of range");
	if (identity[3] < 0)
		throw std::invalid_argument("combination base identity out of range");
	return identity;
}

std::vector<std::vector<ResourceCombinationIntent>> ParseResourceCombinations(const py::list &sequences)
{
	if (sequences.empty() || sequences.size() > ResourceCombinationBatchLimit)
		throw std::invalid_argument("combination batch must contain 1..256 sequences");
	std::vector<std::vector<ResourceCombinationIntent>> parsed;
	for (py::handle input : sequences) {
		if (!py::isinstance<py::list>(input) && !py::isinstance<py::tuple>(input))
			throw std::invalid_argument("combination sequence must be a list or tuple");
		const auto sequence = py::reinterpret_borrow<py::sequence>(input);
		if (sequence.size() > ResourceCombinationStepLimit)
			throw std::invalid_argument("combination sequence exceeds 8 intents");
		std::vector<ResourceCombinationIntent> intents;
		for (size_t step = 0; step < sequence.size(); ++step) {
			py::handle value = sequence[step];
			if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value))
				throw std::invalid_argument("combination intent must be a list or tuple");
			const auto command = py::reinterpret_borrow<py::sequence>(value);
			if (command.empty() || !py::isinstance<py::str>(command[0]))
				throw std::invalid_argument("combination intent lacks operation");
			ResourceCombinationIntent intent;
			intent.operation = py::cast<std::string>(command[0]);
			size_t identityOffset = 0;
			if (intent.operation == "buy_equip" && command.size() == 5) {
				identityOffset = 1;
			} else if (intent.operation == "buy" && command.size() == 7) {
				if (!py::isinstance<py::str>(command[1]) || py::cast<std::string>(command[1]) != "smith")
					throw std::invalid_argument("combination purchases only support known Smith armor");
				intent.index = ResourceCombinationInteger(command[2]);
				identityOffset = 3;
			} else if ((intent.operation == "equip" || intent.operation == "unequip") && command.size() == 6) {
				intent.index = ResourceCombinationInteger(command[1]);
				identityOffset = 2;
			} else if (intent.operation == "repair" && command.size() == 8) {
				intent.index = ResourceCombinationInteger(command[1]);
				identityOffset = 2;
				intent.durability = ResourceCombinationInteger(command[6]);
				intent.price = ResourceCombinationInteger(command[7]);
			} else if (intent.operation == "repair_retained" && command.size() == 1 && step + 1 == sequence.size()) {
				intents.push_back(intent);
				continue;
			} else {
				throw std::invalid_argument("unsupported combination intent or shape");
			}
			intent.identity = ResourceCombinationIdentity(command, identityOffset);
			// Entire request passes this observation whitelist before any stock
			// lookup. A mixed batch cannot query even one unobserved offer.
			if (intent.operation == "buy" || intent.operation == "buy_equip") {
				if (std::find(gResourceSeenSmithItems.begin(), gResourceSeenSmithItems.end(), intent.identity) == gResourceSeenSmithItems.end())
					throw std::invalid_argument("unobserved Smith combination identity");
			}
			intents.push_back(intent);
		}
		parsed.push_back(std::move(intents));
	}
	return parsed;
}

void RecalculateResourceCombinationPlayer(Player &player)
{
	CalcPlrInv(player, false);
	for (int i = 0; i < player._pNumInv; ++i)
		player.InvList[i].updateRequiredStatsCacheForPlayer(player);
	for (Item &item : player.SpdList)
		item.updateRequiredStatsCacheForPlayer(player);
	player.CalcScrolls();
}

py::dict ResourceCombinationItem(const Item &item, int index)
{
	py::dict entry;
	entry["index"] = index;
	entry["empty"] = item.isEmpty();
	if (!item.isEmpty()) {
		AppendItemCombatState(entry, item, 0);
		entry["identity"] = ResourceIdentity(item);
		entry["value"] = item._ivalue;
		entry["price"] = item._iIvalue;
	}
	return entry;
}

py::dict ResourceCombinationState(const Player &player)
{
	py::dict state;
	state["gold"] = player._pGold;
	state["inventory_count"] = player._pNumInv;
	py::list equipment, inventory, belt, grid;
	for (int i = 0; i < NUM_INVLOC; ++i) equipment.append(ResourceCombinationItem(player.InvBody[i], i));
	// Include every slot, including unused slots, and all magical/quest items;
	// no eligibility filter may hide capacity or attribute dependencies.
	for (int i = 0; i < InventoryGridCells; ++i) inventory.append(ResourceCombinationItem(player.InvList[i], i));
	for (int i = 0; i < MaxBeltItems; ++i) belt.append(ResourceCombinationItem(player.SpdList[i], i));
	for (int cell : player.InvGrid) grid.append(cell);
	state["equipment"] = equipment;
	state["inventory"] = inventory;
	state["belt"] = belt;
	state["inventory_grid"] = grid;
	py::dict ready = ResourceReadinessState(player);
	ready["block_enabled"] = player._pBlockFlag;
	ready["block_chance"] = player.GetBlockChance();
	state["readiness"] = ready;
	state["strength"] = player._pStrength;
	state["dexterity"] = player._pDexterity;
	state["magic"] = player._pMagic;
	state["vitality"] = player._pVitality;
	state["base_hp_fixed"] = player._pHPBase;
	state["base_max_hp_fixed"] = player._pMaxHPBase;
	return state;
}

int ResourceCombinationInventoryIndex(const Player &player, const ResourceItemIdentity &identity)
{
	int index = -1;
	for (int i = 0; i < player._pNumInv; ++i) {
		if (ResourceIdentity(player.InvList[i]) != identity || player.InvList[i].isEmpty()) continue;
		if (index != -1) return -2; // Full identity must still resolve uniquely.
		index = i;
	}
	return index;
}

const char *ApplyResourceCombinationEquip(Player &player, int index, py::list &commands)
{
	const Item candidate = player.InvList[index];
	const ResourceItemIdentity identity = ResourceIdentity(candidate);
	const ResourceArmorPlan plan = PlanResourceArmor(player, candidate, index);
	if (!plan.gear.valid || !plan.safe || !plan.canFit) return plan.reason;
	Player staged = ResourceUnequipSimulation(player);
	staged._pGold = player._pGold;
	if (!StageResourceArmorInventory(player, index, plan.gear, staged)) return "no_room_for_replaced_items";
	ApplyResourceArmorPlan(staged, plan.gear);
	RecalculateResourceCombinationPlayer(staged);
	if ((staged._pHitPoints >> 6) <= 0 || !staged.InvBody[plan.gear.target]._iStatFlag
	    || staged.GetArmor() != plan.gear.nextArmorClass
	    || staged._pHitPoints != plan.gear.nextCurrentHitPoints || staged._pMaxHP != plan.gear.nextMaxHitPoints)
		return "equipment_validation_failed";
	commands.append(py::make_tuple("equip", index, identity[0], identity[1], identity[2], identity[3]));
	player = std::move(staged);
	return nullptr;
}

const char *ApplyResourceCombinationRepair(Player &player, int slot, const ResourceItemIdentity &identity,
    int expectedDurability, int expectedPrice, int sourceGold, int maxGoldCost, py::list &commands)
{
	if (slot != INVLOC_HEAD && slot != INVLOC_HAND_LEFT && slot != INVLOC_HAND_RIGHT && slot != INVLOC_CHEST)
		return "invalid_item";
	Item &item = player.InvBody[slot];
	if (item.isEmpty() || ResourceIdentity(item) != identity || item._iDurability != expectedDurability)
		return "stale_item";
	const auto price = GetStoreRepairPrice(item);
	if (!price) return "unrepairable_item";
	if (*price != expectedPrice) return "stale_repair_quote";
	if (*price > player._pGold) return "no_money";
	if (*price > maxGoldCost - (sourceGold - player._pGold)) return "gold_cost_limit";
	// Same ordering as TryRepairStoreItem: repair first, then debit (whose
	// inventory compaction cannot invalidate the equipped item reference).
	item._iDurability = item._iMaxDur;
	if (TakePlayerGold(player, *price, false) != 0) return "personal_gold_inconsistent";
	commands.append(py::make_tuple("repair", slot, identity[0], identity[1], identity[2], identity[3], expectedDurability, *price));
	return nullptr;
}

py::dict PreviewResourceEquipmentCombinations(const py::list &sequences, int maxGoldCost)
{
	EnsureInGame("preview_resource_equipment_combinations");
	if (!gResourceOrdinaryArmorScope || !ResourceTownActionAllowed("preview_resource_equipment_combinations")
	    || MyPlayer->_pmode != PM_STAND || qtextflag)
		throw std::runtime_error("combination preview requires idle ordinary-armor town service");
	if (Stash.gold != 0) throw std::runtime_error("combination preview forbids shared-stash gold");
	if (maxGoldCost < 0 || maxGoldCost > MyPlayer->_pGold)
		throw std::invalid_argument("combination budget must be within personal gold");
	if (gResourceSeenSmithGeneration != gTownRestockSequence || gResourceSeenSmithTownSeed != DungeonSeeds[0])
		throw std::invalid_argument("Smith combination belongs to another town generation");
	const auto parsed = ParseResourceCombinations(sequences);
	const Player &source = *MyPlayer;
	py::dict result;
	result["version"] = ResourceCombinationVersion;
	result["max_gold_cost"] = maxGoldCost;
	result["batch_limit"] = ResourceCombinationBatchLimit;
	result["step_limit"] = ResourceCombinationStepLimit;
	result["truncated"] = false; // Over-limit inputs reject; never silently clip.
	py::dict sourceState = ResourceCombinationState(source);
	sourceState["episode_generation"] = gEpisodeGeneration;
	sourceState["town_seed"] = DungeonSeeds[0];
	sourceState["town_restock_sequence"] = gTownRestockSequence;
	result["source"] = sourceState;
	py::list previews;
	for (size_t candidateIndex = 0; candidateIndex < parsed.size(); ++candidateIndex) {
		Player staged = ResourceUnequipSimulation(source);
		staged._pGold = source._pGold;
		// ResourceUnequipSimulation copies primitive attributes, not derived
		// strength/DEX. Initialize before CanUseItem or store placement probes.
		RecalculateResourceCombinationPlayer(staged);
		std::vector<ResourceItemIdentity> purchased;
		py::list commands;
		const char *failure = nullptr;
		int failedStep = -1;
		for (size_t step = 0; step < parsed[candidateIndex].size(); ++step) {
			const auto &intent = parsed[candidateIndex][step];
			const auto &identity = intent.identity;
			if (intent.operation == "buy_equip" || intent.operation == "buy") {
				if (std::find(purchased.begin(), purchased.end(), identity) != purchased.end()) {
					failure = "duplicate_purchase";
				} else {
					int sourceIndex = -1;
					int currentIndex = -1;
					int availableIndex = 0;
					for (size_t i = 0; i < SmithItems.size(); ++i) {
						const auto itemIdentity = ResourceIdentity(SmithItems[i]);
						if (std::find(purchased.begin(), purchased.end(), itemIdentity) != purchased.end()) continue;
						if (!SmithItems[i].isEmpty() && itemIdentity == identity) {
							if (sourceIndex != -1) { failure = "ambiguous_stock_identity"; break; }
							sourceIndex = static_cast<int>(i);
							currentIndex = availableIndex;
						}
						++availableIndex;
					}
					if (!failure && sourceIndex < 0) failure = "stale_item";
					if (!failure && intent.operation == "buy" && intent.index != sourceIndex) failure = "stale_item";
					if (!failure) {
						Item item = SmithItems[sourceIndex];
						const int price = item._iIvalue;
						if (!IsOrdinaryResourceArmor(item) || price < 0) failure = "unsupported_item";
						else if (!staged.CanUseItem(item)) failure = "cannot_use";
						else if (price > staged._pGold) failure = "no_money";
						else if (price > maxGoldCost - (source._pGold - staged._pGold)) failure = "gold_cost_limit";
						else {
							item._iStatFlag = staged.CanUseItem(item);
							// Exactly the shared kernel's pre-payment capacity check.
							if (!StoreAutoPlaceForPlayer(staged, item, false, false)) failure = "no_room";
							else {
								item._iIdentified = false; // Only normal Smith stock is supported.
								if (TakePlayerGold(staged, price, false) != 0) failure = "personal_gold_inconsistent";
								else if (!StoreAutoPlaceForPlayer(staged, item, true, false)) failure = "no_room_after_payment";
								else {
									RecalculateResourceCombinationPlayer(staged);
									purchased.push_back(identity);
									commands.append(py::make_tuple("buy", "smith", currentIndex, identity[0], identity[1], identity[2], identity[3]));
									if (intent.operation == "buy_equip") {
										const int index = ResourceCombinationInventoryIndex(staged, identity);
										if (index >= 0) failure = ApplyResourceCombinationEquip(staged, index, commands);
										else {
											int equippedCount = 0;
											for (const Item &bodyItem : staged.InvBody)
												if (!bodyItem.isEmpty() && ResourceIdentity(bodyItem) == identity) ++equippedCount;
											if (index == -2 || equippedCount != 1) failure = "purchased_item_not_located";
										}
									}
								}
							}
						}
					}
				}
			} else if (intent.operation == "equip") {
				// Input index is an optimistic guard on the observed source, not
				// a promise that earlier gold debits preserve inventory ordering.
				if (intent.index < 0 || intent.index >= source._pNumInv
				    || ResourceIdentity(source.InvList[intent.index]) != identity) failure = "stale_item";
				else {
					const int index = ResourceCombinationInventoryIndex(staged, identity);
					if (index < 0) failure = index == -2 ? "ambiguous_inventory_identity" : "stale_item";
					else failure = ApplyResourceCombinationEquip(staged, index, commands);
				}
			} else if (intent.operation == "unequip") {
				const int slot = intent.index;
				if (!IsResourceUnequipSlot(staged, slot)) failure = "invalid_item";
				else if (ResourceIdentity(staged.InvBody[slot]) != identity) failure = "stale_item";
				else {
					const auto plan = PlanResourceUnequip(staged, slot);
					if (!plan.canFit) failure = "no_room";
					else if (!plan.safe) failure = "unsafe_life";
					else {
						const Item item = staged.InvBody[slot];
						if (!AutoPlaceItemInInventory(staged, item, false)) failure = "no_room";
						else {
							RemoveEquipment(staged, static_cast<inv_body_loc>(slot), false, false);
							RecalculateResourceCombinationPlayer(staged);
							if ((staged._pHitPoints >> 6) <= 0) failure = "unsafe_life";
							else commands.append(py::make_tuple("unequip", slot, identity[0], identity[1], identity[2], identity[3]));
						}
					}
				}
			} else if (intent.operation == "repair") {
				failure = ApplyResourceCombinationRepair(staged, intent.index, identity, intent.durability, intent.price,
				    source._pGold, maxGoldCost, commands);
			} else {
				// Only the terminal repair_retained shorthand reaches here.
				std::vector<std::pair<int, int>> repairs;
				for (const inv_body_loc slot : { INVLOC_HEAD, INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT, INVLOC_CHEST }) {
					const Item &item = staged.InvBody[slot];
					if (item.isEmpty() || item._iMaxDur == DUR_INDESTRUCTIBLE || item._iDurability >= 15) continue;
					if (item._iMaxDur < 15) { failure = "retained_item_max_durability"; break; }
					const auto price = GetStoreRepairPrice(item);
					if (!price) { failure = "unrepairable_item"; break; }
					repairs.emplace_back(*price, slot);
				}
				std::sort(repairs.begin(), repairs.end());
				if (!failure) for (const auto &[price, slot] : repairs) {
					const Item item = staged.InvBody[slot];
					failure = ApplyResourceCombinationRepair(staged, slot, ResourceIdentity(item), item._iDurability, price,
					    source._pGold, maxGoldCost, commands);
					if (failure) break;
				}
			}
			if (failure) { failedStep = static_cast<int>(step); break; }
		}
		py::dict preview;
		preview["index"] = candidateIndex;
		preview["valid"] = failure == nullptr;
		preview["reason"] = failure ? failure : "projected";
		preview["failed_step"] = failedStep < 0 ? py::none() : py::cast(failedStep);
		preview["gold_cost"] = source._pGold - staged._pGold;
		preview["gold_remaining"] = staged._pGold;
		preview["expanded_commands"] = commands;
		// Failure never grants an executable first command or a complete
		// projection. Expanded commands then document partial simulation only.
		preview["first_command"] = failure || commands.empty() ? py::none() : py::reinterpret_borrow<py::object>(commands[0]);
		if (!failure) {
			py::dict projected = ResourceCombinationState(staged);
			preview["projected_state"] = projected;
			preview["projected_readiness"] = projected["readiness"];
			preview["projected_equipment"] = projected["equipment"];
			preview["projected_inventory"] = projected["inventory"];
		} else {
			preview["projected_state"] = py::none();
			preview["projected_readiness"] = py::none();
			preview["projected_equipment"] = py::none();
			preview["projected_inventory"] = py::none();
		}
		previews.append(preview);
	}
	result["results"] = previews;
	return result;
}
