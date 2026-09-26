// R18-H (2026-09-07) sweep-v1: native observation channel for chests and barrels.
//
// Included inside the bridge's private namespace, immediately before Observe()
// (EnsureEngineProcess / EnsureInGame / gInGame are already defined there).
//
// Design goal (2026-09-07): the agent must be able to finish a complete game, and
// today it cannot see a chest and never smashes a barrel.  This file adds the
// missing EYES only: one new default-false flag and one new raw observation key
// "objects" (each entry carries visible = IsTileLit, the same definition as floor_items).  The HANDS already exist — ACTION_OPERATE reaches OperateChest for
// every chest type and StartAttack (via _oBreak == 1) for every barrel — so no
// new native action and no engine patch is required.
//
// Off path: gObjectObservation stays false, Observe() appends nothing, and the
// observation dict is byte-identical to the frozen one.

// Off by default. When on, Observe() appends raw["objects"]; when off the observation dict is unchanged byte for byte.
bool gObjectObservation = false;

void ConfigureObjectObservation(bool enabled)
{
	EnsureEngineProcess("configure_object_observation");
	// Mirrors ConfigureResourceProtocol: an observation channel may not appear
	// or vanish mid-episode, or a service would read two different worlds.
	if (gInGame && enabled != gObjectObservation)
		throw std::runtime_error("object observation may change only between episodes");
	gObjectObservation = enabled;
}

// Only expose the object kinds relevant to "picking things up". Shrines, bookcases, levers and the like never appear,
// so ordinary interactions do not turn into omniscient automation (same definition as the progression_targets allowlist).
const char *SweepObjectKind(const Object &object)
{
	switch (object._otype) {
	case OBJ_CHEST1:
	case OBJ_CHEST2:
	case OBJ_CHEST3:
		return "chest";
	case OBJ_TCHEST1:
	case OBJ_TCHEST2:
	case OBJ_TCHEST3:
		return "chest_trapped";
	case OBJ_SARC:
		return "sarcophagus";
	case OBJ_BARREL:
		return "barrel";
	case OBJ_BARRELEX:
		return "barrel_explosive";
	default:
		return nullptr;
	}
}

py::list ObserveSweepObjects()
{
	py::list objects;
	for (int i = 0; i < ActiveObjectCount; i++) {
		const Object &object = Objects[ActiveObjects[i]];
		const char *kind = SweepObjectKind(object);
		if (kind == nullptr)
			continue;
		py::dict entry;
		entry["x"] = static_cast<int>(object.position.x);
		entry["y"] = static_cast<int>(object.position.y);
		entry["kind"] = kind;
		// R18-H review round (2026-09-07): visibility label with the same definition as floor_items.
		// The channel itself covers the whole map (like floor_items); SweepService only records objects that have "ever
		// been seen lit" as candidates, so unlit map regions keep the environment partially observable.
		entry["visible"] = IsTileLit(object.position);
		// canInteractWith() == selectionRegion != None, i.e. the modern engine's replacement for _oSelFlag:
		// OperateChest sets it to None after opening a chest, BreakBarrel does the same after smashing a barrel,
		// and both functions also use it at entry as the "can this still be operated" test. Therefore
		// interactable=false means exactly "already opened/smashed/no longer operable".
		entry["interactable"] = object.canInteractWith();
		entry["solid"] = object._oSolidFlag;
		objects.append(entry);
	}
	return objects;
}
