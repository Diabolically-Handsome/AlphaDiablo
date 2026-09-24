// R18-H (2026-09-07) sweep-v1: 宝箱/桶的原生观测通道。
//
// Included inside the bridge's private namespace, immediately before Observe()
// (EnsureEngineProcess / EnsureInGame / gInGame are already defined there).
//
// Chairman ruling 2026-09-07 14:00: the pair must play a complete Diablo, and
// today it cannot see a chest and never smashes a barrel.  This file adds the
// missing EYES only: one new default-false flag and one new raw observation key
// "objects" (每项带 floor_items 同口径的 visible = IsTileLit).  The HANDS already exist — ACTION_OPERATE reaches OperateChest for
// every chest type and StartAttack (via _oBreak == 1) for every barrel — so no
// new native action and no engine patch is required.
//
// Off path: gObjectObservation stays false, Observe() appends nothing, and the
// observation dict is byte-identical to the frozen one.

// 默认关闭。开启后 Observe() 追加 raw["objects"];关闭时观测字典逐字不变。
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

// 只公开与"捡东西"相关的物体类别。神龛/书架/杠杆等一律不出现,
// 免得把普通交互变成全知自动操作(与 progression_targets 白名单同口径)。
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
		// R18-H review round (2026-09-07): 与 floor_items 同口径的可见性标签。
		// 通道本身是全图的(和 floor_items 一样),由 SweepService 只把"曾经
		// 见过亮光"的物体记入候选,未点亮的地图区域仍然属于部分可观测环境。
		entry["visible"] = IsTileLit(object.position);
		// canInteractWith() == selectionRegion != None,即现代引擎里 _oSelFlag
		// 的替代:OperateChest 开箱后置 None,BreakBarrel 碎桶后同样置 None,
		// 两个函数入口也都用它作 "还能操作吗" 的判据。因此
		// interactable=false 恰好等于 "已开/已碎/不可再操作"。
		entry["interactable"] = object.canInteractWith();
		entry["solid"] = object._oSolidFlag;
		objects.append(entry);
	}
	return objects;
}
