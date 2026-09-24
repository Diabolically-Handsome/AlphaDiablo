// Exact native shift-click belt operation, without ambiguous headless UI pixels.
// Historical ManualAction is left intact for complete-prefix replay.
py::dict ManualActionR3(const std::string &kind, py::dict args)
{
    if (kind != "unbelt_exact") return ManualAction(kind, args);
    if (!gManualControl || !CanAcceptPlayerAction("manual_unbelt_exact") || MyPlayer == nullptr)
        return ResourceActionResult(false, "unavailable");
    const int index = py::cast<int>(args["index"]);
    const auto identity = py::cast<ResourceItemIdentity>(args["identity"]);
    if (!ManualCanUnbelt(index, identity))
        return ResourceActionResult(false, "unbelt_unavailable");
    Player &p = *MyPlayer;
    // These are the original inv.cpp CheckInvCut automatic belt branch calls.
    if (!AutoPlaceItemInInventory(p, p.SpdList[index]))
        return ResourceActionResult(false, "inventory_full");
    p.RemoveSpdBarItem(index);
    bool found = false;
    for (int i = 0; i < p._pNumInv; ++i)
        found |= ResourceIdentity(p.InvList[i]) == identity;
    if (!p.SpdList[index].isEmpty() || !found)
        throw std::runtime_error("unbelt native postcondition failed");
    return ResourceActionResult(true, "native_belt_transfer_verified");
}
