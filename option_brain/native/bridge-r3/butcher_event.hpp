// Read-only result evidence. No enemy location, hidden health, or world mutation.
py::dict ButcherEvents()
{
    EnsureInGame("butcher_events");
    if (!gManualControl) throw std::runtime_error("manual mode is off");
    py::dict result;
    result["kills"] = MonsterKillCounts[MT_CLEAVER];
    result["quest_done"] = Quests[Q_BUTCHER]._qactive == QUEST_DONE;
    return result;
}
