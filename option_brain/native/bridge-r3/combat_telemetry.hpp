// Read-only, present-visible telemetry. No RNG reads, commands or simulation.
py::dict ManualCombatTelemetry()
{
    EnsureInGame("manual_combat_telemetry");
    if (!gManualControl || MyPlayer == nullptr) throw std::runtime_error("manual mode required");
    const Player &p = *MyPlayer;
    py::dict d, hero;
    d["tick"] = gManualTicks;
    d["scene"] = py::make_tuple(ConceptualDungeonDepth(), setlevel ? static_cast<int>(setlvlnum) : 0);
    hero["mode"] = static_cast<int>(p._pmode);
    hero["frame"] = p.AnimInfo.currentFrame;
    hero["frame_counter"] = p.AnimInfo.tickCounterOfCurrentFrame;
    hero["attack_frame"] = p._pAFNum - 1;
    hero["x"] = p.position.tile.x;
    hero["y"] = p.position.tile.y;
    hero["hp_fixed"] = p._pHitPoints;
    const Point attackTile = p.position.tile + p._pdir;
    hero["attack_tile"] = py::make_tuple(attackTile.x, attackTile.y);
    d["hero"] = hero;
    py::list enemies;
    for (size_t i = 0; i < ActiveMonsterCount; ++i) {
        const unsigned id = ActiveMonsters[i];
        const Monster &m = Monsters[id];
        if (m.hasNoLife() || m.isPlayerMinion() || (m.flags & MFLAG_HIDDEN) != 0 || !ManualSeen(m.position.tile)) continue;
        py::dict e;
        e["id"] = id; e["king"] = m.type().type == MT_SKING;
        e["hp_fixed"] = m.hitPoints; e["max_hp_fixed"] = m.maxHitPoints;
        e["x"] = m.position.tile.x; e["y"] = m.position.tile.y;
        e["mode"] = static_cast<int>(m.mode);
        e["in_attack_cell"] = m.position.tile == attackTile;
        enemies.append(e);
    }
    d["enemies"] = enemies;
    return d;
}
