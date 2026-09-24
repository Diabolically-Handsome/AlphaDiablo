#pragma once

// Read-only counterpart of the non-set-level entrance branch in CheckQuests.
// The caller supplies native enum comparisons and visibility; no quest state,
// transition guard, RNG, map, or action queue is changed here.
struct QuestEntranceFacts {
    bool shareware;
    bool multiplayerQuests;
    bool inSetLevel;
    bool questPresent;
    bool betrayer;
    bool tileVisible;
    bool tileLit;
    bool inBounds;
    int currentLevel;
    int questLevel;
    int destinationSetLevel;
    int questStage;
};

constexpr bool ExportQuestEntrance(const QuestEntranceFacts &q)
{
    return !q.shareware && !q.multiplayerQuests && !q.inSetLevel
        && q.currentLevel > 0 && q.currentLevel == q.questLevel
        && q.destinationSetLevel > 0 && q.questPresent
        && (!q.betrayer || q.questStage >= 3)
        && q.inBounds && q.tileVisible && q.tileLit;
}
