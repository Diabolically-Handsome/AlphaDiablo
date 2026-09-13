// Read-only companion ABI. No engine initialization, transition, RNG or writes.
#include <cstdint>
#include "quests.h"

extern "C" __attribute__((visibility("default")))
std::uint64_t alphadiablo_butcher_status_v1()
{
    const auto &quest = devilution::Quests[devilution::Q_BUTCHER];
    return std::uint64_t(static_cast<std::uint8_t>(quest._qidx))
        | (std::uint64_t(static_cast<std::uint8_t>(quest._qactive)) << 8)
        | (std::uint64_t(quest._qlevel) << 16)
        | (std::uint64_t(devilution::currlevel) << 24)
        | (std::uint64_t(devilution::setlevel ? 1 : 0) << 32);
}
