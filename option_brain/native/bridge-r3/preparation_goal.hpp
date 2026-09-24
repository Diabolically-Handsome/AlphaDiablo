#pragma once
#include <algorithm>
#include <array>

// Only unfinished goal progress is monotone. Surplus AC/damage may be traded
// once the approved floor is retained; legacy combat utility breaks ties.
using PreparationProgress = std::array<int, 5>;
inline PreparationProgress PreparationGoal(int armor, int low, int high, bool sword, bool shield)
{
	return { std::min(23, armor), std::min(9, low), std::min(14, high), int(sword), int(shield) };
}
inline bool PreparationPreserves(const PreparationProgress &before, const PreparationProgress &after)
{
	for (unsigned i = 0; i < before.size(); ++i)
		if (after[i] < before[i]) return false;
	return true;
}
