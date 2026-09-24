#pragma once

// Unknown affixes remain unknown. Expanding legal shopping does not identify
// an item, generate stock, alter a quote, or waive its attribute requirements.
inline bool ShopCatalogQualityAllowed(bool expanded, bool ordinary, bool identified)
{
	return ordinary || (expanded && identified);
}
constexpr int ResourcePremiumIndexBase = 1000;
