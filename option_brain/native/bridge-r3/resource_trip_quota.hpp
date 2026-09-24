// Pure count policy, shared by actual native guards and a standalone test.
inline bool ResourceTripQuotaAllows(bool limitsEnabled, int completed, int limit)
{
	return !limitsEnabled || completed < limit;
}
