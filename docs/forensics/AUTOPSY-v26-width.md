# v26-leg6 width-problem autopsy (2026-07-11, attribution of the launch failure)

Material: the full-32 per-seed archive `v26-G3-leg6.json` (108.2, paired +16.2, 11/32 wins) against
`v24-G3-leg7.json` (92.0, the archive of the anchor shared by earlier versions). Scalpel: per-seed pairing +
bucket-level behavior (number of farm windows / mean window tau / kills / mode composition / deaths).

## 1. Three pieces of pathological tissue

**Winning bucket (11 seeds, mean paired difference +83.2, median +66.3): deep farming inside the window**:
on rich maps v26 switched to "fewer windows, longer stays": 55.6 farm windows vs 78.2 in the archive, mean window tau
74.3 vs 48.4, kills 63.3 vs 34.1. Extreme cases: 7031 (16 windows @tau-bar 174 vs 73 windows @tau-bar 40, 367.4 vs
122.4, +245) and 7002 (17 windows @tau-bar 152 vs 85 windows @tau-bar 34, +229.5).
**The strategy is real, not evaluation noise.**

**Lesion 1 (6 seeds, -346.4 in total, 87% of all losses): misreading the map**:
- 7008 (-150.8): v26 died (the archive survived comfortably), kills 6 vs 65: collapsed right at the start of a rich
  map, the only "new death" among the three deaths.
- 7011 (-61.0): 107 farm windows in the whole episode with only 1 kill (26 in the archive): never engaged, a
  wandering-ghost map.
- 7017 (-71.9)/7010 (-28.9)/7009 (-17.2)/7013 (-16.6): **did not stay where it should have**: on these maps v26
  instead had more windows with shorter tau (7017: 86 windows @tau 33 vs 45 windows @tau 56 in the archive), misreading
  rich maps as poor ones.
- Direction: map reading fails both ways: rich maps read as poor (no staying), and on individual maps no engagement at
  all.

**Lesion 2 (15 seeds, median -4.2, band [-9.6, -0.4]): exploration friction on ordinary maps**:
the losing bucket has more windows overall (86.7 vs 77.5), fewer kills (34.7 vs 41.6) and slightly shorter tau. The
strategy pays a small "probing for riches" cost on ordinary maps. On its own this is at noise level, but the launch line
counts wins, so all 15 friction maps count as losses.

## 2. Arithmetic implications (from the launch line's point of view)

- The mean difference +16.2 met the line long ago; it is stuck at 11/32 wins < 18.
- Fixing lesion 1 (bringing the 6 maps back to a tie) would only push the mean to ~119, and the win count would stay
  at 11: **the key to launching is lesion 2**: the friction maps have to flip from -4 to +epsilon, at least 7 of them.
- That is: the width problem = a small execution tax that a big-win strategy pays on ordinary maps, at a median rate of
  -4. Flipping it is within reach, and the first hypothesis is **under-training** (leg 4->5->6 = 98.6->112.2->114.5
  shows no plateau).

## 3. Depth definition (a measurement of where the goal of "reliably breaking through to level 4" stands)

Full-32 depth distribution: v26-leg6 = {0:2, 1:26, 2:4}; the archive, v27-leg7 and v24-golden are all the same (median
1, l3=0). **The whole civilization lives on levels 0-2; no system has ever reached level 3.** Mechanism: under the
current returns, the per-tau yield of farming inside a window on level 1 is far higher than diving (+8/level), and the
frozen v22-H manager accordingly almost never chooses DIVE (0-1/episode). **Conclusion: level 4 is not a matter of
"not enough practice" but of "a different objective"**: width training will not and should not change depth; a level-4
demo needs a dive-oriented variant of the objective = an environment-side change, which under the hard rules needs a
case of its own.

## 4. Prescription (for PREREG-v28)

The cheapest prescription with zero new mechanisms = **oasis endurance**: continue training from the leg-6 checkpoint,
beta constant at 0.015625 (double backing from lesson 19 + v27's "cannot settle": never let go), skip-dry unchanged,
and add a width probe to the leg exams. If endurance cannot flip the friction maps, the verdict promotes the mechanism
prescriptions (the anchor follows the king / course sampling) to the workstation line; that is for the next version,
and this version only examines the "under-training hypothesis".
