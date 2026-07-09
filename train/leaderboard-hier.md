# Hierarchy board — manager over frozen options, 32 fixed seeds

Protocol: 3000 micro-steps, argmax + option masks, seeds 9000-9031,
idle machine, engine pinned, world = v20 rules (ladder + death price
+ auto stat-spend). Return = UNDISCOUNTED episode reward (the oracle
ledger). Reference rows are scripted policies via the same wrapper.

| run | ret mean | ret med | died | depth med | notes |
|---|---|---|---|---|---|
| wrapper-retire (scripted ref) | 36.9 | 29.0 | 0/32 | 1.0 | G3 reference |
| wrapper-rush (scripted ref) | -30.1 | 18.6 | 25/32 | 2.0 | G3 reference |
| teacher (scripted ref) | 101.5 | 91.9 | 25/32 | 2.0 | G3 reference |
