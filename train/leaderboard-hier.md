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
| ppo-hier-v22-h | 93.9 | 103.45 | 2/32 | 1.0 | hier; L3+ 0; kills 39.5 |
| ppo-flat-v22-f | 80.2 | 90.1 | 0/32 | 1.0 | flat+clock+BC(devil arm); kills 34.7; train-time 125 deflated -36% (lesson 8) |
| ppo-hier-v22-hbc | 38.5 | 29.0 | 0/32 | 1.0 | hier+manager-BC(insurance, P6); collapsed to FARM-only 3247:1; imitation anchor hurt |
| v23-golden | 77.0 | 91.39 | 3/32 | 1.0 | hier+learned-FARM; L3+ 0; kills 33.4; 换层率 0.0 override 0.0207 |
| v24-golden | 97.2 | 93.75 | 2/32 | 1.0 | hier+learned-FARM; L3+ 0; kills 41.2; 换层率 0.0005 override 0.0248 |
