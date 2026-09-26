# R10 launch record (2026-08-27)

- Case: R10 depth-economy reform (economy v2 and 32 parallel environments). Approved 2026-08-27.
- Pre-registration: [r10-PREREG-FROZEN-20260827.md](r10-PREREG-FROZEN-20260827.md).
- Launch: a bare call of `run_r10_economy.py` (driver not published).
- Pool accounting: consumes 2_114/2_115 (anchor and exam); 2_116 held until after the verdict. Engineering seeds
  60-75 and 555-586 were used for G0 (not formal pools; the embargoed formal range was not touched).
- Fuses: training 21,600 s; each evaluation 7,200 s.

## Addendum 1 (same day, amendment 1)

- Problem: training ran at about 45 steps/min, 1/34 of the 1,554 measured in the stress test. Root cause: the 32
  environments step in lockstep at the manager-step level, and under economy v2 the learned policy's window
  lengths are extremely uneven (dive windows of a few ticks against farm windows of over a thousand ticks), so
  every step waits for the slowest environment. The stress test used an untrained policy with uniform windows
  and did not expose this. Asynchronous collection became an R11 infrastructure topic.
- Early scientific reading (not a verdict): at 10k steps the last 200 games already reached l3+ in 39% of games
  (anchor 11-13%); depth histogram {1: 23, 2: 99, 3: 56, 4: 20, 5: 2}; death rate 98%. The price list works.
- Amendment: total-steps 327,680 -> 12,288 (schedule only, science unchanged). The anchor lines had already been
  burned twice and certified bit for bit, so the relaunch skipped the re-burn. Approved 2026-08-27.
- GPU note: the bottleneck is synchronisation, the engine and regeneration, not compute; a (64,64) network is
  slower on a GPU. A batched inference server was noted as an idea for R11 or later.
