# Smooth replay: the same Butcher kill, recorded properly

[Download the 60-fps MP4](https://github.com/Diabolically-Handsome/AlphaDiablo/releases/download/butcher-first-kill-20260913/butcher-first-kill-smooth-60fps.mp4)
· [Original milestone and research limitations](../2026-09-13-butcher-first-kill/README.md)

The first video was choppy because it sampled only once per four game ticks:
**5 distinct snapshots per second**, duplicated into a 10-fps file. The smooth
version records **every native tick at the unchanged 20-Hz simulation rate**,
then uses the engine's own animation and camera interpolation to render three
subframes per tick at **60 fps**. It is not simply the old images duplicated
more often; no optical-flow or generative frame interpolation is used.

The 90-second video contains a two-second opening still, two continuous earlier
gameplay excerpts (8 and 6 seconds), the full 71-second approach/encounter, and a
three-second ending hold. There are deliberate cuts between excerpts; it is not
an uninterrupted recording of the entire roughly 88-minute game. No audio.

## Verification

- Frozen gameplay engine, bridge and model files were unchanged.
- A recording-only dynamic-link observer calls the original `game_loop` exactly
  once, with the original argument, and preserves its return value.
- After each selected tick, a disposable fork child renders the copied state,
  without advancing gameplay. The engine's display-progress fractions are
  `0/128`, `42/128`, `85/128`. Display work stays outside the parent game state.
- **1,700 native ticks / 5,100 rendered frames**, including every tick of bridge
  steps 26,151–26,505. Output has 5,400 frames including the opening/ending holds.
- Final result, native quest state and all nine physical audit artifacts match
  the original winning run exactly: Butcher defeated, Warrior alive at 83/110 HP.
- This is another replay of development seed 2,133,013, **not an independent
  success or additional model training**. The original video/results are retained.

[Recording verification](capture-verification.json) · [Frame index and hashes](frame-index.json)
· [Media checksum](media.json) · [Evidence checker](verify.py)

```sh
python docs/milestones/2026-09-13-butcher-smooth-video/verify.py
# Optionally also check the downloaded video:
python docs/milestones/2026-09-13-butcher-smooth-video/verify.py /path/to/butcher-first-kill-smooth-60fps.mp4
```

The index identifies locally retained native frames; it does not distribute game
assets or a complete runtime. The original milestone's mixed learned/scripted
method, rule differences and evidence limitations remain unchanged.
